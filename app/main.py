"""findme HTTP API + static UI.

Local-only tool: you give it a folder of photos, it detects and clusters every face,
and you browse the resulting "people". An optional selfie re-ranks the people so the
most likely match for you floats to the top. Nothing leaves the machine.
"""

from __future__ import annotations

import base64
import io
import os
import re
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from .cluster import apply_names, cluster_centroids, rank_clusters_by_selfie, recluster
from .config import BASE_DIR, DATA_DIR
from .engine import IMAGE_EXTS, embed_primary_face
from .pipeline import run_scan
from .store import Job, store

COUNT_CAP = 50000  # stop counting images past this — keeps huge trees responsive

# One scan at a time — the model + CPU are the bottleneck and shared.
_scan_lock = threading.Lock()

DATA_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR = BASE_DIR / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load_existing()  # restore results of past scans on boot
    yield


app = FastAPI(title="findme", lifespan=lifespan)


# ----------------------------- request models -----------------------------
class ScanRequest(BaseModel):
    source: str  # absolute path to a local folder of photos


# ------------------------------- helpers -----------------------------------
def _job_or_404(job_id: str) -> Job:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


def _face_url(job: Job, face_id: int) -> str:
    return f"/data/jobs/{job.id}/faces/{face_id}.jpg"


def _thumb_url(job: Job, photo_id: int) -> str:
    return f"/data/jobs/{job.id}/thumbs/{photo_id}.jpg"


def _job_summary(job: Job) -> dict:
    return {
        "id": job.id,
        "source": job.source,
        "status": job.status,
        "message": job.message,
        "total": job.total,
        "processed": job.processed,
        "num_faces": job.num_faces,
        "num_clusters": job.num_clusters,
        "num_unsorted": len(job.unsorted),
        "params": job.params,
        "error": job.error,
    }


def _clusters_payload(job: Job, limit: int, offset: int) -> dict:
    clusters = [job.clusters[cid] for cid in sorted(job.clusters)]
    window = clusters[offset : offset + limit]
    out = [
        {
            "cluster_id": c.id,
            "name": c.name,
            "face_count": len(c.face_ids),
            "photo_count": len(c.photo_ids),
            "rep_face_url": _face_url(job, c.rep_face_id),
            "sample_face_urls": [_face_url(job, fid) for fid in c.face_ids[:5]],
        }
        for c in window
    ]
    return {"total": len(clusters), "num_unsorted": len(job.unsorted), "clusters": out}


def _read_upload_bgr(raw: bytes) -> np.ndarray | None:
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            rgb = np.asarray(im)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# -------------------------------- routes -----------------------------------
@app.post("/api/scan")
def start_scan(req: ScanRequest) -> dict:
    if not _scan_lock.acquire(blocking=False):
        raise HTTPException(409, "A scan is already running. Please wait for it to finish.")
    try:
        job = Job(id=uuid.uuid4().hex[:12], source=req.source.strip())
        store.add(job)

        def _worker() -> None:
            try:
                run_scan(job)
            finally:
                _scan_lock.release()

        threading.Thread(target=_worker, daemon=True).start()
        return {"job_id": job.id}
    except Exception:
        _scan_lock.release()
        raise


@app.get("/api/jobs")
def list_jobs() -> dict:
    jobs = sorted(store.all(), key=lambda j: j.id)
    return {"jobs": [_job_summary(j) for j in jobs]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return _job_summary(_job_or_404(job_id))


@app.get("/api/jobs/{job_id}/clusters")
def get_clusters(job_id: str, limit: int = 2000, offset: int = 0) -> dict:
    return _clusters_payload(_job_or_404(job_id), limit, offset)


class ReclusterRequest(BaseModel):
    min_det_score: float | None = None
    min_face_px: int | None = None
    distance: float | None = None
    merge_distance: float | None = None
    assign_distance: float | None = None
    margin: float | None = None


@app.post("/api/jobs/{job_id}/recluster")
def do_recluster(job_id: str, req: ReclusterRequest) -> dict:
    job = _job_or_404(job_id)
    if job.embeddings is None or job.num_faces == 0:
        raise HTTPException(409, "This scan has no cached faces to re-group.")
    kwargs = {k: v for k, v in req.model_dump().items() if v is not None}
    recluster(job, **kwargs)
    job.save()
    return _clusters_payload(job, limit=2000, offset=0)


class NameRequest(BaseModel):
    name: str


@app.post("/api/jobs/{job_id}/clusters/{cluster_id}/name")
def set_name(job_id: str, cluster_id: int, req: NameRequest) -> dict:
    """Name a person. Anchored to the cluster's representative face so it survives a
    re-group (the name follows that face into whatever cluster it lands in)."""
    job = _job_or_404(job_id)
    cluster = job.clusters.get(cluster_id)
    if cluster is None:
        raise HTTPException(404, "Unknown cluster")
    name = req.name.strip()
    if name:
        job.names[cluster.rep_face_id] = name
    else:
        job.names.pop(cluster.rep_face_id, None)
    cluster.name = name
    job.save()
    return {"cluster_id": cluster_id, "name": name}


_SAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _safe_folder(name: str, fallback: str) -> str:
    cleaned = _SAFE.sub(" ", name).strip().strip(".")
    return cleaned[:120] or fallback


class ExportRequest(BaseModel):
    dest: str
    named_only: bool = False
    min_photos: int = 1


@app.post("/api/jobs/{job_id}/export")
def export_folders(job_id: str, req: ExportRequest) -> dict:
    """Copy each person's photos into dest/<person name>/. A photo with several people is
    copied into each of their folders. Originals are never moved."""
    job = _job_or_404(job_id)
    dest = Path(req.dest).expanduser()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise HTTPException(400, f"Can't create destination folder: {e}")

    selected = [
        c
        for c in job.clusters.values()
        if len(c.photo_ids) >= req.min_photos and (not req.named_only or c.name)
    ]
    if not selected:
        raise HTTPException(422, "No people match the export filter (try naming some, or lower the photo minimum).")

    used: dict[str, int] = {}
    people = 0
    files = 0
    for c in selected:
        folder_name = _safe_folder(c.name, f"pessoa_{c.id + 1}")
        # de-dupe folder names (two people could share a fallback/name)
        if folder_name in used:
            used[folder_name] += 1
            folder_name = f"{folder_name} ({used[folder_name]})"
        else:
            used[folder_name] = 1
        folder = dest / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        people += 1
        for pid in c.photo_ids:
            src = Path(job.photos[pid].path)
            if not src.exists():
                continue
            target = folder / src.name
            if not target.exists():
                try:
                    shutil.copy2(src, target)
                    files += 1
                except OSError:
                    continue
    return {"dest": str(dest), "people": people, "files": files}


@app.get("/api/jobs/{job_id}/unsorted")
def get_unsorted(job_id: str, limit: int = 300) -> dict:
    """Faces we couldn't confidently place (blurry / side-on / tiny)."""
    job = _job_or_404(job_id)
    out = []
    for fid in job.unsorted[:limit]:
        face = job.faces.get(fid)
        if not face:
            continue
        out.append(
            {
                "face_id": fid,
                "face_url": _face_url(job, fid),
                "photo_id": face.photo_id,
                "thumb_url": _thumb_url(job, face.photo_id),
                "original_url": f"/api/jobs/{job.id}/photos/{face.photo_id}/original",
            }
        )
    return {"total": len(job.unsorted), "faces": out}


@app.get("/api/jobs/{job_id}/metrics")
def get_metrics(job_id: str) -> dict:
    """Numeric audit of clustering quality — no face images, safe to share.

    Per cluster: size, quality, how tight it is (intra distance to centroid) and how far
    the nearest other person is. Plus the most 'suspicious' faces (far from their own
    centroid = likely a wrong grouping)."""
    job = _job_or_404(job_id)
    if job.embeddings is None or not job.clusters:
        return {"job": _job_summary(job), "clusters": [], "suspicious_faces": []}

    cents = cluster_centroids(job)
    cids = sorted(cents)
    cmat = np.vstack([cents[c] for c in cids])

    # nearest-other-centroid distance per cluster
    cc = 1.0 - (cmat @ cmat.T)
    np.fill_diagonal(cc, np.inf)
    nearest_other = cc.min(axis=1)

    clusters_out = []
    suspicious: list[dict] = []
    for row, cid in enumerate(cids):
        c = job.clusters[cid]
        emb = job.embeddings[c.face_ids]
        d_own = 1.0 - (emb @ cents[cid])
        det = [job.faces[f].det_score for f in c.face_ids]
        sizes = [min(job.faces[f].bbox[2] - job.faces[f].bbox[0],
                     job.faces[f].bbox[3] - job.faces[f].bbox[1]) for f in c.face_ids]
        clusters_out.append(
            {
                "cluster_id": cid,
                "faces": len(c.face_ids),
                "photos": len(c.photo_ids),
                "mean_det": round(float(np.mean(det)), 3),
                "mean_face_px": int(np.mean(sizes)),
                "intra_mean": round(float(d_own.mean()), 3),
                "intra_max": round(float(d_own.max()), 3),
                "nearest_other": round(float(nearest_other[row]), 3),
            }
        )
        # flag faces sitting far from their own person's centroid (likely a wrong grouping)
        for f, dv, px, ds in zip(c.face_ids, d_own, sizes, det):
            if dv >= 0.6:
                suspicious.append(
                    {
                        "face_id": int(f),
                        "cluster_id": cid,
                        "photo": job.photos[job.faces[f].photo_id].name,
                        "dist_to_own": round(float(dv), 3),
                        "det_score": ds,
                        "face_px": int(px),
                    }
                )

    clusters_out.sort(key=lambda x: x["photos"], reverse=True)
    suspicious.sort(key=lambda x: x["dist_to_own"], reverse=True)
    return {
        "job": _job_summary(job),
        "cluster_count": len(cids),
        "clusters": clusters_out,
        "suspicious_faces": suspicious[:100],
    }


@app.get("/api/jobs/{job_id}/report")
def build_report(job_id: str, max_clusters: int = 120, per_cluster: int = 8) -> dict:
    """Write a self-contained HTML montage (face crops inlined) to the job folder so it
    can be opened locally and reviewed. Returns its URL under /data."""
    job = _job_or_404(job_id)

    def data_uri(path: Path) -> str:
        try:
            return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()
        except OSError:
            return ""

    cids = sorted(job.clusters, key=lambda c: len(job.clusters[c].photo_ids), reverse=True)
    sections = []
    for cid in cids[:max_clusters]:
        c = job.clusters[cid]
        thumbs = "".join(
            f'<img src="{data_uri(job.face_path(fid))}" title="face {fid}">'
            for fid in c.face_ids[:per_cluster]
        )
        sections.append(
            f'<section><h2>Person {cid} · {len(c.photo_ids)} photos · '
            f'{len(c.face_ids)} faces</h2><div class="row">{thumbs}</div></section>'
        )
    uns = "".join(
        f'<img src="{data_uri(job.face_path(fid))}" title="face {fid}">' for fid in job.unsorted[:120]
    )
    if uns:
        sections.append(f'<section><h2>Unsorted · {len(job.unsorted)} faces</h2><div class="row">{uns}</div></section>')

    html = f"""<!doctype html><meta charset=utf-8><title>findme audit — {job.id}</title>
<style>body{{font:15px -apple-system,system-ui,sans-serif;margin:24px;color:#1d1d1f}}
h1{{font-size:28px}}h2{{font-size:16px;margin:24px 0 8px}}
.row{{display:flex;flex-wrap:wrap;gap:6px}}img{{width:88px;height:88px;object-fit:cover;border-radius:8px;border:1px solid #e0e0e0}}
.meta{{color:#7a7a7a}}</style>
<h1>findme audit report</h1>
<p class=meta>{job.source}<br>{job.num_clusters} people · {len(job.unsorted)} unsorted · {job.num_faces} faces · {job.total} photos<br>params: {job.params}</p>
{''.join(sections)}"""
    out = job.dir / "report.html"
    out.write_text(html)
    return {"url": f"/data/jobs/{job.id}/report.html", "clusters_shown": min(len(cids), max_clusters)}


@app.get("/api/jobs/{job_id}/clusters/{cluster_id}")
def get_cluster(job_id: str, cluster_id: int) -> dict:
    job = _job_or_404(job_id)
    cluster = job.clusters.get(cluster_id)
    if cluster is None:
        raise HTTPException(404, "Unknown cluster")

    # For each photo this person appears in, hand back the box(es) to highlight,
    # normalized to 0..1 so the frontend can overlay regardless of thumbnail size.
    photos = []
    for pid in cluster.photo_ids:
        photo = job.photos[pid]
        boxes = []
        for fid in cluster.face_ids:
            face = job.faces[fid]
            if face.photo_id != pid:
                continue
            x1, y1, x2, y2 = face.bbox
            boxes.append(
                {
                    "x": x1 / photo.width,
                    "y": y1 / photo.height,
                    "w": (x2 - x1) / photo.width,
                    "h": (y2 - y1) / photo.height,
                }
            )
        photos.append(
            {
                "photo_id": pid,
                "name": photo.name,
                "thumb_url": _thumb_url(job, pid),
                "original_url": f"/api/jobs/{job.id}/photos/{pid}/original",
                "boxes": boxes,
            }
        )
    return {
        "cluster_id": cluster.id,
        "name": cluster.name,
        "face_count": len(cluster.face_ids),
        "photo_count": len(cluster.photo_ids),
        "rep_face_url": _face_url(job, cluster.rep_face_id),
        "photos": photos,
    }


@app.post("/api/jobs/{job_id}/match")
async def match_selfie(job_id: str, file: UploadFile = File(...)) -> dict:
    job = _job_or_404(job_id)
    if job.status != "done":
        raise HTTPException(409, "Scan is not finished yet.")

    raw = await file.read()
    img = _read_upload_bgr(raw)
    if img is None:
        raise HTTPException(400, "Could not read that image.")

    query = embed_primary_face(img)
    if query is None:
        raise HTTPException(422, "No face found in the selfie. Try a clearer, front-facing photo.")

    ranked = rank_clusters_by_selfie(job, query)
    return {"matches": ranked}


@app.get("/api/browse")
def browse(path: str | None = None) -> dict:
    """List the sub-folders of a directory so the UI can offer a folder picker.
    Defaults to the user's home. Read-only; only directory names are returned."""
    base = (Path(path).expanduser() if path else Path.home())
    try:
        base = base.resolve()
    except Exception:
        raise HTTPException(400, "Bad path")
    if not base.is_dir():
        raise HTTPException(404, "Not a folder")

    dirs = []
    try:
        for entry in os.scandir(base):
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    dirs.append({"name": entry.name, "path": entry.path})
            except OSError:
                continue
    except PermissionError:
        raise HTTPException(403, "Permission denied for that folder")

    dirs.sort(key=lambda d: d["name"].lower())
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "name": base.name or str(base), "parent": parent, "dirs": dirs}


@app.get("/api/count")
def count_images(path: str) -> dict:
    """Count image files in a folder and ALL its sub-folders (bounded)."""
    base = Path(path).expanduser()
    if not base.is_dir():
        raise HTTPException(404, "Not a folder")

    n = 0
    capped = False
    for root, subdirs, files in os.walk(base):
        subdirs[:] = [d for d in subdirs if not d.startswith(".")]  # skip hidden trees
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXTS:
                n += 1
                if n >= COUNT_CAP:
                    capped = True
                    break
        if capped:
            break
    return {"count": n, "capped": capped}


@app.get("/api/jobs/{job_id}/photos/{photo_id}/original")
def get_original(job_id: str, photo_id: int) -> FileResponse:
    job = _job_or_404(job_id)
    photo = job.photos.get(photo_id)
    if photo is None:
        raise HTTPException(404, "Unknown photo")
    return FileResponse(photo.path, filename=photo.name)


# Static: face crops + photo thumbnails live under DATA_DIR; the UI under web/.
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
