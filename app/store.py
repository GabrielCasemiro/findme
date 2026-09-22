"""In-memory job state with on-disk persistence.

A "job" is one scan of one photo source. Its faces/thumbnails live under
data/jobs/<id>/ and its metadata is mirrored to index.json so results survive a
server restart. Face embeddings are kept in a single (N, 512) array whose row index
equals the face id (ids are assigned sequentially per job).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import JOBS_DIR

Status = str  # "queued" | "scanning" | "clustering" | "done" | "error"


@dataclass
class PhotoRec:
    id: int
    path: str
    name: str
    width: int
    height: int
    num_faces: int = 0


@dataclass
class FaceRec:
    id: int
    photo_id: int
    bbox: list[int]  # x1, y1, x2, y2
    det_score: float
    cluster_id: int = -1


@dataclass
class ClusterRec:
    id: int
    face_ids: list[int]
    photo_ids: list[int]
    rep_face_id: int
    name: str = ""


@dataclass
class Job:
    id: str
    source: str
    status: Status = "queued"
    message: str = ""
    total: int = 0
    processed: int = 0
    num_faces: int = 0
    num_clusters: int = 0
    error: str | None = None
    photos: dict[int, PhotoRec] = field(default_factory=dict)
    faces: dict[int, FaceRec] = field(default_factory=dict)
    clusters: dict[int, ClusterRec] = field(default_factory=dict)
    unsorted: list[int] = field(default_factory=list)  # face ids not confidently placed
    params: dict = field(default_factory=dict)  # clustering params last used
    names: dict[int, str] = field(default_factory=dict)  # face_id -> name (anchors names to faces)
    # not persisted to index.json (stored as .npy instead)
    embeddings: np.ndarray | None = field(default=None, repr=False)

    # --- paths ---
    @property
    def dir(self) -> Path:
        return JOBS_DIR / self.id

    @property
    def faces_dir(self) -> Path:
        return self.dir / "faces"

    @property
    def thumbs_dir(self) -> Path:
        return self.dir / "thumbs"

    def face_path(self, face_id: int) -> Path:
        return self.faces_dir / f"{face_id}.jpg"

    def thumb_path(self, photo_id: int) -> Path:
        return self.thumbs_dir / f"{photo_id}.jpg"

    # --- persistence ---
    def to_index(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "status": self.status,
            "message": self.message,
            "total": self.total,
            "processed": self.processed,
            "num_faces": self.num_faces,
            "num_clusters": self.num_clusters,
            "error": self.error,
            "unsorted": self.unsorted,
            "params": self.params,
            "names": {str(k): v for k, v in self.names.items()},
            "photos": [asdict(p) for p in self.photos.values()],
            "faces": [asdict(f) for f in self.faces.values()],
            "clusters": [asdict(c) for c in self.clusters.values()],
        }

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "index.json.tmp"
        tmp.write_text(json.dumps(self.to_index()))
        tmp.replace(self.dir / "index.json")
        if self.embeddings is not None:
            np.save(self.dir / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, job_dir: Path) -> "Job | None":
        index = job_dir / "index.json"
        if not index.exists():
            return None
        d = json.loads(index.read_text())
        job = cls(
            id=d["id"],
            source=d["source"],
            status=d["status"],
            message=d.get("message", ""),
            total=d.get("total", 0),
            processed=d.get("processed", 0),
            num_faces=d.get("num_faces", 0),
            num_clusters=d.get("num_clusters", 0),
            error=d.get("error"),
        )
        job.unsorted = d.get("unsorted", [])
        job.params = d.get("params", {})
        job.names = {int(k): v for k, v in d.get("names", {}).items()}
        job.photos = {p["id"]: PhotoRec(**p) for p in d.get("photos", [])}
        job.faces = {f["id"]: FaceRec(**f) for f in d.get("faces", [])}
        job.clusters = {c["id"]: ClusterRec(**c) for c in d.get("clusters", [])}
        emb = job_dir / "embeddings.npy"
        if emb.exists():
            job.embeddings = np.load(emb)
        # A job interrupted mid-scan on a previous run can't resume; mark it failed.
        if job.status in ("queued", "scanning", "clustering"):
            job.status = "error"
            job.error = "Interrupted before completion — please re-scan."
        return job


class JobStore:
    """Thread-safe registry of jobs, backed by the jobs directory on disk."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def load_existing(self) -> None:
        if not JOBS_DIR.exists():
            return
        for d in sorted(JOBS_DIR.iterdir()):
            if d.is_dir():
                job = Job.load(d)
                if job:
                    self._jobs[job.id] = job

    def add(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())


store = JobStore()
