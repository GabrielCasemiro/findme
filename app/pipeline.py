"""Scan a photo source end to end: detect faces, write crops + thumbnails, embed,
then cluster. Runs in a background thread; updates job progress as it goes."""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np

from .cluster import cluster_job
from .config import FACE_THUMB_MAX, MIN_DET_SCORE, PHOTO_THUMB_MAX
from .engine import IMAGE_EXTS, crop_face, detect_faces, load_image_bgr, save_jpeg
from .store import FaceRec, Job, PhotoRec


def list_images(folder: Path) -> list[Path]:
    """All decodable image files under a folder, recursively, in stable order."""
    files = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]
    return sorted(files)


def run_scan(job: Job) -> None:
    """Full pipeline for one job. Sets status to done/error and persists results."""
    try:
        folder = Path(job.source).expanduser()
        if not folder.is_dir():
            raise ValueError(f"Not a folder: {folder}")

        paths = list_images(folder)
        job.total = len(paths)
        job.status = "scanning"
        job.message = "Detecting faces…"

        if not paths:
            raise ValueError("No images found in that folder.")

        embeddings: list[np.ndarray] = []
        next_face_id = 0

        for photo_id, path in enumerate(paths):
            img = load_image_bgr(path)
            if img is None:
                job.processed += 1
                continue

            h, w = img.shape[:2]
            faces = detect_faces(img, min_score=MIN_DET_SCORE)

            save_jpeg(img, job.thumb_path(photo_id), max_side=PHOTO_THUMB_MAX)
            job.photos[photo_id] = PhotoRec(
                id=photo_id, path=str(path), name=path.name, width=w, height=h, num_faces=len(faces)
            )

            for df in faces:
                fid = next_face_id
                next_face_id += 1
                save_jpeg(crop_face(img, df.bbox), job.face_path(fid), max_side=FACE_THUMB_MAX)
                job.faces[fid] = FaceRec(
                    id=fid,
                    photo_id=photo_id,
                    bbox=list(df.bbox),
                    det_score=round(df.det_score, 4),
                )
                embeddings.append(df.embedding)

            job.processed += 1
            if job.processed % 25 == 0:
                job.num_faces = len(job.faces)
                job.save()

        job.embeddings = (
            np.vstack(embeddings).astype(np.float32)
            if embeddings
            else np.zeros((0, 512), dtype=np.float32)
        )
        job.num_faces = len(job.faces)

        job.status = "clustering"
        job.message = "Grouping faces into people…"
        job.save()
        cluster_job(job)

        job.status = "done"
        job.message = f"{job.num_clusters} people across {job.num_faces} faces."
        job.save()
    except Exception as exc:  # surface any failure to the UI instead of dying silently
        job.status = "error"
        job.error = str(exc)
        job.message = "Scan failed."
        traceback.print_exc()
        try:
            job.save()
        except Exception:
            pass
