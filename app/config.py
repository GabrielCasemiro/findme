"""Paths and tunables. Everything lives under ./data (gitignored)."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("FINDME_DATA", BASE_DIR / "data"))
JOBS_DIR = DATA_DIR / "jobs"

# Face-crop and photo-thumbnail sizes (longest side, px).
FACE_THUMB_MAX = 320
PHOTO_THUMB_MAX = 1280

# Detection confidence floor when scanning a library.
MIN_DET_SCORE = 0.5

# --- Clustering (quality-aware) --------------------------------------------
# Only faces this confident AND this big (min side, px) become clustering anchors;
# weaker faces are assigned afterward or left unsorted.
CLUSTER_MIN_DET = 0.62
CLUSTER_MIN_FACE_PX = 60

# Cosine-distance thresholds. Same-identity ArcFace embeddings usually sit well under
# ~0.5 apart. `DISTANCE` groups anchors; `MERGE_DISTANCE` fuses near-duplicate identities;
# `ASSIGN_DISTANCE` is the cap for attaching a weak face to an existing person; `MARGIN`
# is how much closer the best identity must be than the runner-up (guards against a
# blurry face grabbing the wrong person).
CLUSTER_DISTANCE = 0.50
CLUSTER_MERGE_DISTANCE = 0.45
CLUSTER_ASSIGN_DISTANCE = 0.58
CLUSTER_MARGIN = 0.08
