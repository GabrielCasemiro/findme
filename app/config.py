"""Paths and tunables.

Works both as a dev app (run from the repo) and as a packaged desktop app (a frozen
PyInstaller build). When frozen: read-only assets (web/, the bundled face model) live
inside the bundle, and mutable app data goes to a per-user writable location
(%LOCALAPPDATA%/findme on Windows) instead of alongside the executable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)


def _bundle_dir() -> Path:
    """Directory holding bundled read-only assets (web/, model)."""
    if FROZEN:
        # PyInstaller extracts data next to the executable (onedir) or to _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def _data_root() -> Path:
    """Writable location for generated data (faces, thumbs, embeddings, jobs)."""
    env = os.environ.get("FINDME_DATA")
    if env:
        return Path(env)
    if FROZEN:
        if sys.platform == "win32":
            base = Path(os.environ.get("LOCALAPPDATA") or Path.home())
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
        return base / "findme"
    return Path(__file__).resolve().parent.parent / "data"


BASE_DIR = _bundle_dir()
DATA_DIR = _data_root()
JOBS_DIR = DATA_DIR / "jobs"
WEB_DIR = BASE_DIR / "web"


def setup_model_home() -> None:
    """Point InsightFace at the model bundled with a frozen build so it works offline
    (no ~300 MB download on the user's machine). No-op in dev."""
    if not FROZEN:
        return
    home = _bundle_dir() / "insightface_home"
    if (home / "models").exists():
        os.environ.setdefault("INSIGHTFACE_HOME", str(home))

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
