# PyInstaller spec for the findme desktop app.
# Build:  pyinstaller findme.spec --noconfirm
# Produces a one-folder app under dist/findme/ (findme[.exe] inside).
#
# The heavy ML deps (insightface / onnxruntime / scikit-learn / scipy / scikit-image /
# opencv) need their data files and native libraries collected explicitly, and the
# buffalo_l face model is bundled so the app runs fully offline.

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []

for pkg in ("insightface", "onnxruntime", "sklearn", "scipy", "skimage", "cv2", "PIL"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Static web UI.
datas += [("web", "web")]

# Bundle the buffalo_l model (offline, no first-run download). Resolved at build time
# from the machine/CI cache at ~/.insightface.
_model = Path(os.path.expanduser("~/.insightface")) / "models" / "buffalo_l"
if _model.exists():
    datas += [(str(_model), "insightface_home/models/buffalo_l")]
else:
    raise SystemExit(
        "buffalo_l model not found at ~/.insightface/models/buffalo_l — "
        "run: python -c \"from insightface.app import FaceAnalysis; "
        "FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)\" before building."
    )

# uvicorn resolves these dynamically; name them so PyInstaller keeps them.
hiddenimports += [
    "app.main",
    "app.desktop",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.lifespan.on",
    "h11",
]

a = Analysis(
    ["app/desktop.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "tensorflow", "torch"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="findme",
    console=True,  # keep the "findme is running" window
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="findme",
)
