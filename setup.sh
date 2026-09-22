#!/usr/bin/env bash
# One-time environment setup. InsightFace ships an sdist that imports numpy/cython
# at build time, so we install those first and build it without isolation.
set -euo pipefail
cd "$(dirname "$0")"

echo ">>> [1/4] create venv (Python 3.12)"
uv venv --python 3.12

echo ">>> [2/4] build prerequisites for insightface's native extension"
uv pip install "numpy<2.2" cython

echo ">>> [3/4] runtime dependencies"
uv pip install \
  "fastapi>=0.115" "uvicorn[standard]>=0.30" "python-multipart>=0.0.9" \
  "onnxruntime>=1.18" "opencv-python-headless>=4.9" "scikit-learn>=1.4" "pillow>=10.3"

echo ">>> [4/4] insightface (no build isolation so it sees numpy/cython)"
uv pip install --no-build-isolation "insightface>=0.7.3"

echo ">>> verifying imports"
.venv/bin/python -c "import insightface, onnxruntime, cv2, sklearn, fastapi; print('OK — findme is ready. Run ./run.sh')"
