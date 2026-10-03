#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Run inside a fresh Python 3.11 environment with a CUDA-12-capable driver.
# A matching CUDA toolkit + compiler is needed for gsplat JIT compilation.
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e '.[research,cluster,data]'
echo 'Environment installed. Obtain authorized base weights using hf auth login.'
