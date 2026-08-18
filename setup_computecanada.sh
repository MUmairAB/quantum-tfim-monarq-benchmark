#!/bin/bash
# Environment setup for Compute Canada / Digital Research Alliance clusters.
# Run from the repo root: bash setup_computecanada.sh [venv_dir]
#
# Their shared JupyterLab kernel sets PIP_PREFIX and EBPYTHONPREFIXES
# globally, which silently redirects pip installs away from the venv and
# leaks the shared kernel's site-packages ahead of it. quspin's compiled
# dependencies also have no working PyPI wheel here, so a plain
# `pip install -r requirements.txt` produces a package that "installs" but
# fails to import — Compute Canada's own wheelhouse has working builds.
set -e

unset PIP_PREFIX
unset EBPYTHONPREFIXES
unset EBPYTHONPREFIXES_PRIORITY

VENV_DIR="${1:-$HOME/quantum-tfim-env}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --index-url https://pypi.org/simple/ -r requirements.txt

# On a GPU allocation, swap in the CUDA build of jax so NNQS training uses
# the GPU (VQE runs on lightning.qubit regardless and doesn't benefit).
if command -v nvidia-smi &>/dev/null; then
    echo "GPU detected — installing jax[cuda12] in place of plain jax..."
    "$VENV_DIR/bin/pip" install --index-url https://pypi.org/simple/ --force-reinstall "jax[cuda12]==0.10.2"
fi

"$VENV_DIR/bin/pip" install --no-cache-dir --force-reinstall --no-deps quspin-extensions
"$VENV_DIR/bin/pip" install --no-cache-dir --force-reinstall --no-deps parallel-sparse-tools

echo "Verifying against the reference value..."
"$VENV_DIR/bin/python" -c "
from src.exact import ground_state_energy
E0 = ground_state_energy(6, 1.0, 1.0)
assert abs(E0 - (-7.29622981)) < 1e-6
print('OK:', E0)
"

echo "JAX backend check (should say 'gpu' if nvidia-smi was found above):"
"$VENV_DIR/bin/python" -c "
import jax
print('  backend:', jax.default_backend(), ' devices:', jax.devices())
"

echo ""
echo "Done. To use this venv as a Jupyter kernel:"
echo "  $VENV_DIR/bin/python -m ipykernel install --user --name quantum-tfim --display-name quantum-tfim"
echo ""
echo "Every new terminal session needs the same env vars unset before using"
echo "this venv — add to your shell profile if this will be a recurring pain:"
echo "  unset PIP_PREFIX EBPYTHONPREFIXES EBPYTHONPREFIXES_PRIORITY"
echo ""
echo "If a GPU run OOMs on a MIG slice (jax preallocates ~75% of GPU memory by"
echo "default, which doesn't leave much room on a 10GB slice once more than one"
echo "circuit depth gets compiled in the same process), set:"
echo "  XLA_PYTHON_CLIENT_PREALLOCATE=false"
echo "before the python invocation. If that alone isn't enough, XLA_PYTHON_CLIENT_MEM_FRACTION"
echo "can cap usage further."
