#!/bin/bash
# Environment setup for Compute Canada / Digital Research Alliance clusters.
# Run from the repo root: bash setup_computecanada.sh [venv_dir]
#
# Two things about their shared JupyterLab kernel environment are not
# obvious from the errors alone:
#
# 1. PIP_PREFIX and EBPYTHONPREFIXES/EBPYTHONPREFIXES_PRIORITY are set
#    globally in the kernel's environment. PIP_PREFIX silently redirects
#    every pip install to a scratch location instead of the active venv,
#    and EBPYTHONPREFIXES injects the shared kernel's site-packages into
#    sys.path ahead of the venv's own — so packages appear to "install"
#    successfully but the venv's Python never sees them, or sees the
#    wrong (older) version. Unsetting these gives a genuinely isolated venv.
#
# 2. quspin's compiled dependencies (quspin-extensions, parallel-sparse-tools)
#    have no prebuilt wheel on PyPI for this platform, so a plain
#    `pip install -r requirements.txt` falls back to a source build that
#    silently fails to compile the C++ extensions (the package "installs"
#    but importing it fails deep inside quspin's own import chain).
#    Compute Canada's wheelhouse has working prebuilt wheels for both —
#    older than the versions PyPI would otherwise select, but functionally
#    correct — so we force pip to use those specifically after the main install.
set -e

unset PIP_PREFIX
unset EBPYTHONPREFIXES
unset EBPYTHONPREFIXES_PRIORITY

VENV_DIR="${1:-$HOME/quantum-tfim-env}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --index-url https://pypi.org/simple/ -r requirements.txt

"$VENV_DIR/bin/pip" install --no-cache-dir --force-reinstall --no-deps quspin-extensions
"$VENV_DIR/bin/pip" install --no-cache-dir --force-reinstall --no-deps parallel-sparse-tools

echo "Verifying against the reference value..."
"$VENV_DIR/bin/python" -c "
from src.exact import ground_state_energy
E0 = ground_state_energy(6, 1.0, 1.0)
assert abs(E0 - (-7.29622981)) < 1e-6
print('OK:', E0)
"

echo ""
echo "Done. To use this venv as a Jupyter kernel:"
echo "  $VENV_DIR/bin/python -m ipykernel install --user --name quantum-tfim --display-name quantum-tfim"
echo ""
echo "Every new terminal session needs the same env vars unset before using"
echo "this venv — add to your shell profile if this will be a recurring pain:"
echo "  unset PIP_PREFIX EBPYTHONPREFIXES EBPYTHONPREFIXES_PRIORITY"
