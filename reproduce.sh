#!/usr/bin/env bash
# FedSSL-MERC — one-command reproduction of the controlled testbed.
# Runs the full federated-EDL experiment battery and regenerates the figures.
# Pure NumPy / SciPy / Matplotlib: no GPU, no PyTorch required.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY="${PY:-python3}"

echo "==> [1/3] Checking dependencies"
$PY - <<'PYCHK'
import importlib.util, sys
missing=[m for m in ("numpy","scipy","matplotlib","sklearn") if importlib.util.find_spec(m) is None]
if missing:
    print("Missing packages:", ", ".join(missing))
    print("Install them with:  pip install -r requirements.txt")
    sys.exit(1)
print("    OK")
PYCHK

echo "==> [2/3] Running the controlled testbed battery (testbed/run_experiments.py)"
echo "    (full multi-seed battery; this can take several minutes)"
( cd testbed && $PY run_experiments.py )

echo "==> [3/3] Regenerating figures from testbed results"
$PY paper/figure_scripts/make_consolidated_figs.py
$PY paper/figure_scripts/make_pipeline.py

echo
echo "Done."
echo "  - JSON results : testbed/results/*.json"
echo "  - Figures      : paper/figures/*.png"
