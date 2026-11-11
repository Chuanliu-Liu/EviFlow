# Reproducibility Supplement

This is the minimal local release of the certifier implementation. It
contains the complete src implementation, self-contained core tests, and the
small synthetic runner used for local experiments.

## Quick start

    python3 -m pip install -r requirements.txt
    python3 -m compileall -q src experiments tests
    python3 examples/run_smoke.py
    PYTHONPATH=. python3 -m pytest -q

The synthetic runner can be used for a small development run:

    PYTHONPATH=. python3 -m experiments.run_main --protocol configs/protocol_main_v1.json --dev --out results/dev

All inputs in this archive are synthetic or frozen configuration values.