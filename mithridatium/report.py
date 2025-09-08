# mithridatium/report.py
"""
Reporting utilities for Mithridatium.

In Sprint 1, this just writes a dummy JSON file so the CLI
can demonstrate the workflow. In later sprints, detection
modules will write their real results here.
"""

import json
import datetime as dt
from pathlib import Path

def write_dummy_report(model_path: str, defense: str, out_path: str, version: str = "0.1.0"):
    """
    Write a placeholder JSON report. Used for Sprint 1 demo.

    Args:
        model_path (str): Path to the model file.
        defense (str): The defense name (currently ignored).
        out_path (str): Path to write the JSON report.
        version (str): Framework version string.
    """
    payload = {
        "mithridatium_version": version,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_path),
        "defense": defense,
        "status": "Not yet implemented"
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"[ok] Dummy report written to {out_file.resolve()}")
    return payload
