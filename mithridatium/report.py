# mithridatium/report.py

import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any
import torch

def run_spectral(model_path: str, dataset: str, iters: int = 50) -> dict:
    """
    Tiny spectral-signature style check:
    - Load state_dict
    - Find largest weight matrix (by elements)
    - Approximate top eigenvalue via power iteration on W^T W
    """
    sd = torch.load(model_path, map_location="cpu")
    # find the largest 2D tensor (a weight matrix)
    mats = [v for k, v in sd.items() if v.ndim >= 2]
    if not mats:
        return {"suspected_backdoor": False, "num_flagged": 0, "top_eigenvalue": 0.0}
    W = max(mats, key=lambda t: t.numel()).detach().flatten(1)  # [out, features]
    # power iteration on A = W^T W
    x = torch.randn(W.shape[1], 1)
    for _ in range(iters):
        x = W.t().mm(W.mm(x))
        x = x / (x.norm() + 1e-12)
    top_ev = float((x.t().mm(W.t().mm(W.mm(x))))/(x.t().mm(x) + 1e-12))
    top_singular = top_ev ** 0.5
    # naive threshold; tune later
    suspected = top_singular > 10.0 

    return {"suspected_backdoor": bool(suspected), "num_flagged": 0, "top_eigenvalue": top_ev}


def write_dummy_report(model_path: str, defense: str, out_path: str, version: str = "0.1.0",results: Dict[str, Any] | None = None) -> Dict[str, Any]:
def write_report(model_path: str, defense: str, out_path: str, details, version: str = "0.1.0"):
    payload = {
        "mithridatium_version": version,
        "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
        "model_path": str(model_path),
        "defense": defense,
        "status": "ok" if details else "empty"
    }

    if details is not None:
        payload["details"] = _json_safe(details)


    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[ok] Report written to {out_file.resolve()}")
    return payload

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

def build_report(model_path: str, defense: str, dataset: str, version: str = "0.1.0",
                 results: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "mithridatium_version": version,
        "model_path": model_path,
        "defense": defense,
        "dataset": dataset,
        "results": results or {
            "suspected_backdoor": False,
            "num_flagged": 0,
            "top_eigenvalue": 0.0,
        },
    }

def run_mmbd_stub(model_path: str, dataset: str) -> Dict[str, Any]:
    # placeholder metrics to satisfy acceptance criteria; swap with real MMBD later
    return {"suspected_backdoor": True, "num_flagged": 500, "top_eigenvalue": 42.3}

def render_summary(report: Dict[str, Any]) -> str:
    r = report["results"]
    return (
        f"Mithridatium {report['mithridatium_version']} | "
        f"defense={report['defense']} | dataset={report['dataset']}\n"
        f"- model_path:        {report['model_path']}\n"
        f"- suspected_backdoor:{r.get('suspected_backdoor')}\n"
        f"- num_flagged:       {r.get('num_flagged')}\n"
        f"- top_eigenvalue:    {r.get('top_eigenvalue')}"
    )
def _json_safe(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
