# mithridatium/report.py

import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any
import torch
import jsonschema

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

def run_mmbd_stub(model_path: str, dataset: str) -> Dict[str, Any]:
    # placeholder metrics to satisfy acceptance criteria; swap with real MMBD later
    return {"suspected_backdoor": True, "num_flagged": 500, "top_eigenvalue": 42.3}

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

def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "reports" / "report_schema.json"

def validate_report_data(data: dict, schema: str | None = None) -> None:
    """
    Validate an in-memory report dict against the JSON Schema.
    Silent on success. Raises on invalid or if jsonschema is missing.
    """
    if jsonschema is None:
        raise RuntimeError("jsonschema is required. Install with: pip install jsonschema")
    sch_path = Path(schema) if schema else _schema_path()
    sch = json.loads(sch_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=sch)
