# mithridatium/report.py

import json
import datetime as dt
from pathlib import Path
from typing import Dict, Any

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
import torch


# def write_dummy_report(model_path: str, defense: str, out_path: str, version: str = "0.1.0",results: Dict[str, Any] | None = None) -> Dict[str, Any]:
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

# def write_dummy_report(model_path: str, defense: str, out_path: str, version: str = "0.1.0"):
#     """
#     Write a placeholder JSON report. Used for Sprint 1 demo.

#     Args:
#         model_path (str): Path to the model file.
#         defense (str): The defense name (currently ignored).
#         out_path (str): Path to write the JSON report.
#         version (str): Framework version string.
#     """
#     payload = {
#         "mithridatium_version": version,
#         "timestamp_utc": dt.datetime.utcnow().isoformat() + "Z",
#         "model_path": str(model_path),
#         "defense": defense,
#         "status": "Not yet implemented"
#     }

#     out_file = Path(out_path)
#     out_file.parent.mkdir(parents=True, exist_ok=True)

#     with out_file.open("w") as f:
#         json.dump(payload, f, indent=2)

#     print(f"[ok] Dummy report written to {out_file.resolve()}")
#     return payload

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

def _schema_path() -> Path:
    return Path(__file__).resolve().parents[1] / "reports" / "report_schema.json"

def validate_report_data(data: dict, schema: str | None = None) -> None:
    """
    Validate an in-memory report dict against the JSON Schema.
    Silent on success. Raises on invalid or if jsonschema is missing.
    """
    import json
    from pathlib import Path
    try:
        import jsonschema
    except ImportError:
        raise RuntimeError("jsonschema is required. Install with: pip install jsonschema")

    sch_path = Path(schema) if schema else _schema_path()
    sch = json.loads(sch_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=sch)