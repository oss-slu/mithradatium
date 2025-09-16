# GPT written tests for mithradatium CLI

import json
from pathlib import Path
from typer.testing import CliRunner
from mithridatium.cli import app, VERSION

runner = CliRunner()

def _write_model(tmp_path: Path) -> Path:
    """Create a small dummy model file."""
    model = tmp_path / "fake.pth"
    model.write_bytes(b"ok")
    return model

def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert VERSION in result.stdout

def test_defenses_lists_spectral():
    result = runner.invoke(app, ["defenses"])
    assert result.exit_code == 0
    assert "spectral" in result.stdout

def test_detect_happy_to_file(tmp_path):
    model = _write_model(tmp_path)
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["detect", "-m", str(model), "-D", "spectral", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["status"] == "Not yet implemented"
    assert report["defense"] == "spectral"
    assert report["model_path"] == str(model)

def test_detect_stdout(tmp_path):
    model = _write_model(tmp_path)
    result = runner.invoke(app, ["detect", "-m", str(model), "-D", "spectral", "-o", "-"])
    assert result.exit_code == 0
    # JSON goes to stdout + success message
    assert '"status": "Not yet implemented"' in result.stdout

def test_missing_model_errors(tmp_path):
    missing = tmp_path / "nope.pth"
    out = tmp_path / "r.json"
    result = runner.invoke(app, ["detect", "-m", str(missing), "-D", "spectral", "-o", str(out)])
    assert result.exit_code == 2
    assert "model path not found" in result.stdout

def test_unreadable_model_errors(tmp_path, monkeypatch):
    model = _write_model(tmp_path)

    # Patch Path.open to raise OSError when opening this model in 'rb' mode
    from pathlib import Path as _P
    original_open = _P.open

    def bad_open(self, mode="r", *args, **kwargs):
        if self == model and "rb" in mode:
            raise OSError("permission denied")
        return original_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(_P, "open", bad_open)

    result = runner.invoke(app, ["detect", "-m", str(model), "-D", "spectral", "-o", str(tmp_path / "r.json")])
    assert result.exit_code == 2
    assert "could not be opened" in result.stdout
    assert "permission denied" in result.stdout

def test_unsupported_defense(tmp_path):
    model = _write_model(tmp_path)
    result = runner.invoke(app, ["detect", "-m", str(model), "-D", "foo", "-o", str(tmp_path / "r.json")])
    assert result.exit_code == 2
    assert "unsupported --defense" in result.stdout
    assert "spectral" in result.stdout  # lists supported options

def test_force_overwrite(tmp_path):
    model = _write_model(tmp_path)
    out = tmp_path / "r.json"

    # first write
    res1 = runner.invoke(app, ["detect", "-m", str(model), "-D", "spectral", "-o", str(out)])
    assert res1.exit_code == 0

    # try to overwrite without --force
    res2 = runner.invoke(app, ["detect", "-m", str(model), "-D", "spectral", "-o", str(out)])
    assert res2.exit_code == 2
    assert "already exists" in res2.stdout

    # now force overwrite
    res3 = runner.invoke(app, ["detect", "-m", str(model), "-D", "spectral", "-o", str(out), "--force"])
    assert res3.exit_code == 0
