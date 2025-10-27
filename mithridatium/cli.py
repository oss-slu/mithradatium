# mithridatium/cli.py
import typer
import json
from pathlib import Path
import sys
from mithridatium import report as rpt
from mithridatium import loader as loader
from mithridatium import data as mdata


VERSION = "0.1.0"
DEFENSES = {"spectral", "mmbd"}

EXIT_USAGE_ERROR = 64     # invalid CLI usage (e.g., unsupported --defense)
EXIT_DATA_ERR = 65  # invalid data format (schema failures)
EXIT_NO_INPUT = 66        # input file missing/not a file
EXIT_CANT_CREATE = 73     # cannot create/overwrite output without --force
EXIT_IO_ERROR = 74        # input exists but can't be opened/read

app = typer.Typer(help="Mithridatium CLI - verify pretrained model integrity")


def _write_json(obj: dict, out_path: str, force: bool) -> None:
    """
    Write JSON to a file or to stdout.
    - Stdout using "--out -"
    - Overwrite using "--force"

    """

    if out_path == "-":
        json.dump(obj, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Checks if file exists and prevents overwriting. Use --force to override.
    if path.exists() and not force:
        typer.secho(
            f"Error: output file already exists: {path}.",
        )
        raise typer.Exit(code=EXIT_CANT_CREATE)

    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

@app.callback(invoke_without_command=True)
def _root(
    # This is a calback that prints the version whenever it is ran.
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show Mithridatium version and exit.",
        is_eager=True, # ensures this runs before any command (including --help
    )
):

    if version:
        typer.echo(VERSION)
        raise typer.Exit()

@app.command()
def defenses() -> None:
    """
    List supported defenses.
    """
    for d in sorted(DEFENSES):
        typer.echo(d)

@app.command()
def detect(
    model: str = typer.Option(
        "models/resnet18.pth",
        "--model",
        "-m",
        help="The model path .pth. E.g. 'models/resnet18.pth'.",
    ),
    data: str = typer.Option(
        "cifar10",
        "--data",
        "-d",
        help="The dataset name. E.g. 'cifar10'.",
    ),
    defense: str = typer.Option(
        "spectral",
        "--defense",
        "-D",
        help="The defense you want to run. E.g. 'spectral'.",
    ),
    arch: str = typer.Option(
        "resnet18",
        "--arch",
        "-a",
        help="The model architecture to use. E.g. 'resnet18'.",
    ),
    out: str = typer.Option(
        "reports/report.json",
        "--out",
        "-o",
        help='The output path for the JSON report. Use "-" for stdout or a file path (e.g. "reports/report.json").',
    ),
    force: bool = typer.Option(
        False, 
        "--force", 
        "-f", 
        help="This allows overwriting. E.g. if the output file already exists --force will overwrite it.",
    ),
):
    """
    Argument validation:
    1) Model path exists and is a file
    2) File exists but can't be loaded
    3) Unsupported defense
    4) Write dummy JSON (stdout allowed via --out -)
    """
    # 1) Model path exists and is a file
    p = Path(model)
    if not p.exists() or not p.is_file():
        typer.secho(
            f"Error: model path not found or not a file: {p}", err=True
        )
        raise typer.Exit(code=EXIT_NO_INPUT)

    # 2) File exists but can't be loaded
    try:
        with p.open("rb"):
            pass
    except OSError as ex:
        typer.secho(
            f"Error: model file could not be opened: {p}\nReason: {ex}", err=True
        )
        raise typer.Exit(code=EXIT_IO_ERROR)
    
    # 3) Unsupported defense
    d = defense.strip().lower()
    if d not in DEFENSES:
        typer.secho(
            "Error: unsupported --defense "
            f"'{defense}'. Supported defenses: {', '.join(sorted(DEFENSES))}", err=True
        )
        raise typer.Exit(code=EXIT_USAGE_ERROR)
    
    # 4) Build model arch
    print(f"[cli] building model architecture '{arch}'…")
    mdl, feature_module = loader.build_model(arch, num_classes=10)

    # 5) Load weights from checkpoint
    print("[cli] loading weights…")
    mdl = loader.load_weights(mdl, str(p))

    # 6) Build dataloader (TEMP: CIFAR-10; replace with PreprocessConfig)
    print("[cli] building dataloader…")
    test_loader = mdata.dataloader_for(str(p), data, split="test", batch_size=256)

    # 7) Run the defenses that are supported
    print(f"[cli] running defense={d}…")
    try:
        if d == "mmbd":
            results = rpt.run_mmbd_stub(str(p), data)
        # elif d == "spectral":
        #     results = rpt.run_spectral(str(p), data)
        else:
            results = {"suspected_backdoor": False, "num_flagged": 0, "top_eigenvalue": 0.0}
    except Exception as ex:
        typer.secho(
            f"Error: failed to run '{d}' on model {p}.\nReason: {ex}", err=True
        )
        raise typer.Exit(code=EXIT_IO_ERROR)
   

    # 8) Build & write report
    rep = rpt.build_report(model_path=str(p), defense=d, dataset=data, version=VERSION, results=results)
    _write_json(rep, out, force)

@app.command("show-report")
def show_report(
    file: str = typer.Option(..., "--file", "-f", help="Path to a JSON report file."),
    mode: str = typer.Option(
        "json",
        "--mode",
        "-m",
        help="Display report as 'json' (pretty JSON) or 'summary' (human-readable).",
    ),
    schema: str = typer.Option(
        None,
        "--schema",
        help="Optional schema path; defaults to bundled schema.",
    ),
) -> None:
    """
    Pretty-print a saved report. Validates first:
    - If valid: prints chosen view (no extra messages).
    - If invalid: prints a single error and exits non-zero.
    """
    p = Path(file)
    if not p.exists() or not p.is_file():
        typer.secho(f"Error: report file not found: {p}", err=True)
        raise typer.Exit(code=EXIT_NO_INPUT)

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as ex:
        typer.secho(f"Error: failed to read/parse JSON: {p}\nReason: {ex}", err=True)
        raise typer.Exit(code=EXIT_IO_ERROR)

    # Validate (silent on success)
    try:
        rpt.validate_report_data(data, schema=schema)
    except RuntimeError as ex:
        # jsonschema missing or similar setup issue
        typer.secho(f"Error: {ex}", err=True)
        raise typer.Exit(code=EXIT_USAGE_ERROR)
    except Exception as ex:
        typer.secho("Report not valid.", err=True)
        typer.secho(f"Reason: {ex}", err=True)
        raise typer.Exit(code=EXIT_DATA_ERR)

    # If we reach here, it's valid—print the chosen view
    if mode.lower() == "json":
        typer.echo(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))
    elif mode.lower() == "summary":
        try:
            summary = rpt.render_summary(data)
            typer.echo(summary)
        except Exception as ex:
            typer.secho("Error: could not render summary. Is this a valid report JSON?", err=True)
            typer.secho(f"Reason: {ex}", err=True)
            raise typer.Exit(code=EXIT_DATA_ERR)
    else:
        typer.secho("Error: --mode must be 'json' or 'summary'", err=True)
        raise typer.Exit(code=EXIT_USAGE_ERROR)
    
if __name__ == "__main__":
    app()
