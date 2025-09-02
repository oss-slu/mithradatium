# mithridatium/cli.py
import typer

app = typer.Typer(help="Mithridatium CLI - verify pretrained model integrity")

@app.command()
def detect(
    model: str = typer.Option("models/resnet18.pth", "--model", "-m", help="Path to model .pth (can be missing for now)"),
    data: str = typer.Option("cifar10", "--data", "-d", help="Dataset name"),
    defense: str = typer.Option("spectral", "--defense", "-D", help="Defense to run"),
    out: str = typer.Option("reports/report.json", "--out", "-o", help="Path to write JSON report"),
):
    typer.echo(f"[args] model={model}  data={data}  defense={defense}  out={out}")

if __name__ == "__main__":
    app()
