import typer
import sys
from pathlib import Path

# Add project root to Python path for clean absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.training import train
from scripts.model import upload_model, load_model
from scripts.preprocessing import load_and_preprocess


app = typer.Typer()
app.command()(train)
app.command()(upload_model)
app.command()(load_model)
app.command()(load_and_preprocess)


if __name__ == "__main__":
    app()