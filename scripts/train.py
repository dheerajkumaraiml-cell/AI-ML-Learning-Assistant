"""QLoRA training entry point. See app/train_qlora.py for implementation."""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "app" / "train_qlora.py"),
        run_name="__main__",
    )
