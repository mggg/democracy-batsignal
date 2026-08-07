import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent

# Data collection settings: add or remove script paths to control this stage of the pipeline.
DATA_COLLECTION_SCRIPTS = (PIPELINE_DIR / "metrics" / "collect_data_vanilla_pa.py",)


def main() -> None:
    """Runs each configured data collection script in order."""
    for script_path in DATA_COLLECTION_SCRIPTS:
        print(f"Running {script_path.relative_to(PROJECT_ROOT)}...", flush=True)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
