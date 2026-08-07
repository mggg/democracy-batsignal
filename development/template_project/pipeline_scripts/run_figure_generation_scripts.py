import subprocess
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parent

# Figure settings: add, remove, or reorder script paths to control this pipeline stage.
FIGURE_GENERATION_SCRIPTS = (
    PIPELINE_DIR / "figure_generators" / "base_plan_figures.py",
    PIPELINE_DIR / "figure_generators" / "cut_edges_histogram.py",
    PIPELINE_DIR / "figure_generators" / "disprop_scatter.py",
    PIPELINE_DIR / "figure_generators" / "reock_boxplot.py",
)


def main() -> None:
    """Runs each configured figure generation script in order."""
    for script_path in FIGURE_GENERATION_SCRIPTS:
        print(f"Running {script_path.relative_to(PROJECT_ROOT)}...", flush=True)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
