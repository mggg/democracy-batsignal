import json
import unittest
from pathlib import Path

import clean_notebooks
import generate_installers

DEVELOPMENT = Path(__file__).resolve().parent
ROOT = DEVELOPMENT.parent
TEMPLATE = DEVELOPMENT / "template_project"
INSTALLER_SRC = DEVELOPMENT / "installer_src"


class ProjectPathsTest(unittest.TestCase):
    def test_selects_platform_scripts_and_preserves_directories(self):
        bash_directories, bash_files = generate_installers.project_paths("bash")
        ps_directories, ps_files = generate_installers.project_paths("powershell")
        bash_paths = {path for path, _ in bash_files}
        ps_paths = {path for path, _ in ps_files}

        self.assertIn("pyproject.toml", bash_paths & ps_paths)
        self.assertNotIn("uv.lock", bash_paths | ps_paths)
        self.assertTrue(any(path.endswith(".sh") for path in bash_paths))
        self.assertFalse(any(path.endswith(".ps1") for path in bash_paths))
        self.assertTrue(any(path.endswith(".ps1") for path in ps_paths))
        self.assertFalse(any(path.endswith(".sh") for path in ps_paths))
        bash_script_stems = {
            Path(path).with_suffix("").as_posix()
            for path in bash_paths
            if path.endswith(".sh")
        }
        ps_script_stems = {
            Path(path).with_suffix("").as_posix()
            for path in ps_paths
            if path.endswith(".ps1")
        }
        self.assertEqual(bash_script_stems, ps_script_stems)
        self.assertIn("chain_outputs", bash_directories)
        self.assertEqual(bash_directories, ps_directories)
        self.assertFalse(
            any(path.endswith(".gitkeep") for path in bash_paths | ps_paths)
        )
        self.assertNotIn(".venv", bash_directories)
        self.assertFalse(any("__pycache__" in path for path in bash_paths | ps_paths))
        self.assertFalse(
            any(path.startswith("chain_outputs/") for path in bash_paths | ps_paths)
        )
        self.assertIn("data/alt_plan_pa.json", bash_paths & ps_paths)

    def test_distributed_notebooks_have_no_execution_state(self):
        _, files = generate_installers.project_paths("bash")
        content = dict(files)["notebooks/gerrychain_cut_edges_walkthrough.ipynb"]
        notebook = json.loads(content)

        self.assertEqual(notebook["metadata"], {})
        for cell in notebook["cells"]:
            self.assertEqual(cell["metadata"], {})
            self.assertNotIn("id", cell)
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


class CleanNotebooksTest(unittest.TestCase):
    def test_removes_transient_state_without_changing_sources(self):
        original = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": 3,
                    "id": "temporary-id",
                    "metadata": {"collapsed": True},
                    "outputs": [{"output_type": "stream", "text": ["result\n"]}],
                    "source": ["print('result')"],
                }
            ],
            "metadata": {"kernelspec": {"display_name": ".venv"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }

        cleaned = json.loads(clean_notebooks.clean_notebook_text(json.dumps(original)))
        cell = cleaned["cells"][0]

        self.assertEqual(cleaned["metadata"], {})
        self.assertEqual(cell["metadata"], {})
        self.assertNotIn("id", cell)
        self.assertIsNone(cell["execution_count"])
        self.assertEqual(cell["outputs"], [])
        self.assertEqual(cell["source"], ["print('result')"])


class InstallerSkeletonTest(unittest.TestCase):
    def test_python_cli_uses_recorded_chain(self):
        text = (TEMPLATE / "pipeline_scripts/chain_runners/example_cli.py").read_text()

        self.assertIn("RecordedChain", text)
        self.assertIn("chain.graph", text)
        self.assertIn("chain.allow_overwrite()", text)
        self.assertNotIn("BenEncoder", text)
        self.assertNotIn("--writeas", text)

    def test_installers_download_the_same_verified_pa_geometry(self):
        bash = (INSTALLER_SRC / "skeleton.sh").read_text()
        powershell = (INSTALLER_SRC / "skeleton.ps1").read_text()
        url_root = "https://raw.githubusercontent.com/mggg/democracy-batsignal"
        revision = "13c1098d244df946263c9353a478ecf66ac8e484"
        path = "template_project/data/pa_gdf.parquet"
        sha256 = "06b3b927b09e3f049623869d0b15e20b1363a2eb4dad43461915382fc165446c"

        for installer in (bash, powershell):
            self.assertIn(url_root, installer)
            self.assertIn(revision, installer)
            self.assertIn(path, installer)
            self.assertIn(sha256, installer)

    def test_installers_use_python_ben_and_pinned_rustrecom(self):
        bash = (INSTALLER_SRC / "skeleton.sh").read_text()
        powershell = (INSTALLER_SRC / "skeleton.ps1").read_text()
        pyproject = (TEMPLATE / "pyproject.toml").read_text()

        self.assertIn('"binary-ensemble>=2.0,<3"', pyproject)
        self.assertIn('"geopandas"', pyproject)
        for installer in (bash, powershell):
            normalized = " ".join(
                installer.replace("\\\n", " ").replace("`\n", " ").split()
            )

            self.assertIn('--tag "v0.2.0" --locked', normalized)
            self.assertNotIn("cargo install binary-ensemble", installer)
            self.assertNotIn("ben-process", installer)

    def test_powershell_uv_bootstrap_supports_windows_and_unix(self):
        powershell = (INSTALLER_SRC / "skeleton.ps1").read_text()

        self.assertIn("https://astral.sh/uv/install.ps1", powershell)
        self.assertIn("https://astral.sh/uv/install.sh", powershell)
        self.assertIn("$IsWindowsPlatform", powershell)
        self.assertIn("[IO.Path]::PathSeparator", powershell)
        self.assertNotIn("$env:Path", powershell)
        self.assertIn(
            '$pyprojectPath = Join-Path $projectPath "pyproject.toml"', powershell
        )
        self.assertIn("https://sh.rustup.rs", powershell)

    def test_bash_rustup_bootstrap_is_noninteractive(self):
        bash = (INSTALLER_SRC / "skeleton.sh").read_text()

        self.assertIn("https://sh.rustup.rs | sh -s -- -y", bash)

    def test_distributed_sources_do_not_reference_frcw(self):
        sources = {
            "README.md": (ROOT / "README.md").read_text(),
            "installer_src/skeleton.sh": (INSTALLER_SRC / "skeleton.sh").read_text(),
            "installer_src/skeleton.ps1": (INSTALLER_SRC / "skeleton.ps1").read_text(),
            "democracy-batsignal.sh": (ROOT / "democracy-batsignal.sh").read_text(),
            "democracy-batsignal.ps1": (ROOT / "democracy-batsignal.ps1").read_text(),
        }
        for platform in ("bash", "powershell"):
            for relative_path, content in generate_installers.project_paths(platform)[
                1
            ]:
                sources[f"template_project/{relative_path}"] = content

        for source, content in sources.items():
            self.assertNotIn("frcw", content.lower(), source)

    def test_rustrecom_examples_use_cli_arguments(self):
        scripts = (
            TEMPLATE / "pipeline_scripts/chain_runners/pa_example_script_vanilla.sh",
            TEMPLATE / "pipeline_scripts/chain_runners/pa_example_script_vanilla.ps1",
        )
        for script in scripts:
            text = script.read_text()

            self.assertIn("rustrecom chain", text)
            self.assertIn("--graph-json", text)
            self.assertNotIn("--config", text)
            self.assertNotIn("frcw", text)

        bash_text = scripts[0].read_text()
        self.assertIn('--rng-seed "$seed"', bash_text)

    def test_rustrecom_opt_example_uses_gingles_partial(self):
        scripts = (
            TEMPLATE / "pipeline_scripts/chain_runners/pa_example_script_opt.sh",
            TEMPLATE / "pipeline_scripts/chain_runners/pa_example_script_opt.ps1",
        )
        for script in scripts:
            text = script.read_text()

            self.assertIn("rustrecom tilted", text)
            self.assertIn("rustrecom_objectives/gingles_partial.json", text)
            self.assertIn("--scores-output-file", text)

        objective_dir = TEMPLATE / "pipeline_scripts/chain_runners/rustrecom_objectives"
        objectives = {
            json.loads(path.read_text())["objective"]
            for path in objective_dir.glob("*.json")
        }
        self.assertEqual(
            objectives,
            {
                "banded_gingles_partial",
                "by_district_abs_deviation",
                "election_wins",
                "gingles_partial",
                "polsby_popper",
            },
        )

    def test_python_first_runner_replaces_platform_batch_wrappers(self):
        runner = (TEMPLATE / "run_chains.py").read_text()
        quickstart = (TEMPLATE / "QUICKSTART.md").read_text()

        self.assertIn("from pipeline_scripts.run_parallel_chains import", runner)
        self.assertIn("run_chains(chains, max_workers=MAX_WORKERS)", runner)
        self.assertIn('ENGINE: ReComEngine = "gerrychain"', runner)
        self.assertIn('ENGINE = "rustrecom-chain"', quickstart)
        self.assertIn('ENGINE = "rustrecom-tilted"', quickstart)
        self.assertEqual(list(TEMPLATE.glob("batch_example*")), [])

    def test_gerrychain_walkthrough_is_a_ten_thousand_step_histogram(self):
        notebook_path = TEMPLATE / "notebooks/gerrychain_cut_edges_walkthrough.ipynb"
        notebook = json.loads(notebook_path.read_text())
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )

        self.assertEqual(notebook["nbformat"], 4)
        self.assertIn("TOTAL_STEPS = 10_000", code)
        self.assertIn("RecordedChain(", code)
        self.assertIn("chain.allow_overwrite()", code)
        self.assertIn("PlanEvaluator(", code)
        self.assertIn("CutEdges()", code)
        self.assertIn("Histogram(", code)
        self.assertIn("histogram.show()", code)


if __name__ == "__main__":
    unittest.main()
