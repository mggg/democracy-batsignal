import json
import unittest
from pathlib import Path

import generate_installers


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
        self.assertIn("chain_outputs", bash_directories)
        self.assertEqual(bash_directories, ps_directories)
        self.assertFalse(any(path.endswith(".gitkeep") for path in bash_paths | ps_paths))
        self.assertNotIn(".venv", bash_directories)
        self.assertFalse(any("__pycache__" in path for path in bash_paths | ps_paths))
        self.assertFalse(any(path.startswith("chain_outputs/") for path in bash_paths | ps_paths))


class InstallerSkeletonTest(unittest.TestCase):
    def test_python_cli_uses_recorded_chain(self):
        text = Path("template_project/pipeline_scripts/example_cli.py").read_text()

        self.assertIn("RecordedChain", text)
        self.assertIn("chain.graph", text)
        self.assertIn("chain.allow_overwrite()", text)
        self.assertNotIn("BenEncoder", text)
        self.assertNotIn("--writeas", text)

    def test_pa_archive_uses_one_temp_file_variable(self):
        skeleton = Path("installer_src/skeleton.sh").read_text()
        pa_download = skeleton[skeleton.index('echo "Downloading PA example data..."') :]

        self.assertNotIn("mn_zip", pa_download)
        self.assertGreaterEqual(pa_download.count('"$pa_zip"'), 3)

    def test_rustrecom_examples_use_cli_arguments(self):
        scripts = (
            Path("template_project/pipeline_scripts/pa_example_script_vanilla.sh"),
            Path("template_project/pipeline_scripts/rust_example_script.ps1"),
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
        text = Path("template_project/pipeline_scripts/pa_example_script_opt.sh").read_text()

        self.assertIn("rustrecom tilted", text)
        self.assertIn("rustrecom_objectives/gingles_partial.json", text)

        objective_dir = Path("template_project/pipeline_scripts/rustrecom_objectives")
        objectives = {
            json.loads(path.read_text())["objective"] for path in objective_dir.glob("*.json")
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


if __name__ == "__main__":
    unittest.main()
