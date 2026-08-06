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
        self.assertFalse(any(path.endswith(".gitkeep") for path in bash_paths | ps_paths))
        self.assertNotIn(".venv", bash_directories)
        self.assertFalse(any("__pycache__" in path for path in bash_paths | ps_paths))
        self.assertFalse(any(path.startswith("chain_outputs/") for path in bash_paths | ps_paths))
        self.assertIn("data/alt_plan_pa.json", bash_paths & ps_paths)


class InstallerSkeletonTest(unittest.TestCase):
    def test_python_cli_uses_recorded_chain(self):
        text = Path("template_project/pipeline_scripts/example_cli.py").read_text()

        self.assertIn("RecordedChain", text)
        self.assertIn("chain.graph", text)
        self.assertIn("chain.allow_overwrite()", text)
        self.assertNotIn("BenEncoder", text)
        self.assertNotIn("--writeas", text)

    def test_installers_download_the_same_verified_pa_geometry(self):
        bash = Path("installer_src/skeleton.sh").read_text()
        powershell = Path("installer_src/skeleton.ps1").read_text()
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
        bash = Path("installer_src/skeleton.sh").read_text()
        powershell = Path("installer_src/skeleton.ps1").read_text()
        pyproject = Path("template_project/pyproject.toml").read_text()

        self.assertIn('"binary-ensemble>=2.0"', pyproject)
        for installer in (bash, powershell):
            normalized = " ".join(installer.replace("\\\n", " ").replace("`\n", " ").split())

            self.assertIn('--tag "v0.2.0" --locked', normalized)
            self.assertNotIn("cargo install binary-ensemble", installer)
            self.assertNotIn("ben-process", installer)

    def test_powershell_uv_bootstrap_supports_windows_and_unix(self):
        powershell = Path("installer_src/skeleton.ps1").read_text()

        self.assertIn("https://astral.sh/uv/install.ps1", powershell)
        self.assertIn("https://astral.sh/uv/install.sh", powershell)
        self.assertIn("$IsWindowsPlatform", powershell)
        self.assertIn("[IO.Path]::PathSeparator", powershell)
        self.assertNotIn("$env:Path", powershell)
        self.assertIn('$pyprojectPath = Join-Path $projectPath "pyproject.toml"', powershell)
        self.assertIn("https://sh.rustup.rs", powershell)

    def test_bash_rustup_bootstrap_is_noninteractive(self):
        bash = Path("installer_src/skeleton.sh").read_text()

        self.assertIn("https://sh.rustup.rs | sh -s -- -y", bash)

    def test_distributed_sources_do_not_reference_frcw(self):
        sources = {
            "README.md": Path("README.md").read_text(),
            "installer_src/skeleton.sh": Path("installer_src/skeleton.sh").read_text(),
            "installer_src/skeleton.ps1": Path("installer_src/skeleton.ps1").read_text(),
            "democracy-batsignal.sh": Path("democracy-batsignal.sh").read_text(),
            "democracy-batsignal.ps1": Path("democracy-batsignal.ps1").read_text(),
        }
        for platform in ("bash", "powershell"):
            for relative_path, content in generate_installers.project_paths(platform)[1]:
                sources[f"template_project/{relative_path}"] = content

        for source, content in sources.items():
            self.assertNotIn("frcw", content.lower(), source)

    def test_rustrecom_examples_use_cli_arguments(self):
        scripts = (
            Path("template_project/pipeline_scripts/pa_example_script_vanilla.sh"),
            Path("template_project/pipeline_scripts/pa_example_script_vanilla.ps1"),
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
            Path("template_project/pipeline_scripts/pa_example_script_opt.sh"),
            Path("template_project/pipeline_scripts/pa_example_script_opt.ps1"),
        )
        for script in scripts:
            text = script.read_text()

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
