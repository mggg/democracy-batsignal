import unittest

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


if __name__ == "__main__":
    unittest.main()
