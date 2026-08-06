#!/usr/bin/env python3
"""Regenerate the installers from template_project/ and installer_src/.

The two installers are single-file, fully self-contained scripts that users can carry
around on their own. Edit the runnable project under template_project/ (or the
installer skeletons under installer_src/) and rerun this script; never edit the
generated installers directly.

Usage:
    python3 generate_installers.py          # rewrite both installers
    python3 generate_installers.py --check  # exit 1 if the installers are out of date
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "template_project"
SRC = ROOT / "installer_src"
MARKER = "# {{GENERATED_PAYLOADS}}"
HEREDOC_EOF = "TEMPLATE_PAYLOAD_EOF"
PLATFORM_SUFFIX = {"bash": ".sh", "powershell": ".ps1"}
SCRIPT_SUFFIXES = set(PLATFORM_SUFFIX.values())
IGNORED_DIRECTORIES = {".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
IGNORED_FILES = {"uv.lock"}
OUTPUT_DIRECTORIES = {"chain_logs", "chain_outputs", "data", "dev_files", "figures", "stats"}
INCLUDED_OUTPUT_FILES = {"data/alt_plan_pa.json"}

BANNER = [
    "====  GENERATED PAYLOADS (from template_project/) -- DO NOT EDIT BY HAND  ====",
    "====  regenerate with: python3 generate_installers.py                     ====",
]


def project_paths(platform):
    """Return the template directories and UTF-8 files for one platform."""
    suffix = PLATFORM_SUFFIX[platform]
    directories = []
    files = []

    for path in sorted(TEMPLATE.rglob("*")):
        relative = path.relative_to(TEMPLATE)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if (
            relative.parts[0] in OUTPUT_DIRECTORIES
            and len(relative.parts) > 1
            and path.name not in {".gitignore", ".gitkeep"}
            and relative.as_posix() not in INCLUDED_OUTPUT_FILES
        ):
            continue
        if path.is_dir():
            directories.append(relative.as_posix())
            continue
        if path.name == ".gitkeep" or path.name in IGNORED_FILES:
            continue
        if path.suffix in SCRIPT_SUFFIXES and path.suffix != suffix:
            continue
        try:
            content = path.read_text()
        except UnicodeDecodeError as error:
            raise SystemExit(f"{relative}: template files must be UTF-8 text") from error
        files.append((relative.as_posix(), content))

    return directories, files


def bash_payload_block():
    directories, items = project_paths("bash")
    lines = [f"# {b}" for b in BANNER]
    lines += ["", "payload_directories=("]
    lines += [f'    "{rel}"' for rel in directories]
    lines += [")", "", "payload_files=("]
    lines += [f'    "{rel}"' for rel, _ in items]
    lines += [")", "", "function write_payload() {", '    case "$1" in']
    for rel, content in items:
        for ln in content.splitlines():
            if ln == HEREDOC_EOF:
                raise SystemExit(f"{rel}: contains the heredoc delimiter line")
        lines.append(f'    "{rel}") cat << \'{HEREDOC_EOF}\'')
        lines.append(content.rstrip("\n"))
        lines.append(HEREDOC_EOF)
        lines.append("        ;;")
    lines += ["    esac", "}"]
    return "\n".join(lines)


def ps1_payload_block():
    directories, items = project_paths("powershell")
    lines = [f"# {b}" for b in BANNER]
    lines += ["", "$PayloadDirectories = @("]
    lines += [f"    '{rel}'" for rel in directories]
    lines += [")", "", "$Payloads = [ordered]@{"]
    for rel, content in items:
        for ln in content.splitlines():
            if ln.startswith("'@"):
                raise SystemExit(f"{rel}: line starts with '@ which ends a here-string")
        lines.append(f"'{rel}' = @'")
        lines.append(content.rstrip("\n"))
        lines.append("'@")
    lines.append("}")
    return "\n".join(lines)


def generate(skeleton_name, block):
    skeleton = (SRC / skeleton_name).read_text()
    if skeleton.count(MARKER) != 1:
        raise SystemExit(f"{skeleton_name}: expected exactly one {MARKER} marker")
    return skeleton.replace(MARKER, block)


def main():
    check = "--check" in sys.argv[1:]
    outputs = {
        "democracy-batsignal.sh": generate("skeleton.sh", bash_payload_block()),
        "democracy-batsignal.ps1": generate("skeleton.ps1", ps1_payload_block()),
    }

    stale = []
    for name, text in outputs.items():
        path = ROOT / name
        if check:
            if not path.exists() or path.read_text() != text:
                stale.append(name)
        else:
            path.write_text(text, newline="\n")
            if name.endswith(".sh"):
                path.chmod(path.stat().st_mode | 0o111)
            print(f"wrote {name}")

    if stale:
        print(f"OUT OF DATE: {', '.join(stale)} -- run: python3 generate_installers.py")
        return 1
    if check:
        print("installers are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
