#!/usr/bin/env python3
"""Regenerate template_maker.sh and template_maker.ps1 from template/ and installer_src/.

The two installers are single-file, fully self-contained scripts that users can carry
around on their own: every project file under template/ is embedded in them as a
payload. Edit the real files under template/ (or the installer skeletons under
installer_src/) and rerun this script; never edit the generated installers directly.

Usage:
    python3 generate_installers.py          # rewrite both installers
    python3 generate_installers.py --check  # exit 1 if the installers are out of date
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "template"
SRC = ROOT / "installer_src"
MARKER = "# {{GENERATED_PAYLOADS}}"
HEREDOC_EOF = "TEMPLATE_PAYLOAD_EOF"

BANNER = [
    "====  GENERATED PAYLOADS (from template/) -- DO NOT EDIT BY HAND  ====",
    "====  regenerate with: python3 generate_installers.py             ====",
]


def payloads(platform):
    """(dest_rel_path, content) pairs from template/common plus template/<platform>."""
    out = []
    for tree in ("common", platform):
        base = TEMPLATE / tree
        for p in sorted(base.rglob("*")):
            if p.is_file():
                out.append((p.relative_to(base).as_posix(), p.read_text()))
    return out


def bash_payload_block():
    items = payloads("bash")
    lines = [f"# {b}" for b in BANNER]
    lines += ["", "payload_files=("]
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
    items = payloads("powershell")
    lines = [f"# {b}" for b in BANNER]
    lines += ["", "$Payloads = [ordered]@{"]
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
        "template_maker.sh": generate("skeleton.sh", bash_payload_block()),
        "template_maker.ps1": generate("skeleton.ps1", ps1_payload_block()),
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
