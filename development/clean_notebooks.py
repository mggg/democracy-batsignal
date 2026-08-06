#!/usr/bin/env python3
"""Remove transient execution state from template notebooks."""

import argparse
import json
from pathlib import Path
from typing import Any

DEVELOPMENT = Path(__file__).resolve().parent
DEFAULT_NOTEBOOKS = DEVELOPMENT / "template_project" / "notebooks"


def clean_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    """Removes outputs, execution counts, cell IDs, and metadata from a notebook.

    Args:
        notebook (dict[str, Any]): Parsed notebook document to clean in place.

    Returns:
        dict[str, Any]: The cleaned notebook document.
    """
    notebook["metadata"] = {}
    for cell in notebook.get("cells", []):
        cell["metadata"] = {}
        cell.pop("id", None)
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def clean_notebook_text(content: str, source: str | Path = "notebook") -> str:
    """Returns a deterministic, cleaned JSON representation of one notebook.

    Args:
        content (str): Notebook JSON text.
        source (str | Path): Name used if the JSON cannot be parsed.

    Returns:
        str: Clean notebook JSON ending with one newline.

    Raises:
        ValueError: If the content is invalid JSON.
        TypeError: If the document does not contain a cell list.
    """
    try:
        notebook = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source}: invalid notebook JSON: {error}") from error
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise TypeError(f"{source}: notebook must contain a cell list")
    return json.dumps(clean_notebook(notebook), indent=1, ensure_ascii=False) + "\n"


def notebook_paths(arguments: list[Path]) -> list[Path]:
    """Resolves explicit notebooks or every template notebook when none are supplied.

    Args:
        arguments (list[Path]): Notebook files or directories supplied on the command line.

    Returns:
        list[Path]: Sorted, unique notebook paths.
    """
    paths: set[Path] = set()
    for argument in arguments or [DEFAULT_NOTEBOOKS]:
        if argument.is_dir():
            paths.update(argument.rglob("*.ipynb"))
        else:
            paths.add(argument)
    return sorted(paths)


def main() -> int:
    """Cleans notebooks in place or reports whether cleaning would change them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path, help="Notebook files or directories."
    )
    parser.add_argument(
        "--check", action="store_true", help="Report dirty notebooks only."
    )
    args = parser.parse_args()

    dirty: list[Path] = []
    for path in notebook_paths(args.paths):
        original = path.read_text()
        cleaned = clean_notebook_text(original, path)
        if original == cleaned:
            continue
        dirty.append(path)
        if not args.check:
            path.write_text(cleaned, newline="\n")
            print(f"cleaned {path}")

    if args.check and dirty:
        for path in dirty:
            print(f"needs cleaning: {path}")
        return 1
    if args.check:
        print("notebooks are clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
