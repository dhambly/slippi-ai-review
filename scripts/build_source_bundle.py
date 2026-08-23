#!/usr/bin/env python3
"""Build a transferable archive containing the app and pinned runtimes."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    ".gitignore",
    "README.md",
    "THIRD_PARTY.md",
    "config.example.toml",
    "dependencies.lock.json",
    "package-lock.json",
    "package.json",
    "pyproject.toml",
    "requirements-inference.txt",
)
ROOT_DIRS = (".github", "docs", "patches", "scripts", "src", "tests", "vendor")
IGNORED_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo", ".log"}


def included_files() -> list[Path]:
    files = [ROOT / name for name in ROOT_FILES]
    for directory in ROOT_DIRS:
        files.extend(
            item
            for item in (ROOT / directory).rglob("*")
            if item.is_file()
            and not (set(item.relative_to(ROOT).parts) & IGNORED_PARTS)
            and item.suffix not in IGNORED_SUFFIXES
        )
    return sorted(set(files))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist" / "slippi-ai-review-source.tar.gz")
    parser.add_argument(
        "--iso",
        type=Path,
        help="Add a local legal ISO as local-assets/GALE01.iso. Never use this for a public artifact.",
    )
    args = parser.parse_args()
    output = args.out.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w:gz" if output.name.endswith((".tar.gz", ".tgz")) else "w"
    with tarfile.open(output, mode, format=tarfile.PAX_FORMAT) as archive:
        for source in included_files():
            archive_name = Path("slippi-ai-review") / source.relative_to(ROOT)
            info = archive.gettarinfo(str(source), arcname=str(archive_name))
            if source.suffix in {".sh", ".py"} and source.parent.name == "scripts":
                info.mode = 0o755
            with source.open("rb") as handle:
                archive.addfile(info, handle)
        if args.iso:
            iso = args.iso.expanduser().resolve()
            if not iso.is_file():
                raise SystemExit(f"ISO not found: {iso}")
            archive.add(
                iso,
                arcname="slippi-ai-review/local-assets/GALE01.iso",
                recursive=False,
            )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
