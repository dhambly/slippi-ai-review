from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
WEB_DIR = PACKAGE_DIR / "web"
JS_DIR = PACKAGE_DIR / "js"
DEFAULT_DATA_DIR = PROJECT_DIR / "data"


def node_executable() -> str:
    override = os.environ.get("SLIPPI_REVIEW_NODE")
    if override:
        return override
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    bundled = Path(sys.prefix) / bin_dir / ("node.exe" if os.name == "nt" else "node")
    if bundled.is_file():
        return str(bundled)
    return shutil.which("node") or "node"


def module_command(module: str, *arguments: object) -> list[str]:
    import sys

    return [sys.executable, "-u", "-m", f"slippi_ai_review.{module}", *(str(value) for value in arguments)]
