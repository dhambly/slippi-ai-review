from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parents[1]
WEB_DIR = PACKAGE_DIR / "web"
JS_DIR = PACKAGE_DIR / "js"
DEFAULT_DATA_DIR = PROJECT_DIR / "data"


def module_command(module: str, *arguments: object) -> list[str]:
    import sys

    return [sys.executable, "-u", "-m", f"slippi_ai_review.{module}", *(str(value) for value in arguments)]
