from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_settings


def doctor() -> int:
    settings = load_settings()
    checks = {
        "config": DEFAULT_CONFIG_PATH.is_file(),
        "node": shutil.which("node") is not None,
        "npm": shutil.which("npm") is not None,
        "wsl": shutil.which("wsl") is not None,
        "model": bool(settings.model and settings.model.exists()),
        "msl_root": bool(settings.msl_root and settings.msl_root.exists()),
        "slippi_ai_root": bool(settings.slippi_ai_root and settings.slippi_ai_root.exists()),
        "melee_iso": bool(settings.melee_iso and settings.melee_iso.exists()),
        "data_dir_writable": settings.data_dir.parent.exists(),
    }
    print(json.dumps({"ok": all(checks.values()), "configPath": str(DEFAULT_CONFIG_PATH), "checks": checks}, indent=2))
    return 0 if all(checks.values()) else 1


def cleanup(days: int, apply: bool) -> int:
    root = load_settings().data_dir / "reviews"
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    matches, bytes_total = [], 0
    for review_path in root.glob("*/review.json") if root.is_dir() else []:
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
            updated = datetime.fromisoformat(str(review["updatedAt"]).replace("Z", "+00:00"))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
        pipeline = review_path.parent / "pipeline"
        if review.get("status") not in {"archived", "failed", "cancelled"} or updated > cutoff or not pipeline.is_dir():
            continue
        size = sum(item.stat().st_size for item in pipeline.rglob("*") if item.is_file())
        matches.append(str(review.get("reviewId")))
        bytes_total += size
        if apply:
            shutil.rmtree(pipeline)
    print(json.dumps({"applied": apply, "reviewIds": matches, "reclaimableBytes": bytes_total}, indent=2))
    return 0


def startup(action: str, port: int) -> int:
    if os.name != "nt":
        raise SystemExit("startup management currently supports Windows only")
    startup_dir = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    launcher = startup_dir / "Slippi AI Review.cmd"
    if action == "install":
        startup_dir.mkdir(parents=True, exist_ok=True)
        launcher.write_text(
            "@echo off\r\n"
            f'start "" /min "{sys.executable}" -m slippi_ai_review serve --host 0.0.0.0 --port {port} --allow-remote --no-open\r\n',
            encoding="ascii",
        )
    elif action == "remove":
        launcher.unlink(missing_ok=True)
    print(json.dumps({"installed": launcher.is_file(), "path": str(launcher)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Installation and storage operations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor")
    clean = subparsers.add_parser("cleanup")
    clean.add_argument("--days", type=int, default=30)
    clean.add_argument("--apply", action="store_true")
    start = subparsers.add_parser("startup")
    start.add_argument("action", choices=("install", "remove", "status"))
    start.add_argument("--port", type=int, default=8877)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "cleanup":
        return cleanup(args.days, args.apply)
    return startup(args.action, args.port)
