from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, load_settings
from .paths import node_executable


def _runtime_mode(configured: str) -> str:
    if configured == "auto":
        return "wsl" if os.name == "nt" else "native"
    return configured


def _executable_exists(value: str) -> bool:
    candidate = Path(value).expanduser()
    return candidate.is_file() if candidate.parent != Path(".") else shutil.which(value) is not None


def _msl_data_ready(root: Path | None, backend: str) -> bool:
    if root is None:
        return False
    marker = root / "data" / ("raw/main.dol" if backend == "decomp" else "manifest.json")
    return marker.is_file()


def _native_imports_ok(settings) -> bool:
    if not _executable_exists(settings.runtime_python):
        return False
    selected_root = settings.msl_decomp_root if settings.simulation_backend == "decomp" else settings.msl_root
    if selected_root is None:
        return False
    environment = os.environ.copy()
    python_path = [str(selected_root)]
    if settings.slippi_ai_root:
        python_path.append(str(settings.slippi_ai_root))
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    if python_path:
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment["MSL_DATA_DIR"] = str(selected_root / "data")
    native_import = (
        "import tools.migration.replay_checkpoint, tools.modelplay.phillip_adapter"
        if settings.simulation_backend == "decomp"
        else "import melee_sim._native, tools.modelplay.state_adapter"
    )
    command = [
        settings.runtime_python,
        "-c",
        (
            "import tensorflow, tensorflow_probability, sonnet, slippi_ai, melee_sim; "
            f"{native_import}"
        ),
    ]
    try:
        return subprocess.run(
            command,
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=90,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def doctor() -> int:
    settings = load_settings()
    mode = _runtime_mode(settings.runtime_mode)
    selected_root = settings.msl_decomp_root if settings.simulation_backend == "decomp" else settings.msl_root
    checks = {
        "config": DEFAULT_CONFIG_PATH.is_file(),
        "node": Path(node_executable()).is_file() or shutil.which(node_executable()) is not None,
        "npm": (
            (Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin") / ("npm.cmd" if os.name == "nt" else "npm")).is_file()
            or shutil.which("npm") is not None
        ),
        "model": bool(settings.model and settings.model.exists()),
        "msl_root": bool(settings.msl_root and settings.msl_root.exists()),
        "msl_data": _msl_data_ready(settings.msl_root, "legacy"),
        "msl_decomp_root": bool(settings.msl_decomp_root and settings.msl_decomp_root.exists()),
        "msl_decomp_data": _msl_data_ready(settings.msl_decomp_root, "decomp"),
        "simulation_backend": settings.simulation_backend in {"legacy", "decomp"},
        "selected_msl_root": bool(selected_root and selected_root.exists()),
        "selected_msl_data": _msl_data_ready(selected_root, settings.simulation_backend),
        "slippi_ai_root": bool(settings.slippi_ai_root and settings.slippi_ai_root.exists()),
        "melee_iso": bool(settings.melee_iso and settings.melee_iso.exists()),
        "data_dir_writable": settings.data_dir.parent.exists(),
    }
    if mode == "wsl":
        checks["wsl"] = shutil.which("wsl") is not None
        checks["micromamba"] = bool(settings.msl_env and settings.micromamba)
    else:
        checks["python_3_12"] = sys.version_info[:2] == (3, 12)
        checks["runtime_imports"] = _native_imports_ok(settings)
    print(json.dumps({
        "ok": all(checks.values()),
        "runtimeMode": mode,
        "simulationBackend": settings.simulation_backend,
        "configPath": str(DEFAULT_CONFIG_PATH),
        "checks": checks,
    }, indent=2))
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
