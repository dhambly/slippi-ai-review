#!/usr/bin/env python3
"""Fail fast when a packaged native installation is incomplete."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check_import(name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__file__", "built-in"))


def data_ready(root: Path | None, backend: str) -> bool:
    if root is None:
        return False
    marker = root / "data" / ("raw/main.dol" if backend == "decomp" else "manifest.json")
    return marker.is_file()


def check_runtime(
    root: Path | None,
    slippi_ai_root: Path | None,
    backend: str,
    *,
    start_simulator: bool,
) -> dict[str, object]:
    if root is None or not root.is_dir():
        return {"ok": False, "detail": f"missing runtime root: {root}"}
    modules = {
        "legacy": (
            "melee_sim._native",
            "tools.eval.validation_dtypes",
            "tools.modelplay.state_adapter",
            "tools.slippi.validation_buffer_builder",
        ),
        "decomp": (
            "melee_sim",
            "tools.migration.replay_checkpoint",
            "tools.modelplay.phillip_adapter",
        ),
    }[backend]
    environment = os.environ.copy()
    python_paths = [str(root)]
    if slippi_ai_root:
        python_paths.append(str(slippi_ai_root))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["MSL_DATA_DIR"] = str(root / "data")
    statement = "; ".join(f"import {name}" for name in modules)
    if start_simulator:
        statement += (
            "\nfrom melee_sim import EnvBatch"
            "\nwith EnvBatch(batch_size=1, length=2) as environment:"
            "\n    pass"
        )
    result = subprocess.run(
        [sys.executable, "-c", statement],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    detail = result.stderr.strip() or result.stdout.strip() or ", ".join(modules)
    return {"ok": result.returncode == 0, "detail": detail}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true", help="Skip private model, ISO, and extracted-data checks.")
    parser.add_argument("--skip-node", action="store_true", help="Skip Node checks when bootstrap omitted Node.")
    args = parser.parse_args()
    from slippi_ai_review.config import DEFAULT_CONFIG_PATH, load_settings

    settings = load_settings()
    imports = {
        name: check_import(name)
        for name in ("numpy", "tensorflow", "tensorflow_probability", "sonnet", "slippi_ai")
    }
    runtimes = {
        "legacy": check_runtime(
            settings.msl_root,
            settings.slippi_ai_root,
            "legacy",
            start_simulator=not args.code_only,
        ),
        "decomp": check_runtime(
            settings.msl_decomp_root,
            settings.slippi_ai_root,
            "decomp",
            start_simulator=not args.code_only,
        ),
    }
    checks: dict[str, object] = {
        "python_3_12": sys.version_info[:2] == (3, 12),
        "platform": sys.platform,
        "config": str(DEFAULT_CONFIG_PATH),
        "config_exists": DEFAULT_CONFIG_PATH.is_file(),
        "runtime_native": settings.runtime_mode == "native",
        "node": shutil.which("node"),
        "npm": shutil.which("npm"),
        "model": str(settings.model) if settings.model else None,
        "model_exists": bool(settings.model and settings.model.exists()),
        "iso_exists": bool(settings.melee_iso and settings.melee_iso.is_file()),
        "legacy_data": data_ready(settings.msl_root, "legacy"),
        "decomp_data": data_ready(settings.msl_decomp_root, "decomp"),
        "imports": {name: {"ok": result[0], "detail": result[1]} for name, result in imports.items()},
        "runtimes": runtimes,
    }
    node_ok = args.skip_node
    if not args.skip_node and shutil.which("node"):
        node = subprocess.run(["node", "--version"], capture_output=True, text=True)
        node_ok = node.returncode == 0 and node.stdout.strip().startswith("v20.")
    checks["node_20"] = node_ok
    private_assets_ok = (
        bool(checks["model_exists"])
        and bool(checks["iso_exists"])
        and bool(checks["legacy_data"])
        and bool(checks["decomp_data"])
    )
    ok = (
        bool(checks["python_3_12"])
        and bool(checks["config_exists"])
        and bool(checks["runtime_native"])
        and bool(checks["node_20"])
        and (args.skip_node or bool(checks["npm"]))
        and (args.code_only or private_assets_ok)
        and all(result[0] for result in imports.values())
        and all(bool(result["ok"]) for result in runtimes.values())
    )
    print(json.dumps({"ok": ok, "checks": checks}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
