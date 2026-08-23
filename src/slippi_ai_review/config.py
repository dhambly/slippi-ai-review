from __future__ import annotations

import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import DEFAULT_DATA_DIR


DEFAULT_CONFIG_PATH = Path(os.environ.get("SLIPPI_REVIEW_CONFIG", Path.home() / ".slippi-ai-review.toml"))


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    model: Path | None = None
    msl_root: Path | None = None
    msl_decomp_root: Path | None = None
    slippi_ai_root: Path | None = None
    melee_iso: Path | None = None
    simulation_backend: str = "legacy"
    runtime_mode: str = "auto"
    runtime_python: str = sys.executable
    micromamba: str = "micromamba"
    msl_env: str = ""
    cuda_library_path: str = ""
    tm_replay: Path | None = None
    ce_iso: Path | None = None
    dolphin: Path | None = None
    dolphin_user_dir: Path | None = None


def _path(value: Any) -> Path | None:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve() if value else None


def load_settings(path: Path | None = None) -> Settings:
    config_path = (path or DEFAULT_CONFIG_PATH).resolve()
    payload: dict[str, Any] = {}
    if config_path.is_file():
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
    paths = payload.get("paths") or {}
    runtime = payload.get("runtime") or {}
    wsl = payload.get("wsl") or {}
    training = payload.get("training_mode") or {}
    return Settings(
        data_dir=_path(os.environ.get("SLIPPI_REVIEW_DATA_DIR") or paths.get("data_dir")) or DEFAULT_DATA_DIR,
        model=_path(os.environ.get("SLIPPI_REVIEW_MODEL") or paths.get("model")),
        msl_root=_path(os.environ.get("SLIPPI_REVIEW_MSL_ROOT") or paths.get("msl_root")),
        msl_decomp_root=_path(
            os.environ.get("SLIPPI_REVIEW_MSL_DECOMP_ROOT")
            or paths.get("msl_decomp_root")
        ),
        slippi_ai_root=_path(os.environ.get("SLIPPI_REVIEW_SLIPPI_AI_ROOT") or paths.get("slippi_ai_root")),
        melee_iso=_path(os.environ.get("SLIPPI_REVIEW_MELEE_ISO") or paths.get("melee_iso")),
        simulation_backend=str(
            os.environ.get("SLIPPI_REVIEW_SIMULATION_BACKEND")
            or runtime.get("simulation_backend")
            or Settings.simulation_backend
        ),
        runtime_mode=str(
            os.environ.get("SLIPPI_REVIEW_RUNTIME")
            or runtime.get("mode")
            or Settings.runtime_mode
        ),
        runtime_python=str(
            os.environ.get("SLIPPI_REVIEW_PYTHON")
            or runtime.get("python")
            or Settings.runtime_python
        ),
        micromamba=str(wsl.get("micromamba") or Settings.micromamba),
        msl_env=str(wsl.get("environment") or Settings.msl_env),
        cuda_library_path=str(wsl.get("cuda_library_path") or Settings.cuda_library_path),
        tm_replay=_path(training.get("tm_replay")),
        ce_iso=_path(training.get("ce_iso")),
        dolphin=_path(training.get("dolphin")),
        dolphin_user_dir=_path(training.get("dolphin_user_dir")),
    )


def require_paths(settings: Settings, *names: str) -> None:
    missing = [name for name in names if getattr(settings, name) is None]
    if missing:
        keys = ", ".join(f"paths.{name}" for name in missing)
        raise SystemExit(f"Missing required configuration: {keys}. See config.example.toml.")
