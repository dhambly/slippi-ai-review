"""Run a resumable whole-session Phillip review and build a nightly report."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from .config import load_settings
from .nightly_report import write_report
from .paths import PROJECT_DIR, module_command
from .server import (
    DEFAULT_UPLOAD_DIR,
    REVIEW_SCHEMA_VERSION,
    UploadError,
    _find_duplicates,
    _load_review,
    _utc_now,
    _write_json_atomic,
    update_review,
    validate_slp,
)
from .stage_geometry import stage_geometry_for_settings


DEFAULT_ALIASES = ("moobs", "bes", "M#0085", "MOOB#964")
GAME_NAME = re.compile(r"Game_(\d{8})T\d{6}\.slp$", re.IGNORECASE)


def _as_pid(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            exit_code = wintypes.DWORD()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _process_command_line(pid: int) -> str | None:
    if not _pid_running(pid):
        return None
    if os.name == "nt":
        command = [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine",
        ]
    elif (proc_path := Path(f"/proc/{pid}/cmdline")).is_file():
        try:
            return proc_path.read_bytes().replace(b"\0", b" ").decode(errors="replace").strip() or None
        except OSError:
            return None
    else:
        command = ["ps", "-p", str(pid), "-o", "command="]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _pid_matches_review_pipeline(pid: int, review_dir: Path) -> bool:
    if not _pid_running(pid):
        return False
    command_line = _process_command_line(pid)
    if command_line is None:
        return True
    normalized = command_line.casefold()
    pipeline_module = "slippi_ai_review.pipeline" in normalized
    review_argument = str(review_dir.resolve()).casefold() in normalized
    return pipeline_module and review_argument


def _claim_active_run(nightly_dir: Path, run_id: str) -> None:
    """Prevent a scheduled and interactive nightly run from sharing the GPU."""
    nightly_dir.mkdir(parents=True, exist_ok=True)
    lock_path = nightly_dir / ".nightly.lock"
    owner_path = lock_path / "owner.json"
    active_path = nightly_dir / "active_nightly.json"
    for attempt in range(3):
        try:
            lock_path.mkdir()
        except FileExistsError:
            legacy_lock = lock_path.is_file()
            metadata_path = lock_path if legacy_lock else owner_path
            try:
                owner = _read_json(metadata_path)
            except (OSError, ValueError):
                owner = {}
            owner_pid = _as_pid(owner.get("pid"))
            if _pid_running(owner_pid):
                raise SystemExit(
                    f"Nightly run {owner.get('runId') or 'unknown'} is already active (PID {owner_pid})."
                )
            try:
                age_seconds = max(0.0, time.time() - lock_path.stat().st_mtime)
            except OSError:
                age_seconds = 0.0
            if not owner and age_seconds < 30:
                raise SystemExit(f"A nightly run is currently acquiring the lock: {lock_path}")
            if attempt < 2:
                if legacy_lock:
                    lock_path.unlink(missing_ok=True)
                else:
                    owner_path.unlink(missing_ok=True)
                    try:
                        lock_path.rmdir()
                    except FileNotFoundError:
                        pass
                continue
            raise SystemExit(f"Could not replace stale nightly lock: {lock_path}")
        else:
            owner = {"runId": run_id, "pid": os.getpid(), "startedAt": _utc_now()}
            _write_json_atomic(owner_path, owner)
            _write_json_atomic(active_path, owner)

            def release() -> None:
                try:
                    current_owner = _read_json(owner_path)
                except (OSError, ValueError):
                    current_owner = {}
                if current_owner.get("runId") == run_id and _as_pid(current_owner.get("pid")) == os.getpid():
                    owner_path.unlink(missing_ok=True)
                    try:
                        lock_path.rmdir()
                    except (FileNotFoundError, OSError):
                        pass
                try:
                    current_active = _read_json(active_path)
                except (OSError, ValueError):
                    current_active = {}
                if current_active.get("runId") == run_id and _as_pid(current_active.get("pid")) == os.getpid():
                    active_path.unlink(missing_ok=True)

            atexit.register(release)
            return


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: Any) -> str:
    return str(value or "").strip().casefold()


def _game_date(path: Path) -> str | None:
    match = GAME_NAME.match(path.name)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def newest_normal_folder(root: Path) -> Path:
    candidates = [
        child for child in root.iterdir()
        if child.is_dir() and "mainline" not in child.name.casefold() and re.fullmatch(r"\d{4}-\d{2}", child.name)
    ]
    if not candidates:
        raise FileNotFoundError(f"No normal YYYY-MM Slippi folder found under {root}")
    with_games = [child for child in candidates if any(child.glob("*.slp"))]
    if not with_games:
        raise FileNotFoundError(f"No Slippi replays found under {root}")
    return max(with_games, key=lambda child: child.name)


def discover_replays(root: Path, requested_date: str | None) -> tuple[Path, str, list[Path]]:
    folder = root if root.name and re.fullmatch(r"\d{4}-\d{2}", root.name) else newest_normal_folder(root)
    dated = [(path, _game_date(path)) for path in folder.glob("*.slp")]
    dated = [(path, date) for path, date in dated if date]
    if not dated:
        raise FileNotFoundError(f"No timestamped Game_*.slp files found in {folder}")
    date = requested_date or max(value for _path, value in dated)
    return folder, date, sorted(path for path, value in dated if value == date)


def target_player(details: dict[str, Any], aliases: set[str]) -> dict[str, Any] | None:
    matches = []
    for player in details.get("players") or []:
        identities = {
            _normalized(player.get(field))
            for field in ("name", "displayName", "nametag", "connectCode")
            if player.get(field)
        }
        if identities & aliases:
            matches.append(player)
    return matches[0] if len(matches) == 1 else None


def _opponent(details: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return next(
        (player for player in details.get("players") or [] if player.get("playerIndex") != target.get("playerIndex")),
        {},
    )


def _review_payload(
    review_id: str,
    source: Path,
    digest: str,
    details: dict[str, Any],
    target: dict[str, Any],
    duplicates: list[dict[str, Any]],
    samples: int,
    segments_per_game: int,
    simulation_backend: str,
) -> dict[str, Any]:
    created_at = _utc_now()
    return {
        "schemaVersion": REVIEW_SCHEMA_VERSION,
        "reviewId": review_id,
        "status": "queued",
        "message": "Waiting for nightly whole-game sweep.",
        "createdAt": created_at,
        "updatedAt": created_at,
        "originalFilename": source.name,
        "replay": {
            "filename": "replay.slp",
            "relativePath": "replay.slp",
            "sizeBytes": source.stat().st_size,
            "sha256": digest,
        },
        "players": details["players"],
        "match": details.get("match", {}),
        "settings": {
            "qualityPreset": "quick",
            "analysisMode": "nightly-sweep",
            "phaseSweepSamples": samples,
            "phaseSweepSegments": segments_per_game,
            "simulationBackend": simulation_backend,
        },
        "artifacts": {
            "review": {"status": "pending"},
            "cePractice": {"status": "available_after_review", "mode": "per_clip"},
        },
        "duplicates": duplicates,
        "urls": {
            "review": f"/reviews/{review_id}/",
            "status": f"/api/reviews/{review_id}",
            "artifacts": f"/review-artifacts/{review_id}/",
            "report": f"/reviews/{review_id}/report",
            "artifactReport": None,
        },
        "targetPlayer": target,
    }


def create_review(
    upload_dir: Path,
    source: Path,
    digest: str,
    details: dict[str, Any],
    target: dict[str, Any],
    samples: int,
    segments_per_game: int,
    simulation_backend: str,
) -> tuple[str, Path]:
    review_id = str(uuid.uuid4())
    review_dir = upload_dir / review_id
    review_dir.mkdir(parents=True)
    shutil.copy2(source, review_dir / "replay.slp")
    payload = _review_payload(
        review_id,
        source,
        digest,
        details,
        target,
        _find_duplicates(upload_dir, digest),
        samples,
        segments_per_game,
        simulation_backend,
    )
    _write_json_atomic(review_dir / "review.json", payload)
    return review_id, review_dir


def _patch_review(upload_dir: Path, review_id: str, **values: Any) -> None:
    payload = _load_review(upload_dir, review_id)
    if payload is None:
        return
    payload.update(values)
    payload["updatedAt"] = _utc_now()
    _write_json_atomic(upload_dir / review_id / "review.json", payload)


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()


def run_review(
    args: argparse.Namespace,
    review_id: str,
    review_dir: Path,
    details: dict[str, Any],
    target: dict[str, Any],
    deadline: float,
) -> tuple[bool, float, str | None]:
    started = time.perf_counter()
    update_review(args.upload_dir, review_id, "processing", message="Running nightly whole-game Phillip sweep.")
    log_path = review_dir / "logs" / "nightly_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = module_command(
        "pipeline",
        "--replay", review_dir / "replay.slp",
        "--controlled-port", int(target["port"]),
        "--job-dir", review_dir,
        "--display-name", Path(str(review_dir / "replay.slp")).stem,
        "--slp-version", str((details.get("match") or {}).get("slpVersion") or ""),
        "--phase-sweep-samples", args.samples,
        "--phase-sweep-max-segments", args.segments_per_game,
        "--max-batch-lanes", args.max_batch_lanes,
        "--render-workers", args.render_workers,
        "--gpu-duty-cycle", args.gpu_duty_cycle,
        "--simulation-backend", args.simulation_backend,
        "--sweep-only",
    )
    command = [str(value) for value in command if str(value)]
    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _patch_review(args.upload_dir, review_id, worker={"pid": os.getpid(), "pipelinePid": process.pid, "startedAt": _utc_now()})
    remaining = deadline - time.monotonic()
    try:
        output, _ = process.communicate(timeout=max(1.0, remaining))
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        output = "Nightly deadline reached while this replay was processing.\n"
        log_path.write_text(" ".join(command) + "\n\n" + output, encoding="utf-8")
        update_review(args.upload_dir, review_id, "failed", message="Nightly deadline reached; retry will reuse completed stages.")
        return False, time.perf_counter() - started, "deadline"
    log_path.write_text(" ".join(command) + "\n\n" + output, encoding="utf-8")
    seconds = time.perf_counter() - started
    if process.returncode != 0:
        update_review(args.upload_dir, review_id, "failed", message=f"Nightly pipeline failed; see {log_path}.")
        _patch_review(args.upload_dir, review_id, timing={"processingSeconds": round(seconds, 3)})
        return False, seconds, f"pipeline exit {process.returncode}"
    reports = [review_dir / "artifacts" / f"{phase}_review.html" for phase in ("advantage", "neutral", "disadvantage")]
    if not all(path.is_file() for path in reports):
        update_review(args.upload_dir, review_id, "failed", message="Nightly pipeline did not publish all phase reports.")
        return False, seconds, "missing report"
    update_review(args.upload_dir, review_id, "complete", message="Nightly sweep complete.", report="advantage_review.html")
    _patch_review(args.upload_dir, review_id, timing={"processingSeconds": round(seconds, 3)})
    return True, seconds, None


def _duration(seconds: float) -> str:
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def _load_state(path: Path) -> dict[str, Any]:
    try:
        value = _read_json(path)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _review_compatible(
    review: dict[str, Any],
    review_dir: Path,
    digest: str,
    target: dict[str, Any],
    *,
    samples: int,
    segments_per_game: int,
    simulation_backend: str,
) -> bool:
    settings = review.get("settings") if isinstance(review.get("settings"), dict) else {}
    replay = review.get("replay") if isinstance(review.get("replay"), dict) else {}
    saved_target = review.get("targetPlayer") if isinstance(review.get("targetPlayer"), dict) else {}
    if settings.get("analysisMode") != "nightly-sweep":
        return False
    if replay.get("sha256") != digest:
        return False
    try:
        saved_port = int(saved_target.get("port") or 0)
        target_port = int(target.get("port") or 0)
    except (TypeError, ValueError):
        return False
    if saved_port != target_port:
        return False
    numeric_requirements = {
        "phaseSweepSamples": samples,
        "phaseSweepSegments": segments_per_game,
    }
    for key, requested in numeric_requirements.items():
        configured = settings.get(key)
        if configured is not None:
            try:
                if int(configured) < requested:
                    return False
            except (TypeError, ValueError):
                return False
    configured_backend = settings.get("simulationBackend")
    if configured_backend is not None and configured_backend != simulation_backend:
        return False
    queue_path = review_dir / "pipeline" / "phase_sweep_queue.json"
    if queue_path.is_file():
        try:
            queue = _read_json(queue_path)
        except (OSError, ValueError):
            return False
        try:
            queue_port = int(queue.get("controlled_port") or 0)
        except (TypeError, ValueError):
            return False
        if queue_port != target_port:
            return False
    return (review_dir / "replay.slp").is_file()


def _existing_review(
    upload_dir: Path,
    state: dict[str, Any],
    digest: str,
    target: dict[str, Any],
    *,
    samples: int,
    segments_per_game: int,
    simulation_backend: str,
) -> tuple[str, Path, str] | None:
    record = (state.get("replays") or {}).get(digest) or {}
    candidate_ids = [str(record.get("reviewId") or "")]
    candidate_ids.extend(str(item.get("reviewId") or "") for item in _find_duplicates(upload_dir, digest))
    seen: set[str] = set()
    for review_id in candidate_ids:
        if not review_id or review_id in seen:
            continue
        seen.add(review_id)
        review_dir = upload_dir / review_id
        review = _load_review(upload_dir, review_id)
        if not review or not _review_compatible(
            review,
            review_dir,
            digest,
            target,
            samples=samples,
            segments_per_game=segments_per_game,
            simulation_backend=simulation_backend,
        ):
            continue
        status = str(review.get("status") or "")
        reports = [review_dir / "artifacts" / f"{phase}_review.html" for phase in ("advantage", "neutral", "disadvantage")]
        queue_path = review_dir / "pipeline" / "phase_sweep_queue.json"
        if status == "complete" and queue_path.is_file() and all(path.is_file() for path in reports):
            return review_id, review_dir, "complete"
        if status in {"queued", "failed", "processing"}:
            worker = review.get("worker") if isinstance(review.get("worker"), dict) else {}
            pipeline_pid = _as_pid(worker.get("pipelinePid"))
            if status == "processing" and _pid_matches_review_pipeline(pipeline_pid, review_dir):
                continue
            return review_id, review_dir, "resume"
    return None


def _session_game(
    source: Path,
    details: dict[str, Any],
    target: dict[str, Any],
    *,
    review_id: str | None,
    status: str,
    seconds: float = 0.0,
    reason: str | None = None,
) -> dict[str, Any]:
    opponent = _opponent(details, target)
    opponent_character = ((opponent.get("character") or {}).get("name") if isinstance(opponent.get("character"), dict) else None)
    stage = stage_geometry_for_settings(details.get("match") or {}) or {}
    return {
        "filename": source.name,
        "reviewId": review_id,
        "status": status,
        "reason": reason,
        "seconds": round(seconds, 3),
        "stage": str(stage.get("stageName") or (details.get("match") or {}).get("stageName") or ""),
        "target": str(target.get("displayName") or target.get("connectCode") or ""),
        "opponent": str(opponent.get("displayName") or opponent.get("connectCode") or opponent_character or "Opponent"),
        "matchup": f"{((target.get('character') or {}).get('name') if isinstance(target.get('character'), dict) else 'Unknown')} vs {opponent_character or 'Unknown'}",
    }


def _session_metadata(
    run_id: str,
    date: str,
    folder: Path,
    aliases: list[str],
    games: list[dict[str, Any]],
    status: str,
    started_at: str,
    elapsed: float,
) -> dict[str, Any]:
    completed = [game for game in games if game.get("status") in {"complete", "reused"}]
    failed = [game for game in games if game.get("status") == "failed"]
    skipped = [game for game in games if game.get("status") == "skipped"]
    deferred = [game for game in games if game.get("status") == "deferred"]
    return {
        "schemaVersion": 1,
        "nightlyId": run_id,
        "status": status,
        "date": date,
        "title": f"Nightly review - {date}",
        "headline": "The habits worth practicing tomorrow",
        "createdAt": started_at,
        "updatedAt": _utc_now(),
        "sourceFolder": str(folder),
        "aliases": aliases,
        "report": "report.html",
        "url": f"/nightly/{run_id}/report",
        "stats": {
            "discoveredGames": len(games),
            "analyzedGames": len(completed),
            "failedGames": len(failed),
            "skippedGames": len(skipped),
            "deferredGames": len(deferred),
            "processingSeconds": round(elapsed, 3),
            "processingTime": _duration(elapsed),
        },
        "games": games,
    }


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slippi-root", type=Path, default=Path.home() / "Documents" / "Slippi")
    parser.add_argument("--date", help="YYYY-MM-DD; defaults to the newest normal Slippi replay date")
    parser.add_argument("--alias", action="append", dest="aliases")
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--nightly-dir", type=Path, default=settings.data_dir / "nightly")
    parser.add_argument("--run-id", default=str(uuid.uuid4()))
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--segments-per-game", type=int, default=8)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--min-game-frames", type=int, default=1800)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-batch-lanes", type=int, default=4096)
    parser.add_argument("--render-workers", type=int, default=6)
    parser.add_argument("--gpu-duty-cycle", type=float, default=1.0)
    parser.add_argument("--simulation-backend", choices=("legacy", "decomp"), default=settings.simulation_backend)
    parser.add_argument("--force", action="store_true", help="Ignore the replay-hash state and create fresh reviews")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        args.run_id = str(uuid.UUID(args.run_id))
    except ValueError as exc:
        raise SystemExit("--run-id must be a UUID") from exc
    if args.samples < 4 or args.max_hours <= 0 or not 0 < args.gpu_duty_cycle <= 1:
        raise SystemExit("Use at least 4 samples, a positive time budget, and GPU duty cycle in (0, 1].")
    aliases = args.aliases or list(DEFAULT_ALIASES)
    alias_set = {_normalized(alias) for alias in aliases}
    args.upload_dir = args.upload_dir.resolve()
    args.nightly_dir = args.nightly_dir.resolve()
    if not args.dry_run:
        _claim_active_run(args.nightly_dir, args.run_id)
    folder, date, replays = discover_replays(args.slippi_root.resolve(), args.date)
    if args.max_games is not None:
        replays = replays[:args.max_games]
    args.upload_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.nightly_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.nightly_dir / "state.json"
    state = _load_state(state_path)
    state.setdefault("schemaVersion", 1)
    state.setdefault("replays", {})
    started = time.perf_counter()
    started_at = _utc_now()
    deadline = time.monotonic() + args.max_hours * 3600
    games: list[dict[str, Any]] = []
    completed_dirs: list[Path] = []
    log = run_dir / "nightly.log"

    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"[{started_at}] discovered {len(replays)} replay(s) in {folder} for {date}\n")
        for number, source in enumerate(replays, 1):
            if time.monotonic() >= deadline:
                stream.write("Nightly deadline reached before the next replay.\n")
                games.extend(
                    {"filename": pending.name, "status": "deferred", "reason": "nightly deadline reached"}
                    for pending in replays[number - 1 :]
                )
                break
            try:
                try:
                    details = validate_slp(source)
                except UploadError as exc:
                    games.append({"filename": source.name, "status": "skipped", "reason": exc.message})
                    stream.write(f"[{number}/{len(replays)}] skipped {source.name}: {exc.message}\n")
                    continue
                target = target_player(details, alias_set)
                if target is None:
                    games.append({"filename": source.name, "status": "skipped", "reason": "target alias not uniquely identified"})
                    continue
                last_frame = int((details.get("match") or {}).get("lastFrame") or 0)
                if last_frame < args.min_game_frames:
                    games.append(_session_game(source, details, target, review_id=None, status="skipped", reason=f"short game ({last_frame} frames)"))
                    continue
                digest = _sha256(source)
                existing = None if args.force else _existing_review(
                    args.upload_dir,
                    state,
                    digest,
                    target,
                    samples=args.samples,
                    segments_per_game=args.segments_per_game,
                    simulation_backend=args.simulation_backend,
                )
                if existing and existing[2] == "complete":
                    review_id, review_dir, _mode = existing
                    completed_dirs.append(review_dir)
                    games.append(_session_game(source, details, target, review_id=review_id, status="reused"))
                    stream.write(f"[{number}/{len(replays)}] reused {source.name} as {review_id}\n")
                    continue
                if args.dry_run:
                    games.append(_session_game(source, details, target, review_id=None, status="validated"))
                    stream.write(f"[{number}/{len(replays)}] validated {source.name}, P{target.get('port')}\n")
                    continue
                if existing:
                    review_id, review_dir, _mode = existing
                    stream.write(f"[{number}/{len(replays)}] resuming {source.name} as {review_id}\n")
                else:
                    review_id, review_dir = create_review(
                        args.upload_dir,
                        source,
                        digest,
                        details,
                        target,
                        args.samples,
                        args.segments_per_game,
                        args.simulation_backend,
                    )
                    stream.write(f"[{number}/{len(replays)}] processing {source.name} as {review_id}\n")
                stream.flush()
                ok, seconds, error = run_review(args, review_id, review_dir, details, target, deadline)
                status = "complete" if ok else "failed"
                games.append(_session_game(source, details, target, review_id=review_id, status=status, seconds=seconds, reason=error))
                if ok:
                    completed_dirs.append(review_dir)
                    state["replays"][digest] = {"reviewId": review_id, "date": date, "updatedAt": _utc_now()}
                    _write_json_atomic(state_path, state)
                stream.write(f"[{number}/{len(replays)}] {status} in {seconds:.1f}s{f': {error}' if error else ''}\n")
                stream.flush()
                partial = _session_metadata(
                    args.run_id, date, folder, aliases, games, "processing", started_at, time.perf_counter() - started
                )
                _write_json_atomic(run_dir / "nightly.json", partial)
                write_report(partial, completed_dirs, run_dir)
            except Exception as exc:
                games.append({"filename": source.name, "status": "failed", "reason": str(exc)})
                stream.write(f"[{number}/{len(replays)}] exception for {source.name}: {exc}\n{traceback.format_exc()}\n")
                stream.flush()

    elapsed = time.perf_counter() - started
    incomplete = {"failed", "deferred"}
    status = "complete" if len(games) == len(replays) and not any(game.get("status") in incomplete for game in games) else "partial"
    if args.dry_run:
        status = "dry-run"
    session = _session_metadata(args.run_id, date, folder, aliases, games, status, started_at, elapsed)
    _write_json_atomic(run_dir / "nightly.json", session)
    payload = write_report(session, completed_dirs, run_dir)
    summary = {
        "nightlyId": args.run_id,
        "status": status,
        "date": date,
        "games": len(games),
        "analyzed": session["stats"]["analyzedGames"],
        "patterns": len(payload.get("recurringPatterns") or []),
        "seconds": round(elapsed, 3),
        "report": str((run_dir / "report.html").resolve()),
        "url": session["url"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if status in {"complete", "dry-run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
