"""Run a resumable whole-session Phillip review and build a nightly report."""

from __future__ import annotations

import argparse
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
    except (OSError, json.JSONDecodeError):
        return {}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_review(upload_dir: Path, state: dict[str, Any], digest: str) -> tuple[str, Path] | None:
    record = (state.get("replays") or {}).get(digest) or {}
    review_id = str(record.get("reviewId") or "")
    if review_id:
        review_dir = upload_dir / review_id
        review = _load_review(upload_dir, review_id)
        if review and review.get("status") == "complete" and (review_dir / "pipeline" / "phase_sweep_queue.json").is_file():
            return review_id, review_dir
    for duplicate in _find_duplicates(upload_dir, digest):
        review_id = str(duplicate.get("reviewId") or "")
        review_dir = upload_dir / review_id
        review = _load_review(upload_dir, review_id)
        if review and review.get("status") == "complete" and (review_dir / "pipeline" / "phase_sweep_queue.json").is_file():
            return review_id, review_dir
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
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--segments-per-game", type=int, default=12)
    parser.add_argument("--max-hours", type=float, default=6.0)
    parser.add_argument("--min-game-frames", type=int, default=1800)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-batch-lanes", type=int, default=4096)
    parser.add_argument("--render-workers", type=int, default=6)
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.5)
    parser.add_argument("--simulation-backend", choices=("legacy", "decomp"), default=settings.simulation_backend)
    parser.add_argument("--force", action="store_true", help="Ignore the replay-hash state and create fresh reviews")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        uuid.UUID(args.run_id)
    except ValueError as exc:
        raise SystemExit("--run-id must be a UUID") from exc
    if args.samples < 4 or args.max_hours <= 0 or not 0 < args.gpu_duty_cycle <= 1:
        raise SystemExit("Use at least 4 samples, a positive time budget, and GPU duty cycle in (0, 1].")
    aliases = args.aliases or list(DEFAULT_ALIASES)
    alias_set = {_normalized(alias) for alias in aliases}
    args.upload_dir = args.upload_dir.resolve()
    args.nightly_dir = args.nightly_dir.resolve()
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
                existing = None if args.force else _existing_review(args.upload_dir, state, digest)
                if existing:
                    review_id, review_dir = existing
                    completed_dirs.append(review_dir)
                    games.append(_session_game(source, details, target, review_id=review_id, status="reused"))
                    stream.write(f"[{number}/{len(replays)}] reused {source.name} as {review_id}\n")
                    continue
                if args.dry_run:
                    games.append(_session_game(source, details, target, review_id=None, status="validated"))
                    stream.write(f"[{number}/{len(replays)}] validated {source.name}, P{target.get('port')}\n")
                    continue
                review_id, review_dir = create_review(args.upload_dir, source, digest, details, target, args.samples)
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
    status = "complete" if games and not any(game.get("status") == "failed" for game in games) else "partial"
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
