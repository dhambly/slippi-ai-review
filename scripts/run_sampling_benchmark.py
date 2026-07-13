"""Run a deadline-bounded sampling benchmark over completed Slippi games.

The expensive reference runs use a large sample corpus. Lower sampling budgets can
then be evaluated by filtering those corpora without rerunning the simulator.
Progress is persisted after every stage and simulation chunk so interrupted runs
can resume without losing completed work or the original wall-clock deadline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from slippi_ai_review.config import load_settings
from slippi_ai_review.paths import PROJECT_DIR, module_command
from slippi_ai_review.pipeline import msl_command


REFERENCE_PREFLIGHT_SAMPLES = 128
REFERENCE_REFINEMENT_SAMPLES = 256
PREFLIGHT_MIN_OPTION_SAMPLES = 2
REFINEMENT_MIN_OPTION_SAMPLES = 8


@dataclass(frozen=True)
class Game:
    id: str
    replay: Path
    controlled_port: int
    matchup: str
    stage: str
    last_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--max-replicates", type=int, default=3)
    parser.add_argument("--preflight-samples", type=int, default=REFERENCE_PREFLIGHT_SAMPLES)
    parser.add_argument("--refinement-samples", type=int, default=REFERENCE_REFINEMENT_SAMPLES)
    parser.add_argument("--max-batch-lanes", type=int, default=4096)
    parser.add_argument(
        "--minimum-start-minutes",
        type=float,
        default=8.0,
        help="Do not start another reference run with less time remaining.",
    )
    return parser.parse_args()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_games(path: Path) -> list[Game]:
    payload = read_json(path.resolve())
    games = [
        Game(
            id=str(item["id"]),
            replay=Path(item["replay"]).resolve(),
            controlled_port=int(item["controlled_port"]),
            matchup=str(item["matchup"]),
            stage=str(item["stage"]),
            last_frame=int(item["last_frame"]),
        )
        for item in payload["games"]
    ]
    if len(games) != 10:
        raise ValueError(f"Benchmark manifest must contain exactly 10 games, got {len(games)}")
    for game in games:
        if game.controlled_port not in (1, 2):
            raise ValueError(f"Invalid controlled port for {game.id}: {game.controlled_port}")
        if not game.replay.is_file():
            raise FileNotFoundError(game.replay)
    return games


def initial_state(args: argparse.Namespace, games: list[Game]) -> dict[str, Any]:
    started = utc_now()
    return {
        "status": "running",
        "started_at": iso(started),
        "deadline_at": iso(started + timedelta(hours=float(args.hours))),
        "requested_hours": float(args.hours),
        "preflight_samples": int(args.preflight_samples),
        "refinement_samples": int(args.refinement_samples),
        "max_batch_lanes": int(args.max_batch_lanes),
        "games": [game.id for game in games],
        "active": None,
        "completed_jobs": [],
        "failed_jobs": [],
        "events": [],
    }


def update_clock(state: dict[str, Any]) -> None:
    started = datetime.fromisoformat(state["started_at"])
    deadline = datetime.fromisoformat(state["deadline_at"])
    now = utc_now()
    state["updated_at"] = iso(now)
    state["elapsed_seconds"] = round((now - started).total_seconds(), 3)
    state["remaining_seconds"] = round((deadline - now).total_seconds(), 3)


def save_state(path: Path, state: dict[str, Any], event: dict[str, Any] | None = None) -> None:
    if event is not None:
        event = {"at": iso(utc_now()), **event}
        state["events"] = (state.get("events") or [])[-199:] + [event]
    update_clock(state)
    write_json(path, state)


def run_command(
    *,
    command: list[str],
    log_path: Path,
    state_path: Path,
    state: dict[str, Any],
    stage: str,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    save_state(state_path, state, {"stage": stage, "event": "started"})
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    child = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(child, dict) and child.get("chunk") is not None:
                    save_state(
                        state_path,
                        state,
                        {
                            "stage": stage,
                            "event": "chunk",
                            "chunk": int(child["chunk"]),
                            "lanes": int(child.get("lanes") or 0),
                            "seconds": float(child.get("seconds") or 0.0),
                        },
                    )
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(f"{stage} failed with exit code {return_code}; see {log_path}")
    save_state(
        state_path,
        state,
        {"stage": stage, "event": "completed", "seconds": round(elapsed, 3)},
    )
    return elapsed


def selection_command(
    *,
    run_dir: Path,
    candidates: Path,
    replay: Path,
    iso_path: Path,
    controlled_port: int,
    out: Path,
    min_option_samples: int,
    preflight_queue: Path | None = None,
) -> list[str]:
    command = module_command(
        "selection",
        "--run-dir", str(run_dir),
        "--candidates-json", str(candidates),
        "--replay", str(replay),
        "--iso", str(iso_path),
        "--controlled-port", str(controlled_port),
        "--top", "0",
        "--min-option-samples", str(min_option_samples),
        "--min-option-share", "0.1",
        "--min-improvement-rate", "0.25",
        "--alternative-routes", "3",
        "--out", str(out),
    )
    if preflight_queue is not None:
        command.extend(["--preflight-queue", str(preflight_queue)])
    return command


def pipeline_namespace(args: argparse.Namespace, game: Game, settings: Any) -> argparse.Namespace:
    return argparse.Namespace(
        replay=game.replay,
        controlled_port=game.controlled_port,
        model=settings.model.resolve(),
        msl_root=settings.msl_root.resolve(),
        slippi_ai_root=settings.slippi_ai_root.resolve(),
        iso=settings.melee_iso.resolve(),
        micromamba=settings.micromamba,
        msl_env=settings.msl_env,
        cuda_library_path=settings.cuda_library_path,
        max_batch_lanes=int(args.max_batch_lanes),
    )


def ensure_candidates(
    *,
    game: Game,
    game_root: Path,
    state_path: Path,
    state: dict[str, Any],
) -> tuple[Path, float]:
    candidates = game_root / "candidates.json"
    if candidates.is_file():
        return candidates, 0.0
    elapsed = run_command(
        command=module_command(
            "candidates",
            "--replay", str(game.replay),
            "--analyzed-port", str(game.controlled_port),
            "--max-candidates", "0",
            "--coverage-report", str(game_root / "candidate_coverage.md"),
            "--raw-events-out", str(game_root / "raw_events.json"),
            "--out", str(candidates),
        ),
        log_path=game_root / "logs" / "candidates.log",
        state_path=state_path,
        state=state,
        stage=f"{game.id}:candidates",
    )
    return candidates, elapsed


def run_reference_job(
    *,
    args: argparse.Namespace,
    game: Game,
    replicate: int,
    out_root: Path,
    state_path: Path,
    state: dict[str, Any],
    settings: Any,
) -> dict[str, Any]:
    game_root = out_root / "games" / game.id
    run_root = game_root / "replicates" / f"r{replicate:02d}"
    completion = run_root / "job_summary.json"
    if completion.is_file() and read_json(completion).get("status") == "complete":
        return read_json(completion)

    state["active"] = {"game": game.id, "replicate": replicate, "stage": "starting"}
    save_state(state_path, state)
    started_at = utc_now()
    timings: dict[str, float] = {}
    candidates, timings["candidates"] = ensure_candidates(
        game=game,
        game_root=game_root,
        state_path=state_path,
        state=state,
    )
    ns = pipeline_namespace(args, game, settings)

    preflight = run_root / "preflight"
    if not (preflight / "summary.json").is_file():
        timings["preflight"] = run_command(
            command=msl_command(
                ns,
                metadata=candidates,
                out_dir=preflight,
                offsets="0",
                samples=int(args.preflight_samples),
                dump_streams=False,
            ),
            log_path=run_root / "logs" / "preflight.log",
            state_path=state_path,
            state=state,
            stage=f"{game.id}:r{replicate}:preflight",
        )
    else:
        timings["preflight"] = 0.0

    preflight_queue = run_root / "preflight_selection.json"
    if not preflight_queue.is_file():
        timings["preflight_selection"] = run_command(
            command=selection_command(
                run_dir=preflight,
                candidates=candidates,
                replay=game.replay,
                iso_path=settings.melee_iso,
                controlled_port=game.controlled_port,
                out=preflight_queue,
                min_option_samples=PREFLIGHT_MIN_OPTION_SAMPLES,
            ),
            log_path=run_root / "logs" / "preflight_selection.log",
            state_path=state_path,
            state=state,
            stage=f"{game.id}:r{replicate}:preflight_selection",
        )
    else:
        timings["preflight_selection"] = 0.0

    preflight_payload = read_json(preflight_queue)
    refinement = run_root / "refinement"
    final_queue = run_root / "final_selection.json"
    if preflight_payload.get("targets"):
        if not (refinement / "summary.json").is_file():
            timings["refinement"] = run_command(
                command=msl_command(
                    ns,
                    metadata=preflight_queue,
                    out_dir=refinement,
                    offsets="0,1,2",
                    samples=int(args.refinement_samples),
                    dump_streams=False,
                ),
                log_path=run_root / "logs" / "refinement.log",
                state_path=state_path,
                state=state,
                stage=f"{game.id}:r{replicate}:refinement",
            )
        else:
            timings["refinement"] = 0.0
        if not final_queue.is_file():
            timings["final_selection"] = run_command(
                command=selection_command(
                    run_dir=refinement,
                    candidates=candidates,
                    replay=game.replay,
                    iso_path=settings.melee_iso,
                    controlled_port=game.controlled_port,
                    out=final_queue,
                    min_option_samples=REFINEMENT_MIN_OPTION_SAMPLES,
                    preflight_queue=preflight_queue,
                ),
                log_path=run_root / "logs" / "final_selection.log",
                state_path=state_path,
                state=state,
                stage=f"{game.id}:r{replicate}:final_selection",
            )
        else:
            timings["final_selection"] = 0.0
        final_payload = read_json(final_queue)
    else:
        timings["refinement"] = 0.0
        timings["final_selection"] = 0.0
        final_payload = {"targets": [], "selection_audit": {}}

    preflight_summary = read_json(preflight / "summary.json")
    refinement_summary = read_json(refinement / "summary.json") if (refinement / "summary.json").is_file() else {}
    result = {
        "status": "complete",
        "game": game.id,
        "replicate": replicate,
        "started_at": iso(started_at),
        "completed_at": iso(utc_now()),
        "timings": {key: round(value, 3) for key, value in timings.items()},
        "candidate_count": len((read_json(candidates).get("frames") or [])),
        "preflight_target_count": len(preflight_payload.get("targets") or []),
        "final_target_count": len(final_payload.get("targets") or []),
        "preflight_lanes": int(preflight_summary.get("laneCount") or 0),
        "refinement_lanes": int(refinement_summary.get("laneCount") or 0),
        "preflight": str(preflight.resolve()),
        "refinement": str(refinement.resolve()),
        "preflight_selection": str(preflight_queue.resolve()),
        "final_selection": str(final_queue.resolve()) if final_queue.is_file() else None,
    }
    write_json(completion, result)
    return result


def main() -> int:
    args = parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    games = load_games(args.manifest)
    settings = load_settings()
    state_path = args.out / "benchmark_state.json"
    state = read_json(state_path) if state_path.is_file() else initial_state(args, games)
    save_state(state_path, state, {"event": "runner_started", "pid": __import__("os").getpid()})
    deadline = datetime.fromisoformat(state["deadline_at"])
    completed_keys = {
        (str(item["game"]), int(item["replicate"]))
        for item in state.get("completed_jobs") or []
    }

    for replicate in range(1, int(args.max_replicates) + 1):
        for game in games:
            if (game.id, replicate) in completed_keys:
                continue
            remaining = (deadline - utc_now()).total_seconds()
            if remaining < float(args.minimum_start_minutes) * 60.0:
                state["status"] = "deadline_reached"
                state["active"] = None
                save_state(state_path, state, {"event": "deadline_stop", "remaining_seconds": remaining})
                return 0
            try:
                result = run_reference_job(
                    args=args,
                    game=game,
                    replicate=replicate,
                    out_root=args.out,
                    state_path=state_path,
                    state=state,
                    settings=settings,
                )
            except Exception as exc:
                failure = {
                    "game": game.id,
                    "replicate": replicate,
                    "error": f"{type(exc).__name__}: {exc}",
                    "at": iso(utc_now()),
                }
                state["failed_jobs"].append(failure)
                state["active"] = None
                save_state(state_path, state, {"event": "job_failed", **failure})
                continue
            state["completed_jobs"].append(result)
            completed_keys.add((game.id, replicate))
            state["active"] = None
            save_state(
                state_path,
                state,
                {
                    "event": "job_completed",
                    "game": game.id,
                    "replicate": replicate,
                    "final_targets": result["final_target_count"],
                },
            )

    state["status"] = "all_planned_jobs_complete"
    state["active"] = None
    save_state(state_path, state, {"event": "runner_completed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
