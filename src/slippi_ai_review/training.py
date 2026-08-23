"""Export a selected MSL route to Training Mode CE and optionally launch it."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import load_settings
from .paths import PROJECT_DIR, module_command
from .practice_start import select_practice_start


WORK_DIR = PROJECT_DIR
SETTINGS = load_settings()
DEFAULT_DOLPHIN = SETTINGS.dolphin
DEFAULT_DOLPHIN_USER = SETTINGS.dolphin_user_dir
DEFAULT_CARD_DIR = DEFAULT_DOLPHIN_USER / "GC" / "USA" / "Card A" if DEFAULT_DOLPHIN_USER else None
DEFAULT_ISO = SETTINGS.ce_iso
DEFAULT_TM_REPLAY = SETTINGS.tm_replay
DEFAULT_MSL_ROOT = SETTINGS.msl_root
CARD_FILENAME = "01-GTME-TMREC_CODEX_AUTOLOAD.gci"
INTERNAL_FILENAME = "TMREC_CODEX_AUTOLOAD"
LAUNCH_STATE = SETTINGS.data_dir / "training_mode_ce_state.json"
DEFAULT_PREROLL_FRAMES = 30
MAX_PRACTICE_LEADIN_FRAMES = 200
SCENARIO_MODES = ("replay", "phillip", "variations")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def host_path(value: str | Path) -> Path:
    raw = str(value)
    if raw.startswith("/mnt/") and len(raw) >= 7 and raw[6] == "/":
        suffix = raw[7:].replace("/", "\\")
        return Path(f"{raw[5].upper()}:\\{suffix}")
    return Path(raw)


def select_route(
    queue: dict[str, Any], target_index: int, alternative_index: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    targets = queue.get("targets") or []
    if target_index < 1 or target_index > len(targets):
        raise ValueError(f"target index {target_index} outside 1..{len(targets)}")
    target = targets[target_index - 1]
    if alternative_index < 0:
        raise ValueError("alternative index must be zero or greater")
    if alternative_index == 0:
        route = target
    else:
        alternatives = target.get("alternative_routes") or []
        if alternative_index > len(alternatives):
            raise ValueError(
                f"alternative index {alternative_index} outside 0..{len(alternatives)}"
            )
        route = alternatives[alternative_index - 1]
    lane = route.get("representative_lane") or {}
    stream = lane.get("controllerStream") or {}
    if not stream:
        raise ValueError("selected route has no controller stream")
    return target, lane, stream


def derive_practice_window(
    target: dict[str, Any], stream: dict[str, Any], preroll_frames: int
) -> tuple[int, int, int]:
    if preroll_frames < 0:
        raise ValueError("preroll frames must be zero or greater")
    takeover_frame = int(stream.get("takeoverFrame") or target.get("takeover_frame"))
    end_frame = int(stream.get("endFrame") or takeover_frame)
    baseline = target.get("replay_baseline") or {}
    presentation = baseline.get("presentation_segment") or {}
    opening_move = baseline.get("sequence_opening_move") or {}
    opening_candidates = [
        opening_move.get("frame"),
        baseline.get("segment_start_frame"),
        presentation.get("openingFrame"),
    ]
    valid_openings = [
        int(frame)
        for frame in opening_candidates
        if frame is not None and int(frame) <= takeover_frame
    ]
    opening_frame = valid_openings[0] if valid_openings else takeover_frame
    practice_start_frame = max(
        -123,
        takeover_frame - MAX_PRACTICE_LEADIN_FRAMES,
        opening_frame - max(DEFAULT_PREROLL_FRAMES, preroll_frames),
    )
    return practice_start_frame, opening_frame, end_frame


def build_tm_replay_command(
    tm_replay: Path,
    synthetic_slp: Path,
    generated_gci: Path,
    *,
    start_frame: int,
    duration: int,
    display_name: str,
    human_port: int,
    cpu_handoff_frame: int | None = None,
) -> list[str]:
    command = [
        str(tm_replay.resolve()),
        "--slp-file",
        str(synthetic_slp),
        "--slippi-frame",
        str(start_frame),
        "--num-frames",
        str(duration),
        "--output-file",
        str(generated_gci),
        "--name",
        display_name,
        "--human-port",
        str(human_port),
        "--hmn-mode",
        "playback",
        "--cpu-mode",
        "playback",
        "--auto-restore",
        "--internal-name",
        INTERNAL_FILENAME,
    ]
    if cpu_handoff_frame is not None:
        command.extend(("--cpu-handoff-frame", str(cpu_handoff_frame)))
    return command


def run_logged(command: list[str], log_path: Path, *, cwd: Path = WORK_DIR) -> float:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ " + subprocess.list2cmdline(command) + "\n\n"
        + result.stdout
        + ("\n[stderr]\n" + result.stderr if result.stderr else "")
        + f"\n[exit={result.returncode} elapsed={elapsed:.3f}s]\n",
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log_path}")
    return elapsed


def resolved_safe_frame(log_path: Path) -> int:
    match = re.search(
        r"^Resolved safe Slippi frame: (-?\d+)$",
        log_path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise RuntimeError(f"tm_replay did not report its resolved safe frame; see {log_path}")
    return int(match.group(1))


def install_gci(source: Path, card_dir: Path) -> Path:
    card_dir.mkdir(parents=True, exist_ok=True)
    destination = card_dir / CARD_FILENAME
    with tempfile.NamedTemporaryFile(
        prefix=f".{CARD_FILENAME}.", suffix=".tmp", dir=card_dir, delete=False
    ) as stream:
        temporary = Path(stream.name)
        with source.open("rb") as source_stream:
            shutil.copyfileobj(source_stream, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def tracked_dolphin() -> dict[str, Any] | None:
    if not LAUNCH_STATE.is_file():
        return None
    try:
        state = load_json(LAUNCH_STATE)
        pid = int(state.get("pid") or 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    if pid <= 0:
        return None
    script = (
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}';"
        "if($p){$p|Select-Object ProcessId,ExecutablePath,CommandLine|ConvertTo-Json -Compress}"
    )
    probe = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode or not probe.stdout.strip():
        return None
    try:
        process = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return None
    executable = Path(str(process.get("ExecutablePath") or ""))
    command_line = str(process.get("CommandLine") or "")
    if executable.name.lower() != "slippi dolphin.exe" or str(DEFAULT_ISO).lower() not in command_line.lower():
        return None
    return {**state, "pid": pid}


def stop_tracked_dolphin() -> int | None:
    state = tracked_dolphin()
    if state is None:
        return None
    pid = int(state["pid"])
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        text=True,
        capture_output=True,
        check=False,
    )
    return pid


def launch_dolphin(dolphin: Path, iso: Path, user_dir: Path) -> int:
    stop_tracked_dolphin()
    process = subprocess.Popen(
        [str(dolphin), "-e", str(iso), "-u", str(user_dir), "-b"],
        cwd=dolphin.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    LAUNCH_STATE.parent.mkdir(parents=True, exist_ok=True)
    LAUNCH_STATE.write_text(
        json.dumps(
            {
                "pid": process.pid,
                "iso": str(iso.resolve()),
                "dolphin": str(dolphin.resolve()),
                "userDir": str(user_dir.resolve()),
                "launchedAt": time.time(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return process.pid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", required=True, type=Path)
    parser.add_argument("--target-index", required=True, type=int)
    parser.add_argument("--alternative-index", type=int, default=0)
    parser.add_argument(
        "--scenario-mode",
        choices=SCENARIO_MODES,
        default="phillip",
        help="replay preserves both original controller streams; phillip applies the staged MSL rollout",
    )
    parser.add_argument("--preroll-frames", type=int, default=DEFAULT_PREROLL_FRAMES)
    parser.add_argument(
        "--variation-start-frame",
        type=int,
        help="First Slippi frame controlled by Training Mode CPU logic in variations mode",
    )
    parser.add_argument(
        "--variation-source",
        choices=("replay", "rollout"),
        default="replay",
        help="Controller trajectory used before the random-defense handoff",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--card-dir", type=Path, default=DEFAULT_CARD_DIR)
    parser.add_argument("--tm-replay", type=Path, default=DEFAULT_TM_REPLAY)
    parser.add_argument("--msl-root", type=Path, default=DEFAULT_MSL_ROOT)
    parser.add_argument("--dolphin", type=Path, default=DEFAULT_DOLPHIN)
    parser.add_argument("--dolphin-user-dir", type=Path, default=DEFAULT_DOLPHIN_USER)
    parser.add_argument("--iso", type=Path, default=DEFAULT_ISO)
    parser.add_argument("--launch", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    queue_path = args.queue_json.resolve()
    queue = load_json(queue_path)
    target, lane, stream = select_route(queue, args.target_index, args.alternative_index)
    takeover_frame = int(stream.get("takeoverFrame") or target.get("takeover_frame"))
    try:
        practice_start_frame, opening_frame, end_frame = derive_practice_window(
            target, stream, args.preroll_frames
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    baseline = target.get("replay_baseline") or {}
    opening_move = baseline.get("sequence_opening_move") or {}
    opening_action_frame = int(opening_move.get("actionStartFrame") or opening_frame)
    analyzed_port = int(stream.get("analyzedPort") or queue.get("controlled_port") or 0)
    defender_port = int(stream.get("defenderPort") or 0)
    source_replay = host_path(queue.get("replay") or "").resolve()
    selection_started = time.perf_counter()
    start_selection: dict[str, Any]
    try:
        if analyzed_port < 1 or defender_port < 1 or not source_replay.is_file():
            raise ValueError("source replay or player ports are unavailable")
        decision = select_practice_start(
            source_replay,
            players={analyzed_port - 1, defender_port - 1},
            takeover_frame=takeover_frame,
            opening_frame=opening_frame,
            opening_action_frame=opening_action_frame,
            preroll_frames=args.preroll_frames,
            maximum_leadin_frames=MAX_PRACTICE_LEADIN_FRAMES,
            default_frame=practice_start_frame,
            opening_type=str(baseline.get("opening_type") or ""),
        )
        practice_start_frame = decision.frame
        start_selection = decision.to_dict()
        if decision.mode == "default_no_stable_candidate":
            raise ValueError(
                f"no restorable playback start was found within {MAX_PRACTICE_LEADIN_FRAMES} frames"
            )
    except (OSError, ValueError, struct.error) as exc:
        raise SystemExit(f"Training Mode export blocked: {exc}") from exc
    selection_seconds = time.perf_counter() - selection_started
    rollout_duration = end_frame - takeover_frame + 1
    duration = end_frame - practice_start_frame + 1
    if duration <= 0 or duration > 3600:
        raise SystemExit(f"Invalid controller stream duration: {duration} frames")
    human_port = analyzed_port
    if human_port < 1 or human_port > 4:
        raise SystemExit(f"Invalid analyzed port: {human_port}")
    variation_start_frame = None
    if args.scenario_mode == "variations":
        variation_start_frame = int(
            args.variation_start_frame if args.variation_start_frame is not None else opening_frame
        )
        if variation_start_frame <= practice_start_frame or variation_start_frame > end_frame:
            raise SystemExit(
                f"Variation start frame must be within {practice_start_frame + 1}..{end_frame}"
            )
    variation_source = args.variation_source if args.scenario_mode == "variations" else None
    uses_rollout = args.scenario_mode == "phillip" or variation_source == "rollout"

    out_dir = (
        args.out_dir
        or queue_path.parent
        / "training_mode"
        / (
            f"target_{args.target_index:02d}_route_{args.alternative_index:02d}_{args.scenario_mode}"
            + (f"_{variation_source}" if variation_source else "")
        )
    ).resolve()
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic_slp = out_dir / "scenario.slp"
    generated_gci = out_dir / CARD_FILENAME

    missing = [name for name in ("tm_replay", "msl_root", "card_dir") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing Training Mode configuration: {', '.join(missing)}")
    for required in (args.tm_replay, args.msl_root):
        if not required.exists():
            raise SystemExit(f"Required path does not exist: {required}")

    export_command = module_command(
        "synthetic_slp",
        "--queue-json",
        str(queue_path),
        "--target-index",
        str(args.target_index),
        "--alternative-index",
        str(args.alternative_index),
        "--out",
        str(synthetic_slp),
        "--msl-root",
        str(args.msl_root),
        "--max-frames",
        str(rollout_duration),
        "--patch-mode",
        "full" if uses_rollout else "none",
    )
    timings = {
        "practiceStartSelectionSeconds": selection_seconds,
        "syntheticSlpSeconds": run_logged(export_command, logs_dir / "synthetic_slp.log"),
    }

    display_name = (
        f"CODEX REPLAY s{practice_start_frame}"
        if args.scenario_mode == "replay"
        else (
            f"CODEX RND {'MSL' if variation_source == 'rollout' else 'RPL'} s{practice_start_frame} f{variation_start_frame}"
            if args.scenario_mode == "variations"
            else f"CODEX P{human_port} s{practice_start_frame} t{takeover_frame}"
        )
    )
    tm_command = build_tm_replay_command(
        args.tm_replay,
        synthetic_slp,
        generated_gci,
        start_frame=practice_start_frame,
        duration=duration,
        display_name=display_name,
        human_port=human_port,
        cpu_handoff_frame=variation_start_frame,
    )
    tm_replay_log = logs_dir / "tm_replay.log"
    timings["gciExportSeconds"] = run_logged(tm_command, tm_replay_log)
    safe_start_frame = resolved_safe_frame(tm_replay_log)
    if safe_start_frame > practice_start_frame:
        raise RuntimeError(
            f"tm_replay resolved safe frame {safe_start_frame} after requested frame {practice_start_frame}"
        )
    installed_gci = install_gci(generated_gci, args.card_dir.resolve())

    pid = None
    if args.launch:
        missing = [name for name in ("dolphin", "iso", "dolphin_user_dir") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"Missing Training Mode launch configuration: {', '.join(missing)}")
        for required in (args.dolphin, args.iso, args.dolphin_user_dir):
            if not required.exists():
                raise SystemExit(f"Required launch path does not exist: {required}")
        launch_started = time.perf_counter()
        pid = launch_dolphin(args.dolphin.resolve(), args.iso.resolve(), args.dolphin_user_dir.resolve())
        timings["launchSeconds"] = time.perf_counter() - launch_started

    timings["totalSeconds"] = time.perf_counter() - started
    status = {
        "ok": True,
        "queueJson": str(queue_path),
        "targetIndex": args.target_index,
        "alternativeIndex": args.alternative_index,
        "scenarioMode": args.scenario_mode,
        "variationSource": variation_source,
        "laneId": lane.get("laneId"),
        "requestedPracticeStartFrame": practice_start_frame,
        "practiceStartFrame": safe_start_frame,
        "practiceStartSelection": start_selection,
        "openingHitFrame": opening_frame,
        "openingHitIncluded": safe_start_frame <= opening_frame,
        "preRollFrames": max(0, opening_frame - safe_start_frame),
        "omittedContextFrames": max(0, safe_start_frame - opening_frame),
        "takeoverLeadInFrames": takeover_frame - safe_start_frame,
        "takeoverFrame": takeover_frame if uses_rollout else None,
        "variationStartFrame": variation_start_frame,
        "endFrame": end_frame,
        "durationFrames": end_frame - safe_start_frame + 1,
        "rolloutDurationFrames": rollout_duration,
        "humanPort": human_port,
        "humanMode": "playback",
        "cpuPort": stream.get("defenderPort"),
        "cpuMode": "playback",
        "defenderTakeoverFrame": (
            stream.get("defenderTakeoverFrame") if uses_rollout else None
        ),
        "syntheticSlp": str(synthetic_slp),
        "generatedGci": str(generated_gci),
        "installedGci": str(installed_gci),
        "dolphinPid": pid,
        "timings": {key: round(value, 3) for key, value in timings.items()},
    }
    status_path = out_dir / "training_mode_export.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
