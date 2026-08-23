"""Export the replay and selected MSL route as synchronized interactive traces."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .config import load_settings
from .paths import PROJECT_DIR, module_command
from .pipeline import runtime_path, simulation_runtime_prefix, windows_to_wsl


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_target(queue: dict[str, Any], target_index: int, alternative_index: int) -> dict[str, Any]:
    targets = queue.get("targets") or []
    if target_index < 1 or target_index > len(targets):
        raise SystemExit(f"target index {target_index} outside 1..{len(targets)}")
    target = dict(targets[target_index - 1])
    if alternative_index:
        alternatives = target.get("alternative_routes") or []
        if alternative_index > len(alternatives):
            raise SystemExit(f"alternative index {alternative_index} outside 1..{len(alternatives)}")
        target.update(alternatives[alternative_index - 1])
        target["alternative_routes"] = []
    return target


def trace_alignment_frames(
    target: dict[str, Any], lane: dict[str, Any], stream: dict[str, Any]
) -> tuple[int, int]:
    trace_start = int(stream.get("takeoverFrame") or lane.get("takeoverFrame") or 0)
    model_control = int(
        stream.get("modelControlFrame")
        or lane.get("modelControlFrame")
        or target.get("takeover_frame")
        or trace_start
    )
    return trace_start, model_control


def run(command: list[str], log: Path) -> float:
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"trace export failed; see {log}")
    return time.perf_counter() - started


def decomp_command(module: str, values: list[object]) -> list[str]:
    settings = load_settings()
    if settings.msl_decomp_root is None or settings.slippi_ai_root is None:
        raise SystemExit("Decomp traces require paths.msl_decomp_root and paths.slippi_ai_root")
    runtime = SimpleNamespace(
        simulation_backend="decomp",
        msl_root=settings.msl_root,
        msl_decomp_root=settings.msl_decomp_root,
        slippi_ai_root=settings.slippi_ai_root,
        runtime_mode=settings.runtime_mode,
        runtime_python=settings.runtime_python,
        micromamba=settings.micromamba,
        msl_env=settings.msl_env,
        cuda_library_path=settings.cuda_library_path,
    )
    command = simulation_runtime_prefix(runtime, include_package_source=True) + ["-u", "-m", module]
    for value in values:
        command.append(runtime_path(runtime, value) if isinstance(value, Path) else str(value))
    return command


def decomp_source_path(value: object) -> str:
    raw = str(value)
    if raw.startswith("/"):
        return raw
    return windows_to_wsl(Path(raw))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", type=Path, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--alternative-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--msl-root", type=Path, default=load_settings().msl_root)
    args = parser.parse_args()

    queue_path = args.queue_json.resolve()
    queue = load_json(queue_path)
    backend = str(queue.get("simulation_backend") or "legacy")
    if backend == "legacy" and args.msl_root is None:
        raise SystemExit("Missing paths.msl_root configuration")
    if backend not in {"legacy", "decomp"}:
        raise SystemExit(f"Unsupported queue simulation backend: {backend}")
    target = selected_target(queue, args.target_index, args.alternative_index)
    lane = target.get("representative_lane") or {}
    stream = lane.get("controllerStream") or {}
    if not stream:
        raise SystemExit("selected route has no controller stream")
    # Source alignment always follows the trace start. The model-control frame
    # is normally identical, but remains separate for explicit bridge diagnostics.
    branch, model_control = trace_alignment_frames(target, lane, stream)
    end = int(stream.get("endFrame") or branch + int(queue.get("rollout_frames") or 180) - 1)
    clip_start = max(-123, branch - int(queue.get("video_preroll_frames") or 60))
    max_frames = max(1, end - branch + 1)
    out = args.out_dir.resolve()
    logs = out / "logs"
    msl_trace = out / "msl_lane.msltrace.json"
    replay_trace = out / "replay_window.msltrace.json"
    out.mkdir(parents=True, exist_ok=True)

    # lane_trace reads the primary route from a queue; create an isolated queue
    # so alternative routes are exported from their own controller stream.
    isolated = dict(queue)
    isolated["targets"] = [target]
    if backend == "decomp":
        isolated["replay"] = decomp_source_path(queue["replay"])
    isolated_queue = out / "selected_route.json"
    isolated_queue.write_text(json.dumps(isolated, indent=2) + "\n", encoding="utf-8")
    if backend == "decomp":
        decomp_root = load_settings().msl_decomp_root
        if decomp_root is None:
            raise SystemExit("Missing paths.msl_decomp_root configuration")
        agent_command = decomp_command(
            "slippi_ai_review.decomp_lane_trace",
            [
                "--queue-json", isolated_queue,
                "--target-index", 1,
                "--out", msl_trace,
                "--msl-root", decomp_root,
                "--max-frames", max_frames,
            ],
        )
        replay_command = decomp_command(
            "slippi_ai_review.decomp_replay_trace",
            [
                "--replay", decomp_source_path(queue["replay"]),
                "--start-frame", clip_start,
                "--end-frame", end,
                "--out", replay_trace,
                "--msl-root", decomp_root,
            ],
        )
    else:
        agent_command = module_command(
            "lane_trace", "--queue-json", isolated_queue, "--target-index", 1,
            "--out", msl_trace, "--msl-root", args.msl_root, "--max-frames", max_frames,
        )
        replay_command = module_command(
            "replay_trace", "--replay", queue["replay"], "--start-frame", clip_start,
            "--end-frame", end, "--out", replay_trace, "--msl-root", args.msl_root,
        )
    with ThreadPoolExecutor(max_workers=2) as pool:
        agent = pool.submit(run, agent_command, logs / "agent_trace.log")
        replay = pool.submit(run, replay_command, logs / "replay_trace.log")
        timings = {"agentTraceSeconds": agent.result(), "replayTraceSeconds": replay.result()}
    manifest = {
        "queue_json": str(queue_path), "target_index": args.target_index,
        "alternative_index": args.alternative_index, "target": target,
        "artifact_mode": "interactive-traces", "simulation_backend": backend,
        "branch_start_frame": branch,
        "model_control_frame": model_control,
        "clip_start_frame": clip_start, "branch_switch_frame": branch - clip_start,
        "msl_trace": str(msl_trace), "replay_trace": str(replay_trace), "timings": timings,
    }
    manifest_path = out / "trace_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "timings": timings}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
