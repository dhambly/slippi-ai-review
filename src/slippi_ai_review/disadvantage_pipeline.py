"""Run refined defensive simulations for every target-player disadvantage segment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .disadvantage_report import write_placeholder
from .paths import PROJECT_DIR, module_command
from .phase_sweep_selection import build_queue, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--analyzed-port", type=int, choices=(1, 2), required=True)
    parser.add_argument("--display-name")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--rollout-frames", type=int, default=180)
    parser.add_argument("--opponent-takeover-max-delay-frames", type=int, default=60)
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--max-batch-lanes", type=int, default=4096)
    parser.add_argument("--artifact-workers", type=int, default=2)
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.20)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--msl-root", type=Path)
    parser.add_argument("--slippi-ai-root", type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            [str(value) for value in command],
            cwd=PROJECT_DIR,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed with exit {completed.returncode}; see {log}")
    return time.perf_counter() - started


def disadvantage_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    segments = [item for item in inventory.get("segments") or [] if item.get("phase") == "disadvantage"]
    segment_ids = {item.get("id") for item in segments}
    frames = [
        item
        for item in inventory.get("frames") or []
        if ((item.get("point_evidence") or {}).get("sweep_id")) in segment_ids
    ]
    return {
        **inventory,
        "segments": segments,
        "frames": frames,
        "counts": {"advantage": 0, "neutral": 0, "disadvantage": len(segments)},
    }


def optional_path_flags(args: argparse.Namespace) -> list[Any]:
    flags: list[Any] = []
    for flag, value in (
        ("--model", args.model),
        ("--msl-root", args.msl_root),
        ("--slippi-ai-root", args.slippi_ai_root),
    ):
        if value is not None:
            flags.extend([flag, value.resolve()])
    return flags


def publish_disadvantage_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "disadvantage_review.html", destination / "disadvantage_review.html")
    defense_manifest = source / "disadvantage_defense.json"
    if defense_manifest.is_file():
        shutil.copy2(defense_manifest, destination / defense_manifest.name)
    destination_traces = destination / "traces"
    destination_traces.mkdir(parents=True, exist_ok=True)
    for trace in (source / "traces").glob("disadvantage_*"):
        shutil.copy2(trace, destination_traces / trace.name)
    destination_viewer = destination / "viewer"
    destination_viewer.mkdir(parents=True, exist_ok=True)
    source_viewer = source / "viewer"
    if source_viewer.is_dir():
        for item in source_viewer.iterdir():
            if item.is_file():
                shutil.copy2(item, destination_viewer / item.name)


def main() -> int:
    args = parse_args()
    if args.samples < 1 or args.rollout_frames < 1 or args.opponent_takeover_max_delay_frames < 1:
        raise SystemExit("--samples, --rollout-frames, and --opponent-takeover-max-delay-frames must be positive")
    if not 0 < args.gpu_duty_cycle <= 1:
        raise SystemExit("--gpu-duty-cycle must be in (0, 1]")

    out = args.out.resolve()
    logs = out / "logs"
    simulation_dir = out / "simulation"
    artifacts_root = out / "artifacts_build"
    final_artifacts = artifacts_root / "final_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    inventory = disadvantage_inventory(json.loads(args.inventory.resolve().read_text(encoding="utf-8")))
    # The whole-game inventory may have been produced by the Windows
    # orchestrator, while this pipeline runs inside WSL. Keep all downstream
    # trace exporters in the current path namespace.
    inventory["replay"] = str(args.replay.resolve())
    inventory_path = out / "disadvantage_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    queue_path = out / "disadvantage_queue.json"

    if inventory["segments"]:
        lanes_path = simulation_dir / "lanes.jsonl"
        if args.force or not lanes_path.is_file():
            command = module_command(
                "simulation",
                "--replay", args.replay.resolve(),
                "--analyzed-port", args.analyzed_port,
                "--takeover-frames-json", inventory_path,
                "--offsets", "0",
                "--samples-per-point", args.samples,
                "--max-batch-lanes", args.max_batch_lanes,
                "--objective", "general",
                "--rollout-frames", args.rollout_frames,
                "--defense-resolution-extra-frames", 0,
                "--defense-resolution-min-frames", args.rollout_frames,
                "--defender-delay-frames", args.opponent_takeover_max_delay_frames,
                "--defender-takeover-mode", "observed-opponent-followup",
                "--warmup-frames", args.warmup_frames,
                "--history-mode", "teacher-forced",
                "--opponent-mode", "replay",
                "--rng-mode", "replay",
                "--sample-temperature", "1.0",
                "--option-horizon-frames", min(120, args.rollout_frames),
                "--dump-controller-streams",
                "--out", simulation_dir,
                *optional_path_flags(args),
            )
            if args.enable_gpu:
                command.extend(["--enable-gpu", "--gpu-duty-cycle", args.gpu_duty_cycle])
            timings["simulation"] = run(command, logs / "simulation.log")

        queue = build_queue(inventory, read_jsonl(lanes_path))
        queue["mode"] = "disadvantage-defense"
        queue["display_name"] = args.display_name or args.replay.stem
        queue["rollout_frames"] = int(args.rollout_frames)
        queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")

        artifact_command = module_command(
            "artifacts",
            "--queue-json", queue_path,
            "--out-root", artifacts_root,
            "--workers", args.artifact_workers,
            "--trace-prefix", "disadvantage_",
        )
        if args.msl_root is not None:
            artifact_command.extend(["--msl-root", args.msl_root.resolve()])
        timings["artifacts"] = run(artifact_command, logs / "artifacts.log")
        generic_manifest = final_artifacts / "advantage_improvements.json"
        defense_manifest = final_artifacts / "disadvantage_defense.json"
        shutil.copy2(generic_manifest, defense_manifest)
        timings["report"] = run(
            module_command(
                "disadvantage_report",
                "--manifest", defense_manifest,
                "--out", final_artifacts / "disadvantage_review.html",
            ),
            logs / "report.log",
        )
    else:
        final_artifacts.mkdir(parents=True, exist_ok=True)
        write_placeholder(
            final_artifacts / "disadvantage_review.html",
            display_name=args.display_name or args.replay.stem,
            controlled_port=args.analyzed_port,
        )

    timings["total"] = time.perf_counter() - started
    summary = {
        "status": "ok",
        "replay": str(args.replay.resolve()),
        "analyzedPort": int(args.analyzed_port),
        "segments": len(inventory["segments"]),
        "samplesPerSegment": int(args.samples),
        "opponentTakeoverMaxDelayFrames": int(args.opponent_takeover_max_delay_frames),
        "report": str((final_artifacts / "disadvantage_review.html").resolve()),
        "timings": {key: round(value, 3) for key, value in timings.items()},
    }
    (out / "disadvantage_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
