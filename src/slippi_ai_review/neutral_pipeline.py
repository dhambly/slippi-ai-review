"""Run neutral-loss lookback discovery, refinement, trace export, and reporting."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .neutral_selection import build_neutral_queue, preflight_boundaries, read_jsonl
from .paths import PROJECT_DIR, module_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--raw-events-json", type=Path, help="Reuse canonical replay conversions from candidate generation.")
    parser.add_argument("--analyzed-port", type=int, choices=(1, 2), default=1)
    parser.add_argument("--display-name")
    parser.add_argument("--slp-version")
    parser.add_argument("--experiment-json", type=Path, help="Reuse a completed neutral_experiment.json preflight.")
    parser.add_argument("--refinement-dir", type=Path, help="Reuse a completed refinement directory with lane streams.")
    parser.add_argument("--step-frames", type=int, default=10)
    parser.add_argument("--max-lookback-frames", type=int, default=80)
    parser.add_argument("--post-opening-frames", type=int, default=30)
    parser.add_argument("--preflight-samples", type=int, default=48)
    parser.add_argument("--refinement-samples", type=int, default=192)
    parser.add_argument("--route-grid-samples", type=int, default=64, help="Samples for each visible 10f insertion route.")
    parser.add_argument("--route-grid-post-opening-frames", type=int, default=90, help="Continuation shown after the recorded opening.")
    parser.add_argument("--consistency-threshold", type=float, default=0.75)
    parser.add_argument("--confidence-z", type=float, default=1.645)
    parser.add_argument("--min-option-samples", type=int, default=8)
    parser.add_argument("--min-option-share", type=float, default=0.10)
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.20)
    parser.add_argument("--video-preroll-frames", type=int, default=60)
    parser.add_argument("--max-batch-lanes", type=int, default=1024)
    parser.add_argument("--artifact-workers", type=int, default=2)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--msl-root", type=Path)
    parser.add_argument("--slippi-ai-root", type=Path)
    parser.add_argument("--publish-dir", type=Path, help="Copy neutral artifacts beside an existing advantage report.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def run(command: list[str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            text=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(f"command failed with exit {completed.returncode}; see {log}")
    return time.perf_counter() - started


def optional_path_flags(args: argparse.Namespace) -> list[str]:
    result = []
    for flag, value in (
        ("--model", args.model),
        ("--msl-root", args.msl_root),
        ("--slippi-ai-root", args.slippi_ai_root),
    ):
        if value is not None:
            result.extend([flag, str(value.resolve())])
    return result


def publish_neutral_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("neutral_review.html", "disadvantage_review.html", "neutral_avoidance.json"):
        shutil.copy2(source / name, destination / name)
    for folder in ("traces", "viewer"):
        source_folder = source / folder
        destination_folder = destination / folder
        destination_folder.mkdir(parents=True, exist_ok=True)
        for item in source_folder.iterdir():
            if item.is_file():
                shutil.copy2(item, destination_folder / item.name)


def main() -> int:
    args = parse_args()
    if args.experiment_json is None and args.replay is None:
        raise SystemExit("--replay is required unless --experiment-json is supplied")
    if not 0 < args.consistency_threshold <= 1:
        raise SystemExit("--consistency-threshold must be in (0, 1]")
    if not 0 < args.gpu_duty_cycle <= 1:
        raise SystemExit("--gpu-duty-cycle must be in (0, 1]")
    out = args.out.resolve()
    logs = out / "logs"
    out.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    cached_experiment = out / "preflight" / "neutral_experiment.json"
    if args.experiment_json is not None:
        experiment_path = args.experiment_json.resolve()
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        replay = Path(experiment["replay"]).resolve()
        analyzed_port = int(experiment["analyzedPort"])
    elif cached_experiment.is_file() and not args.force:
        experiment_path = cached_experiment
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
        replay = Path(experiment["replay"]).resolve()
        analyzed_port = int(experiment["analyzedPort"])
    else:
        replay = args.replay.resolve()
        analyzed_port = int(args.analyzed_port)
        preflight_dir = out / "preflight"
        command = module_command(
            "neutral_experiment",
            "--replay", replay,
            "--analyzed-port", analyzed_port,
            "--kind", "avoid",
            "--max-events-per-kind", 0,
            "--step-frames", args.step_frames,
            "--max-lookback-frames", args.max_lookback_frames,
            "--post-opening-frames", args.post_opening_frames,
            "--samples-per-point", args.preflight_samples,
            "--consistency-threshold", args.consistency_threshold,
            "--warmup-frames", args.warmup_frames,
            "--max-batch-lanes", args.max_batch_lanes,
            "--out", preflight_dir,
            *optional_path_flags(args),
        )
        if args.raw_events_json is not None:
            command.extend(["--raw-events-json", args.raw_events_json.resolve()])
        if args.enable_gpu:
            command.extend(["--enable-gpu", "--gpu-duty-cycle", str(args.gpu_duty_cycle)])
        timings["preflight"] = run(command, logs / "preflight.log")
        experiment_path = preflight_dir / "neutral_experiment.json"
        experiment = json.loads(experiment_path.read_text(encoding="utf-8"))

    selected, preflight_omitted = preflight_boundaries(
        experiment,
        threshold=float(args.consistency_threshold),
        z=float(args.confidence_z),
    )
    refinement_dir = out / "refinement"
    refinement_rows: list[dict[str, Any]] = []
    cached_refinement = refinement_dir / "lanes.jsonl"
    if selected and args.refinement_dir is not None:
        refinement_dir = args.refinement_dir.resolve()
        refinement_rows = read_jsonl(refinement_dir / "lanes.jsonl")
    elif selected and cached_refinement.is_file() and not args.force:
        refinement_rows = read_jsonl(cached_refinement)
    elif selected:
        unique_points = {}
        for event in selected:
            frame = int(event["boundaryPoint"]["injectionFrame"])
            unique_points[frame] = {
                "frame": frame,
                "opportunity_role": "neutral_loss_avoidance",
                "point_evidence": {
                    "opening_frame": int(event["openingFrame"]),
                    "lookback_frames": int(event["boundaryPoint"]["lookbackFrames"]),
                },
            }
        refinement_points = out / "refinement_points.json"
        refinement_points.write_text(
            json.dumps({"frames": [unique_points[key] for key in sorted(unique_points)]}, indent=2) + "\n",
            encoding="utf-8",
        )
        max_selected_lookback = max(int(event["boundaryPoint"]["lookbackFrames"]) for event in selected)
        rollout_frames = max_selected_lookback + int(experiment["postOpeningFrames"]) + 1
        command = module_command(
            "simulation",
            "--replay", replay,
            "--analyzed-port", analyzed_port,
            "--takeover-frames-json", refinement_points,
            "--offsets", "0",
            "--samples-per-point", args.refinement_samples,
            "--max-batch-lanes", args.max_batch_lanes,
            "--objective", "general",
            "--rollout-frames", rollout_frames,
            "--defense-resolution-extra-frames", 0,
            "--defense-resolution-min-frames", rollout_frames,
            "--defender-delay-frames", rollout_frames + 1,
            "--warmup-frames", args.warmup_frames,
            "--history-mode", "teacher-forced",
            "--opponent-mode", "replay",
            "--rng-mode", "replay",
            "--dump-controller-streams",
            "--out", refinement_dir,
            *optional_path_flags(args),
        )
        if args.enable_gpu:
            command.extend(["--enable-gpu", "--gpu-duty-cycle", str(args.gpu_duty_cycle)])
        # GPU is deliberately not enabled. This analysis may run beside active MSL development.
        timings["refinement"] = run(command, logs / "refinement.log")
        refinement_rows = read_jsonl(refinement_dir / "lanes.jsonl")
    else:
        refinement_dir.mkdir(parents=True, exist_ok=True)
        (refinement_dir / "lanes.jsonl").write_text("", encoding="utf-8")

    # The confidence refinement proves that a loss is worth inspecting. A second,
    # smaller grid then records Phillip's actual best route at every 10f lead time
    # so the report can show the difference between early agency and late escapes.
    route_grid_dir = out / "route_grid"
    route_grid_rows: list[dict[str, Any]] = []
    cached_route_grid = route_grid_dir / "lanes.jsonl"
    if selected and cached_route_grid.is_file() and not args.force:
        route_grid_rows = read_jsonl(cached_route_grid)
    elif selected:
        route_grid_points: dict[int, dict[str, Any]] = {}
        for event in selected:
            opening = int(event["openingFrame"])
            for lookback in range(int(args.step_frames), int(args.max_lookback_frames) + 1, int(args.step_frames)):
                frame = opening - lookback
                route_grid_points[frame] = {
                    "frame": frame,
                    "opportunity_role": "neutral_loss_insertion_explorer",
                    "point_evidence": {"opening_frame": opening, "lookback_frames": lookback},
                }
        grid_points_path = out / "route_grid_points.json"
        grid_points_path.write_text(
            json.dumps({"frames": [route_grid_points[key] for key in sorted(route_grid_points)]}, indent=2) + "\n",
            encoding="utf-8",
        )
        grid_rollout = int(args.max_lookback_frames) + int(args.route_grid_post_opening_frames) + 1
        command = module_command(
            "simulation",
            "--replay", replay,
            "--analyzed-port", analyzed_port,
            "--takeover-frames-json", grid_points_path,
            "--offsets", "0",
            "--samples-per-point", args.route_grid_samples,
            "--max-batch-lanes", args.max_batch_lanes,
            "--objective", "general",
            "--rollout-frames", grid_rollout,
            "--defense-resolution-extra-frames", 0,
            "--defense-resolution-min-frames", grid_rollout,
            "--defender-delay-frames", grid_rollout + 1,
            "--warmup-frames", args.warmup_frames,
            "--history-mode", "teacher-forced",
            "--opponent-mode", "replay",
            "--rng-mode", "replay",
            "--dump-controller-streams",
            "--out", route_grid_dir,
            *optional_path_flags(args),
        )
        if args.enable_gpu:
            command.extend(["--enable-gpu", "--gpu-duty-cycle", str(args.gpu_duty_cycle)])
        timings["routeGrid"] = run(command, logs / "route_grid.log")
        route_grid_rows = read_jsonl(route_grid_dir / "lanes.jsonl")
    else:
        route_grid_dir.mkdir(parents=True, exist_ok=True)
        (route_grid_dir / "lanes.jsonl").write_text("", encoding="utf-8")

    queue = build_neutral_queue(
        experiment=experiment,
        selected_events=selected,
        refinement_rows=refinement_rows,
        refinement_dir=refinement_dir,
        route_grid_rows=route_grid_rows,
        threshold=float(args.consistency_threshold),
        min_option_samples=int(args.min_option_samples),
        min_option_share=float(args.min_option_share),
        z=float(args.confidence_z),
        video_preroll_frames=int(args.video_preroll_frames),
    )
    queue["display_name"] = args.display_name or replay.stem
    if args.slp_version:
        queue["slp_version"] = args.slp_version
    queue["selection_audit"]["preflightOmitted"] = preflight_omitted
    queue_path = out / "neutral_queue.json"
    queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    artifact_root = out / "artifacts_build"
    artifact_command = module_command(
        "neutral_artifacts",
        "--queue-json", queue_path,
        "--out-root", artifact_root,
        "--workers", args.artifact_workers,
    )
    if args.msl_root is not None:
        artifact_command.extend(["--msl-root", args.msl_root.resolve()])
    timings["artifacts"] = run(artifact_command, logs / "artifacts.log")
    final_artifacts = artifact_root / "final_artifacts"
    if args.publish_dir is not None:
        publish_neutral_artifacts(final_artifacts, args.publish_dir.resolve())

    timings["total"] = time.perf_counter() - started
    summary = {
        "status": "ok",
        "replay": str(replay),
        "analyzedPort": analyzed_port,
        "experiment": str(experiment_path),
        "preflightSelected": len(selected),
        "finalTargets": len(queue.get("targets") or []),
        "routeGridRows": len(route_grid_rows),
        "report": str((final_artifacts / "neutral_review.html").resolve()),
        "publishedReport": (
            str((args.publish_dir.resolve() / "neutral_review.html"))
            if args.publish_dir is not None
            else None
        ),
        "gpuEnabled": bool(args.enable_gpu),
        "timings": {key: round(value, 3) for key, value in timings.items()},
    }
    (out / "neutral_pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
