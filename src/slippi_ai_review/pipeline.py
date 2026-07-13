"""Run the current MSL advantage-review pipeline for one UUID job."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from .config import load_settings
from .paths import PACKAGE_DIR, PROJECT_DIR, module_command


WORK_DIR = PROJECT_DIR
PYTHON = sys.executable


def windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    posix = resolved.as_posix()
    return f"/mnt/{drive}{posix[len(resolved.drive):]}" if drive else posix


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_queue_metadata(path: Path, *, display_name: str, slp_version: str | None) -> None:
    payload = read_json(path)
    changed = payload.get("display_name") != display_name
    payload["display_name"] = display_name
    if slp_version and payload.get("slp_version") != slp_version:
        payload["slp_version"] = slp_version
        changed = True
    if not changed:
        return
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def emit(stage: str, message: str, **details: object) -> None:
    print(json.dumps({"event": "progress", "stage": stage, "message": message, **details}), flush=True)


def run_stage(stage: str, command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    emit(stage, f"Running {stage}.")
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=WORK_DIR,
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
                    child_event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(child_event, dict) and child_event.get("chunk") is not None:
                    emit(
                        stage,
                        f"{stage.replace('_', ' ').title()}: finished simulation chunk {int(child_event['chunk']) + 1}.",
                        chunk=int(child_event["chunk"]),
                        lanes=int(child_event.get("lanes") or 0),
                        seconds=float(child_event.get("seconds") or 0.0),
                    )
        return_code = process.wait()
    elapsed = time.perf_counter() - started
    if return_code != 0:
        raise RuntimeError(f"{stage} failed with exit code {return_code}; see {log_path}")
    emit(stage, f"Finished {stage}.", seconds=round(elapsed, 3))
    return elapsed


def msl_command(
    args: argparse.Namespace,
    *,
    metadata: Path,
    out_dir: Path,
    offsets: str,
    samples: int,
    dump_streams: bool,
) -> list[str]:
    runner = PACKAGE_DIR / "simulation.py"
    command = [
        "wsl",
        "env",
        f"PYTHONPATH={windows_to_wsl(args.slippi_ai_root)}",
        f"MSL_DATA_DIR={windows_to_wsl(args.msl_root)}/data",
        f"LD_LIBRARY_PATH={args.cuda_library_path}",
        args.micromamba,
        "run",
        "-p",
        args.msl_env,
        "python",
        windows_to_wsl(runner),
        "--replay", windows_to_wsl(args.replay),
        "--analyzed-port", str(args.controlled_port),
        "--takeover-frames-json", windows_to_wsl(metadata),
        "--offsets", offsets,
        "--samples-per-point", str(samples),
        "--max-batch-lanes", str(args.max_batch_lanes),
        "--objective", "combo-extension",
        "--rollout-frames", "180",
        "--defense-resolution-extra-frames", "180",
        "--combo-max-rollout-frames", "420",
        "--defense-resolution-min-frames", "180",
        "--defense-neutral-stable-frames", "20",
        "--combo-escape-stable-frames", "20",
        "--unresolved-defense-penalty", "60",
        "--warmup-frames", "90",
        "--defender-delay-frames", "60",
        "--anchor-recorded-contact",
        "--history-mode", "teacher-forced",
        "--opponent-mode", "replay",
        "--rng-mode", "replay",
        "--sample-temperature", "1.0",
        "--option-horizon-frames", "90",
        "--option-max-action-segments", "6",
        "--option-max-input-segments", "8",
        "--msl-root", windows_to_wsl(args.msl_root),
        "--slippi-ai-root", windows_to_wsl(args.slippi_ai_root),
        "--model", windows_to_wsl(args.model),
        "--enable-gpu",
        "--out", windows_to_wsl(out_dir),
    ]
    if dump_streams:
        command.append("--dump-controller-streams")
    return command


def publish_artifacts(source: Path, destination: Path) -> None:
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    shutil.copytree(source, staging)
    if destination.exists():
        backup = destination.parent / f"{destination.name}.previous.{int(time.time())}"
        os.replace(destination, backup)
    os.replace(staging, destination)


def build_empty_report(args: argparse.Namespace, work: Path, candidates: dict[str, Any]) -> Path:
    artifacts = work / "empty_report" / "final_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    queue_path = artifacts / "empty_queue.json"
    selection_audit = {
        "candidate_count": len(candidates.get("frames") or []),
        "selected_improvement_count": 0,
        "candidate_generation": candidates.get("coverage") or {},
    }
    queue_path.write_text(json.dumps({
        "replay": str(args.replay),
        "display_name": args.display_name,
        "slp_version": args.slp_version,
        "controlled_port": args.controlled_port,
        "selection_audit": selection_audit,
        "targets": [],
    }, indent=2) + "\n", encoding="utf-8")
    manifest = artifacts / "advantage_improvements.json"
    manifest.write_text(json.dumps({
        "status": "ok",
        "queue_json": str(queue_path),
        "render_mode": "msl-direct",
        "artifact_mode": "interactive-traces",
        "video_fallback": False,
        "workers": 0,
        "target_count": 0,
        "successful_count": 0,
        "failed_count": 0,
        "wall_seconds": 0.0,
        "html_review": None,
        "clips_dir": str((artifacts / "clips").resolve()),
        "selection_audit": selection_audit,
        "results": [],
    }, indent=2) + "\n", encoding="utf-8")
    run_stage("html", module_command("report", "--manifest", manifest, "--out", artifacts / "advantage_review.html"), work / "logs" / "html.log")
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--controlled-port", type=int, choices=(1, 2), required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--display-name")
    parser.add_argument("--slp-version")
    settings = load_settings()
    parser.add_argument("--model", type=Path, default=settings.model)
    parser.add_argument("--msl-root", type=Path, default=settings.msl_root)
    parser.add_argument("--slippi-ai-root", type=Path, default=settings.slippi_ai_root)
    parser.add_argument("--iso", type=Path, default=settings.melee_iso)
    parser.add_argument("--micromamba", default=settings.micromamba)
    parser.add_argument("--msl-env", default=settings.msl_env)
    parser.add_argument("--cuda-library-path", default=settings.cuda_library_path)
    parser.add_argument("--preflight-samples", type=int, default=64)
    parser.add_argument("--refinement-samples", type=int, default=128)
    parser.add_argument("--max-batch-lanes", type=int, default=4096)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    missing = [name for name in ("model", "msl_root", "slippi_ai_root", "iso") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required paths: {', '.join(missing)}. Configure them or pass explicit flags.")

    args.replay = args.replay.resolve()
    args.job_dir = args.job_dir.resolve()
    args.model = args.model.resolve()
    args.msl_root = args.msl_root.resolve()
    args.slippi_ai_root = args.slippi_ai_root.resolve()
    args.iso = args.iso.resolve()
    args.display_name = args.display_name or args.replay.stem
    for path in (args.replay, args.model, args.msl_root, args.slippi_ai_root, args.iso):
        if not path.exists():
            raise FileNotFoundError(path)

    pipeline = args.job_dir / "pipeline"
    logs = pipeline / "logs"
    candidates_path = pipeline / "candidates.json"
    coverage_path = pipeline / "candidate_coverage.md"
    raw_events_path = pipeline / "raw_events.json"
    preflight = pipeline / "preflight"
    preflight_queue = pipeline / "preflight_selection.json"
    refinement = pipeline / "refinement"
    render_queue = pipeline / "render_queue.json"
    report_root = pipeline / "report"
    pipeline.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    if args.force or not candidates_path.is_file():
        timings["candidates"] = run_stage("candidates", module_command(
            "candidates",
            "--replay", str(args.replay),
            "--analyzed-port", str(args.controlled_port),
            "--max-candidates", "0",
            "--coverage-report", str(coverage_path),
            "--raw-events-out", str(raw_events_path),
            "--out", candidates_path,
        ), logs / "candidates.log")
    else:
        emit("candidates", "Reusing completed candidates stage.")
    candidates = read_json(candidates_path)

    if args.force or not (preflight / "summary.json").is_file():
        timings["preflight"] = run_stage(
            "preflight",
            msl_command(args, metadata=candidates_path, out_dir=preflight, offsets="0", samples=args.preflight_samples, dump_streams=False),
            logs / "preflight.log",
        )
    else:
        emit("preflight", "Reusing completed preflight stage.")

    if args.force or not preflight_queue.is_file():
        timings["preflight_selection"] = run_stage("preflight_selection", module_command(
            "selection",
            "--run-dir", str(preflight),
            "--candidates-json", str(candidates_path),
            "--replay", str(args.replay),
            "--iso", str(args.iso),
            "--controlled-port", str(args.controlled_port),
            "--top", "0",
            "--min-option-samples", "8",
            "--min-option-share", "0.1",
            "--min-improvement-rate", "0.25",
            "--alternative-routes", "3",
            "--out", preflight_queue,
        ), logs / "preflight_selection.log")
    else:
        emit("preflight_selection", "Reusing completed preflight selection.")
    set_queue_metadata(preflight_queue, display_name=args.display_name, slp_version=args.slp_version)

    preflight_payload = read_json(preflight_queue)
    if not (preflight_payload.get("targets") or []):
        emit("publish", "No robust improvements qualified; publishing an empty review.")
        source_artifacts = build_empty_report(args, pipeline, candidates)
    else:
        if args.force or not (refinement / "summary.json").is_file():
            timings["refinement"] = run_stage(
                "refinement",
                msl_command(args, metadata=preflight_queue, out_dir=refinement, offsets="0,1,2", samples=args.refinement_samples, dump_streams=True),
                logs / "refinement.log",
            )
        else:
            emit("refinement", "Reusing completed refinement stage.")

        if args.force or not render_queue.is_file():
            timings["route_selection"] = run_stage("route_selection", module_command(
                "selection",
                "--run-dir", str(refinement),
                "--candidates-json", str(candidates_path),
                "--preflight-queue", str(preflight_queue),
                "--replay", str(args.replay),
                "--iso", str(args.iso),
                "--controlled-port", str(args.controlled_port),
                "--top", "0",
                "--min-option-samples", "8",
                "--min-option-share", "0.1",
                "--min-improvement-rate", "0.25",
                "--alternative-routes", "3",
                "--out", render_queue,
            ), logs / "route_selection.log")
        else:
            emit("route_selection", "Reusing completed route selection.")
        set_queue_metadata(render_queue, display_name=args.display_name, slp_version=args.slp_version)

        queue_payload = read_json(render_queue)
        if not (queue_payload.get("targets") or []):
            source_artifacts = build_empty_report(args, pipeline, candidates)
        else:
            expected_report = report_root / "final_artifacts" / "advantage_review.html"
            if args.force or not expected_report.is_file():
                timings["artifacts"] = run_stage("artifacts", module_command(
                    "artifacts",
                    "--queue-json", str(render_queue),
                    "--out-root", str(report_root),
                    "--workers", str(args.render_workers),
                ), logs / "artifacts.log")
            else:
                emit("artifacts", "Reusing completed report artifacts.")
            source_artifacts = report_root / "final_artifacts"

    emit("publish", "Publishing review artifacts.")
    publish_artifacts(source_artifacts, args.job_dir / "artifacts")
    summary = {
        "status": "complete",
        "replay": str(args.replay),
        "controlledPort": args.controlled_port,
        "candidateCount": len(candidates.get("frames") or []),
        "timings": timings,
        "report": str((args.job_dir / "artifacts" / "advantage_review.html").resolve()),
    }
    (pipeline / "pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    emit("complete", "Analysis artifacts are ready.", report="advantage_review.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
