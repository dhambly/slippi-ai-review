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
from .disadvantage_pipeline import publish_disadvantage_artifacts
from .neutral_pipeline import publish_neutral_artifacts
from .paths import PACKAGE_DIR, PROJECT_DIR, module_command


WORK_DIR = PROJECT_DIR
PYTHON = sys.executable
PREFLIGHT_MIN_OPTION_SAMPLES = 2
REFINEMENT_MIN_OPTION_SAMPLES = 8


def windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    posix = resolved.as_posix()
    for prefix in ("//wsl.localhost/", "//wsl$/"):
        if posix.lower().startswith(prefix):
            remainder = posix[len(prefix) :]
            _distribution, separator, linux_path = remainder.partition("/")
            if not separator:
                raise ValueError(f"WSL UNC path has no Linux path component: {resolved}")
            return f"/{linux_path}"
    return f"/mnt/{drive}{posix[len(resolved.drive):]}" if drive else posix


def runtime_mode(args: argparse.Namespace, *, platform_name: str | None = None) -> str:
    requested = str(getattr(args, "runtime_mode", "auto") or "auto").lower()
    if requested not in {"auto", "native", "wsl"}:
        raise ValueError(f"Unsupported runtime mode: {requested}")
    if requested != "auto":
        return requested
    return "wsl" if (platform_name or sys.platform) == "win32" else "native"


def runtime_path(args: argparse.Namespace, path: Path) -> str:
    return windows_to_wsl(path) if runtime_mode(args) == "wsl" else str(path.resolve())


def simulation_backend(args: argparse.Namespace) -> str:
    backend = str(getattr(args, "simulation_backend", "legacy") or "legacy").lower()
    if backend not in {"legacy", "decomp"}:
        raise ValueError(f"Unsupported simulation backend: {backend}")
    return backend


def simulation_msl_root(args: argparse.Namespace) -> Path:
    if simulation_backend(args) == "decomp":
        root = getattr(args, "msl_decomp_root", None)
        if root is None:
            raise ValueError("decomp backend requires paths.msl_decomp_root or --msl-decomp-root")
        return Path(root)
    return Path(args.msl_root)


def simulation_runtime_prefix(
    args: argparse.Namespace,
    *,
    include_package_source: bool,
    msl_root: Path | None = None,
) -> list[str]:
    package_source = PROJECT_DIR / "src"
    active_root = (msl_root or simulation_msl_root(args)).resolve()
    python_paths = [active_root, args.slippi_ai_root]
    if include_package_source:
        python_paths.insert(0, package_source)
    separator = ":" if runtime_mode(args) == "wsl" else os.pathsep
    environment = [
        f"PYTHONPATH={separator.join(runtime_path(args, path) for path in python_paths)}",
        f"MSL_DATA_DIR={runtime_path(args, active_root / 'data')}",
    ]
    cuda_library_path = str(getattr(args, "cuda_library_path", "") or "")
    if cuda_library_path:
        environment.append(f"LD_LIBRARY_PATH={cuda_library_path}")
    if runtime_mode(args) == "wsl":
        msl_env = str(getattr(args, "msl_env", "") or "")
        if not msl_env:
            raise ValueError("WSL runtime requires wsl.environment in the configuration")
        return [
            "wsl",
            "env",
            *environment,
            str(getattr(args, "micromamba", "micromamba")),
            "run",
            "-p",
            msl_env,
            "python",
        ]
    return ["env", *environment, str(getattr(args, "runtime_python", sys.executable))]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def set_queue_metadata(
    path: Path,
    *,
    display_name: str,
    slp_version: str | None,
    backend: str = "legacy",
) -> None:
    payload = read_json(path)
    changed = payload.get("display_name") != display_name
    payload["display_name"] = display_name
    if payload.get("simulation_backend") != backend:
        payload["simulation_backend"] = backend
        changed = True
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
    backend = simulation_backend(args)
    root = simulation_msl_root(args)
    runner = PACKAGE_DIR / ("decomp_simulation.py" if backend == "decomp" else "simulation.py")
    gpu_duty_cycle = getattr(args, "gpu_duty_cycle", 0.20)
    command = simulation_runtime_prefix(
        args,
        include_package_source=backend == "decomp",
        msl_root=root,
    ) + [
        "-u",
        runtime_path(args, runner),
        "--replay", runtime_path(args, args.replay),
        "--analyzed-port", str(args.controlled_port),
        "--takeover-frames-json", runtime_path(args, metadata),
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
        "--msl-root", runtime_path(args, root),
        "--slippi-ai-root", runtime_path(args, args.slippi_ai_root),
        "--model", runtime_path(args, args.model),
        "--enable-gpu",
        "--gpu-duty-cycle", str(gpu_duty_cycle),
        "--out", runtime_path(args, out_dir),
    ]
    if dump_streams:
        command.append("--dump-controller-streams")
    return command


def neutral_command(args: argparse.Namespace, *, out_dir: Path, raw_events: Path) -> list[str]:
    gpu_duty_cycle = getattr(args, "gpu_duty_cycle", 0.20)
    # Neutral discovery still uses legacy validation buffers. Decomp is used by
    # the advantage/phase rollout lane until neutral's lookback runner migrates.
    command = simulation_runtime_prefix(args, include_package_source=True, msl_root=args.msl_root) + [
        "-u",
        "-m",
        "slippi_ai_review.neutral_pipeline",
        "--replay", runtime_path(args, args.replay),
        "--raw-events-json", runtime_path(args, raw_events),
        "--analyzed-port", str(args.controlled_port),
        "--display-name", args.display_name,
        "--preflight-samples", str(args.neutral_preflight_samples),
        "--refinement-samples", str(args.neutral_refinement_samples),
        "--max-batch-lanes", str(args.max_batch_lanes),
        "--artifact-workers", str(max(1, min(2, args.render_workers))),
        "--model", runtime_path(args, args.model),
        "--msl-root", runtime_path(args, args.msl_root),
        "--slippi-ai-root", runtime_path(args, args.slippi_ai_root),
        "--out", runtime_path(args, out_dir),
    ] + (["--slp-version", args.slp_version] if args.slp_version else [])
    if getattr(args, "force", False):
        command.append("--force")
    if getattr(args, "neutral_enable_gpu", True):
        command.append("--enable-gpu")
    command.extend(["--gpu-duty-cycle", str(gpu_duty_cycle)])
    return command


def disadvantage_command(args: argparse.Namespace, *, inventory: Path, out_dir: Path) -> list[str]:
    command = simulation_runtime_prefix(args, include_package_source=True, msl_root=args.msl_root) + [
        "-u", "-m", "slippi_ai_review.disadvantage_pipeline",
        "--replay", runtime_path(args, args.replay),
        "--inventory", runtime_path(args, inventory),
        "--analyzed-port", str(args.controlled_port),
        "--display-name", args.display_name,
        "--samples", str(args.disadvantage_samples),
        "--rollout-frames", str(args.disadvantage_rollout_frames),
        "--opponent-takeover-max-delay-frames", str(args.disadvantage_opponent_delay_frames),
        "--max-batch-lanes", str(args.max_batch_lanes),
        "--artifact-workers", str(max(1, min(2, args.render_workers))),
        "--model", runtime_path(args, args.model),
        "--msl-root", runtime_path(args, args.msl_root),
        "--slippi-ai-root", runtime_path(args, args.slippi_ai_root),
        "--out", runtime_path(args, out_dir),
    ]
    if args.force:
        command.append("--force")
    if args.disadvantage_enable_gpu:
        command.extend(["--enable-gpu", "--gpu-duty-cycle", str(args.gpu_duty_cycle)])
    return command


def phase_sweep_command(args: argparse.Namespace, *, inventory: Path, out_dir: Path) -> list[str]:
    """A cheap, renderable probe of every playable timeline segment."""
    gpu_duty_cycle = getattr(args, "gpu_duty_cycle", 0.20)
    backend = simulation_backend(args)
    root = simulation_msl_root(args)
    runner = PACKAGE_DIR / ("decomp_simulation.py" if backend == "decomp" else "simulation.py")
    command = simulation_runtime_prefix(args, include_package_source=True, msl_root=root) + [
        "-u", runtime_path(args, runner),
        "--replay", runtime_path(args, args.replay),
        "--analyzed-port", str(args.controlled_port),
        "--takeover-frames-json", runtime_path(args, inventory),
        "--offsets", "0",
        "--samples-per-point", str(args.phase_sweep_samples),
        "--max-batch-lanes", str(args.max_batch_lanes),
        "--objective", "general",
        "--rollout-frames", "120",
        "--defense-resolution-extra-frames", "0",
        "--defense-resolution-min-frames", "120",
        "--defender-delay-frames", "121",
        "--defender-takeover-mode", "observed-phase-followup",
        "--warmup-frames", "90",
        "--history-mode", "teacher-forced",
        "--opponent-mode", "replay",
        "--rng-mode", "replay",
        "--dump-controller-streams",
        "--msl-root", runtime_path(args, root),
        "--slippi-ai-root", runtime_path(args, args.slippi_ai_root),
        "--model", runtime_path(args, args.model),
        "--out", runtime_path(args, out_dir),
    ]
    if getattr(args, "phase_sweep_enable_gpu", True):
        command.extend(["--enable-gpu", "--gpu-duty-cycle", str(gpu_duty_cycle)])
    return command


def publish_artifacts(source: Path, destination: Path) -> None:
    staging = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    shutil.copytree(source, staging)
    if destination.exists():
        backup = destination.parent / f"{destination.name}.previous.{int(time.time())}"
        os.replace(destination, backup)
    os.replace(staging, destination)


def publish_phase_sweep_artifacts(source: Path, destination: Path) -> None:
    """Overlay phase decks and their self-contained viewer data on the review artifact set."""
    for name in ("advantage_review.html", "neutral_review.html", "disadvantage_review.html"):
        if name == "advantage_review.html" and (destination / name).is_file():
            continue
        shutil.copy2(source / name, destination / name)
    for folder in ("traces", "viewer"):
        source_folder = source / folder
        destination_folder = destination / folder
        destination_folder.mkdir(parents=True, exist_ok=True)
        for item in source_folder.iterdir():
            if item.is_file():
                shutil.copy2(item, destination_folder / item.name)


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
    parser.add_argument("--msl-decomp-root", type=Path, default=settings.msl_decomp_root)
    parser.add_argument("--slippi-ai-root", type=Path, default=settings.slippi_ai_root)
    parser.add_argument("--iso", type=Path, default=settings.melee_iso)
    parser.add_argument("--runtime-mode", choices=("auto", "native", "wsl"), default=settings.runtime_mode)
    parser.add_argument("--runtime-python", default=settings.runtime_python)
    parser.add_argument(
        "--simulation-backend",
        choices=("legacy", "decomp"),
        default=settings.simulation_backend,
    )
    parser.add_argument("--micromamba", default=settings.micromamba)
    parser.add_argument("--msl-env", default=settings.msl_env)
    parser.add_argument("--cuda-library-path", default=settings.cuda_library_path)
    parser.add_argument("--preflight-samples", type=int, default=64)
    parser.add_argument("--refinement-samples", type=int, default=128)
    parser.add_argument("--neutral-preflight-samples", type=int, default=48)
    parser.add_argument("--neutral-refinement-samples", type=int, default=192)
    parser.add_argument("--neutral-enable-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disadvantage-samples", type=int, default=16)
    parser.add_argument("--disadvantage-rollout-frames", type=int, default=180)
    parser.add_argument("--disadvantage-opponent-delay-frames", type=int, default=60)
    parser.add_argument("--disadvantage-enable-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.20)
    parser.add_argument("--phase-sweep-samples", type=int, default=12)
    parser.add_argument("--phase-sweep-enable-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-batch-lanes", type=int, default=4096)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 < args.gpu_duty_cycle <= 1:
        raise SystemExit("--gpu-duty-cycle must be in (0, 1]")

    missing = [name for name in ("model", "msl_root", "slippi_ai_root", "iso") if getattr(args, name) is None]
    if args.simulation_backend == "decomp" and args.msl_decomp_root is None:
        missing.append("msl_decomp_root")
    if missing:
        raise SystemExit(f"Missing required paths: {', '.join(missing)}. Configure them or pass explicit flags.")

    args.replay = args.replay.resolve()
    args.job_dir = args.job_dir.resolve()
    args.model = args.model.resolve()
    args.msl_root = args.msl_root.resolve()
    if args.msl_decomp_root is not None:
        args.msl_decomp_root = args.msl_decomp_root.resolve()
    args.slippi_ai_root = args.slippi_ai_root.resolve()
    args.iso = args.iso.resolve()
    args.display_name = args.display_name or args.replay.stem
    required_paths = [args.replay, args.model, args.msl_root, args.slippi_ai_root, args.iso]
    if args.simulation_backend == "decomp":
        required_paths.append(args.msl_decomp_root)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    pipeline = args.job_dir / "pipeline"
    logs = pipeline / "logs"
    candidates_path = pipeline / "candidates.json"
    coverage_path = pipeline / "candidate_coverage.md"
    raw_events_path = pipeline / "raw_events.json"
    phase_sweep_inventory = pipeline / "phase_sweep_inventory.json"
    phase_sweep_run = pipeline / "phase_sweep"
    phase_sweep_queue = pipeline / "phase_sweep_queue.json"
    phase_sweep_artifacts = pipeline / "phase_sweep_artifacts"
    preflight = pipeline / "preflight"
    preflight_queue = pipeline / "preflight_selection.json"
    refinement = pipeline / "refinement"
    render_queue = pipeline / "render_queue.json"
    report_root = pipeline / "report"
    neutral_root = pipeline / "neutral"
    disadvantage_root = pipeline / "disadvantage"
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
    if args.force or not phase_sweep_inventory.is_file():
        timeline = Path(str(candidates.get("timeline") or ""))
        if not timeline.is_file():
            raise FileNotFoundError(f"Candidate timeline missing: {timeline}")
        timings["phase_sweep_inventory"] = run_stage("phase_sweep_inventory", module_command(
            "phase_sweep",
            "--timeline", str(timeline),
            "--analyzed-port", str(args.controlled_port),
            "--out", str(phase_sweep_inventory),
        ), logs / "phase_sweep_inventory.log")
    else:
        emit("phase_sweep_inventory", "Reusing phase sweep inventory.")
    if args.force or not (phase_sweep_run / "summary.json").is_file():
        timings["phase_sweep"] = run_stage(
            "phase_sweep",
            phase_sweep_command(args, inventory=phase_sweep_inventory, out_dir=phase_sweep_run),
            logs / "phase_sweep.log",
        )
    else:
        emit("phase_sweep", "Reusing low-sample phase sweep.")

    if args.force or not (preflight / "summary.json").is_file():
        timings["preflight"] = run_stage(
            "preflight",
            # Preserve low-sample controller streams so the coverage sweep can
            # promote any reviewed phase into a visible slide without re-running it.
            msl_command(args, metadata=candidates_path, out_dir=preflight, offsets="0", samples=args.preflight_samples, dump_streams=True),
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
            "--min-option-samples", str(PREFLIGHT_MIN_OPTION_SAMPLES),
            "--min-option-share", "0.1",
            "--min-improvement-rate", "0.25",
            "--alternative-routes", "3",
            "--out", preflight_queue,
        ), logs / "preflight_selection.log")
    else:
        emit("preflight_selection", "Reusing completed preflight selection.")
    set_queue_metadata(
        preflight_queue,
        display_name=args.display_name,
        slp_version=args.slp_version,
        backend=args.simulation_backend,
    )

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
                "--min-option-samples", str(REFINEMENT_MIN_OPTION_SAMPLES),
                "--min-option-share", "0.1",
                "--min-improvement-rate", "0.25",
                "--alternative-routes", "3",
                "--out", render_queue,
            ), logs / "route_selection.log")
        else:
            emit("route_selection", "Reusing completed route selection.")
        set_queue_metadata(
            render_queue,
            display_name=args.display_name,
            slp_version=args.slp_version,
            backend=args.simulation_backend,
        )

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

    neutral_artifacts = neutral_root / "artifacts_build" / "final_artifacts"
    expected_neutral_report = neutral_artifacts / "neutral_review.html"
    expected_neutral_grid = neutral_root / "route_grid" / "lanes.jsonl"
    if args.force or not expected_neutral_report.is_file() or not expected_neutral_grid.is_file():
        timings["neutral"] = run_stage(
            "neutral",
            neutral_command(args, out_dir=neutral_root, raw_events=raw_events_path),
            logs / "neutral.log",
        )
    else:
        emit("neutral", "Reusing completed neutral-loss analysis.")
    if not expected_neutral_report.is_file():
        raise FileNotFoundError(f"Neutral pipeline did not create {expected_neutral_report}")

    disadvantage_artifacts = disadvantage_root / "artifacts_build" / "final_artifacts"
    expected_disadvantage_report = disadvantage_artifacts / "disadvantage_review.html"
    expected_disadvantage_summary = disadvantage_root / "disadvantage_pipeline_summary.json"
    if args.force or not expected_disadvantage_report.is_file() or not expected_disadvantage_summary.is_file():
        timings["disadvantage"] = run_stage(
            "disadvantage",
            disadvantage_command(args, inventory=phase_sweep_inventory, out_dir=disadvantage_root),
            logs / "disadvantage.log",
        )
    else:
        emit("disadvantage", "Reusing completed disadvantage analysis.")
    if not expected_disadvantage_report.is_file():
        raise FileNotFoundError(f"Disadvantage pipeline did not create {expected_disadvantage_report}")

    advantage_manifest = source_artifacts / "advantage_improvements.json"
    if not advantage_manifest.is_file():
        raise FileNotFoundError(f"Advantage pipeline did not create {advantage_manifest}")
    timings["report_navigation"] = run_stage(
        "report_navigation",
        module_command(
            "report",
            "--manifest", advantage_manifest,
            "--out", source_artifacts / "advantage_review.html",
        ),
        logs / "report_navigation.log",
    )
    phase_sweep_final = phase_sweep_artifacts / "final_artifacts"
    if args.force or not phase_sweep_queue.is_file():
        timings["phase_sweep_selection"] = run_stage("phase_sweep_selection", module_command(
            "phase_sweep_selection",
            "--inventory", str(phase_sweep_inventory),
            "--run-dir", str(phase_sweep_run),
            "--out", str(phase_sweep_queue),
        ), logs / "phase_sweep_selection.log")
    else:
        emit("phase_sweep_selection", "Reusing phase sweep selection.")
    set_queue_metadata(
        phase_sweep_queue,
        display_name=args.display_name,
        slp_version=args.slp_version,
        backend=args.simulation_backend,
    )
    if args.force or not (phase_sweep_final / "advantage_improvements.json").is_file():
        timings["phase_sweep_artifacts"] = run_stage("phase_sweep_artifacts", module_command(
            "artifacts",
            "--queue-json", str(phase_sweep_queue),
            "--out-root", str(phase_sweep_artifacts),
            "--workers", str(args.render_workers),
        ), logs / "phase_sweep_artifacts.log")
    else:
        emit("phase_sweep_artifacts", "Reusing phase sweep traces.")
    phase_sweep_reports = tuple(
        phase_sweep_final / f"{phase}_review.html"
        for phase in ("advantage", "neutral", "disadvantage")
    )
    if args.force or any(not report.is_file() for report in phase_sweep_reports):
        timings["phase_sweep_report"] = run_stage("phase_sweep_report", module_command(
            "phase_sweep_report",
            "--queue-json", str(phase_sweep_queue),
            "--manifest", str(phase_sweep_final / "advantage_improvements.json"),
            "--out-dir", str(phase_sweep_final),
        ), logs / "phase_sweep_report.log")
    else:
        emit("phase_sweep_report", "Reusing phase sweep slide decks.")
    publish_phase_sweep_artifacts(phase_sweep_final, source_artifacts)
    # The low-sample phase sweep provides complete fallback navigation. Refined
    # phase-specific decks then replace only their own pages and trace families.
    publish_neutral_artifacts(neutral_artifacts, source_artifacts)
    publish_disadvantage_artifacts(disadvantage_artifacts, source_artifacts)

    emit("publish", "Publishing review artifacts.")
    publish_artifacts(source_artifacts, args.job_dir / "artifacts")
    summary = {
        "status": "complete",
        "replay": str(args.replay),
        "controlledPort": args.controlled_port,
        "simulationBackend": args.simulation_backend,
        "phaseBackends": {
            "advantage": args.simulation_backend,
            "phaseSweep": args.simulation_backend,
            "neutral": "legacy",
            "disadvantage": "legacy",
        },
        "candidateCount": len(candidates.get("frames") or []),
        "timings": timings,
        "report": str((args.job_dir / "artifacts" / "advantage_review.html").resolve()),
        "neutralReport": str((args.job_dir / "artifacts" / "neutral_review.html").resolve()),
        "disadvantageReport": str((args.job_dir / "artifacts" / "disadvantage_review.html").resolve()),
    }
    (pipeline / "pipeline_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    emit("complete", "Analysis artifacts are ready.", report="advantage_review.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
