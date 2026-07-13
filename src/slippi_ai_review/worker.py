"""Poll UUID review jobs and run the MSL advantage pipeline one at a time."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .server import (
    DEFAULT_UPLOAD_DIR,
    _list_reviews,
    _load_review,
    _utc_now,
    _write_json_atomic,
    update_review,
)
from .paths import PROJECT_DIR, module_command


WORK_DIR = PROJECT_DIR
PYTHON = sys.executable


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


@contextmanager
def singleton_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0)
        if handle.read(1) == b"":
            handle.seek(0)
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another advantage review worker is already running") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another advantage review worker is already running") from exc
        yield
    finally:
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def write_worker_state(upload_dir: Path, *, status: str, review_id: str | None = None, message: str = "") -> None:
    _write_json_atomic(upload_dir / "worker.json", {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "status": status,
        "reviewId": review_id,
        "message": message,
        "updatedAt": _utc_now(),
    })


def patch_job(upload_dir: Path, review_id: str, **values: object) -> dict[str, object]:
    payload = _load_review(upload_dir, review_id)
    if payload is None:
        raise FileNotFoundError(review_id)
    payload.update(values)
    payload["updatedAt"] = _utc_now()
    _write_json_atomic(upload_dir / review_id / "review.json", payload)
    return payload


def recover_interrupted_jobs(upload_dir: Path) -> None:
    for review in _list_reviews(upload_dir):
        if review.get("status") != "processing":
            continue
        worker = review.get("worker") if isinstance(review.get("worker"), dict) else {}
        pid = int(worker.get("pid") or 0)
        if process_is_alive(pid):
            continue
        update_review(
            upload_dir,
            str(review["reviewId"]),
            "queued",
            message="Recovered after an interrupted worker; completed stages will be reused.",
        )


def next_job(upload_dir: Path, requested_id: str | None = None) -> dict[str, object] | None:
    reviews = _list_reviews(upload_dir)
    if requested_id:
        reviews = [review for review in reviews if review.get("reviewId") == requested_id]
    eligible = [review for review in reviews if review.get("status") == "queued" and review.get("targetPlayer")]
    eligible.sort(key=lambda review: str(review.get("createdAt") or ""))
    return eligible[0] if eligible else None


def pipeline_command(args: argparse.Namespace, review: dict[str, object]) -> list[str]:
    review_id = str(review["reviewId"])
    target = review.get("targetPlayer") if isinstance(review.get("targetPlayer"), dict) else {}
    settings = review.get("settings") if isinstance(review.get("settings"), dict) else {}
    preset = str(settings.get("qualityPreset") or "standard")
    presets = {
        "quick": (16, 32),
        "standard": (64, 128),
        "deep": (128, 256),
    }
    preflight_samples, refinement_samples = presets.get(preset, presets["standard"])
    if args.preflight_samples is not None:
        preflight_samples = args.preflight_samples
    if args.refinement_samples is not None:
        refinement_samples = args.refinement_samples
    command = module_command(
        "pipeline",
        "--replay", str(args.upload_dir / review_id / "replay.slp"),
        "--controlled-port", str(target.get("port")),
        "--job-dir", str(args.upload_dir / review_id),
        "--display-name", str(Path(str(review.get("originalFilename") or "replay.slp")).stem),
        "--preflight-samples", str(preflight_samples),
        "--refinement-samples", str(refinement_samples),
        "--max-batch-lanes", str(args.max_batch_lanes),
        "--render-workers", str(args.render_workers),
    )
    if args.force:
        command.append("--force")
    return command


def process_job(args: argparse.Namespace, review: dict[str, object]) -> bool:
    review_id = str(review["reviewId"])
    target = review.get("targetPlayer") if isinstance(review.get("targetPlayer"), dict) else {}
    target_label = f"P{target.get('port')} {(target.get('character') or {}).get('name', '')}"
    started_monotonic = time.perf_counter()
    started_at = _utc_now()
    update_review(args.upload_dir, review_id, "processing", message=f"Starting analysis for {target_label}.")
    patch_job(args.upload_dir, review_id, worker={
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "startedAt": started_at,
    })
    write_worker_state(args.upload_dir, status="processing", review_id=review_id, message="Starting pipeline")

    job_dir = args.upload_dir / review_id
    log_path = job_dir / "logs" / "worker_pipeline.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = pipeline_command(args, review)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{_utc_now()}] {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=WORK_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            patch_job(args.upload_dir, review_id, worker={
                "pid": os.getpid(),
                "pipelinePid": process.pid,
                "host": socket.gethostname(),
                "startedAt": started_at,
            })
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") != "progress":
                        continue
                    stage = str(event.get("stage") or "processing")
                    message = str(event.get("message") or stage)
                    patch_job(args.upload_dir, review_id, message=message, progress={
                        "stage": stage,
                        "message": message,
                        "updatedAt": _utc_now(),
                    })
                    write_worker_state(args.upload_dir, status="processing", review_id=review_id, message=message)
            return_code = process.wait()
        latest = _load_review(args.upload_dir, review_id)
        if latest and latest.get("status") == "cancelled":
            patch_job(args.upload_dir, review_id, timing={
                "startedAt": started_at,
                "finishedAt": _utc_now(),
                "processingSeconds": round(time.perf_counter() - started_monotonic, 3),
            })
            write_worker_state(args.upload_dir, status="idle", message=f"Cancelled {review_id}")
            return False
        if return_code != 0:
            raise RuntimeError(f"Pipeline exited with code {return_code}; see {log_path}")
        report = job_dir / "artifacts" / "advantage_review.html"
        if not report.is_file():
            raise FileNotFoundError(f"Pipeline did not publish {report}")
        update_review(
            args.upload_dir,
            review_id,
            "complete",
            message="Analysis complete.",
            report="advantage_review.html",
        )
        patch_job(args.upload_dir, review_id, progress={"stage": "complete", "message": "Analysis complete.", "updatedAt": _utc_now()})
        patch_job(args.upload_dir, review_id, timing={
            "startedAt": started_at,
            "finishedAt": _utc_now(),
            "processingSeconds": round(time.perf_counter() - started_monotonic, 3),
        })
        write_worker_state(args.upload_dir, status="idle", message=f"Completed {review_id}")
        return True
    except Exception as exc:
        error_path = job_dir / "logs" / "worker_error.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        update_review(
            args.upload_dir,
            review_id,
            "failed",
            message=f"Analysis failed: {exc}",
        )
        patch_job(args.upload_dir, review_id, error={"message": str(exc), "log": str(error_path), "updatedAt": _utc_now()})
        write_worker_state(args.upload_dir, status="idle", message=f"Failed {review_id}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--review-id")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--preflight-samples", type=int)
    parser.add_argument("--refinement-samples", type=int)
    parser.add_argument("--max-batch-lanes", type=int, default=4096)
    parser.add_argument("--render-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.upload_dir = args.upload_dir.resolve()
    args.upload_dir.mkdir(parents=True, exist_ok=True)
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    with singleton_lock(args.upload_dir / "worker.lock"):
        recover_interrupted_jobs(args.upload_dir)
        while True:
            review = next_job(args.upload_dir, args.review_id)
            if review is not None:
                process_job(args, review)
                if args.once or args.review_id:
                    return 0
                continue
            write_worker_state(args.upload_dir, status="idle", message="Waiting for queued reviews")
            if args.once or args.review_id:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
