"""Build one portable interactive HTML report from selected MSL routes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .paths import PROJECT_DIR, WEB_DIR, module_command
from .timeline import timeline_payload


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_route(queue: Path, target_index: int, alternative_index: int, out: Path) -> dict[str, Any]:
    command = module_command("render_target", "--queue-json", queue, "--target-index", target_index, "--out-dir", out)
    if alternative_index:
        command.extend(["--alternative-index", str(alternative_index)])
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = out / "export.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    manifest = out / "trace_manifest.json"
    return {"status": "ok" if result.returncode == 0 and manifest.is_file() else "failed", "manifest": str(manifest), "log": str(log), "seconds": time.perf_counter() - started}


def consolidate(manifest_path: Path, traces: Path, stem: str) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    replay_source, agent_source = Path(manifest["replay_trace"]), Path(manifest["msl_trace"])
    replay_out, agent_out = traces / f"{stem}_replay.msltrace.json", traces / f"{stem}_agent.msltrace.json"
    shutil.copy2(replay_source, replay_out)
    shutil.copy2(agent_source, agent_out)
    replay, agent = load_json(replay_out), load_json(agent_out)
    switch = int(manifest["branch_switch_frame"])
    timeline_out = traces / f"{stem}_timeline.json"
    timeline_out.write_text(json.dumps(timeline_payload(replay, agent, switch_frame=switch), indent=2) + "\n", encoding="utf-8")
    lane = (manifest.get("target") or {}).get("representative_lane") or {}
    clip_start = int(manifest.get("clip_start_frame") or 0)
    defender = int(lane.get("defenderTakeoverFrame") or manifest.get("branch_start_frame") or clip_start)
    return {
        "replay_trace": f"../traces/{replay_out.name}", "agent_trace": f"../traces/{agent_out.name}",
        "timeline_events": f"../traces/{timeline_out.name}", "switch_frame": switch,
        "defender_switch_frame": max(0, defender - clip_start), "start_frame": clip_start,
        "frame_count": len((replay.get("frames") or {}).get("rows") or []), "alternative_routes": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    queue_path, out_root = args.queue_json.resolve(), args.out_root.resolve()
    queue = load_json(queue_path)
    targets = queue.get("targets") or []
    if not targets:
        raise SystemExit("render queue has no targets")
    started = time.perf_counter()
    jobs = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(targets))) as pool:
        futures = {}
        for index, target in enumerate(targets, 1):
            root = out_root / "routes" / f"{index:03d}"
            futures[pool.submit(export_route, queue_path, index, 0, root / "primary")] = (index, 0, target)
            for alternative in range(1, len(target.get("alternative_routes") or []) + 1):
                futures[pool.submit(export_route, queue_path, index, alternative, root / f"alternative_{alternative:02d}")] = (index, alternative, target)
        for future in as_completed(futures):
            index, alternative, target = futures[future]
            jobs.append({"target_index": index, "alternative_index": alternative, "base_frame": int(target.get("base_frame") or target.get("takeover_frame") or 0), **future.result()})
    failures = [job for job in jobs if job["status"] != "ok"]
    if failures:
        raise RuntimeError(f"{len(failures)} trace exports failed; first log: {failures[0]['log']}")
    artifacts, traces, viewer = out_root / "final_artifacts", out_root / "final_artifacts" / "traces", out_root / "final_artifacts" / "viewer"
    traces.mkdir(parents=True, exist_ok=True)
    viewer.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEB_DIR / "viewer" / "compare.html", viewer / "compare.html")
    shutil.copy2(WEB_DIR / "viewer" / "frame.html", viewer / "frame.html")
    results = []
    for index, target in enumerate(targets, 1):
        primary_job = next(job for job in jobs if job["target_index"] == index and job["alternative_index"] == 0)
        primary = consolidate(Path(primary_job["manifest"]), traces, f"{index:03d}_f{primary_job['base_frame']}")
        for job in sorted((j for j in jobs if j["target_index"] == index and j["alternative_index"]), key=lambda j: j["alternative_index"]):
            route = consolidate(Path(job["manifest"]), traces, f"{index:03d}_f{job['base_frame']}_route_{job['alternative_index']:02d}")
            route.update({"route_index": job["alternative_index"], "option": (target.get("alternative_routes") or [])[job["alternative_index"] - 1].get("option") or {}})
            primary["alternative_routes"].append(route)
        results.append({"target_index": index, "base_frame": primary_job["base_frame"], "status": "ok", "interactive": primary, "wall_seconds": primary_job["seconds"], "resolution": ((target.get("representative_lane") or {}).get("resolution") or {})})
    payload = {
        "status": "ok", "queue_json": str(queue_path), "render_mode": "interactive-traces",
        "artifact_mode": "interactive-traces", "video_fallback": False, "workers": args.workers,
        "target_count": len(targets), "successful_count": len(targets), "failed_count": 0,
        "wall_seconds": round(time.perf_counter() - started, 3), "clips_dir": str(artifacts / "clips"),
        "selection_audit": queue.get("selection_audit"), "results": results,
    }
    manifest = artifacts / "advantage_improvements.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(module_command("report", "--manifest", manifest, "--out", artifacts / "advantage_review.html"), cwd=PROJECT_DIR, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out_root / "report.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"report generation failed; see {out_root / 'report.log'}")
    print(json.dumps({"status": "ok", "report": str(artifacts / "advantage_review.html"), "targets": len(targets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
