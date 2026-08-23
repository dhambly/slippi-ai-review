"""Export synchronized neutral-loss traces and build the neutral review page."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .artifacts import consolidate, export_route, load_json
from .paths import PROJECT_DIR, WEB_DIR, module_command


def opening_display_name(target: dict[str, Any], interactive: dict[str, Any], traces: Path) -> str | None:
    timeline_name = Path(str(interactive.get("timeline_events") or "")).name
    timeline_path = traces / timeline_name
    if not timeline_name or not timeline_path.is_file():
        return None
    timeline = load_json(timeline_path)
    opening_frame = int((target.get("neutral_loss") or {}).get("openingFrame") or 0)
    trace_frame = opening_frame - int(interactive.get("start_frame") or 0)
    attacks = [
        event
        for event in ((timeline.get("lanes") or {}).get("replay") or [])
        if event.get("kind") == "attack"
        and int(event.get("frame") or 0) <= trace_frame <= int(event.get("endFrame") or event.get("frame") or 0)
    ]
    if not attacks:
        return None
    return str(min(attacks, key=lambda event: abs(int(event.get("frame") or 0) - trace_frame)).get("name") or "") or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--msl-root", type=Path)
    args = parser.parse_args()
    queue_path = args.queue_json.resolve()
    out_root = args.out_root.resolve()
    queue = load_json(queue_path)
    targets = queue.get("targets") or []
    started = time.perf_counter()
    jobs = []
    if targets:
        route_count = sum(1 + len(target.get("alternative_routes") or []) for target in targets)
        with ThreadPoolExecutor(max_workers=min(max(1, args.workers), route_count)) as pool:
            futures = {}
            for index, target in enumerate(targets, start=1):
                root = out_root / "routes" / f"{index:03d}"
                futures[pool.submit(export_route, queue_path, index, 0, root / "primary", args.msl_root)] = (index, 0, target)
                for alternative in range(1, len(target.get("alternative_routes") or []) + 1):
                    futures[pool.submit(
                        export_route,
                        queue_path,
                        index,
                        alternative,
                        root / f"alternative_{alternative:02d}",
                        args.msl_root,
                    )] = (index, alternative, target)
            for future in as_completed(futures):
                index, alternative, target = futures[future]
                jobs.append(
                    {
                        "target_index": index,
                        "alternative_index": alternative,
                        "base_frame": int(target.get("base_frame") or 0),
                        **future.result(),
                    }
                )
        failures = [job for job in jobs if job["status"] != "ok"]
        if failures:
            raise RuntimeError(f"{len(failures)} neutral trace exports failed; first log: {failures[0]['log']}")

    artifacts = out_root / "final_artifacts"
    traces = artifacts / "traces"
    viewer = artifacts / "viewer"
    traces.mkdir(parents=True, exist_ok=True)
    viewer.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEB_DIR / "viewer" / "compare.html", viewer / "compare.html")
    shutil.copy2(WEB_DIR / "viewer" / "frame.html", viewer / "frame.html")
    results = []
    for index, target in enumerate(targets, start=1):
        primary_job = next(job for job in jobs if job["target_index"] == index and job["alternative_index"] == 0)
        primary = consolidate(Path(primary_job["manifest"]), traces, f"neutral_{index:03d}_f{primary_job['base_frame']}")
        primary["option"] = target.get("option") or {}
        primary["lookback_frames"] = (target.get("neutral_loss") or {}).get("selectedRouteLookbackFrames") or (target.get("neutral_loss") or {}).get("lookbackFrames")
        primary["injection_frame"] = int(target.get("base_frame") or 0)
        primary["route_kind"] = "avoid"
        primary["opening_display_name"] = opening_display_name(target, primary, traces)
        for job in sorted(
            (item for item in jobs if item["target_index"] == index and item["alternative_index"]),
            key=lambda item: item["alternative_index"],
        ):
            route = consolidate(
                Path(job["manifest"]),
                traces,
                f"neutral_{index:03d}_f{job['base_frame']}_route_{job['alternative_index']:02d}",
            )
            route["route_index"] = job["alternative_index"]
            source_route = (target.get("alternative_routes") or [])[job["alternative_index"] - 1]
            route["option"] = source_route.get("option") or {}
            route["lookback_frames"] = source_route.get("lookback_frames")
            route["injection_frame"] = source_route.get("injection_frame")
            route["route_kind"] = source_route.get("route_kind") or "avoid"
            primary["alternative_routes"].append(route)
        results.append(
            {
                "target_index": index,
                "base_frame": primary_job["base_frame"],
                "status": "ok",
                "interactive": primary,
                "wall_seconds": primary_job["seconds"],
            }
        )
    payload: dict[str, Any] = {
        "status": "ok",
        "queue_json": str(queue_path),
        "artifact_mode": "interactive-traces",
        "target_count": len(targets),
        "successful_count": len(targets),
        "failed_count": 0,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "selection_audit": queue.get("selection_audit") or {},
        "results": results,
    }
    manifest = artifacts / "neutral_avoidance.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    report = artifacts / "neutral_review.html"
    result = subprocess.run(
        module_command("neutral_report", "--manifest", manifest, "--out", report),
        cwd=PROJECT_DIR,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (out_root / "neutral_report.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"neutral report generation failed; see {out_root / 'neutral_report.log'}")
    print(json.dumps({"status": "ok", "report": str(report), "targets": len(targets)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
