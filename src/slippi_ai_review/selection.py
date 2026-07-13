"""Choose representative combo-extension lanes and build a Dolphin render queue."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


NON_ACTIONABLE_OPTIONS = frozenset({"NO_COMMITMENT", "WAIT", "NONE"})


def meaningful_option(row: dict[str, Any]) -> bool:
    signature = str(row.get("comboOptionSignature") or row.get("optionSignature") or "unknown").upper()
    return signature not in NON_ACTIONABLE_OPTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare replay-vs-MSL renders for robust combo extensions.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidates-json", type=Path, required=True)
    parser.add_argument(
        "--preflight-queue",
        type=Path,
        default=None,
        help="Optional first-pass queue used to preserve full candidate coverage in a refined run.",
    )
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--iso", type=Path, required=True)
    parser.add_argument("--controlled-port", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Maximum qualifying improvements to render; zero keeps every qualifying opening.",
    )
    parser.add_argument("--min-option-samples", type=int, default=8)
    parser.add_argument("--min-option-share", type=float, default=0.1)
    parser.add_argument(
        "--min-improvement-rate",
        type=float,
        default=0.25,
        help="Minimum robust option improvement rate; defaults to 25%% to retain marginal teaching candidates.",
    )
    parser.add_argument("--video-preroll-frames", type=int, default=60)
    parser.add_argument(
        "--alternative-routes",
        type=int,
        default=3,
        help="Maximum additional reliable option signatures retained per scenario.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_map(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(item["frame"]): item for item in payload.get("frames") or []}, payload


def lane_outperformed(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    original_kill = bool(baseline.get("original_followup_kill"))
    model_kill = int(row.get("defenderStocksLost") or 0) > 0
    if original_kill and model_kill:
        return False
    if "outperformedReplay" in row:
        return bool(row["outperformedReplay"])
    original_damage = float(baseline.get("original_followup_damage") or 0.0)
    resolution_reason = str((row.get("resolution") or {}).get("reason") or "")
    return (
        not bool(row.get("comboReversed"))
        and resolution_reason != "combo_timeout"
        and (
            (model_kill and not original_kill)
            or (
                not model_kill and not original_kill
                and float(row.get("followupDamage") or 0.0) >= original_damage + 3.0
            )
        )
    )


def option_summary(rows: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    original_damage = float(baseline.get("original_followup_damage") or 0.0)
    original_score = float(baseline.get("original_followup_score") or 0.0)
    improved = [row for row in rows if lane_outperformed(row, baseline)]
    return {
        "optionSignature": rows[0].get("comboOptionSignature") or rows[0].get("optionSignature"),
        "samples": len(rows),
        "improvementRate": round(len(improved) / len(rows), 4),
        "extensionRate": round(sum(int(row.get("followupHits") or 0) > 0 for row in rows) / len(rows), 4),
        "killRate": round(sum(int(row.get("defenderStocksLost") or 0) > 0 for row in rows) / len(rows), 4),
        "reversalRate": round(sum(bool(row.get("comboReversed")) for row in rows) / len(rows), 4),
        "meanFollowupDamage": round(sum(float(row.get("followupDamage") or 0.0) for row in rows) / len(rows), 3),
        "meanDamageDeltaVsReplay": round(
            sum(float(row.get("followupDamage") or 0.0) - original_damage for row in rows) / len(rows),
            3,
        ),
        "medianScoreDeltaVsReplay": round(
            median(float(row.get("score") or 0.0) - original_score for row in rows),
            3,
        ),
    }


def representative_lane(rows: list[dict[str, Any]], baseline: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    pool = [row for row in rows if lane_outperformed(row, baseline)]
    if not pool:
        pool = [row for row in rows if not bool(row.get("comboReversed"))] or rows
    if float(summary["killRate"]) >= 0.5:
        killed = [row for row in pool if int(row.get("defenderStocksLost") or 0) > 0]
        if killed:
            pool = killed
    target_score = median(float(row.get("score") or 0.0) for row in pool)
    return min(
        pool,
        key=lambda row: (
            abs(float(row.get("score") or 0.0) - target_score),
            int(row.get("laneId") or 0),
        ),
    )


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(run_dir / "lanes.jsonl")
    baselines, candidate_payload = candidate_map(args.candidates_json.resolve())
    preflight_payload = (
        json.loads(args.preflight_queue.resolve().read_text(encoding="utf-8"))
        if args.preflight_queue is not None
        else None
    )

    buckets: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        signature = str(row.get("comboOptionSignature") or row.get("optionSignature") or "unknown")
        buckets[(int(row["baseFrame"]), int(row["offset"]), signature)].append(row)

    best_by_opening: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    qualified_by_opening: dict[int, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    rejection_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    point_counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        point_counts[(int(row["baseFrame"]), int(row["offset"]))] += 1
    for (base_frame, offset, _signature), option_rows in buckets.items():
        if base_frame not in baselines:
            rejection_counts[base_frame]["missing_candidate_metadata"] += 1
            continue
        if not meaningful_option(option_rows[0]):
            rejection_counts[base_frame]["no_meaningful_commitment"] += 1
            continue
        if len(option_rows) < int(args.min_option_samples):
            rejection_counts[base_frame]["too_few_option_samples"] += 1
            continue
        option_share = len(option_rows) / max(1, point_counts[(base_frame, offset)])
        if option_share < float(args.min_option_share):
            rejection_counts[base_frame]["option_share_below_threshold"] += 1
            continue
        metrics = option_summary(option_rows, baselines[base_frame])
        if float(metrics["improvementRate"]) < float(args.min_improvement_rate):
            rejection_counts[base_frame]["improvement_rate_below_threshold"] += 1
            continue
        metrics["baseFrame"] = base_frame
        metrics["offset"] = offset
        metrics["optionShare"] = round(option_share, 4)
        existing = best_by_opening.get(base_frame)
        rank = (
            float(metrics["improvementRate"]),
            float(metrics["killRate"]),
            float(metrics["medianScoreDeltaVsReplay"]),
            float(metrics["meanDamageDeltaVsReplay"]),
            int(metrics["samples"]),
        )
        metrics["_rank"] = rank
        qualified_by_opening[base_frame].append((metrics, option_rows))
        if existing is None or rank > existing[0]["_rank"]:
            best_by_opening[base_frame] = (metrics, option_rows)

    ranked_selected = sorted(
        best_by_opening.values(),
        key=lambda item: item[0]["_rank"],
        reverse=True,
    )
    if args.top > 0:
        ranked_selected = ranked_selected[: int(args.top)]
    for rank, (metrics, _option_rows) in enumerate(ranked_selected, start=1):
        metrics["selectionRank"] = rank
    selected = sorted(ranked_selected, key=lambda item: int(item[0]["baseFrame"]))

    targets = []
    for queue_index, (metrics, option_rows) in enumerate(selected, start=1):
        base_frame = int(metrics["baseFrame"])
        baseline = baselines[base_frame]
        lane = representative_lane(option_rows, baseline, metrics)
        public_metrics = {key: value for key, value in metrics.items() if key != "_rank"}
        takeover_frame = int(lane["takeoverFrame"])
        alternatives = []
        seen_signatures = {str(public_metrics["optionSignature"])}
        for alternative_metrics, alternative_rows in sorted(
            qualified_by_opening[base_frame], key=lambda item: item[0]["_rank"], reverse=True
        ):
            signature = str(alternative_metrics["optionSignature"])
            if signature in seen_signatures or signature in NON_ACTIONABLE_OPTIONS:
                continue
            seen_signatures.add(signature)
            alternative_lane = representative_lane(alternative_rows, baseline, alternative_metrics)
            alternative_public = {
                key: value for key, value in alternative_metrics.items() if key != "_rank"
            }
            alternatives.append({
                "route_rank": len(alternatives) + 2,
                "offset": int(alternative_lane["offset"]),
                "branch_start_frame": int(alternative_lane["takeoverFrame"]),
                "takeover_frame": int(alternative_lane["takeoverFrame"]),
                "option": alternative_public,
                "representative_lane": alternative_lane,
            })
            if len(alternatives) >= max(0, int(args.alternative_routes)):
                break
        targets.append({
            "queue_index": queue_index,
            "base_frame": base_frame,
            "offset": int(lane["offset"]),
            "branch_start_frame": takeover_frame,
            "takeover_frame": takeover_frame,
            "takeover_time": round(takeover_frame / 60.0, 3),
            "label": (
                f"Combo extension after {baseline['opening_move'].get('moveShortName')} at f{base_frame}: "
                f"Phillip {public_metrics['optionSignature']}, "
                f"{public_metrics['improvementRate']:.0%} improvement rate"
            ),
            "replay_baseline": baseline,
            "option": public_metrics,
            "representative_lane": lane,
            "alternative_routes": alternatives,
        })

    refined_frames = sorted({int(row["baseFrame"]) for row in rows})
    candidate_frames = sorted(baselines)
    selected_frames = [int(target["base_frame"]) for target in targets]
    preflight_audit = (preflight_payload or {}).get("selection_audit") or {}
    if preflight_audit:
        coverage_simulated_frames = sorted(
            set(int(frame) for frame in preflight_audit.get("candidate_frames") or [])
            - set(int(frame) for frame in preflight_audit.get("missing_simulation_frames") or [])
        )
        refinement_requested_frames = sorted(
            int(frame) for frame in preflight_audit.get("selected_frames") or []
        )
        preflight_rejections = {
            int(item["frame"]): dict(item.get("reasons") or {})
            for item in preflight_audit.get("not_selected") or []
        }
    else:
        coverage_simulated_frames = refined_frames
        refinement_requested_frames = refined_frames
        preflight_rejections = {}

    not_selected = []
    for frame in candidate_frames:
        if frame in best_by_opening:
            continue
        if frame in refined_frames:
            stage = "refinement"
            reasons = dict(sorted(rejection_counts.get(frame, {}).items()))
        elif frame in preflight_rejections:
            stage = "preflight"
            reasons = preflight_rejections[frame]
        elif frame in refinement_requested_frames:
            stage = "refinement"
            reasons = {"missing_refinement_rows": 1}
        else:
            stage = "coverage"
            reasons = {"missing_simulation_rows": 1}
        not_selected.append({"frame": frame, "stage": stage, "reasons": reasons})

    selection_audit = {
        "candidate_generation": (
            candidate_payload.get("coverage")
            or candidate_payload.get("coverage_diagnostics")
            or {}
        ),
        "candidate_count": len(candidate_frames),
        "simulated_candidate_count": len(set(candidate_frames) & set(coverage_simulated_frames)),
        "refinement_requested_count": len(refinement_requested_frames),
        "refined_candidate_count": len(set(candidate_frames) & set(refined_frames)),
        "selected_improvement_count": len(selected_frames),
        "candidate_frames": candidate_frames,
        "refinement_requested_frames": refinement_requested_frames,
        "refined_candidate_frames": refined_frames,
        "selected_frames": selected_frames,
        "missing_simulation_frames": sorted(set(candidate_frames) - set(coverage_simulated_frames)),
        "missing_refinement_frames": sorted(set(refinement_requested_frames) - set(refined_frames)),
        "simulated_without_candidate_metadata": sorted(set(coverage_simulated_frames) - set(candidate_frames)),
        "not_selected": not_selected,
    }

    queue = {
        "version": 1,
        "source_run_dir": str(run_dir),
        "preflight_queue": str(args.preflight_queue.resolve()) if args.preflight_queue else None,
        "summary": str((run_dir / "summary.json").resolve()),
        "lanes_jsonl": str((run_dir / "lanes.jsonl").resolve()),
        "replay": str(args.replay.resolve()),
        "iso": str(args.iso.resolve()),
        "controlled_port": int(args.controlled_port),
        "rollout_frames": int(summary.get("maxRolloutFrames") or summary.get("rolloutFrames") or 180),
        "video_preroll_frames": int(args.video_preroll_frames),
        "selection": {
            "top": int(args.top),
            "min_option_samples": int(args.min_option_samples),
            "min_option_share": float(args.min_option_share),
            "min_improvement_rate": float(args.min_improvement_rate),
            "alternative_routes": max(0, int(args.alternative_routes)),
        },
        "selection_audit": selection_audit,
        "targets": targets,
    }
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out.resolve()),
        "targets": [
            {
                "frame": target["base_frame"],
                "option": target["option"]["optionSignature"],
                "improvement_rate": target["option"]["improvementRate"],
                "lane_id": target["representative_lane"]["laneId"],
            }
            for target in targets
        ],
        "selection_audit": selection_audit,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
