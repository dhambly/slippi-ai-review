"""Evaluate smaller sampling budgets from completed benchmark lane corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from slippi_ai_review.selection import meaningful_option, option_summary


PREFLIGHT_BUDGETS = (16, 24, 32, 48, 64, 96, 128)
REFINEMENT_BUDGETS = (32, 48, 64, 80, 96, 112, 128, 160, 192, 256)
THRESHOLD_SWEEPS = {
    "min_option_samples": (1, 2, 4, 8, 12, 16, 24),
    "min_option_share": (0.05, 0.1, 0.15, 0.2),
    "min_improvement_rate": (0.1, 0.2, 0.25, 0.33, 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def sample_indices(full: int, budget: int, *, mode: str, seed: int) -> set[int]:
    budget = min(full, budget)
    if mode == "prefix":
        return set(range(budget))
    return set(random.Random(seed).sample(range(full), budget))


def filter_rows(rows: list[dict[str, Any]], indices: set[int], frames: set[int] | None = None) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if int(row["sampleIndex"]) in indices
        and (frames is None or int(row["baseFrame"]) in frames)
    ]


def select_options(
    rows: Iterable[dict[str, Any]],
    baselines: dict[int, dict[str, Any]],
    *,
    min_option_samples: int,
    min_option_share: float = 0.1,
    min_improvement_rate: float = 0.25,
) -> dict[int, dict[str, Any]]:
    rows = list(rows)
    buckets: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    point_counts: dict[tuple[int, int], int] = defaultdict(int)
    for row in rows:
        frame = int(row["baseFrame"])
        offset = int(row["offset"])
        signature = str(row.get("comboOptionSignature") or row.get("optionSignature") or "unknown")
        buckets[(frame, offset, signature)].append(row)
        point_counts[(frame, offset)] += 1

    qualified: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (frame, offset, _), option_rows in buckets.items():
        if frame not in baselines or not meaningful_option(option_rows[0]):
            continue
        if len(option_rows) < min_option_samples:
            continue
        share = len(option_rows) / max(1, point_counts[(frame, offset)])
        if share < min_option_share:
            continue
        metrics = option_summary(option_rows, baselines[frame])
        if float(metrics["improvementRate"]) < min_improvement_rate:
            continue
        metrics["offset"] = offset
        metrics["optionShare"] = round(share, 4)
        metrics["_rank"] = (
            float(metrics["improvementRate"]),
            float(metrics["killRate"]),
            float(metrics["medianScoreDeltaVsReplay"]),
            float(metrics["meanDamageDeltaVsReplay"]),
            int(metrics["samples"]),
        )
        qualified[frame].append(metrics)

    result: dict[int, dict[str, Any]] = {}
    ranked = []
    for frame, options in qualified.items():
        options.sort(key=lambda item: item["_rank"], reverse=True)
        primary = options[0]
        signatures = []
        for item in options:
            signature = str(item["optionSignature"])
            if signature not in signatures:
                signatures.append(signature)
        result[frame] = {
            "signature": str(primary["optionSignature"]),
            "offset": int(primary["offset"]),
            "improvement_rate": float(primary["improvementRate"]),
            "rank_key": primary["_rank"],
            "alternatives": signatures[1:4],
        }
        ranked.append((frame, primary["_rank"]))
    ranked.sort(key=lambda item: item[1], reverse=True)
    for rank, (frame, _) in enumerate(ranked, 1):
        result[frame]["rank"] = rank
    return result


def select_with_threshold(
    rows: list[dict[str, Any]],
    baselines: dict[int, dict[str, Any]],
    *,
    stage: str,
    knob: str,
    value: float,
) -> dict[int, dict[str, Any]]:
    settings: dict[str, float | int] = {
        "min_option_samples": 2 if stage == "preflight" else 8,
        "min_option_share": 0.1,
        "min_improvement_rate": 0.25,
    }
    settings[knob] = int(value) if knob == "min_option_samples" else float(value)
    return select_options(rows, baselines, **settings)


def queue_options(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result = {}
    for target in payload.get("targets") or []:
        option = target["option"]
        result[int(target["base_frame"])] = {
            "signature": str(option["optionSignature"]),
            "offset": int(target["offset"]),
            "improvement_rate": float(option["improvementRate"]),
            "rank": int(option.get("selectionRank") or target.get("queue_index") or 0),
            "alternatives": [
                str(item["option"]["optionSignature"])
                for item in target.get("alternative_routes") or []
            ],
        }
    return result


def ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def compare_options(candidate: dict[int, dict[str, Any]], reference: dict[int, dict[str, Any]]) -> dict[str, Any]:
    candidate_frames = set(candidate)
    reference_frames = set(reference)
    common = candidate_frames & reference_frames
    primary_matches = sum(candidate[frame]["signature"] == reference[frame]["signature"] for frame in common)
    exact_matches = sum(
        candidate[frame]["signature"] == reference[frame]["signature"]
        and candidate[frame]["offset"] == reference[frame]["offset"]
        for frame in common
    )
    rate_errors = [
        abs(candidate[frame]["improvement_rate"] - reference[frame]["improvement_rate"])
        for frame in common
        if candidate[frame]["signature"] == reference[frame]["signature"]
    ]
    rank_errors = [abs(candidate[frame]["rank"] - reference[frame]["rank"]) for frame in common]
    return {
        "selected_count": len(candidate_frames),
        "reference_count": len(reference_frames),
        "frame_precision": round(ratio(len(common), len(candidate_frames)), 6),
        "frame_recall": round(ratio(len(common), len(reference_frames)), 6),
        "frame_jaccard": round(ratio(len(common), len(candidate_frames | reference_frames)), 6),
        "primary_agreement": round(ratio(primary_matches, len(common)), 6),
        "exact_route_agreement": round(ratio(exact_matches, len(common)), 6),
        "mean_improvement_rate_error": round(mean(rate_errors), 6) if rate_errors else None,
        "mean_rank_error": round(mean(rank_errors), 6) if rank_errors else None,
    }


def projected_stage_seconds(summary: dict[str, Any], actual_seconds: float, projected_lanes: int) -> float:
    actual_lanes = int(summary.get("laneCount") or 0)
    if not actual_lanes or not actual_seconds:
        return 0.0
    timings = summary.get("timings") or {}
    fixed = float(timings.get("build_validation_buffers_s") or 0.0) + float(timings.get("load_state_s") or 0.0)
    variable = max(0.0, actual_seconds - fixed)
    return fixed + variable * projected_lanes / actual_lanes


def evaluate_job(
    summary_path: Path,
    random_trials: int,
    game_metadata: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    job = read_json(summary_path)
    run_root = summary_path.parent
    game_root = run_root.parents[1]
    baselines_payload = read_json(game_root / "candidates.json")
    baselines = {int(item["frame"]): item for item in baselines_payload.get("frames") or []}
    preflight_rows = read_jsonl(Path(job["preflight"]) / "lanes.jsonl")
    refinement_rows = read_jsonl(Path(job["refinement"]) / "lanes.jsonl") if job.get("final_selection") else []
    reference_preflight = queue_options(read_json(Path(job["preflight_selection"])))
    reference_final = queue_options(read_json(Path(job["final_selection"]))) if job.get("final_selection") else {}
    preflight_summary = read_json(Path(job["preflight"]) / "summary.json")
    refinement_summary = read_json(Path(job["refinement"]) / "summary.json") if refinement_rows else {}
    full_preflight = max(int(row["sampleIndex"]) for row in preflight_rows) + 1
    full_refinement = max((int(row["sampleIndex"]) for row in refinement_rows), default=-1) + 1
    trials = [("prefix", 0)] + [("random", trial) for trial in range(1, random_trials + 1)]
    output = []
    for mode, trial in trials:
        for preflight_budget in PREFLIGHT_BUDGETS:
            if preflight_budget > full_preflight:
                continue
            pre_indices = sample_indices(
                full_preflight,
                preflight_budget,
                mode=mode,
                seed=stable_seed(job["game"], job["replicate"], "preflight", trial, preflight_budget),
            )
            preflight = select_options(
                filter_rows(preflight_rows, pre_indices),
                baselines,
                min_option_samples=2,
            )
            preflight_comparison = compare_options(preflight, reference_preflight)
            for refinement_budget in REFINEMENT_BUDGETS:
                if refinement_budget > full_refinement:
                    continue
                ref_indices = sample_indices(
                    full_refinement,
                    refinement_budget,
                    mode=mode,
                    seed=stable_seed(job["game"], job["replicate"], "refinement", trial, refinement_budget),
                )
                refined = select_options(
                    filter_rows(refinement_rows, ref_indices, set(preflight)),
                    baselines,
                    min_option_samples=8,
                )
                projected_preflight_lanes = len(baselines) * preflight_budget
                projected_refinement_lanes = len(preflight) * 3 * refinement_budget
                projected_seconds = (
                    float(job.get("timings", {}).get("candidates") or 0.0)
                    + projected_stage_seconds(
                        preflight_summary,
                        float(job.get("timings", {}).get("preflight") or 0.0),
                        projected_preflight_lanes,
                    )
                    + projected_stage_seconds(
                        refinement_summary,
                        float(job.get("timings", {}).get("refinement") or 0.0),
                        projected_refinement_lanes,
                    )
                )
                output.append({
                    "game": job["game"],
                    "matchup": game_metadata.get(job["game"], {}).get("matchup"),
                    "stage": game_metadata.get(job["game"], {}).get("stage"),
                    "replicate": int(job["replicate"]),
                    "mode": mode,
                    "trial": trial,
                    "preflight_budget": preflight_budget,
                    "refinement_budget": refinement_budget,
                    "projected_lanes": projected_preflight_lanes + projected_refinement_lanes,
                    "projected_seconds": round(projected_seconds, 3),
                    **{f"preflight_{key}": value for key, value in preflight_comparison.items()},
                    **compare_options(refined, reference_final),
                })
    return output


def evaluate_thresholds(summary_path: Path, game_metadata: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    job = read_json(summary_path)
    run_root = summary_path.parent
    game_root = run_root.parents[1]
    baselines_payload = read_json(game_root / "candidates.json")
    baselines = {int(item["frame"]): item for item in baselines_payload.get("frames") or []}
    preflight_rows = read_jsonl(Path(job["preflight"]) / "lanes.jsonl")
    refinement_rows = read_jsonl(Path(job["refinement"]) / "lanes.jsonl") if job.get("final_selection") else []
    reference_preflight = queue_options(read_json(Path(job["preflight_selection"])))
    reference_final = queue_options(read_json(Path(job["final_selection"]))) if job.get("final_selection") else {}
    output = []
    for stage in ("preflight", "refinement"):
        source_rows = preflight_rows if stage == "preflight" else refinement_rows
        if not source_rows:
            continue
        for knob, values in THRESHOLD_SWEEPS.items():
            for value in values:
                selected = select_with_threshold(
                    source_rows,
                    baselines,
                    stage=stage,
                    knob=knob,
                    value=value,
                )
                if stage == "preflight":
                    downstream = select_options(
                        [row for row in refinement_rows if int(row["baseFrame"]) in selected],
                        baselines,
                        min_option_samples=8,
                    )
                    stage_comparison = compare_options(selected, reference_preflight)
                    final_comparison = compare_options(downstream, reference_final)
                else:
                    stage_comparison = compare_options(selected, reference_final)
                    final_comparison = stage_comparison
                output.append({
                    "game": job["game"],
                    "matchup": game_metadata.get(job["game"], {}).get("matchup"),
                    "replicate": int(job["replicate"]),
                    "stage": stage,
                    "knob": knob,
                    "value": value,
                    **{f"stage_{key}": metric for key, metric in stage_comparison.items()},
                    **{f"final_{key}": metric for key, metric in final_comparison.items()},
                })
    return output


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["preflight_budget"], row["refinement_budget"])].append(row)
    metrics = (
        "preflight_frame_precision",
        "preflight_frame_recall",
        "frame_precision",
        "frame_recall",
        "frame_jaccard",
        "primary_agreement",
        "exact_route_agreement",
        "mean_improvement_rate_error",
        "mean_rank_error",
        "projected_seconds",
    )
    output = []
    for (mode, preflight, refinement), items in groups.items():
        record = {
            "mode": mode,
            "preflight_budget": preflight,
            "refinement_budget": refinement,
            "observations": len(items),
        }
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            record[f"mean_{metric}"] = round(mean(values), 6) if values else None
            record[f"median_{metric}"] = round(median(values), 6) if values else None
        output.append(record)
    return sorted(output, key=lambda item: (item["mode"], item["preflight_budget"], item["refinement_budget"]))


def aggregate_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["stage"], row["knob"], float(row["value"]))].append(row)
    metrics = (
        "stage_frame_precision",
        "stage_frame_recall",
        "stage_frame_jaccard",
        "stage_primary_agreement",
        "final_frame_precision",
        "final_frame_recall",
        "final_frame_jaccard",
        "final_primary_agreement",
    )
    output = []
    for (stage, knob, value), items in groups.items():
        record = {"stage": stage, "knob": knob, "value": value, "observations": len(items)}
        for metric in metrics:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            record[f"mean_{metric}"] = round(mean(values), 6) if values else None
        output.append(record)
    return sorted(output, key=lambda item: (item["stage"], item["knob"], item["value"]))


def cross_replicate(benchmark: Path) -> list[dict[str, Any]]:
    by_game: dict[str, list[Path]] = defaultdict(list)
    for path in benchmark.glob("games/*/replicates/*/job_summary.json"):
        if read_json(path).get("status") == "complete":
            by_game[path.parents[2].name].append(path)
    output = []
    for game, paths in by_game.items():
        paths.sort()
        for left, right in zip(paths, paths[1:]):
            left_job = read_json(left)
            right_job = read_json(right)
            if not left_job.get("final_selection") or not right_job.get("final_selection"):
                continue
            comparison = compare_options(
                queue_options(read_json(Path(right_job["final_selection"]))),
                queue_options(read_json(Path(left_job["final_selection"]))),
            )
            output.append({
                "game": game,
                "left_replicate": left_job["replicate"],
                "right_replicate": right_job["replicate"],
                **comparison,
            })
    return output


def write_markdown(path: Path, aggregate_rows: list[dict[str, Any]], replicate_rows: list[dict[str, Any]], jobs: int) -> None:
    prefix = [row for row in aggregate_rows if row["mode"] == "prefix"]
    ranked = sorted(
        prefix,
        key=lambda row: (
            float(row.get("mean_primary_agreement") or 0),
            float(row.get("mean_frame_jaccard") or 0),
            -(row["preflight_budget"] + 3 * row["refinement_budget"]),
        ),
        reverse=True,
    )
    lines = [
        "# Sampling Benchmark",
        "",
        f"Completed reference jobs: {jobs}",
        f"Evaluated virtual configurations: {len(aggregate_rows)}",
        "",
        "## Highest-agreement prefix configurations",
        "",
        "| Preflight | Refinement | Primary agreement | Frame Jaccard | Frame recall | Est. game time |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked[:20]:
        lines.append(
            f"| {row['preflight_budget']} | {row['refinement_budget']} | "
            f"{float(row.get('mean_primary_agreement') or 0):.1%} | "
            f"{float(row.get('mean_frame_jaccard') or 0):.1%} | "
            f"{float(row.get('mean_frame_recall') or 0):.1%} | "
            f"{float(row.get('mean_projected_seconds') or 0) / 60:.1f} min |"
        )
    lines.extend(["", "## Cross-replicate reference stability", ""])
    if replicate_rows:
        lines.extend([
            "| Game | Replicates | Primary agreement | Frame Jaccard |",
            "| --- | --- | ---: | ---: |",
        ])
        for row in replicate_rows:
            lines.append(
                f"| {row['game']} | {row['left_replicate']} vs {row['right_replicate']} | "
                f"{row['primary_agreement']:.1%} | {row['frame_jaccard']:.1%} |"
            )
    else:
        lines.append("Independent replicate pairs are not complete yet.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    benchmark = args.benchmark.resolve()
    out = (args.out or benchmark / "evaluation").resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = benchmark / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {"games": []}
    game_metadata = {str(game["id"]): game for game in manifest.get("games") or []}
    detail = []
    threshold_detail = []
    summaries = sorted(benchmark.glob("games/*/replicates/*/job_summary.json"))
    for summary in summaries:
        if read_json(summary).get("status") == "complete":
            detail.extend(evaluate_job(summary, args.random_trials, game_metadata))
            threshold_detail.extend(evaluate_thresholds(summary, game_metadata))
    aggregate_rows = aggregate(detail)
    threshold_rows = aggregate_thresholds(threshold_detail)
    replicate_rows = cross_replicate(benchmark)
    (out / "virtual_budget_results.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in detail),
        encoding="utf-8",
    )
    (out / "aggregate.json").write_text(json.dumps(aggregate_rows, indent=2) + "\n", encoding="utf-8")
    (out / "cross_replicate.json").write_text(json.dumps(replicate_rows, indent=2) + "\n", encoding="utf-8")
    (out / "threshold_sensitivity.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in threshold_detail),
        encoding="utf-8",
    )
    (out / "threshold_aggregate.json").write_text(json.dumps(threshold_rows, indent=2) + "\n", encoding="utf-8")
    write_markdown(out / "sampling_report.md", aggregate_rows, replicate_rows, len(summaries))
    print(json.dumps({
        "completed_jobs": len(summaries),
        "detail_rows": len(detail),
        "aggregate_rows": len(aggregate_rows),
        "threshold_rows": len(threshold_detail),
        "cross_replicate_pairs": len(replicate_rows),
        "report": str((out / "sampling_report.md").resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
