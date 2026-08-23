"""Compare legacy and decomp simulation runs at shared takeover points."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


Point = tuple[int, int]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def point_key(row: dict[str, Any]) -> Point:
    return int(row["baseFrame"]), int(row["offset"])


def group_points(rows: Iterable[dict[str, Any]]) -> dict[Point, list[dict[str, Any]]]:
    grouped: dict[Point, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[point_key(row)].append(row)
    return dict(grouped)


def option_signature(row: dict[str, Any], objective: str) -> str:
    key = "comboOptionSignature" if objective == "combo-extension" else "optionSignature"
    return str(row.get(key) or row.get("optionSignature") or "UNKNOWN")


def distribution(rows: list[dict[str, Any]], objective: str) -> dict[str, float]:
    counts = Counter(option_signature(row, objective) for row in rows)
    total = max(1, len(rows))
    return {key: count / total for key, count in sorted(counts.items())}


def total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return fmean(values) if values else 0.0


def reason_distribution(rows: list[dict[str, Any]]) -> dict[str, float]:
    reasons = Counter(str((row.get("resolution") or {}).get("reason") or "unknown") for row in rows)
    total = max(1, len(rows))
    return {key: count / total for key, count in sorted(reasons.items())}


def compare_point(
    point: Point,
    legacy_rows: list[dict[str, Any]],
    decomp_rows: list[dict[str, Any]],
    *,
    objective: str,
    option_tv_limit: float,
) -> dict[str, Any]:
    legacy_states = {json.dumps(row.get("startState"), sort_keys=True) for row in legacy_rows}
    decomp_states = {json.dumps(row.get("startState"), sort_keys=True) for row in decomp_rows}
    legacy_takeovers = {int(row["takeoverFrame"]) for row in legacy_rows}
    decomp_takeovers = {int(row["takeoverFrame"]) for row in decomp_rows}
    legacy_options = distribution(legacy_rows, objective)
    decomp_options = distribution(decomp_rows, objective)
    option_tv = total_variation(legacy_options, decomp_options)
    branch_state_exact = len(legacy_states) == 1 and legacy_states == decomp_states
    takeover_frames_exact = len(legacy_takeovers) == 1 and legacy_takeovers == decomp_takeovers
    hard_parity = branch_state_exact and takeover_frames_exact

    return {
        "baseFrame": point[0],
        "offset": point[1],
        "legacySamples": len(legacy_rows),
        "decompSamples": len(decomp_rows),
        "branchStateExact": branch_state_exact,
        "takeoverFramesExact": takeover_frames_exact,
        "legacyTakeoverFrames": sorted(legacy_takeovers),
        "decompTakeoverFrames": sorted(decomp_takeovers),
        "hardParity": hard_parity,
        "optionDistributionTotalVariation": round(option_tv, 6),
        "optionDistributionWithinLimit": option_tv <= option_tv_limit,
        "legacyOptionDistribution": {key: round(value, 6) for key, value in legacy_options.items()},
        "decompOptionDistribution": {key: round(value, 6) for key, value in decomp_options.items()},
        "outcomes": {
            "legacy": {
                "meanScore": round(mean(legacy_rows, "score"), 4),
                "meanDamageDealt": round(mean(legacy_rows, "damageDealt"), 4),
                "meanDamageTaken": round(mean(legacy_rows, "damageTaken"), 4),
                "improvementRate": round(mean(legacy_rows, "outperformedReplay"), 6),
                "resolutionReasons": reason_distribution(legacy_rows),
            },
            "decomp": {
                "meanScore": round(mean(decomp_rows, "score"), 4),
                "meanDamageDealt": round(mean(decomp_rows, "damageDealt"), 4),
                "meanDamageTaken": round(mean(decomp_rows, "damageTaken"), 4),
                "improvementRate": round(mean(decomp_rows, "outperformedReplay"), 6),
                "resolutionReasons": reason_distribution(decomp_rows),
            },
        },
    }


def compare_runs(legacy_run: Path, decomp_run: Path, *, option_tv_limit: float = 0.4) -> dict[str, Any]:
    legacy_summary_path = legacy_run / "summary.json"
    decomp_summary_path = decomp_run / "summary.json"
    legacy_summary = (
        json.loads(legacy_summary_path.read_text(encoding="utf-8"))
        if legacy_summary_path.is_file()
        else {}
    )
    decomp_summary = (
        json.loads(decomp_summary_path.read_text(encoding="utf-8"))
        if decomp_summary_path.is_file()
        else {}
    )
    legacy_objective = str(legacy_summary.get("objective") or "unknown")
    decomp_objective = str(decomp_summary.get("objective") or "unknown")
    objectives_match = legacy_objective == decomp_objective
    objective = legacy_objective if objectives_match else "unknown"
    legacy = group_points(read_jsonl(legacy_run / "lanes.jsonl"))
    decomp = group_points(read_jsonl(decomp_run / "lanes.jsonl"))
    shared = sorted(set(legacy) & set(decomp))
    points = [
        compare_point(
            point,
            legacy[point],
            decomp[point],
            objective=objective,
            option_tv_limit=option_tv_limit,
        )
        for point in shared
    ]
    hard_parity_points = sum(int(point["hardParity"]) for point in points)
    option_parity_points = sum(int(point["optionDistributionWithinLimit"]) for point in points)

    return {
        "legacyRun": str(legacy_run.resolve()),
        "decompRun": str(decomp_run.resolve()),
        "legacyObjective": legacy_objective,
        "decompObjective": decomp_objective,
        "objectivesMatch": objectives_match,
        "optionDistributionTotalVariationLimit": option_tv_limit,
        "sharedPointCount": len(points),
        "legacyOnlyPoints": [list(point) for point in sorted(set(legacy) - set(decomp))],
        "decompOnlyPoints": [list(point) for point in sorted(set(decomp) - set(legacy))],
        "hardParityPointCount": hard_parity_points,
        "hardParityRate": round(hard_parity_points / max(1, len(points)), 6),
        "optionParityPointCount": option_parity_points,
        "optionParityRate": round(option_parity_points / max(1, len(points)), 6),
        "passed": (
            bool(points)
            and objectives_match
            and hard_parity_points == len(points)
            and option_parity_points == len(points)
        ),
        "points": points,
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        "# MSL Backend Comparison",
        "",
        f"**Status:** {status}",
        "",
        f"- Shared takeover points: {report['sharedPointCount']}",
        f"- Objective: {report['legacyObjective']}",
        f"- Exact branch/takeover parity: {report['hardParityPointCount']}/{report['sharedPointCount']}",
        f"- Option distribution parity: {report['optionParityPointCount']}/{report['sharedPointCount']}",
        f"- Option TV limit: {report['optionDistributionTotalVariationLimit']}",
        "",
        "Outcome differences are diagnostic only. They are not used as an injection-parity gate.",
        "",
        "| Base frame | Offset | Branch | Takeover | Option TV | Legacy score | Decomp score |",
        "| ---: | ---: | :---: | :---: | ---: | ---: | ---: |",
    ]
    for point in report["points"]:
        outcomes = point["outcomes"]
        lines.append(
            f"| {point['baseFrame']} | {point['offset']} | "
            f"{'yes' if point['branchStateExact'] else 'no'} | "
            f"{'yes' if point['takeoverFramesExact'] else 'no'} | "
            f"{point['optionDistributionTotalVariation']:.3f} | "
            f"{outcomes['legacy']['meanScore']:.2f} | {outcomes['decomp']['meanScore']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run", type=Path, required=True)
    parser.add_argument("--decomp-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--option-tv-limit", type=float, default=0.4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = compare_runs(args.legacy_run, args.decomp_run, option_tv_limit=args.option_tv_limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("passed", "sharedPointCount", "hardParityRate", "optionParityRate")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
