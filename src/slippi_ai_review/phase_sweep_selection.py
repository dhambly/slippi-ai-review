"""Select transparent representative lanes from a low-sample phase sweep."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def option_name(row: dict[str, Any]) -> str:
    return str(row.get("comboOptionSignature") or row.get("optionSignature") or "NO_COMMITMENT")


def option_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row.get("score") or 0.0) for row in rows]
    return {
        "optionSignature": option_name(rows[0]),
        "samples": len(rows),
        "meanScore": round(sum(scores) / len(scores), 3),
        "medianScore": round(float(median(scores)), 3),
        "bestScore": round(max(scores), 3),
        "damageDealt": round(sum(float(row.get("damageDealt") or 0.0) for row in rows) / len(rows), 3),
        "damageTaken": round(sum(float(row.get("damageTaken") or 0.0) for row in rows) / len(rows), 3),
        "reversalRate": round(sum(bool(row.get("comboReversed")) for row in rows) / len(rows), 4),
        "killRate": round(sum(int(row.get("defenderStocksLost") or 0) > 0 for row in rows) / len(rows), 4),
    }


def representative(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    target = float(summary["medianScore"])
    return min(rows, key=lambda row: (abs(float(row.get("score") or 0.0) - target), int(row.get("laneId") or 0)))


def build_queue(inventory: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[int(row["baseFrame"])].append(row)
    targets = []
    omitted = []
    for segment in inventory.get("segments") or []:
        frame = int(segment["injectionFrame"])
        point_rows = rows_by_frame.get(frame) or []
        if not point_rows:
            omitted.append({"id": segment["id"], "reason": "missing_sweep_rows"})
            continue
        options: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in point_rows:
            options[option_name(row)].append(row)
        summaries = [(option_summary(option_rows), option_rows) for option_rows in options.values()]
        chosen, chosen_rows = max(
            summaries,
            key=lambda item: (float(item[0]["meanScore"]), float(item[0]["bestScore"]), int(item[0]["samples"])),
        )
        chosen["sweepSamples"] = len(point_rows)
        chosen["optionShare"] = round(len(chosen_rows) / len(point_rows), 4)
        lane = representative(chosen_rows, chosen)
        inference_start = int(lane.get("takeoverFrame") or frame)
        model_control = int(lane.get("modelControlFrame") or inference_start)
        targets.append({
            "queue_index": len(targets) + 1,
            "phase": segment["phase"],
            "phase_segment": segment,
            "base_frame": frame,
            "inference_start_frame": inference_start,
            "takeover_frame": model_control,
            "branch_start_frame": inference_start,
            "label": f"{segment['phase'].title()} sweep: {segment['label']}",
            "option": chosen,
            "representative_lane": lane,
            "alternative_routes": [],
            "replay_baseline": {
                "opening_move": {"frame": frame, "moveShortName": segment["phase"]},
                "opening_type": "phase-sweep",
                "opportunity_role": f"phase_sweep_{segment['phase']}",
                "original_followup_damage": 0.0,
                "original_followup_hits": 0,
                "original_followup_kill": False,
            },
        })
    return {
        "version": 1,
        "mode": "phase-sweep",
        "replay": inventory.get("replay"),
        "game": inventory.get("game") or {},
        "controlled_port": inventory.get("analyzedPort"),
        "rollout_frames": 120,
        "video_preroll_frames": 60,
        "selection_audit": {"segmentCounts": inventory.get("counts"), "omitted": omitted},
        "targets": targets,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.resolve().read_text(encoding="utf-8"))
    queue = build_queue(inventory, read_jsonl(args.run_dir.resolve() / "lanes.jsonl"))
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "targets": len(queue["targets"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
