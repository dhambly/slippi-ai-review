"""Probe how early Phillip needs control to change a recorded neutral opening."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .candidates import extract_raw_conversions, player_index_for_port
from .paths import module_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--raw-events-json", type=Path, help="Reuse canonical replay conversions from candidate generation.")
    parser.add_argument("--analyzed-port", type=int, choices=(1, 2), required=True)
    parser.add_argument("--step-frames", type=int, default=10)
    parser.add_argument("--max-lookback-frames", type=int, default=80)
    parser.add_argument("--post-opening-frames", type=int, default=30)
    parser.add_argument("--samples-per-point", type=int, default=48)
    parser.add_argument("--consistency-threshold", type=float, default=0.75)
    parser.add_argument("--max-events-per-kind", type=int, default=4)
    parser.add_argument("--kind", choices=("both", "avoid", "secure"), default="both")
    parser.add_argument("--max-batch-lanes", type=int, default=4096)
    parser.add_argument(
        "--defender-delay-frames",
        type=int,
        default=None,
        help="Override recorded-opponent duration. By default replay inputs remain active through the evaluation deadline.",
    )
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.20)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--msl-root", type=Path)
    parser.add_argument("--slippi-ai-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def neutral_opening_events(raw: dict[str, Any], analyzed_port: int) -> list[dict[str, Any]]:
    settings = raw.get("settings") or {}
    analyzed_index = player_index_for_port(settings, analyzed_port)
    events = []
    for conversion in raw.get("conversions") or []:
        if str(conversion.get("openingType") or "") != "neutral-win":
            continue
        moves = conversion.get("moves") or []
        if not moves:
            continue
        attacker = int(conversion.get("lastHitBy", -1))
        defender = int(conversion.get("playerIndex", -1))
        if analyzed_index == attacker:
            kind = "secure"
        elif analyzed_index == defender:
            kind = "avoid"
        else:
            continue
        opening = moves[0]
        events.append(
            {
                "eventId": f"{kind}_{int(conversion.get('conversionIndex', len(events))):03d}_f{int(opening['frame'])}",
                "kind": kind,
                "openingFrame": int(opening["frame"]),
                "openingMove": opening.get("moveShortName") or opening.get("moveName") or "unknown",
                "attackerPlayerIndex": attacker,
                "defenderPlayerIndex": defender,
                "conversionIndex": int(conversion.get("conversionIndex", len(events))),
            }
        )
    return events


def choose_events(events: list[dict[str, Any]], kind: str, maximum: int) -> list[dict[str, Any]]:
    allowed = {"avoid", "secure"} if kind == "both" else {kind}
    counts: dict[str, int] = defaultdict(int)
    chosen = []
    for event in events:
        event_kind = str(event["kind"])
        if event_kind not in allowed:
            continue
        if maximum > 0 and counts[event_kind] >= maximum:
            continue
        counts[event_kind] += 1
        chosen.append(event)
    return chosen


def build_points(
    events: list[dict[str, Any]],
    *,
    first_frame: int,
    step_frames: int,
    max_lookback_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tests = []
    unique_frames = {}
    for event in events:
        for lookback in range(step_frames, max_lookback_frames + 1, step_frames):
            injection_frame = int(event["openingFrame"]) - lookback
            if injection_frame < first_frame:
                continue
            test = {**event, "lookbackFrames": lookback, "injectionFrame": injection_frame}
            tests.append(test)
            unique_frames.setdefault(
                injection_frame,
                {
                    "frame": injection_frame,
                    "opportunity_role": "neutral_lookback",
                    "point_evidence": {"neutral_probe": True},
                },
            )
    return tests, [unique_frames[frame] for frame in sorted(unique_frames)]


def _within(frame: Any, deadline: int) -> bool:
    return frame is not None and int(frame) <= deadline


def summarize_test(
    test: dict[str, Any],
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    deadline = int(test["openingFrame"]) + int(test["postOpeningFrames"])
    landed = 0
    avoided = 0
    won_first = 0
    no_contact = 0
    for row in rows:
        contact = row.get("neutralContact") or {}
        dealt = contact.get("firstDamageDealtFrame")
        taken = contact.get("firstDamageTakenFrame")
        dealt_in_window = _within(dealt, deadline)
        taken_in_window = _within(taken, deadline)
        landed += int(dealt_in_window)
        avoided += int(not taken_in_window)
        won_first += int(dealt_in_window and (not taken_in_window or int(dealt) < int(taken)))
        no_contact += int(not dealt_in_window and not taken_in_window)
    samples = len(rows)
    denominator = max(1, samples)
    baseline_contact = ((baseline_rows or [{}])[0].get("neutralContact") or {})
    baseline_landed = (
        _within(baseline_contact.get("firstDamageDealtFrame"), deadline)
        if baseline_rows
        else None
    )
    baseline_took_hit = (
        _within(baseline_contact.get("firstDamageTakenFrame"), deadline)
        if baseline_rows
        else None
    )
    baseline_reproduced = (
        (baseline_took_hit if test["kind"] == "avoid" else baseline_landed)
        if baseline_rows
        else None
    )
    return {
        **test,
        "deadlineFrame": deadline,
        "samples": samples,
        "avoidHitRate": round(avoided / denominator, 4),
        "landHitRate": round(landed / denominator, 4),
        "winFirstHitRate": round(won_first / denominator, 4),
        "noContactRate": round(no_contact / denominator, 4),
        "replayInputBaselineLandedHit": baseline_landed,
        "replayInputBaselineTookHit": baseline_took_hit,
        "baselineReproduced": baseline_reproduced,
    }


def boundary(points: list[dict[str, Any]], metric: str, threshold: float) -> dict[str, Any]:
    ordered = sorted(
        (item for item in points if item.get("baselineReproduced") is not False),
        key=lambda item: int(item["lookbackFrames"]),
    )
    first = next((item for item in ordered if float(item[metric]) >= threshold), None)
    sustained = next(
        (
            item
            for index, item in enumerate(ordered)
            if all(float(later[metric]) >= threshold for later in ordered[index:])
        ),
        None,
    )
    contiguous = None
    for item in ordered:
        if float(item[metric]) < threshold:
            break
        contiguous = item
    return {
        "metric": metric,
        "threshold": threshold,
        "firstPassingLookbackFrames": int(first["lookbackFrames"]) if first else None,
        "furthestPassingLookbackFrames": int(
            next((item["lookbackFrames"] for item in reversed(ordered) if float(item[metric]) >= threshold), 0)
        ) or None,
        "contiguousPassingLookbackFrames": int(contiguous["lookbackFrames"]) if contiguous else None,
        "sustainedPassingLookbackFrames": int(sustained["lookbackFrames"]) if sustained else None,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Neutral Lookback Experiment",
        "",
        f"Replay: `{payload['replay']}`",
        f"Analyzed port: P{payload['analyzedPort']}",
        f"Consistency threshold: {payload['consistencyThreshold']:.0%}",
        "",
    ]
    for event in payload["events"]:
        summary = event["boundary"]
        label = "avoid being hit" if event["kind"] == "avoid" else "land a hit"
        relevant_boundary = (
            summary["firstPassingLookbackFrames"]
            if event["kind"] == "avoid"
            else summary["contiguousPassingLookbackFrames"]
        )
        lines.extend(
            [
                f"## {event['eventId']}: {event['kind']} {event['openingMove']} at f{event['openingFrame']}",
                "",
                f"Relevant lookback boundary for **{label}**: `{relevant_boundary}` frames.",
                f"Nearest / furthest passing lookback: `{summary['firstPassingLookbackFrames']}` / `{summary['furthestPassingLookbackFrames']}` frames.",
                f"Contiguous passing lookback from the recorded hit: `{summary['contiguousPassingLookbackFrames']}` frames.",
                f"Sustained passing lookback: `{summary['sustainedPassingLookbackFrames']}` frames.",
                "",
                "| Lookback | Replay valid | Avoid hit | Land hit | Win first hit | No contact |",
                "|---:|:---:|---:|---:|---:|---:|",
            ]
        )
        for point in event["points"]:
            lines.append(
                f"| {point['lookbackFrames']}f | {'yes' if point['baselineReproduced'] else 'NO'} | "
                f"{point['avoidHitRate']:.1%} | "
                f"{point['landHitRate']:.1%} | {point['winFirstHitRate']:.1%} | {point['noContactRate']:.1%} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if args.step_frames <= 0 or args.max_lookback_frames < args.step_frames:
        raise ValueError("lookback range must contain at least one positive step")
    if args.samples_per_point <= 0:
        raise ValueError("--samples-per-point must be positive")
    if not 0 < args.consistency_threshold <= 1:
        raise ValueError("--consistency-threshold must be in (0, 1]")
    if not 0 < args.gpu_duty_cycle <= 1:
        raise ValueError("--gpu-duty-cycle must be in (0, 1]")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    replay = args.replay.resolve()
    started = time.perf_counter()
    raw = (
        json.loads(args.raw_events_json.resolve().read_text(encoding="utf-8"))
        if args.raw_events_json is not None
        else extract_raw_conversions(replay)
    )
    earliest_injection_frame = int(raw.get("firstFrame") or -123) + int(args.warmup_frames)
    eligible_events = [
        event
        for event in neutral_opening_events(raw, int(args.analyzed_port))
        if int(event["openingFrame"]) - int(args.max_lookback_frames) >= earliest_injection_frame
    ]
    events = choose_events(
        eligible_events,
        str(args.kind),
        int(args.max_events_per_kind),
    )
    tests, points = build_points(
        events,
        first_frame=earliest_injection_frame,
        step_frames=int(args.step_frames),
        max_lookback_frames=int(args.max_lookback_frames),
    )
    if not points:
        payload = {
            "version": 1,
            "replay": str(replay),
            "analyzedPort": int(args.analyzed_port),
            "stepFrames": int(args.step_frames),
            "maxLookbackFrames": int(args.max_lookback_frames),
            "postOpeningFrames": int(args.post_opening_frames),
            "samplesPerPoint": int(args.samples_per_point),
            "warmupFrames": int(args.warmup_frames),
            "defenderDelayFrames": int(args.max_lookback_frames) + int(args.post_opening_frames) + 2,
            "consistencyThreshold": float(args.consistency_threshold),
            "runtimeSeconds": round(time.perf_counter() - started, 3),
            "events": [],
        }
        (out / "neutral_experiment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (out / "neutral_experiment.md").write_text(render_markdown(payload), encoding="utf-8")
        print(json.dumps({"out": str(out), "events": 0, "runtimeSeconds": payload["runtimeSeconds"]}))
        return 0
    for test in tests:
        test["postOpeningFrames"] = int(args.post_opening_frames)

    points_path = out / "neutral_points.json"
    points_path.write_text(json.dumps({"frames": points}, indent=2) + "\n", encoding="utf-8")
    simulation_dir = out / "simulation"
    rollout_frames = int(args.max_lookback_frames) + int(args.post_opening_frames) + 1
    defender_delay_frames = (
        int(args.defender_delay_frames)
        if args.defender_delay_frames is not None
        else rollout_frames + 1
    )
    command = module_command(
        "simulation",
        "--replay", replay,
        "--analyzed-port", args.analyzed_port,
        "--takeover-frames-json", points_path,
        "--offsets", "0",
        "--samples-per-point", args.samples_per_point,
        "--max-batch-lanes", args.max_batch_lanes,
        "--objective", "general",
        "--rollout-frames", rollout_frames,
        "--defense-resolution-extra-frames", 0,
        "--defense-resolution-min-frames", rollout_frames,
        "--defender-delay-frames", defender_delay_frames,
        "--warmup-frames", args.warmup_frames,
        "--history-mode", "teacher-forced",
        "--opponent-mode", "replay",
        "--rng-mode", "replay",
        "--dump-controller-streams",
        "--out", simulation_dir,
    )
    for flag, value in (
        ("--model", args.model),
        ("--msl-root", args.msl_root),
        ("--slippi-ai-root", args.slippi_ai_root),
    ):
        if value is not None:
            command.extend([flag, str(value.resolve())])
    if args.enable_gpu:
        command.extend(["--enable-gpu", "--gpu-duty-cycle", str(args.gpu_duty_cycle)])
    log_path = out / "simulation.log"
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)

    baseline_dir = out / "replay_input_control"
    baseline_command = list(command)
    baseline_command[baseline_command.index("--samples-per-point") + 1] = "1"
    baseline_command[baseline_command.index("--out") + 1] = str(baseline_dir)
    baseline_command.remove("--dump-controller-streams")
    baseline_command.extend(["--analyzed-mode", "replay"])
    with (out / "replay_input_control.log").open("w", encoding="utf-8") as log:
        subprocess.run(baseline_command, check=True, stdout=log, stderr=subprocess.STDOUT)

    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with (simulation_dir / "lanes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_by_frame[int(row["baseFrame"])].append(row)
    baseline_rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with (baseline_dir / "lanes.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            baseline_rows_by_frame[int(row["baseFrame"])].append(row)
    summarized = [
        summarize_test(
            test,
            rows_by_frame[int(test["injectionFrame"])],
            baseline_rows_by_frame[int(test["injectionFrame"])],
        )
        for test in tests
    ]
    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in summarized:
        by_event[str(point["eventId"])].append(point)
    event_payload = []
    event_map = {str(event["eventId"]): event for event in events}
    for event_id, event_points in by_event.items():
        event = event_map[event_id]
        metric = "avoidHitRate" if event["kind"] == "avoid" else "landHitRate"
        event_payload.append(
            {
                **event,
                "boundary": boundary(event_points, metric, float(args.consistency_threshold)),
                "points": sorted(event_points, key=lambda item: int(item["lookbackFrames"])),
            }
        )
    payload = {
        "version": 1,
        "replay": str(replay),
        "analyzedPort": int(args.analyzed_port),
        "stepFrames": int(args.step_frames),
        "maxLookbackFrames": int(args.max_lookback_frames),
        "postOpeningFrames": int(args.post_opening_frames),
        "samplesPerPoint": int(args.samples_per_point),
        "warmupFrames": int(args.warmup_frames),
        "defenderDelayFrames": defender_delay_frames,
        "consistencyThreshold": float(args.consistency_threshold),
        "runtimeSeconds": round(time.perf_counter() - started, 3),
        "events": event_payload,
    }
    (out / "neutral_experiment.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "neutral_experiment.md").write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"out": str(out), "events": len(event_payload), "runtimeSeconds": payload["runtimeSeconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
