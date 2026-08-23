"""Select reliable neutral-loss avoidance boundaries and representative lanes."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable


NON_ACTIONABLE_OPTIONS = frozenset({"NO_COMMITMENT", "WAIT", "NONE"})
EMERGENCY_OPTIONS = frozenset({"ROLL_FORWARD", "ROLL_BACKWARD", "SPOTDODGE", "AIRDODGE"})


def wilson_lower(successes: int, samples: int, *, z: float = 1.645) -> float:
    """One-sided Wilson lower confidence bound for a binomial rate."""
    if samples <= 0:
        return 0.0
    successes = max(0, min(int(successes), int(samples)))
    n = float(samples)
    p = successes / n
    denominator = 1.0 + z * z / n
    center = p + z * z / (2.0 * n)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (center - spread) / denominator)


def success_count(point: dict[str, Any], metric: str = "avoidHitRate") -> int:
    return int(round(float(point.get(metric) or 0.0) * int(point.get("samples") or 0)))


def nearest_reliable_boundary(
    points: Iterable[dict[str, Any]],
    *,
    threshold: float,
    z: float = 1.645,
) -> dict[str, Any] | None:
    for point in sorted(points, key=lambda item: int(item["lookbackFrames"])):
        if point.get("baselineReproduced") is False:
            continue
        samples = int(point.get("samples") or 0)
        successes = success_count(point)
        lower = wilson_lower(successes, samples, z=z)
        if float(point.get("avoidHitRate") or 0.0) >= threshold and lower >= threshold:
            return {**point, "avoidHitWilsonLower": round(lower, 4)}
    return None


def preflight_boundaries(
    payload: dict[str, Any],
    *,
    threshold: float | None = None,
    z: float = 1.645,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    threshold = float(threshold if threshold is not None else payload.get("consistencyThreshold") or 0.75)
    selected = []
    omitted = []
    for event in payload.get("events") or []:
        if event.get("kind") != "avoid":
            continue
        boundary = nearest_reliable_boundary(event.get("points") or [], threshold=threshold, z=z)
        if boundary is None:
            omitted.append({**event, "reason": "no_reliable_avoidance_boundary"})
            continue
        selected.append({**event, "boundaryPoint": boundary})
    return selected, omitted


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def avoided_by_deadline(row: dict[str, Any], deadline: int) -> bool:
    frame = (row.get("neutralContact") or {}).get("firstDamageTakenFrame")
    return frame is None or int(frame) > int(deadline)


def landed_by_deadline(row: dict[str, Any], deadline: int) -> bool:
    frame = (row.get("neutralContact") or {}).get("firstDamageDealtFrame")
    return frame is not None and int(frame) <= int(deadline)


def won_neutral_by_deadline(row: dict[str, Any], deadline: int) -> bool:
    """True when Phillip lands the first non-trade hit in the neutral window."""
    contact = row.get("neutralContact") or {}
    dealt = contact.get("firstDamageDealtFrame")
    taken = contact.get("firstDamageTakenFrame")
    return dealt is not None and int(dealt) <= int(deadline) and (taken is None or int(dealt) < int(taken))


CAUSAL_ACTIONS = {
    "ROLL_FORWARD": "ROLL_FORWARD",
    "ROLL_BACKWARD": "ROLL_BACKWARD",
    "SPOTDODGE": "SPOTDODGE",
    "AIRDODGE": "AIRDODGE",
    "PLATFORM_DROP": "PLATFORM_DROP",
    "EDGE_CATCHING": "LEDGE_GRAB",
    "EDGE_HANGING": "LEDGE_HANG",
}
ATTACK_PREFIXES = (
    "NEUTRAL_ATTACK",
    "DASH_ATTACK",
    "FTILT",
    "UPTILT",
    "DOWNTILT",
    "FSMASH",
    "UPSMASH",
    "DOWNSMASH",
    "NAIR",
    "FAIR",
    "BAIR",
    "UAIR",
    "DAIR",
    "NEUTRAL_B",
    "SIDE_B",
    "UP_B",
    "DOWN_B",
    "GRAB",
)


def model_control_frame(row: dict[str, Any], fallback: int = 0) -> int:
    return int(row.get("modelControlFrame") or row.get("takeoverFrame") or row.get("baseFrame") or fallback)


def causal_option_name(row: dict[str, Any], *, decision_frame: int | None = None) -> str:
    if decision_frame is None:
        return str(row.get("comboOptionSignature") or row.get("optionSignature") or "NO_COMMITMENT").upper()
    takeover = model_control_frame(row)
    decision_step = max(0, int(decision_frame) - takeover)
    option = row.get("option") or {}
    segments = option.get("actionSegments") or option.get("rawActionSegments") or []
    for segment in segments:
        if int(segment.get("startStep") or 0) > decision_step:
            continue
        name = str(segment.get("actionName") or "").upper()
        if name in CAUSAL_ACTIONS:
            return CAUSAL_ACTIONS[name]
        for prefix in ATTACK_PREFIXES:
            if name.startswith(prefix):
                return prefix
        if name.startswith("SHIELD"):
            return "SHIELD"
        if name == "KNEE_BEND":
            return "JUMP"
        if name in {"DASHING", "RUNNING", "TURNING"}:
            return "GROUND_MOVEMENT"
    initial = str(option.get("initialActionName") or "").upper()
    if any(token in initial for token in ("JUMP", "FALL", "AERIAL", "AIR_")):
        return "AIR_DRIFT"
    if initial in {"DASHING", "RUNNING", "TURNING", "WALK_SLOW", "WALK_MIDDLE", "WALK_FAST"}:
        return "GROUND_MOVEMENT"
    return "NO_COMMITMENT"


def option_statistics(
    rows: list[dict[str, Any]],
    *,
    deadline: int,
    threshold: float,
    min_samples: int = 8,
    min_share: float = 0.10,
    z: float = 1.645,
    decision_frame: int | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[causal_option_name(row, decision_frame=decision_frame)].append(row)
    total = max(1, len(rows))
    total_successes = max(1, sum(avoided_by_deadline(row, deadline) for row in rows))
    options = []
    for signature, option_rows in grouped.items():
        successes = [row for row in option_rows if avoided_by_deadline(row, deadline)]
        wins = [row for row in option_rows if won_neutral_by_deadline(row, deadline)]
        samples = len(option_rows)
        lower = wilson_lower(len(successes), samples, z=z)
        options.append(
            {
                "optionSignature": signature,
                "samples": samples,
                "successfulSamples": len(successes),
                "optionShare": round(samples / total, 4),
                "successfulPolicyShare": round(len(successes) / total_successes, 4),
                "avoidRate": round(len(successes) / max(1, samples), 4),
                "avoidWilsonLower": round(lower, 4),
                "winFirstHitRate": round(sum(landed_by_deadline(row, deadline) for row in successes) / max(1, len(successes)), 4),
                "neutralWinSamples": len(wins),
                "neutralWinRate": round(len(wins) / max(1, samples), 4),
                "cleanEscapeRate": round(
                    sum(
                        avoided_by_deadline(row, deadline)
                        and float(row.get("damageTaken") or 0.0) <= 0.0
                        and int(row.get("analyzedStocksLost") or 0) == 0
                        for row in option_rows
                    ) / max(1, samples),
                    4,
                ),
                "stableNeutralRate": round(
                    sum(
                        avoided_by_deadline(row, deadline)
                        and str((row.get("resolution") or {}).get("reason") or "") == "stable_neutral"
                        for row in option_rows
                    ) / max(1, samples),
                    4,
                ),
                "emergency": signature in EMERGENCY_OPTIONS,
                "actionable": signature not in NON_ACTIONABLE_OPTIONS,
                "qualified": (
                    signature not in NON_ACTIONABLE_OPTIONS
                    and samples >= min_samples
                    and len(successes) / max(1, samples) >= threshold
                    and lower >= threshold
                ),
                "rows": option_rows,
                "successfulRows": successes,
                "winningRows": wins,
            }
        )
    return sorted(
        options,
        key=lambda item: (
            bool(item["qualified"]),
            int(item["successfulSamples"]),
            float(item["successfulPolicyShare"]),
            float(item["avoidWilsonLower"]),
            int(item["samples"]),
        ),
        reverse=True,
    )


def representative_lane(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot choose a representative lane without rows")
    detailed = Counter(str(row.get("optionSignature") or "") for row in rows)
    common_signature, _ = detailed.most_common(1)[0]
    pool = [row for row in rows if str(row.get("optionSignature") or "") == common_signature]
    target_damage = median(float(row.get("damageDealt") or 0.0) for row in pool)
    target_resolution = median(int((row.get("resolution") or {}).get("step") or 0) for row in pool)
    return min(
        pool,
        key=lambda row: (
            abs(float(row.get("damageDealt") or 0.0) - target_damage),
            abs(int((row.get("resolution") or {}).get("step") or 0) - target_resolution),
            int(row.get("laneId") or 0),
        ),
    )


def route_option(
    rows: list[dict[str, Any]],
    *,
    deadline: int,
    threshold: float,
    min_samples: int,
    decision_frame: int,
    z: float,
    objective: str = "avoid",
) -> dict[str, Any] | None:
    """Choose one transparent representative option for a particular lead time.

    This deliberately does not filter emergency actions. The UI is an explorer:
    late insertion often leaves only a roll or air dodge, and that is useful evidence.
    """
    options = option_statistics(
        rows,
        deadline=deadline,
        threshold=threshold,
        min_samples=min_samples,
        decision_frame=decision_frame,
        z=z,
    )
    actionable = [item for item in options if item["actionable"]]
    if objective == "win":
        actionable = [item for item in actionable if item["winningRows"]]
    if not actionable:
        return None
    if objective == "win":
        return max(
            actionable,
            key=lambda item: (
                float(item["neutralWinRate"]),
                float(item["avoidRate"]),
                float(item["cleanEscapeRate"]),
                not bool(item["emergency"]),
                int(item["neutralWinSamples"]),
                int(item["samples"]),
            ),
        )
    return max(
        actionable,
        key=lambda item: (
            float(item["avoidRate"]),
            float(item["cleanEscapeRate"]),
            float(item["stableNeutralRate"]),
            float(item["winFirstHitRate"]),
            not bool(item["emergency"]),
            int(item["successfulSamples"]),
            int(item["samples"]),
        ),
    )


def trim_lane_stream(lane: dict[str, Any], *, end_frame: int) -> dict[str, Any]:
    lane = dict(lane)
    stream = dict(lane.get("controllerStream") or {})
    if not stream:
        raise ValueError("representative lane is missing its controller stream")
    takeover = int(stream.get("takeoverFrame") or lane.get("takeoverFrame") or 0)
    frame_count = max(1, int(end_frame) - takeover + 1)
    stream["frames"] = min(frame_count, int(stream.get("frames") or frame_count))
    stream["endFrame"] = takeover + int(stream["frames"]) - 1
    lane["controllerStream"] = stream
    return lane


def build_neutral_queue(
    *,
    experiment: dict[str, Any],
    selected_events: list[dict[str, Any]],
    refinement_rows: list[dict[str, Any]],
    refinement_dir: Path,
    route_grid_rows: list[dict[str, Any]] | None = None,
    threshold: float,
    min_option_samples: int = 8,
    min_option_share: float = 0.10,
    z: float = 1.645,
    video_preroll_frames: int = 60,
) -> dict[str, Any]:
    rows_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in refinement_rows:
        rows_by_frame[int(row["baseFrame"])].append(row)

    grid_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in route_grid_rows or []:
        grid_by_frame[int(row["baseFrame"])].append(row)

    targets = []
    omitted = []
    for event in selected_events:
        boundary = event["boundaryPoint"]
        injection_frame = int(boundary["injectionFrame"])
        deadline = int(boundary["deadlineFrame"])
        rows = rows_by_frame.get(injection_frame) or []
        successes = [row for row in rows if avoided_by_deadline(row, deadline)]
        lower = wilson_lower(len(successes), len(rows), z=z)
        if not rows or len(successes) / max(1, len(rows)) < threshold or lower < threshold:
            omitted.append(
                {
                    "eventId": event["eventId"],
                    "openingFrame": event["openingFrame"],
                    "injectionFrame": injection_frame,
                    "reason": "refinement_did_not_confirm_boundary",
                    "samples": len(rows),
                    "avoidRate": round(len(successes) / max(1, len(rows)), 4),
                    "avoidWilsonLower": round(lower, 4),
                }
            )
            continue
        options = option_statistics(
            rows,
            deadline=deadline,
            decision_frame=int(event["openingFrame"]),
            threshold=threshold,
            min_samples=min_option_samples,
            min_share=min_option_share,
            z=z,
        )
        qualified_options = [option for option in options if option["qualified"]]
        chosen = qualified_options[0] if qualified_options else None
        if chosen is None:
            chosen = next(
                (option for option in options if option["actionable"] and option["successfulRows"]),
                None,
            )
            qualified_options = [chosen] if chosen is not None else []
        if chosen is None:
            omitted.append(
                {
                    "eventId": event["eventId"],
                    "openingFrame": event["openingFrame"],
                    "injectionFrame": injection_frame,
                    "reason": "no_actionable_successful_option",
                }
            )
            continue
        lane = trim_lane_stream(representative_lane(chosen["successfulRows"]), end_frame=deadline)
        public_option = {key: value for key, value in chosen.items() if key not in {"rows", "successfulRows", "winningRows"}}
        public_option["selectionRank"] = len(targets) + 1
        alternative_routes = []
        for route_rank, alternative in enumerate(qualified_options[1:], start=2):
            alternative_lane = trim_lane_stream(
                representative_lane(alternative["successfulRows"]),
                end_frame=deadline,
            )
            alternative_option = {
                key: value for key, value in alternative.items() if key not in {"rows", "successfulRows", "winningRows"}
            }
            alternative_inference_start = int(alternative_lane.get("takeoverFrame") or injection_frame)
            alternative_control = model_control_frame(alternative_lane, injection_frame)
            alternative_routes.append(
                {
                    "route_rank": route_rank,
                    "offset": 0,
                    "inference_start_frame": alternative_inference_start,
                    "branch_start_frame": alternative_inference_start,
                    "takeover_frame": alternative_control,
                    "option": alternative_option,
                    "representative_lane": alternative_lane,
                }
            )
        inference_start = int(lane.get("takeoverFrame") or injection_frame)
        control_frame = model_control_frame(lane, injection_frame)
        target = {
                "queue_index": len(targets) + 1,
                "base_frame": injection_frame,
                "inference_start_frame": inference_start,
                "takeover_frame": control_frame,
                "branch_start_frame": inference_start,
                "label": (
                    f"Avoid {event['openingMove']} at f{event['openingFrame']} from "
                    f"{int(boundary['lookbackFrames'])}f earlier with {chosen['optionSignature']}"
                ),
                "option": public_option,
                "representative_lane": lane,
                "alternative_routes": alternative_routes,
                "replay_baseline": {
                    "opening_move": {
                        "frame": int(event["openingFrame"]),
                        "moveShortName": event["openingMove"],
                    },
                    "opening_type": "neutral-loss",
                    "opportunity_role": "neutral_loss_avoidance",
                    "original_followup_damage": 0.0,
                    "original_followup_hits": 0,
                    "original_followup_kill": False,
                },
                "neutral_loss": {
                    "eventId": event["eventId"],
                    "openingFrame": int(event["openingFrame"]),
                    "openingMove": event["openingMove"],
                    "injectionFrame": injection_frame,
                    "lookbackFrames": int(boundary["lookbackFrames"]),
                    "deadlineFrame": deadline,
                    "samples": len(rows),
                    "successfulSamples": len(successes),
                    "avoidRate": round(len(successes) / len(rows), 4),
                    "avoidWilsonLower": round(lower, 4),
                    "replayInputBaselineReproduced": bool(boundary.get("baselineReproduced")),
                    "qualifiedOptionCount": len(qualified_options),
                    "optionDecisionDeadlineFrame": int(event["openingFrame"]),
                },
        }

        # The verified boundary remains useful context, but expose the best model
        # route at every requested insertion time. This makes late panic escapes
        # visible rather than presenting them as a universal recommendation.
        if grid_by_frame:
            lead_routes = []
            for lookback in range(10, int(experiment["maxLookbackFrames"]) + 1, 10):
                route_frame = int(event["openingFrame"]) - lookback
                route_rows = grid_by_frame.get(route_frame) or []
                if not route_rows:
                    continue
                for route_kind in ("avoid", "win"):
                    option = route_option(
                        route_rows,
                        deadline=deadline,
                        threshold=threshold,
                        min_samples=min_option_samples,
                        decision_frame=int(event["openingFrame"]),
                        z=z,
                        objective=route_kind,
                    )
                    if option is None:
                        continue
                    preferred_rows = (
                        option["winningRows"] if route_kind == "win" else option["successfulRows"]
                    ) or option["rows"]
                    route_lane = trim_lane_stream(representative_lane(preferred_rows), end_frame=max(
                        int(row.get("controllerStream", {}).get("endFrame") or deadline) for row in preferred_rows
                    ))
                    public_route_option = {
                        key: value for key, value in option.items()
                        if key not in {"rows", "successfulRows", "winningRows"}
                    }
                    route_inference_start = int(route_lane.get("takeoverFrame") or route_frame)
                    route_control = model_control_frame(route_lane, route_frame)
                    lead_routes.append({
                        "route_rank": len(lead_routes) + 1,
                        "route_kind": route_kind,
                        "offset": 0,
                        "inference_start_frame": route_inference_start,
                        "branch_start_frame": route_inference_start,
                        "takeover_frame": route_control,
                        "lookback_frames": lookback,
                        "injection_frame": route_frame,
                        "deadline_frame": deadline,
                        "option": public_route_option,
                        "representative_lane": route_lane,
                    })
            if lead_routes:
                primary_route = next((route for route in lead_routes if route["route_kind"] == "avoid"), lead_routes[0])
                target.update({
                    "base_frame": primary_route["injection_frame"],
                    "takeover_frame": primary_route["takeover_frame"],
                    "branch_start_frame": primary_route["branch_start_frame"],
                    "option": primary_route["option"],
                    "representative_lane": primary_route["representative_lane"],
                    "alternative_routes": [route for route in lead_routes if route is not primary_route],
                })
                target["neutral_loss"]["insertionExplorer"] = True
                target["neutral_loss"]["selectedRouteLookbackFrames"] = primary_route["lookback_frames"]
                target["neutral_loss"]["availableLookbackFrames"] = sorted({route["lookback_frames"] for route in lead_routes})
        targets.append(target)
    return {
        "version": 2,
        "mode": "neutral-loss-avoidance",
        "source_run_dir": str(refinement_dir.resolve()),
        "lanes_jsonl": str((refinement_dir / "lanes.jsonl").resolve()),
        "replay": experiment["replay"],
        "controlled_port": int(experiment["analyzedPort"]),
        "rollout_frames": int(experiment["maxLookbackFrames"]) + int(experiment["postOpeningFrames"]) + 1,
        "video_preroll_frames": int(video_preroll_frames),
        "selection": {
            "consistencyThreshold": threshold,
            "confidenceZ": z,
            "minOptionSamples": min_option_samples,
            "minOptionShare": min_option_share,
        },
        "selection_audit": {
            "neutralLossCount": sum(event.get("kind") == "avoid" for event in experiment.get("events") or []),
            "preflightSelectedCount": len(selected_events),
            "finalSelectedCount": len(targets),
            "omitted": omitted,
        },
        "targets": targets,
    }
