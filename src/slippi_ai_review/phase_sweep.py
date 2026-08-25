"""Prepare one low-cost Phillip probe per playable phase-timeline segment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PLAYABLE_PHASES = frozenset({"advantage", "neutral", "one_hit_kill"})


def player_index_for_port(settings: dict[str, Any], port: int) -> int:
    for player in settings.get("players") or []:
        if int(player.get("port") or -1) == port:
            return int(player.get("playerIndex") or 0)
    raise ValueError(f"No player on port {port}")


def review_phase(segment: dict[str, Any], analyzed_index: int) -> str | None:
    phase = str(segment.get("phase") or "")
    if phase not in PLAYABLE_PHASES:
        return None
    if phase == "neutral":
        return "neutral"
    owner = segment.get("owner")
    if owner is None:
        return None
    return "advantage" if int(owner) == analyzed_index else "disadvantage"


def injection_frame(segment: dict[str, Any], phase: str, first_frame: int) -> int:
    start = max(0, int(first_frame), int(segment.get("startFrame") or first_frame))
    end = int(segment.get("endFrame") or start)
    opening = int(segment.get("openingFrame") or start)
    # Neutral benefits from a few live frames of context. Defensive and advantage
    # probes begin at the state that actually begins the phase.
    return min(end, start + 12) if phase == "neutral" else max(start, opening)


def game_metadata(settings: dict[str, Any]) -> dict[str, Any]:
    """Keep the game identity once, outside the per-slide event labels."""
    return {
        "stageName": settings.get("stageName"),
        "players": [
            {
                "port": player.get("port"),
                "playerIndex": player.get("playerIndex"),
                "displayName": player.get("displayName") or player.get("connectCode") or f"P{player.get('port')}",
                "connectCode": player.get("connectCode"),
                "characterName": player.get("characterName"),
            }
            for player in settings.get("players") or []
        ],
    }


def matching_opportunity(segment: dict[str, Any], opportunities: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the opener that was live at the beginning of this timeline segment."""
    owner = segment.get("owner")
    if owner is None:
        return None
    start = int(segment.get("startFrame") or 0)
    containing = [
        item for item in opportunities
        if item.get("owner") == owner
        and int(item.get("startFrame") or 0) <= start <= int(item.get("endFrame") or start)
    ]
    return max(containing, key=lambda item: int(item.get("startFrame") or 0)) if containing else None


def segment_presentation(segment: dict[str, Any], opportunities: list[dict[str, Any]]) -> tuple[str, str]:
    """A short player-facing name, with technical phase context kept secondary."""
    if segment.get("phase") == "neutral":
        return "Neutral scramble", "Find the commitment that decided the exchange"

    opportunity = matching_opportunity(segment, opportunities)
    if not opportunity:
        return ("One-hit stock" if segment.get("phase") == "one_hit_kill" else "Advantage sequence"), "Opening sequence"

    sequence = opportunity.get("moveSequence") or []
    move = str((sequence[0] if sequence else {}).get("moveShortName") or "opening").replace("-", " ").title()
    opening = str(opportunity.get("openingType") or "opening").replace("-", " ")
    if bool(opportunity.get("didKill") or opportunity.get("oneHitKill")):
        outcome = "stock"
    else:
        hits = int(opportunity.get("moves") or len(sequence) or 1)
        damage = float(opportunity.get("damage") or 0)
        outcome = f"{hits} hit{'s' if hits != 1 else ''} · +{damage:.0f}%"
    return f"{move} {opening} · {outcome}", opening.title()


def _original_outcome(
    timeline_segments: list[dict[str, Any]],
    timeline_index: int,
    analyzed_index: int,
    opportunities: list[dict[str, Any]],
) -> dict[str, Any]:
    source = timeline_segments[timeline_index]
    candidate = source
    phase = review_phase(source, analyzed_index)
    if phase == "neutral":
        candidate = next(
            (
                item for item in timeline_segments[timeline_index + 1 :]
                if review_phase(item, analyzed_index) in {"advantage", "disadvantage"}
            ),
            source,
        )
        phase = review_phase(candidate, analyzed_index)
    opportunity = matching_opportunity(candidate, opportunities)
    sequence = (opportunity or {}).get("moveSequence") or []
    return {
        "phase": phase,
        "damage": float((opportunity or {}).get("damage") or 0.0),
        "hits": int((opportunity or {}).get("moves") or len((opportunity or {}).get("moveSequence") or [])),
        "killed": bool((opportunity or {}).get("didKill") or (opportunity or {}).get("oneHitKill")),
        "openingFrame": (opportunity or {}).get("startFrame"),
        "openingMove": (sequence[0] if sequence else {}).get("moveShortName"),
    }


def _priority(segment: dict[str, Any]) -> float:
    outcome = segment.get("originalOutcome") or {}
    phase = str(segment.get("phase") or "")
    damage = float(outcome.get("damage") or 0.0)
    killed = bool(outcome.get("killed"))
    duration = min(240, int(segment.get("endFrame") or 0) - int(segment.get("startFrame") or 0))
    if phase == "neutral":
        return (160 if outcome.get("phase") == "disadvantage" else 20) + damage + (60 if killed else 0) + duration / 60
    if phase == "disadvantage":
        return 120 + damage + (60 if killed else 0) + duration / 120
    # Successful one-hit stocks are useful game events but poor practice probes.
    return (20 if killed else 100) + max(0.0, 22 - damage) + duration / 120


def limit_segments(segments: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if maximum <= 0 or len(segments) <= maximum:
        return segments
    neutral_quota = max(1, math.ceil(maximum * 0.5))
    remaining = maximum - neutral_quota
    quotas = {
        "neutral": neutral_quota,
        "advantage": max(1, math.ceil(remaining / 2)),
        "disadvantage": max(1, remaining // 2),
    }
    selected: list[dict[str, Any]] = []
    for phase, quota in quotas.items():
        candidates = sorted(
            (segment for segment in segments if segment["phase"] == phase),
            key=lambda segment: (-_priority(segment), int(segment["timelineIndex"])),
        )
        selected.extend(candidates[:quota])
    if len(selected) < maximum:
        selected_ids = {id(segment) for segment in selected}
        remainder = sorted(
            (segment for segment in segments if id(segment) not in selected_ids),
            key=lambda segment: (-_priority(segment), int(segment["timelineIndex"])),
        )
        selected.extend(remainder[: maximum - len(selected)])
    selected = sorted(selected[:maximum], key=lambda segment: int(segment["timelineIndex"]))
    for index, segment in enumerate(selected, 1):
        segment["id"] = f"{segment['phase']}_{index:03d}_f{segment['injectionFrame']}"
    return selected


def build_inventory(timeline: dict[str, Any], *, analyzed_port: int, max_segments: int = 0) -> dict[str, Any]:
    settings = timeline.get("settings") or {}
    analyzed_index = player_index_for_port(settings, analyzed_port)
    first_frame = int(timeline.get("firstFrame") or -123)
    opportunities = timeline.get("advantageOpportunities") or []
    timeline_segments = timeline.get("segments") or []
    segments = []
    for timeline_index, segment in enumerate(timeline_segments):
        phase = review_phase(segment, analyzed_index)
        if phase is None:
            continue
        start, end = int(segment.get("startFrame") or first_frame), int(segment.get("endFrame") or first_frame)
        if end <= start or end - start < 8:
            continue
        frame = injection_frame(segment, phase, first_frame)
        title, subtitle = segment_presentation(segment, opportunities)
        segments.append({
            "id": f"{phase}_{len(segments) + 1:03d}_f{frame}",
            "timelineIndex": timeline_index,
            "phase": phase,
            "owner": segment.get("owner"),
            "startFrame": start,
            "endFrame": end,
            "injectionFrame": frame,
            "label": segment.get("label") or f"{phase} f{start}-{end}",
            "title": title,
            "subtitle": subtitle,
            "openingFrame": segment.get("openingFrame"),
            "notes": segment.get("notes") or [],
            "originalOutcome": _original_outcome(
                timeline_segments,
                timeline_index,
                analyzed_index,
                opportunities,
            ),
        })
    source_count = len(segments)
    segments = limit_segments(segments, max_segments)
    points = [
        {
            "frame": item["injectionFrame"],
            "opportunity_role": f"phase_sweep_{item['phase']}",
            "point_evidence": {
                "sweep_id": item["id"],
                "phase": item["phase"],
                "segment_start_frame": item["startFrame"],
                "segment_end_frame": item["endFrame"],
            },
        }
        for item in segments
    ]
    return {
        "version": 1,
        "replay": timeline.get("replay"),
        "analyzedPort": analyzed_port,
        "analyzedPlayerIndex": analyzed_index,
        "game": game_metadata(settings),
        "segments": segments,
        # Keep the simulation-compatible shape at top level. The segment list
        # remains the presentation identity used later by the slide deck.
        "frames": points,
        "counts": {phase: sum(item["phase"] == phase for item in segments) for phase in ("advantage", "neutral", "disadvantage")},
        "selection": {"sourceSegments": source_count, "selectedSegments": len(segments), "maximum": max_segments},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--analyzed-port", type=int, choices=(1, 2), required=True)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_inventory(
        json.loads(args.timeline.resolve().read_text(encoding="utf-8")),
        analyzed_port=args.analyzed_port,
        max_segments=args.max_segments,
    )
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "counts": payload["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
