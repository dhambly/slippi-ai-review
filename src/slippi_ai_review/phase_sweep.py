"""Prepare one low-cost Phillip probe per playable phase-timeline segment."""

from __future__ import annotations

import argparse
import json
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


def build_inventory(timeline: dict[str, Any], *, analyzed_port: int) -> dict[str, Any]:
    settings = timeline.get("settings") or {}
    analyzed_index = player_index_for_port(settings, analyzed_port)
    first_frame = int(timeline.get("firstFrame") or -123)
    opportunities = timeline.get("advantageOpportunities") or []
    segments = []
    for timeline_index, segment in enumerate(timeline.get("segments") or []):
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
        })
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--analyzed-port", type=int, choices=(1, 2), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_inventory(
        json.loads(args.timeline.resolve().read_text(encoding="utf-8")),
        analyzed_port=args.analyzed_port,
    )
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "counts": payload["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
