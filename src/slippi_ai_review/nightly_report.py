"""Aggregate whole-session Phillip sweeps into a compact practice report."""

from __future__ import annotations

import html
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable
from urllib.parse import urlencode


PHASE_LABELS = {
    "neutral": "Neutral",
    "advantage": "Punish",
    "disadvantage": "Defense",
}
DIRECT_RESPONSE_MAX_FRAMES = 30
CHECKMATE_MIN_SAMPLES = 4
CHECKMATE_LARGE_SAMPLE_THRESHOLD = 0.875


@dataclass
class Evidence:
    review_id: str
    target_index: int
    phase: str
    frame: int
    elapsed: str
    game_name: str
    matchup: str
    stage: str
    replay_action: str
    opponent_action: str
    phillip_action: str
    original_result: str
    phillip_result: str
    improvement: float
    option_samples: int
    sweep_samples: int
    option_share: float
    reversal_rate: float
    self_death_rate: float
    favorable_samples: int
    favorable_rate: float
    source_url: str
    viewer_url: str | None


@dataclass
class Pattern:
    key: str
    phase: str
    title: str
    matchup: str
    occurrences: int
    games: int
    confidence: str
    summary: str
    drill: str
    evidence: list[Evidence]
    score: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _elapsed(frame: int) -> str:
    seconds = max(0, frame) / 60
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def _humanize(value: Any) -> str:
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip().lower()


def _move_label(value: Any, character: str = "") -> str:
    move = _humanize(str(value or "").split(">")[0])
    if move.startswith("aerial "):
        move = move.removeprefix("aerial ")
    aliases = {
        "neutral attack 1": "jab",
        "attack air n": "nair",
        "attack air f": "fair",
        "attack air b": "bair",
        "attack air hi": "uair",
        "attack air lw": "dair",
        "throw up": "up throw",
        "throw down": "down throw",
        "throw forward": "forward throw",
        "throw back": "back throw",
        "lying ground up": "getup option",
        "forward tech": "tech forward",
        "backward tech": "tech back",
        "edge catching": "grab ledge",
        "no commitment": "drift / no immediate button",
        "tech miss up": "missed tech",
        "tech miss down": "missed tech",
        "ground roll backward up": "roll backward after missed tech",
        "ground roll backward down": "roll backward after missed tech",
        "ground roll forward up": "roll forward after missed tech",
        "ground roll forward down": "roll forward after missed tech",
        "edge hanging": "hold ledge",
        "grab pummel": "pummel",
        "shine turn": "shine",
        "uthrow": "up throw",
        "dthrow": "down throw",
        "fthrow": "forward throw",
        "bthrow": "back throw",
    }
    move = aliases.get(move, move)
    if move.startswith("down b") and character in {"Fox", "Falco"}:
        return "shine"
    if move.startswith("sword dance") and character in {"Fox", "Falco"}:
        return "side b"
    return move or "no clear commitment"


def _move_key(value: str) -> str:
    return (
        value.replace("forward", "f")
        .replace("back", "b")
        .replace("down", "d")
        .replace("up", "u")
        .replace(" ", "")
    )


def _phillip_action(target: dict[str, Any], character: str, opening: str = "") -> str:
    """Describe the first meaningful model choice, skipping forced hit/knockdown states."""
    lane = target.get("representative_lane") or {}
    signature = lane.get("comboOptionSignature") or ""
    segments = ((lane.get("option") or {}).get("actionSegments") or [])
    names = [str(segment.get("actionName") or "") for segment in segments]
    if opening:
        opening_key = _move_key(opening)
        saw_opening = False
        pummels = False
        passive = (
            "LANDING", "STANDING", "WALK", "DASH", "RUNNING", "TURNING", "CROUCH", "KNEE_BEND",
            "JUMPING", "FALL", "GRAB_WAIT", "GRAB_PULL", "GRABBED", "DAMAGE", "THROWN", "SHIELD_STUN",
        )
        for name in names:
            if name == "GRAB_PUMMEL":
                pummels = True
                continue
            label = _move_label(name, character)
            if _move_key(label) == opening_key:
                saw_opening = True
                continue
            if "LANDING" in name or any(name.startswith(prefix) for prefix in passive):
                continue
            if saw_opening or _move_key(label) != opening_key:
                return label
        if pummels:
            return f"extra pummels before {opening}"
    if str(signature).startswith("TECH_MISS"):
        roll = next((name for name in names if name.startswith("GROUND_ROLL_")), "")
        return _move_label(roll or signature, character)
    if str(signature).startswith("THROWN_"):
        response = next(
            (
                name for name in names
                if "TECH" in name or name.startswith("GROUND_ROLL_") or name in {"AIRDODGE", "JUMPING_ARIAL_FORWARD", "JUMPING_ARIAL_BACKWARD"}
            ),
            "",
        )
        return _move_label(response, character) if response else "DI through the throw"
    return _move_label(signature, character)


def _action_group(action: str) -> str:
    value = action.split(" (", 1)[0]
    if value in {"nair", "fair", "bair", "uair", "dair", "aerial landing"}:
        return "aerial commitment"
    if value in {"grab", "dash grab"} or "throw" in value:
        return "grab commitment"
    if value in {"shield timing", "spotdodge", "airdodge"} or "roll" in value or "tech" in value:
        return "defensive timing"
    if value in {"dash movement", "jump", "movement"}:
        return "movement"
    if value == "stood still":
        return "no action"
    return "grounded attack commitment"


def _segment_result(segment: dict[str, Any]) -> tuple[float, int, bool]:
    text = f"{segment.get('title') or ''} {segment.get('label') or ''}"
    damages = re.findall(r"\+(\d+(?:\.\d+)?)%", text)
    hits = re.findall(r"(\d+)\s+hits?", text, flags=re.IGNORECASE)
    damage = float(damages[0]) if damages else 0.0
    hit_count = int(hits[0]) if hits else 0
    killed = bool(re.search(r"\bstock\b|one.hit.kill", text, flags=re.IGNORECASE))
    return damage, hit_count, killed


def _matchup(queue: dict[str, Any], controlled_port: int) -> tuple[str, str, str, int]:
    players = (queue.get("game") or {}).get("players") or []
    target = next((p for p in players if int(p.get("port") or 0) == controlled_port), {})
    opponent = next((p for p in players if int(p.get("port") or 0) != controlled_port), {})
    target_character = str(target.get("characterName") or "Unknown")
    opponent_character = str(opponent.get("characterName") or "Unknown")
    target_index = int(target.get("playerIndex") or max(0, controlled_port - 1))
    return f"{target_character} vs {opponent_character}", target_character, opponent_character, target_index


def _last_commitment(segment: dict[str, Any], player_index: int, character: str) -> tuple[str, str]:
    runs = (((segment.get("actionTimeline") or {}).get("runsByPlayer") or {}).get(str(player_index)) or [])
    meaningful: list[tuple[int, int, str, str]] = []
    keywords = (
        "nair", "fair", "bair", "uair", "dair", "jab", "tilt", "smash", "attack", "grab", "throw",
        "shield", "roll", "spotdodge", "airdodge", "dash", "run", "jump", "knee bend", "shine", "down b",
        "side b", "up b", "neutral b", "landing", "turn", "wait", "crouch",
    )
    ignored = ("damage", "hitstun", "dead", "rebirth", "capture", "thrown", "sleep")
    for run in runs:
        name = _humanize(run.get("actionStateName"))
        if not name or any(token in name for token in ignored) or not any(token in name for token in keywords):
            continue
        end = int(run.get("endFrame") or run.get("startFrame") or 0)
        start = int(run.get("startFrame") or end)
        side = str(((run.get("end") or {}).get("stageSide") or "")).strip()
        meaningful.append((end, start, name, side))
    if not meaningful:
        return "movement", ""
    _end, _start, action, side = max(meaningful)
    if "landing" in action:
        action = "aerial landing"
    elif "knee bend" in action or "jump" in action:
        action = "jump"
    elif action == "wait":
        action = "stood still"
    elif "dash" in action or "run" in action or action == "turn":
        action = "dash movement"
    elif "shield" in action:
        action = "shield timing"
    return _move_label(action, character), side


def _interactive_url(review_id: str, interactive: dict[str, Any], queue: dict[str, Any]) -> str | None:
    if not interactive:
        return None
    players = sorted((queue.get("game") or {}).get("players") or [], key=lambda p: int(p.get("port") or 99))
    names = {
        f"p{int(player.get('port') or 0)}": str(player.get("displayName") or player.get("connectCode") or "").strip()
        for player in players
        if int(player.get("port") or 0) in (1, 2)
    }
    params = {
        "replay": interactive.get("replay_trace"),
        "agent": interactive.get("agent_trace"),
        "switch": int(interactive.get("switch_frame") or 0),
        "takeover": int(interactive.get("model_control_frame") or interactive.get("switch_frame") or 0),
        "defenderSwitch": int(interactive.get("defender_switch_frame") or 0),
        "start": int(interactive.get("start_frame") or 0),
        "frames": int(interactive.get("frame_count") or 1),
        **names,
    }
    if interactive.get("timeline_events"):
        params["events"] = interactive["timeline_events"]
    return f"/review-artifacts/{review_id}/viewer/compare.html?{urlencode(params)}"


def _option_rows(rows: Iterable[dict[str, Any]], frame: int, signature: str) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if int(row.get("baseFrame") or -1) == frame
        and str(row.get("comboOptionSignature") or str(row.get("optionSignature") or "").split(">", 1)[0]) == signature
    ]


def _is_direct_response(frame: int, outcome: dict[str, Any]) -> bool:
    opening = outcome.get("openingFrame")
    if opening is None:
        return False
    delta = frame - int(opening)
    return 0 <= delta <= DIRECT_RESPONSE_MAX_FRAMES


def _lane_is_favorable(
    phase: str,
    row: dict[str, Any],
    *,
    original_damage: float,
    original_kill: bool,
) -> bool:
    if int(row.get("analyzedStocksLost") or 0) > 0 or bool(row.get("comboReversed")):
        return False
    reason = str((row.get("resolution") or {}).get("reason") or "")
    if reason.startswith("timeout_unresolved"):
        return False
    dealt = float(row.get("damageDealt") or 0.0)
    taken = float(row.get("damageTaken") or 0.0)
    if phase == "disadvantage":
        if original_kill:
            return taken < original_damage
        return taken <= max(0.0, original_damage - 4.0)
    if phase == "neutral":
        return float(row.get("score") or 0.0) > 0.0 and dealt > taken
    return int(row.get("defenderStocksLost") or 0) > 0 or dealt >= original_damage + 4.0


def _is_checkmate_candidate(evidence: Evidence) -> bool:
    if evidence.option_samples < CHECKMATE_MIN_SAMPLES:
        return False
    if evidence.option_samples < 8:
        return evidence.favorable_samples == evidence.option_samples
    return evidence.favorable_rate >= CHECKMATE_LARGE_SAMPLE_THRESHOLD


def collect_review_evidence(review_dir: Path) -> list[tuple[str, Evidence]]:
    review = _read_json(review_dir / "review.json")
    review_id = str(review["reviewId"])
    pipeline = review_dir / "pipeline"
    queue = _read_json(pipeline / "phase_sweep_queue.json")
    timeline = _read_json(pipeline / "candidates_phase_timeline.json")
    manifest = _read_json(pipeline / "phase_sweep_artifacts" / "final_artifacts" / "advantage_improvements.json")
    rows = _read_jsonl(pipeline / "phase_sweep" / "lanes.jsonl")
    interactive_by_index = {
        int(item.get("target_index") or 0): item.get("interactive") or {}
        for item in manifest.get("results") or []
    }
    rich_segments = list(timeline.get("segments") or [])
    rich_by_bounds = {
        (int(segment.get("startFrame") or 0), int(segment.get("endFrame") or 0)): segment
        for segment in rich_segments
    }
    controlled_port = int(queue.get("controlled_port") or 1)
    matchup, character, opponent_character, player_index = _matchup(queue, controlled_port)
    stage = str((queue.get("game") or {}).get("stageName") or "Unknown stage")
    game_name = str(review.get("originalFilename") or review_id)
    evidence: list[tuple[str, Evidence]] = []

    for target in queue.get("targets") or []:
        phase = str(target.get("phase") or "")
        if phase not in PHASE_LABELS:
            continue
        frame = int(target.get("base_frame") or 0)
        index = int(target.get("queue_index") or 0)
        segment = target.get("phase_segment") or {}
        option = target.get("option") or {}
        signature = str(option.get("optionSignature") or "")
        selected_rows = _option_rows(rows, frame, signature)
        samples = int(option.get("samples") or len(selected_rows))
        sweep_samples = int(option.get("sweepSamples") or 0)
        share = float(option.get("optionShare") or 0.0)
        reversal_rate = float(option.get("reversalRate") or 0.0)
        self_death_rate = mean(float(int(row.get("analyzedStocksLost") or 0) > 0) for row in selected_rows) if selected_rows else 0.0
        unresolved_rate = (
            mean(float(str((row.get("resolution") or {}).get("reason") or "") == "timeout_unresolved_defense") for row in selected_rows)
            if selected_rows else 1.0
        )
        representative_reason = str(
            (((target.get("representative_lane") or {}).get("resolution") or {}).get("reason") or "")
        )
        minimum_samples = max(2, math.ceil(max(1, sweep_samples) * 0.15))
        if (
            len(selected_rows) != samples
            or samples < minimum_samples
            or share < 0.15
            or reversal_rate > 0.35
            or self_death_rate > 0.20
            or (
                phase == "disadvantage"
                and (unresolved_rate > 0.50 or representative_reason == "timeout_unresolved_defense")
            )
        ):
            continue

        outcome = segment.get("originalOutcome") or {}
        fallback_damage, fallback_hits, fallback_kill = _segment_result(segment)
        original_damage = float(outcome.get("damage") or fallback_damage)
        original_hits = int(outcome.get("hits") or fallback_hits)
        original_kill = bool(outcome.get("killed") or fallback_kill)
        model_dealt = float(option.get("damageDealt") or 0.0)
        model_taken = float(option.get("damageTaken") or 0.0)
        model_kill = float(option.get("killRate") or 0.0)
        replay_action = _move_label(
            outcome.get("openingMove") or str(segment.get("title") or "").split()[0],
            character,
        )
        phillip_action = _phillip_action(target, character, replay_action if phase == "advantage" else "")
        opponent_action = replay_action
        position = ""

        if phase == "advantage":
            if original_kill:
                continue
            improvement = model_dealt - original_damage
            if model_kill < 0.20 and improvement < 6.0:
                continue
            key = f"advantage|{matchup}|{replay_action}"
            original_result = f"{original_hits or 1} hit, {original_damage:.0f}% before the sequence ended"
            phillip_result = f"{model_dealt:.0f}% average" + (f", {model_kill:.0%} kill rate" if model_kill else "")
        elif phase == "disadvantage":
            if not _is_direct_response(frame, outcome):
                continue
            improvement = original_damage - model_taken + model_dealt * 0.25
            if not original_kill and improvement < 6.0:
                continue
            key = f"disadvantage|{matchup}|{opponent_action}"
            original_result = "lost the stock" if original_kill else f"took {original_damage:.0f}%"
            phillip_result = f"took {model_taken:.0f}% and dealt {model_dealt:.0f}%"
            replay_action = "defense after hit"
        else:
            if str(outcome.get("phase") or "") != "disadvantage":
                continue
            opening_frame = int(outcome.get("openingFrame") or frame)
            if opening_frame - frame > 45 or (model_kill >= 0.5 and model_dealt < 1.0):
                continue
            rich = rich_by_bounds.get((int(segment.get("startFrame") or 0), int(segment.get("endFrame") or 0)), {})
            replay_action, position = _last_commitment(rich, player_index, character)
            opponent_action = _move_label(outcome.get("openingMove") or "opening hit", opponent_character)
            improvement = original_damage - model_taken + model_dealt * 0.35
            if improvement < 4.0 or float(option.get("meanScore") or 0.0) <= 0:
                continue
            key = f"neutral|{matchup}|{_action_group(replay_action)} against {opponent_action}"
            original_result = ("lost the stock" if original_kill else f"got opened by {opponent_action} for {original_damage:.0f}%")
            phillip_result = f"dealt {model_dealt:.0f}% and took {model_taken:.0f}%"

        if position:
            replay_action = f"{replay_action} ({position})"
        favorable_samples = sum(
            _lane_is_favorable(
                phase,
                row,
                original_damage=original_damage,
                original_kill=original_kill,
            )
            for row in selected_rows
        )
        source_url = f"/review-artifacts/{review_id}/{phase}_review.html#slide-{index}"
        viewer_url = _interactive_url(review_id, interactive_by_index.get(index) or {}, queue)
        evidence.append((key, Evidence(
            review_id=review_id,
            target_index=index,
            phase=phase,
            frame=frame,
            elapsed=_elapsed(frame),
            game_name=game_name,
            matchup=matchup,
            stage=stage,
            replay_action=replay_action,
            opponent_action=opponent_action,
            phillip_action=phillip_action,
            original_result=original_result,
            phillip_result=phillip_result,
            improvement=round(improvement, 2),
            option_samples=samples,
            sweep_samples=sweep_samples,
            option_share=share,
            reversal_rate=reversal_rate,
            self_death_rate=self_death_rate,
            favorable_samples=favorable_samples,
            favorable_rate=round(favorable_samples / max(1, samples), 4),
            source_url=source_url,
            viewer_url=viewer_url,
        )))
    return evidence


def _pattern_copy(phase: str, action: str, items: list[Evidence]) -> tuple[str, str, str]:
    phillip_actions = defaultdict(int)
    opponent_actions = defaultdict(int)
    for item in items:
        phillip_actions[item.phillip_action] += 1
        opponent_actions[item.opponent_action] += 1
    ranked_phillip = sorted(phillip_actions, key=lambda value: (-phillip_actions[value], value))
    if len(ranked_phillip) <= 2:
        phillip_clause = f"used {' or '.join(ranked_phillip)}"
    elif phillip_actions[ranked_phillip[0]] > phillip_actions[ranked_phillip[1]]:
        phillip_clause = f"most often used {ranked_phillip[0]}, with other position-specific responses"
    else:
        phillip_clause = "used different position-specific responses shown in the examples"
    opponent = sorted(opponent_actions, key=lambda value: (-opponent_actions[value], value))[0]
    replay_actions = defaultdict(int)
    for item in items:
        replay_actions[item.replay_action.split(" (", 1)[0]] += 1
    replay_examples = sorted(replay_actions, key=lambda value: (-replay_actions[value], value))[:2]
    replay_example = " or ".join(replay_examples)
    if phase == "neutral":
        commitment, separator, exact_opener = action.partition(" against ")
        title = {
            "aerial commitment": "Getting opened during aerial commitments",
            "grab commitment": "Getting opened during grab commitments",
            "defensive timing": "Getting opened during defensive timing",
            "movement": "Getting opened while moving",
            "no action": "Getting opened while standing still",
            "grounded attack commitment": "Getting opened during grounded attacks",
        }.get(commitment, f"Getting opened during {commitment}")
        if separator:
            title = title.replace("Getting opened", f"Getting opened by {exact_opener}")
            opponent = exact_opener
        commitment_phrase = {
            "aerial commitment": "an aerial",
            "grab commitment": "a grab",
            "defensive timing": "a defensive option",
            "movement": "moving or jumping",
            "no action": "standing still",
            "grounded attack commitment": "a grounded attack",
        }.get(commitment, replay_example)
        summary = (
            f"This showed up {len(items)} times after the replay committed to {commitment_phrase}: it got caught by {opponent}, "
            f"while sampled Phillip branches {phillip_clause}."
        )
        drill = (
            f"Open each linked position and practice its displayed response where the replay committed to {commitment_phrase}. "
            "Use the recorded side first; mirror it only when the setup stays equivalent."
        )
    elif phase == "advantage":
        title = f"Leaving damage after {action} openings"
        summary = (
            f"Across {len(items)} conversions, the replay's {action} opening ended materially earlier than sampled Phillip branches, "
            f"which {phillip_clause}."
        )
        drill = (
            f"Start from each linked {action} setup and practice its displayed continuation. "
            "Continue until neutral or the stock ends. "
            "Count 10 completed routes from the recorded side, then mirror only equivalent setups."
        )
    else:
        title = f"Taking extra damage after {action}"
        summary = (
            f"After {opponent}, the replay repeatedly took more damage or lost the stock while sampled Phillip branches "
            f"{phillip_clause}."
        )
        drill = (
            f"Start from each linked {opponent} hit and practice its displayed response for that position. "
            "Count the rep only after stable neutral or ledge; use the recorded side before mirroring equivalent setups."
        )
    return title, summary, drill


def build_patterns(evidence_rows: Iterable[tuple[str, Evidence]]) -> tuple[list[Pattern], list[Pattern]]:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for key, evidence in evidence_rows:
        grouped[key].append(evidence)
    recurring = []
    watchlist = []
    for key, items in grouped.items():
        phase, matchup, action = key.split("|", 2)
        games = len({item.review_id for item in items})
        avg_improvement = mean(item.improvement for item in items)
        avg_share = mean(item.option_share for item in items)
        confidence = "high" if games >= 4 and len(items) >= 5 and avg_share >= 0.25 else "medium" if games >= 2 else "low"
        title, summary, drill = _pattern_copy(phase, action, items)
        score = games * 12 + len(items) * 3 + min(30.0, avg_improvement) + avg_share * 10
        pattern = Pattern(
            key=key,
            phase=phase,
            title=title,
            matchup=matchup,
            occurrences=len(items),
            games=games,
            confidence=confidence,
            summary=summary,
            drill=drill,
            evidence=sorted(items, key=lambda item: (-item.improvement, item.game_name, item.frame)),
            score=round(score, 2),
        )
        (recurring if games >= 2 and len(items) >= 2 else watchlist).append(pattern)
    recurring.sort(key=lambda pattern: (-pattern.score, pattern.phase, pattern.title))
    watchlist.sort(key=lambda pattern: (-pattern.score, pattern.phase, pattern.title))
    return recurring, watchlist


def report_payload(session: dict[str, Any], review_dirs: Iterable[Path]) -> dict[str, Any]:
    rows: list[tuple[str, Evidence]] = []
    errors = []
    for review_dir in review_dirs:
        try:
            rows.extend(collect_review_evidence(review_dir))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            errors.append({"reviewId": review_dir.name, "error": str(exc)})
    recurring, watchlist = build_patterns(rows)
    checkmates = sorted(
        (evidence for _key, evidence in rows if _is_checkmate_candidate(evidence)),
        key=lambda evidence: (-evidence.favorable_rate, -evidence.option_samples, -evidence.improvement, evidence.game_name, evidence.frame),
    )
    return {
        "schemaVersion": 1,
        "session": session,
        "evidenceCount": len(rows),
        "recurringPatterns": [asdict(pattern) for pattern in recurring],
        "watchlist": [asdict(pattern) for pattern in watchlist],
        "checkmateCandidates": [asdict(evidence) for evidence in checkmates],
        "aggregationErrors": errors,
    }


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _evidence_card(item: dict[str, Any]) -> str:
    practice = json.dumps({
        "reviewId": item["review_id"],
        "targetIndex": item["target_index"],
        "variationStartFrame": item["frame"],
    }, separators=(",", ":"))
    viewer = (
        f'<iframe loading="lazy" allowfullscreen title="Replay versus Phillip" data-src="{_esc(item.get("viewer_url"))}"></iframe>'
        if item.get("viewer_url") else '<div class="viewer-missing">Comparison unavailable</div>'
    )
    replay_context = item["opponent_action"] if item.get("phase") == "disadvantage" else item["replay_action"]
    replay_outcome = f"{item['original_result']} after {replay_context}"
    favorable_samples = int(item.get("favorable_samples") or 0)
    option_samples = int(item.get("option_samples") or 0)
    checkmate_note = ""
    if option_samples and favorable_samples == option_samples:
        checkmate_note = f'<p class="checkmate-note"><b>{favorable_samples}/{option_samples} favorable branches</b> in this route family.</p>'
    elif option_samples >= 8 and favorable_samples / option_samples >= CHECKMATE_LARGE_SAMPLE_THRESHOLD:
        checkmate_note = f'<p class="checkmate-note"><b>{favorable_samples}/{option_samples} favorable branches</b> in this route family.</p>'
    return f"""
    <article class="evidence" id="scenario-{_esc(item['review_id'])}-{int(item['target_index'])}">
      <div class="viewer">{viewer}</div>
      <div class="evidence-copy">
        <div class="evidence-head"><strong>{_esc(item['elapsed'])} on {_esc(item['stage'])}</strong><span>{_esc(item['matchup'])}<br>{_esc(item['game_name'])}</span></div>
        <p><b>Replay:</b> {_esc(replay_outcome)}.</p>
        <p><b>Phillip used {_esc(item['phillip_action'])}:</b> {_esc(item['phillip_result'])}.</p>
        {checkmate_note}
        <p class="support">This branch family appeared in {int(item['option_samples'])} of {int(item['sweep_samples'])} simulations.</p>
        <div class="evidence-actions"><a href="{_esc(item['source_url'])}">Open full comparison</a><button class="practice" data-practice="{_esc(practice)}">Practice this in TMCE</button><span role="status"></span></div>
      </div>
    </article>"""


def _checkmate_card(item: dict[str, Any]) -> str:
    phase = str(item.get("phase") or "")
    if phase == "advantage":
        title = f"{item['replay_action'].capitalize()} consistently converted in this position"
    elif phase == "disadvantage":
        title = f"{item['phillip_action'].capitalize()} consistently escaped {item['opponent_action']}"
    else:
        title = f"{item['phillip_action'].capitalize()} consistently won this neutral position"
    return f"""
    <section class="checkmate">
      <header><span class="phase">{_esc(PHASE_LABELS.get(phase, phase))}</span><h2>{_esc(title)}</h2><p>{int(item['favorable_samples'])}/{int(item['option_samples'])} sampled branches from this exact position produced a favorable result.</p></header>
      {_evidence_card(item)}
    </section>"""


def _pattern_card(pattern: dict[str, Any], index: int) -> str:
    items = pattern.get("evidence") or []
    evidence = "".join(_evidence_card(item) for item in items[:3])
    if len(items) > 3:
        remaining_count = len(items) - 3
        remaining = "".join(_evidence_card(item) for item in items[3:])
        noun = "example" if remaining_count == 1 else "examples"
        evidence += f'<details class="more-evidence"><summary>{remaining_count} more {noun}</summary>{remaining}</details>'
    return f"""
    <section class="pattern" id="priority-{index}">
      <header><div><span class="phase">{_esc(PHASE_LABELS.get(pattern['phase'], pattern['phase']))}</span><h2>{_esc(pattern['title'])}</h2></div><div class="confidence"><strong>{int(pattern['occurrences'])} times</strong><span>{int(pattern['games'])} games - {_esc(pattern['confidence'])} confidence</span></div></header>
      <p class="summary">{_esc(pattern['summary'])}</p>
      <div class="drill"><span>Practice cue</span><p>{_esc(pattern['drill'])}</p></div>
      <div class="evidence-list">{evidence}</div>
    </section>"""


def build_html(payload: dict[str, Any]) -> str:
    session = payload.get("session") or {}
    patterns = payload.get("recurringPatterns") or []
    top = patterns[:5]
    additional = patterns[5:]
    recurring_evidence = sum(len(pattern.get("evidence") or []) for pattern in patterns)
    recurring_ids = {
        (str(item.get("review_id") or ""), int(item.get("target_index") or 0))
        for pattern in patterns
        for item in pattern.get("evidence") or []
    }
    one_off_checkmates = [
        item for item in payload.get("checkmateCandidates") or []
        if (str(item.get("review_id") or ""), int(item.get("target_index") or 0)) not in recurring_ids
    ]
    stats = session.get("stats") or {}
    cards = "".join(_pattern_card(pattern, index) for index, pattern in enumerate(top, 1))
    if not cards:
        cards = '<section class="empty"><h2>No repeated high-confidence mistake survived the filters.</h2><p>Open the game reports below to inspect individual moments; the nightly job did not invent a drill from weak evidence.</p></section>'
    more = ""
    if additional:
        extra_cards = "".join(_pattern_card(pattern, index) for index, pattern in enumerate(additional, 6))
        more = f'<details class="more-patterns"><summary>{len(additional)} more repeated patterns</summary>{extra_cards}</details>'
    checkmate_cards = "".join(_checkmate_card(item) for item in one_off_checkmates[:5])
    checkmate_more = ""
    if len(one_off_checkmates) > 5:
        extra_checkmates = "".join(_checkmate_card(item) for item in one_off_checkmates[5:])
        checkmate_more = f'<details class="more-patterns"><summary>{len(one_off_checkmates) - 5} more checkmate candidates</summary>{extra_checkmates}</details>'
    checkmate_section = ""
    if one_off_checkmates:
        checkmate_section = f'<div class="section-head"><div><span class="eyebrow">Single-position certainty</span><h2>Checkmate candidates</h2></div><p>One-off positions with consistently favorable sampled branches.</p></div>{checkmate_cards}{checkmate_more}'
    game_links = "".join(
        f'<a href="/reviews/{_esc(game.get("reviewId"))}/report"><span>{_esc(game.get("opponent") or game.get("filename"))}</span><small>{_esc(game.get("stage") or "")} - {_esc(game.get("status") or "")}</small></a>'
        for game in session.get("games") or [] if game.get("reviewId")
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Nightly Melee Review - {_esc(session.get('date'))}</title><style>
    :root{{color-scheme:dark;--bg:#0b0e0c;--panel:#151a17;--raised:#1d241f;--line:#344039;--text:#f2f5f3;--muted:#9eaaa3;--green:#66d89b;--cyan:#67c6dd;--amber:#efb55e}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 Inter,Segoe UI,system-ui,sans-serif}}a{{color:inherit}}.top{{display:flex;align-items:center;justify-content:space-between;gap:16px;min-height:58px;padding:10px 22px;border-bottom:3px solid var(--green);background:#070908}}.top strong{{font-size:17px}}.top a{{padding:7px 9px;border:1px solid var(--line);border-radius:4px;text-decoration:none;font-size:12px;font-weight:800}}main{{width:min(1120px,calc(100% - 28px));margin:0 auto 80px}}.hero{{padding:36px 0 24px;border-bottom:1px solid var(--line)}}.eyebrow,.phase,.drill span{{color:var(--green);font-size:11px;font-weight:900;text-transform:uppercase}}h1{{max-width:760px;margin:5px 0 8px;font-size:34px;line-height:1.15;letter-spacing:0}}.hero>p{{max-width:760px;margin:0;color:var(--muted)}}.stats{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));margin-top:24px;border:1px solid var(--line)}}.stats div{{padding:13px;border-right:1px solid var(--line)}}.stats div:last-child{{border:0}}.stats strong,.stats span{{display:block}}.stats strong{{font-size:21px}}.stats span{{color:var(--muted);font-size:11px}}.section-head{{display:flex;align-items:end;justify-content:space-between;gap:12px;padding:26px 0 10px}}.section-head h2{{margin:0;font-size:18px}}.section-head p{{margin:0;color:var(--muted);font-size:12px}}.pattern,.checkmate{{margin-bottom:24px;border-top:1px solid var(--line)}}.pattern>header{{display:flex;justify-content:space-between;gap:20px;padding:20px 0 10px}}.checkmate>header{{padding:18px 0 10px}}.checkmate>header h2{{font-size:19px}}.checkmate>header p{{margin:5px 0 0;color:var(--muted);font-size:12px}}h2{{margin:3px 0 0;font-size:22px}}.confidence{{text-align:right}}.confidence strong,.confidence span{{display:block}}.confidence span{{color:var(--muted);font-size:11px}}.summary{{max-width:800px;margin:0 0 12px;font-size:16px}}.drill{{display:grid;grid-template-columns:110px 1fr;gap:14px;padding:12px;border-left:3px solid var(--green);background:var(--raised)}}.drill p{{margin:0}}.evidence-list{{display:grid;gap:10px;margin-top:12px}}.more-evidence{{display:grid;gap:10px}}.more-evidence>summary{{padding:10px;color:var(--green);font-weight:800;cursor:pointer}}.more-evidence[open]>.evidence{{margin-top:10px}}.evidence{{display:grid;grid-template-columns:minmax(320px,1.15fr) minmax(280px,1fr);border:1px solid var(--line);background:var(--panel)}}.viewer{{position:relative;min-height:260px;background:#030504}}.viewer iframe{{position:absolute;inset:0;width:100%;height:100%;border:0}}.viewer-missing{{display:grid;place-items:center;height:100%;color:var(--muted)}}.evidence-copy{{padding:15px}}.evidence-head{{display:flex;justify-content:space-between;gap:10px;padding-bottom:9px;border-bottom:1px solid var(--line)}}.evidence-head span{{color:var(--muted);font-size:11px}}.evidence-copy p{{margin:10px 0}}.evidence-copy .support{{color:var(--muted);font-size:12px}}.checkmate-note{{padding:7px 9px;border-left:3px solid var(--amber);background:#282015;color:#f3ce91}}.evidence-actions{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:14px}}button,.evidence-actions a{{min-height:36px;padding:7px 10px;border:1px solid #4b745e;border-radius:4px;background:var(--raised);color:var(--text);font:inherit;font-size:12px;font-weight:800;cursor:pointer;text-decoration:none}}button.practice{{background:var(--green);color:#07120c}}button:disabled{{opacity:.55;cursor:wait}}.evidence-actions span{{color:var(--muted);font-size:11px}}.more-patterns{{margin:8px 0 30px;border-top:1px solid var(--line)}}.more-patterns>summary{{padding:15px 0;color:var(--green);font-weight:800;cursor:pointer}}.games{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-top:1px solid var(--line);border-left:1px solid var(--line)}}.games a{{display:block;min-width:0;padding:11px;border-right:1px solid var(--line);border-bottom:1px solid var(--line);text-decoration:none}}.games span,.games small{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.games small{{color:var(--muted)}}.empty{{padding:30px 0;border-top:1px solid var(--line)}}@media(max-width:760px){{h1{{font-size:28px}}.stats{{grid-template-columns:repeat(2,1fr)}}.stats div:nth-child(2){{border-right:0}}.stats div:nth-child(-n+2){{border-bottom:1px solid var(--line)}}.pattern>header{{display:block}}.confidence{{margin-top:8px;text-align:left}}.evidence{{grid-template-columns:1fr}}.viewer{{min-height:240px}}.drill{{grid-template-columns:1fr;gap:4px}}.games{{grid-template-columns:1fr}}}}
    </style></head><body><header class="top"><strong>Nightly Melee Review</strong><a href="/">Dashboard</a></header><main><section class="hero"><span class="eyebrow">{_esc(session.get('date'))} session</span><h1>{_esc(session.get('headline') or 'The habits worth practicing tomorrow')}</h1><p>{recurring_evidence} of {int(payload.get('evidenceCount') or 0)} qualified comparisons repeated across games. Checkmate candidates may come from one game, but require consistently favorable branches from the same position.</p><div class="stats"><div><strong>{int(stats.get('analyzedGames') or 0)}</strong><span>games analyzed</span></div><div><strong>{int(payload.get('evidenceCount') or 0)}</strong><span>qualified comparisons</span></div><div><strong>{len(patterns)}</strong><span>repeated patterns</span></div><div><strong>{_esc(stats.get('processingTime') or 'n/a')}</strong><span>processing time</span></div></div></section><div class="section-head"><div><span class="eyebrow">Start here</span><h2>Practice priorities</h2></div><p>Ordered by recurrence, sample support, and outcome gap.</p></div>{cards}{more}{checkmate_section}<div class="section-head"><div><span class="eyebrow">Evidence</span><h2>Every processed game</h2></div><p>Open a full phase deck for context.</p></div><div class="games">{game_links}</div></main><script>
    const loadViewer=entry=>{{const frame=entry.target.querySelector('iframe[data-src]');if(frame&&!frame.src)frame.src=frame.dataset.src}};const observer=new IntersectionObserver(entries=>entries.filter(entry=>entry.isIntersecting).forEach(entry=>{{loadViewer(entry);observer.unobserve(entry.target)}}),{{rootMargin:'300px'}});document.querySelectorAll('.viewer').forEach(viewer=>observer.observe(viewer));document.addEventListener('click',async event=>{{const button=event.target.closest('.practice');if(!button)return;const status=button.parentElement.querySelector('[role=status]'),payload=JSON.parse(button.dataset.practice);button.disabled=true;status.textContent='Preparing Training Mode...';try{{const response=await fetch(`/api/reviews/${{payload.reviewId}}/training-mode`,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{targetIndex:payload.targetIndex,alternativeIndex:0,scenarioMode:'variations',variationStartFrame:payload.variationStartFrame,variationSource:'replay',queueMode:'phase-sweep'}})}});const body=await response.json();if(!response.ok||!body.ok)throw new Error(body.error?.message||'Training Mode launch failed.');status.textContent=`Ready from f${{body.scenario.practiceStartFrame}}.`}}catch(error){{status.textContent=error.message||'Training Mode launch failed.'}}finally{{button.disabled=false}}}});
    </script></body></html>"""


def apply_curation(payload: dict[str, Any], curation: dict[str, Any]) -> dict[str, Any]:
    excluded = {str(key) for key in curation.get("excludeKeys") or []}
    overrides = curation.get("overrides") if isinstance(curation.get("overrides"), dict) else {}
    patterns = []
    for pattern in payload.get("recurringPatterns") or []:
        key = str(pattern.get("key") or "")
        if key in excluded:
            continue
        override = overrides.get(key) if isinstance(overrides.get(key), dict) else {}
        updated = dict(pattern)
        for field in ("title", "summary", "drill"):
            if override.get(field):
                updated[field] = str(override[field])
        updated["curated"] = bool(override)
        updated["curationOrder"] = int(override.get("order") or 1000)
        patterns.append(updated)
    patterns.sort(key=lambda pattern: (int(pattern["curationOrder"]), -float(pattern.get("score") or 0.0)))
    for pattern in patterns:
        pattern.pop("curationOrder", None)
    payload["recurringPatterns"] = patterns
    payload["curation"] = {
        "applied": True,
        "reviewedBy": [str(value) for value in curation.get("reviewedBy") or []],
    }
    return payload


def write_report(session: dict[str, Any], review_dirs: Iterable[Path], out_dir: Path) -> dict[str, Any]:
    payload = report_payload(session, review_dirs)
    out_dir.mkdir(parents=True, exist_ok=True)
    curation_path = out_dir / "curation.json"
    if curation_path.is_file():
        payload = apply_curation(payload, _read_json(curation_path))
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.html").write_text(build_html(payload), encoding="utf-8")
    return payload
