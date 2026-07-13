"""Build comprehensive attacker advantage takeover candidates from a Slippi replay.

The phase timeline is useful presentation context, but it is intentionally lossy:
minor conversions are hidden, short resets are merged, and overlapping conversions
are serialized. Candidate coverage therefore uses raw slippi-js conversions as the
opportunity universe and maps them back to presentation segments when possible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .paths import JS_DIR, PROJECT_DIR
from .stage_geometry import stage_geometry_for_settings


WORK_DIR = PROJECT_DIR

THROW_MOVES = {"fthrow", "bthrow", "uthrow", "dthrow"}
TECH_ACTIONS = set(range(0xB7, 0xCD))
LEDGE_ACTIONS = {0xFC, 0xFD}
DEAD_ACTIONS = set(range(0x00, 0x0B))
GROUND_ATTACK_NAMES = {
    0x2C: "jab1",
    0x2D: "jab2",
    0x2E: "jab3",
    0x2F: "rapid-jab-start",
    0x30: "rapid-jab",
    0x31: "rapid-jab-end",
    0x32: "dash",
    0x33: "ftilt",
    0x34: "ftilt",
    0x35: "ftilt",
    0x36: "ftilt",
    0x37: "ftilt",
    0x38: "utilt",
    0x39: "dtilt",
    0x3A: "fsmash",
    0x3B: "fsmash",
    0x3C: "fsmash",
    0x3D: "fsmash",
    0x3E: "fsmash",
    0x3F: "usmash",
    0x40: "dsmash",
}
AERIAL_NAMES = {0x41: "nair", 0x42: "fair", 0x43: "bair", 0x44: "uair", 0x45: "dair"}
GRAB_THROW_NAMES = {
    0xD4: "grab",
    0xD5: "grab-pull",
    0xD6: "dash-grab",
    0xD7: "grab-hold",
    0xD8: "pummel",
    0xD9: "fthrow",
    0xDA: "bthrow",
    0xDB: "uthrow",
    0xDC: "dthrow",
}


# This stays embedded so the candidate builder remains one owned implementation
# file. It uses slippi-js structured APIs; no replay bytes are parsed ad hoc.
RAW_CONVERSION_EXTRACTOR_JS = r"""
const path = require("path");
const slippi = require("@slippi/slippi-js/node");
const replay = path.resolve(process.argv[1]);
const game = new slippi.SlippiGame(replay);
const settings = game.getSettings();
const stats = game.getStats();
const frames = game.getFrames();
if (!settings || !stats || !frames) throw new Error("Could not load Slippi settings/stats/frames.");

const frameIds = Object.keys(frames).map(Number).filter(Number.isFinite).sort((a, b) => a - b);
const firstFrame = frameIds[0];
const lastFrame = frameIds[frameIds.length - 1];
const localActionNames = {
  0x00: "dead down", 0x01: "dead left", 0x02: "dead right", 0x03: "dead up",
  0x0e: "wait", 0x14: "dash", 0x15: "run", 0x18: "jump squat",
  0x1d: "fall", 0x2a: "landing", 0x2b: "landing fall special",
  0xb7: "knockdown bounce face up", 0xb8: "knockdown wait face up",
  0xbc: "tech roll forward face up", 0xbd: "tech roll back face up",
  0xbe: "missed tech face up", 0xbf: "knockdown bounce face down",
  0xc0: "knockdown wait face down", 0xc4: "tech roll forward face down",
  0xc5: "tech roll back face down", 0xc6: "missed tech face down",
  0xc7: "tech in place", 0xc8: "wall tech", 0xc9: "ceiling tech",
  0xca: "missed tech getup", 0xcb: "wall jump tech", 0xcc: "missed wall tech",
  0xd4: "grab", 0xd5: "grab pull", 0xd6: "dash grab", 0xd7: "grab hold",
  0xd8: "pummel", 0xd9: "forward throw", 0xda: "back throw",
  0xdb: "up throw", 0xdc: "down throw", 0xfc: "ledge catch", 0xfd: "ledge hang",
};

function post(frame, playerIndex) {
  return frames?.[frame]?.players?.[playerIndex]?.post || null;
}

function actionName(actionStateId) {
  if (actionStateId == null) return "unknown";
  if (localActionNames[actionStateId]) return localActionNames[actionStateId];
  try {
    const value = slippi.actions?.getName ? slippi.actions.getName(actionStateId) : null;
    if (value) return value;
  } catch (_) {}
  return `state ${actionStateId}`;
}

function snapshot(frame, playerIndex) {
  const value = post(frame, playerIndex);
  if (!value) return null;
  return {
    frame,
    playerIndex,
    percent: value.percent ?? null,
    stocksRemaining: value.stocksRemaining ?? null,
    positionX: value.positionX ?? null,
    positionY: value.positionY ?? null,
    actionStateId: value.actionStateId ?? null,
    actionStateName: actionName(value.actionStateId),
    actionStateCounter: value.actionStateCounter ?? null,
    isAirborne: value.isAirborne ?? null,
    facingDirection: value.facingDirection ?? null,
    hitlagRemaining: value.hitlagRemaining ?? null,
    hitstunRemaining: value.hitstunRemaining ?? null,
    lastAttackLanded: value.lastAttackLanded ?? null,
    lastHitBy: value.lastHitBy ?? null,
  };
}

function actionStartFrame(frame, playerIndex) {
  const value = post(frame, playerIndex);
  if (!value || value.actionStateId == null) return frame;
  const action = value.actionStateId;
  let start = frame;
  while (start > firstFrame && frame - start < 120) {
    const before = post(start - 1, playerIndex);
    if (!before || before.actionStateId !== action) break;
    start -= 1;
  }
  return start;
}

function actionRuns(startFrame, endFrame, playerIndex) {
  const runs = [];
  let active = null;
  const flush = (end) => {
    if (!active) return;
    runs.push({
      startFrame: active.startFrame,
      endFrame: end,
      durationFrames: end - active.startFrame + 1,
      actionStateId: active.actionStateId,
      actionStateName: actionName(active.actionStateId),
      start: snapshot(active.startFrame, playerIndex),
      end: snapshot(end, playerIndex),
    });
    active = null;
  };
  for (let frame = Math.max(firstFrame, startFrame); frame <= Math.min(lastFrame, endFrame); frame += 1) {
    const value = post(frame, playerIndex);
    if (!value) {
      flush(frame - 1);
      continue;
    }
    if (!active || active.actionStateId !== value.actionStateId) {
      flush(frame - 1);
      active = { startFrame: frame, actionStateId: value.actionStateId };
    }
  }
  flush(Math.min(lastFrame, endFrame));
  return runs;
}

function positionSummary(startFrame, endFrame, playerIndex) {
  let minX = null, maxX = null, minY = null, maxY = null;
  const actionStateIds = new Set();
  for (let frame = Math.max(firstFrame, startFrame); frame <= Math.min(lastFrame, endFrame); frame += 1) {
    const value = post(frame, playerIndex);
    if (!value) continue;
    const x = value.positionX;
    const y = value.positionY;
    if (x != null) { minX = minX == null ? x : Math.min(minX, x); maxX = maxX == null ? x : Math.max(maxX, x); }
    if (y != null) { minY = minY == null ? y : Math.min(minY, y); maxY = maxY == null ? y : Math.max(maxY, y); }
    if (value.actionStateId != null) actionStateIds.add(value.actionStateId);
  }
  return { minX, maxX, minY, maxY, actionStateIds: [...actionStateIds].sort((a, b) => a - b) };
}

const conversions = (stats.conversions || []).filter((item) => item.startFrame != null).map((conversion, conversionIndex) => {
  const attacker = conversion.lastHitBy;
  const defender = conversion.playerIndex;
  const startFrame = conversion.startFrame;
  const endFrame = conversion.endFrame ?? conversion.startFrame;
  const moves = (conversion.moves || []).map((move, moveIndex) => {
    const moveAttacker = move.playerIndex ?? attacker;
    let moveName = `Move ${move.moveId}`;
    let moveShortName = `move${move.moveId}`;
    try { moveName = slippi.moves.getMoveName(move.moveId); } catch (_) {}
    try { moveShortName = slippi.moves.getMoveShortName(move.moveId); } catch (_) {}
    return {
      index: moveIndex + 1,
      frame: move.frame,
      playerIndex: moveAttacker,
      moveId: move.moveId,
      moveName,
      moveShortName,
      hitCount: move.hitCount,
      damage: move.damage,
      actionStartFrame: actionStartFrame(move.frame, moveAttacker),
      attacker: snapshot(move.frame, moveAttacker),
      attackerBefore: snapshot(move.frame - 1, moveAttacker),
      defender: snapshot(move.frame, defender),
      defenderBefore: snapshot(move.frame - 1, defender),
    };
  });
  return {
    conversionIndex,
    startFrame,
    endFrame,
    lastHitBy: attacker,
    playerIndex: defender,
    openingType: conversion.openingType || "unknown",
    startPercent: conversion.startPercent ?? null,
    endPercent: conversion.endPercent ?? conversion.currentPercent ?? null,
    currentPercent: conversion.currentPercent ?? null,
    didKill: Boolean(conversion.didKill),
    moves,
    attackerStart: snapshot(startFrame, attacker),
    attackerEnd: snapshot(endFrame, attacker),
    defenderStart: snapshot(startFrame, defender),
    defenderEnd: snapshot(endFrame, defender),
    attackerPositionSummary: positionSummary(startFrame, endFrame, attacker),
    defenderPositionSummary: positionSummary(startFrame, endFrame, defender),
    attackerActionRuns: actionRuns(startFrame, endFrame, attacker),
    defenderActionRuns: actionRuns(startFrame, endFrame, defender),
  };
});

const players = (settings.players || []).map((player) => ({
  playerIndex: player.playerIndex,
  port: player.port,
  characterId: player.characterId,
  characterName: slippi.characters.getCharacterInfo(player.characterId)?.name || `Character ${player.characterId}`,
  displayName: player.displayName || "",
  connectCode: player.connectCode || "",
}));

console.log(JSON.stringify({
  replay,
  firstFrame,
  lastFrame,
  settings: {
    stageId: settings.stageId,
    stageName: slippi.stages.getStageName(settings.stageId),
    players,
  },
  conversions,
}));
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build comprehensive advantage takeover candidates from Slippi.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--replay", type=Path)
    source.add_argument("--timeline", type=Path)
    parser.add_argument("--analyzed-port", type=int, choices=(1, 2), required=True)
    parser.add_argument("--min-damage", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=0, help="Zero keeps every qualifying opportunity.")
    parser.add_argument(
        "--include-kills",
        action="store_true",
        help="Compatibility flag. Multi-hit kill routes are now included by default; one-hit kills remain excluded.",
    )
    parser.add_argument("--dedupe-frames", type=int, default=3)
    parser.add_argument("--coverage-report", type=Path)
    parser.add_argument("--raw-events-out", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def build_timeline(replay: Path, out: Path) -> dict[str, Any]:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "node",
            str(JS_DIR / "build_phase_timeline.js"),
            "--replay",
            str(replay.resolve()),
            "--out",
            str(out.resolve()),
        ],
        cwd=WORK_DIR,
        check=True,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def extract_raw_conversions(replay: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["node", "-e", RAW_CONVERSION_EXTRACTOR_JS, str(replay.resolve())],
        cwd=WORK_DIR,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return json.loads(completed.stdout)


def player_index_for_port(settings: dict[str, Any], port: int) -> int:
    for player in settings.get("players") or []:
        if int(player.get("port") or -1) == port:
            return int(player["playerIndex"])
    raise ValueError(f"could not resolve Slippi player index for P{port}")


def player_labels(settings: dict[str, Any]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for player in settings.get("players") or []:
        player_index = int(player.get("playerIndex") or 0)
        port = int(player.get("port") or player_index + 1)
        name = player.get("displayName") or player.get("connectCode") or ""
        labels[player_index] = f"P{port} {name}".strip()
    return labels


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def conversion_damage(conversion: dict[str, Any]) -> float:
    start = _number(conversion.get("startPercent"))
    end = _number(conversion.get("endPercent"), _number(conversion.get("currentPercent"), start))
    damage = max(0.0, end - start)
    if damage <= 0:
        damage = sum(_number(move.get("damage")) for move in conversion.get("moves") or [])
    return round(damage, 3)


def normalize_moves(conversion: dict[str, Any], labels: dict[int, str]) -> list[dict[str, Any]]:
    attacker = _integer(conversion.get("lastHitBy"), -1)
    moves = []
    for index, raw_move in enumerate(conversion.get("moves") or []):
        move = dict(raw_move)
        player_index = _integer(move.get("playerIndex"), attacker)
        move.update(
            {
                "index": index + 1,
                "frame": _integer(move.get("frame")),
                "playerIndex": player_index,
                "playerLabel": labels.get(player_index, f"P{player_index + 1}"),
                "moveId": move.get("moveId"),
                "moveName": move.get("moveName") or f"Move {move.get('moveId')}",
                "moveShortName": move.get("moveShortName") or f"move{move.get('moveId')}",
                "hitCount": max(1, _integer(move.get("hitCount"), 1)),
                "damage": round(_number(move.get("damage")), 3),
            }
        )
        if move.get("actionStartFrame") is not None:
            move["actionStartFrame"] = _integer(move["actionStartFrame"])
        moves.append(move)
    return moves


def presentation_fallback_conversions(timeline: dict[str, Any], labels: dict[int, str]) -> list[dict[str, Any]]:
    conversions = []
    for index, segment in enumerate(timeline.get("segments") or []):
        if segment.get("phase") not in {"advantage", "one_hit_kill"}:
            continue
        conversion = segment.get("conversion") or {}
        moves = normalize_moves({"moves": conversion.get("moveSequence") or [], "lastHitBy": segment.get("owner")}, labels)
        conversions.append(
            {
                "conversionIndex": index,
                "startFrame": _integer(segment.get("startFrame")),
                "endFrame": _integer(segment.get("endFrame")),
                "lastHitBy": segment.get("owner"),
                "playerIndex": segment.get("defender"),
                "openingType": conversion.get("openingType") or "unknown",
                "startPercent": conversion.get("startPercent"),
                "endPercent": conversion.get("endPercent"),
                "didKill": bool(conversion.get("didKill")),
                "moves": moves,
                "attackerStart": (conversion.get("frameFacts") or {}).get("attackerStart"),
                "attackerEnd": (conversion.get("frameFacts") or {}).get("attackerEnd"),
                "defenderStart": (conversion.get("frameFacts") or {}).get("defenderStart"),
                "defenderEnd": (conversion.get("frameFacts") or {}).get("defenderEnd"),
                "attackerActionRuns": ((segment.get("actionTimeline") or {}).get("runsByPlayer") or {}).get(str(segment.get("owner")), []),
                "defenderActionRuns": ((segment.get("actionTimeline") or {}).get("runsByPlayer") or {}).get(str(segment.get("defender")), []),
                "dataSource": "presentation_timeline_fallback",
            }
        )
    return conversions


def make_sequences(
    conversions: Iterable[dict[str, Any]],
    *,
    analyzed_index: int,
    analyzed_port: int,
    labels: dict[int, str],
) -> list[dict[str, Any]]:
    sequences = []
    for conversion in conversions:
        if _integer(conversion.get("lastHitBy"), -1) != analyzed_index:
            continue
        normalized = dict(conversion)
        normalized["moves"] = normalize_moves(conversion, labels)
        start = _integer(normalized.get("startFrame"))
        end = _integer(normalized.get("endFrame"), start)
        conversion_index = _integer(normalized.get("conversionIndex"), len(sequences))
        normalized.update(
            {
                "sequence_id": f"p{analyzed_port}_adv_{conversion_index:03d}_f{start}_{end}",
                "startFrame": start,
                "endFrame": end,
                "damage": conversion_damage(normalized),
                "didKill": bool(normalized.get("didKill")),
                "openingType": normalized.get("openingType") or "unknown",
            }
        )
        sequences.append(normalized)
    return sorted(sequences, key=lambda item: (item["startFrame"], item["endFrame"], item["sequence_id"]))


def segment_opening_frame(segment: dict[str, Any]) -> int | None:
    moves = (segment.get("conversion") or {}).get("moveSequence") or []
    if not moves or moves[0].get("frame") is None:
        return None
    return _integer(moves[0]["frame"])


def map_presentation_context(sequences: list[dict[str, Any]], timeline: dict[str, Any], analyzed_index: int) -> None:
    segments = [
        segment
        for segment in timeline.get("segments") or []
        if segment.get("phase") in {"advantage", "one_hit_kill"} and segment.get("owner") == analyzed_index
    ]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        moves = sequence.get("moves") or []
        opening_frame = _integer(moves[0].get("frame"), sequence["startFrame"]) if moves else sequence["startFrame"]
        best: tuple[int, dict[str, Any]] | None = None
        for segment in segments:
            segment_start = _integer(segment.get("startFrame"))
            segment_end = _integer(segment.get("endFrame"), segment_start)
            overlap = max(0, min(sequence["endFrame"], segment_end) - max(sequence["startFrame"], segment_start) + 1)
            contains_opening = segment_start <= opening_frame <= segment_end
            exact_opening = segment_opening_frame(segment) == opening_frame
            if overlap <= 0 and not contains_opening:
                continue
            score = overlap + 100_000 * int(contains_opening) + 200_000 * int(exact_opening)
            if best is None or score > best[0]:
                best = (score, segment)
        if best is None:
            sequence["presentation"] = None
            continue
        segment = best[1]
        sequence["presentation"] = {
            "phase": segment.get("phase"),
            "startFrame": _integer(segment.get("startFrame")),
            "endFrame": _integer(segment.get("endFrame")),
            "label": segment.get("label"),
            "notes": segment.get("notes") or [],
            "openingFrame": segment_opening_frame(segment),
        }
        grouped[(sequence["presentation"]["startFrame"], sequence["presentation"]["endFrame"])].append(sequence)

    for group in grouped.values():
        ordered = sorted(group, key=lambda item: item["startFrame"])
        for index, sequence in enumerate(ordered):
            presentation_opening = (sequence.get("presentation") or {}).get("openingFrame")
            raw_opening = (sequence.get("moves") or [{}])[0].get("frame")
            sequence["is_reopening"] = index > 0 or (
                presentation_opening is not None and raw_opening is not None and _integer(raw_opening) != _integer(presentation_opening)
            )


def link_baseline_horizons(sequences: list[dict[str, Any]]) -> None:
    """Link raw conversions that the presentation timeline treats as one advantage.

    The raw conversion remains the simulation/candidate identity. Only its replay
    comparison horizon is extended through later same-owner conversions inside the
    same merged presentation segment.
    """
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        presentation = sequence.get("presentation") or {}
        if presentation.get("phase") != "advantage":
            continue
        key = (_integer(presentation.get("startFrame")), _integer(presentation.get("endFrame")))
        grouped[key].append(sequence)

    for sequence in sequences:
        sequence["baseline_horizon_sequence_ids"] = [sequence["sequence_id"]]
        sequence["baseline_horizon_end_frame"] = sequence["endFrame"]
        sequence["baseline_later_moves"] = []
        sequence["baseline_later_kill"] = False
        sequence["baseline_horizon_damage"] = _number(sequence.get("damage"))
        sequence["baseline_horizon_hit_events"] = len(sequence.get("moves") or [])

    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (item["startFrame"], item["endFrame"], item["sequence_id"]))
        if len(ordered) <= 1:
            continue
        for index, sequence in enumerate(ordered):
            horizon = ordered[index:]
            later = ordered[index + 1 :]
            sequence["baseline_horizon_sequence_ids"] = [item["sequence_id"] for item in horizon]
            sequence["baseline_horizon_end_frame"] = max(item["endFrame"] for item in horizon)
            sequence["baseline_later_moves"] = [move for item in later for move in item.get("moves") or []]
            sequence["baseline_later_kill"] = any(bool(item.get("didKill")) for item in later)
            sequence["baseline_horizon_damage"] = round(sum(_number(item.get("damage")) for item in horizon), 3)
            sequence["baseline_horizon_hit_events"] = sum(len(item.get("moves") or []) for item in horizon)


def legacy_candidates(timeline: dict[str, Any], analyzed_index: int, min_damage: float) -> list[dict[str, Any]]:
    candidates = []
    for segment in timeline.get("segments") or []:
        conversion = segment.get("conversion") or {}
        moves = conversion.get("moveSequence") or []
        if segment.get("phase") != "advantage" or segment.get("owner") != analyzed_index:
            continue
        if conversion.get("didKill") or not moves or _number(conversion.get("damage")) < min_damage:
            continue
        candidates.append(
            {
                "frame": _integer(moves[0].get("frame")),
                "segment_start_frame": _integer(segment.get("startFrame")),
                "segment_end_frame": _integer(segment.get("endFrame")),
            }
        )
    return candidates


def find_missed_reason(sequence: dict[str, Any], timeline: dict[str, Any], legacy_frames: set[int]) -> str | None:
    moves = sequence.get("moves") or []
    opening_frame = _integer(moves[0].get("frame"), sequence["startFrame"]) if moves else sequence["startFrame"]
    if opening_frame in legacy_frames:
        return None
    if sequence.get("didKill") and len(moves) > 1:
        return "multi_hit_kill_route_excluded"
    if sequence.get("is_reopening"):
        return "reopening_hidden_by_merged_phase_segment"
    presentation = sequence.get("presentation")
    if presentation is None:
        containing = [
            segment
            for segment in timeline.get("segments") or []
            if _integer(segment.get("startFrame")) <= opening_frame <= _integer(segment.get("endFrame"))
        ]
        if any(segment.get("phase") == "advantage" for segment in containing):
            return "overlapping_conversion_clipped_by_serial_phase_timeline"
        return "minor_conversion_filtered_from_phase_timeline"
    return "legacy_opening_only_filter"


def mutual_trade_resolution(sequence: dict[str, Any], all_conversions: list[dict[str, Any]]) -> tuple[bool, str | None]:
    if sequence.get("openingType") != "trade":
        return True, None
    attacker = _integer(sequence.get("lastHitBy"), -1)
    defender = _integer(sequence.get("playerIndex"), -1)
    pair = next(
        (
            other
            for other in all_conversions
            if other is not sequence
            and other.get("openingType") == "trade"
            and _integer(other.get("lastHitBy"), -1) == defender
            and _integer(other.get("playerIndex"), -1) == attacker
            and abs(_integer(other.get("startFrame")) - sequence["startFrame"]) <= 3
        ),
        None,
    )
    if pair is None:
        return True, "unpaired_trade_with_advantage_conversion"
    pair_end = _integer(pair.get("endFrame"), _integer(pair.get("startFrame")))
    if sequence["endFrame"] >= pair_end + 5:
        return True, "trade_hitstun_resolved_in_analyzed_player_favor"
    next_owned = next(
        (
            other
            for other in sorted(all_conversions, key=lambda item: _integer(item.get("startFrame")))
            if _integer(other.get("lastHitBy"), -1) == attacker
            and other.get("openingType") != "trade"
            and _integer(other.get("startFrame")) > sequence["startFrame"]
            and _integer(other.get("startFrame")) - max(sequence["endFrame"], pair_end) <= 120
        ),
        None,
    )
    if next_owned is not None:
        return True, f"trade_led_to_advantage_at_f{_integer(next_owned.get('startFrame'))}"
    return False, "mutual_trade_without_clear_following_advantage"


def sequence_eligibility(
    sequence: dict[str, Any],
    *,
    min_damage: float,
    all_conversions: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    moves = sequence.get("moves") or []
    if not moves:
        return False, "no_recorded_hit_events"
    if sequence.get("didKill") and len(moves) <= 1:
        return False, "one_hit_kill"
    if _number(sequence.get("damage")) < min_damage and len(moves) <= 1 and not sequence.get("didKill"):
        return False, "below_min_damage"
    trade_eligible, trade_reason = mutual_trade_resolution(sequence, all_conversions)
    sequence["trade_resolution_evidence"] = trade_reason
    if not trade_eligible:
        return False, trade_reason
    return True, None


def _run_action_ids(sequence: dict[str, Any], key: str) -> set[int]:
    ids = set()
    for run in sequence.get(key) or []:
        if run.get("actionStateId") is not None:
            ids.add(_integer(run["actionStateId"]))
    return ids


def _ledge_x(stage_geometry: dict[str, Any] | None) -> float | None:
    try:
        return abs(float(stage_geometry["ledges"]["right"]["x"]))
    except (KeyError, TypeError, ValueError):
        return None


def _snapshot_offstage(snapshot: dict[str, Any] | None, ledge_x: float | None) -> bool:
    if not snapshot or ledge_x is None:
        return False
    x = abs(_number(snapshot.get("positionX")))
    y = _number(snapshot.get("positionY"))
    return x >= ledge_x - 2.0 or (y < -2.0 and x >= ledge_x - 18.0)


def classify_sequence(sequence: dict[str, Any], stage_geometry: dict[str, Any] | None) -> dict[str, Any]:
    moves = sequence.get("moves") or []
    move_names = [str(move.get("moveShortName") or "unknown").lower() for move in moves]
    defender_actions = _run_action_ids(sequence, "defenderActionRuns")
    ledge_x = _ledge_x(stage_geometry)
    offstage_hits = [
        _integer(move.get("frame"))
        for move in moves
        if _snapshot_offstage(move.get("defenderBefore") or move.get("defender"), ledge_x)
    ]
    ledge_runs = [
        run
        for run in sequence.get("defenderActionRuns") or []
        if _integer(run.get("actionStateId"), -1) in LEDGE_ACTIONS
    ]
    platform_hits = []
    vertical_moves = {"uair", "utilt", "usmash", "bair", "nair"}
    for move in moves:
        defender = move.get("defenderBefore") or move.get("defender") or {}
        attacker = move.get("attackerBefore") or move.get("attacker") or {}
        defender_y = _number(defender.get("positionY"))
        attacker_y = _number(attacker.get("positionY"))
        defender_x = abs(_number(defender.get("positionX")))
        if (
            str(move.get("moveShortName") or "").lower() in vertical_moves
            and defender_y >= 10.0
            and defender_y >= attacker_y + 3.0
            and (ledge_x is None or defender_x <= ledge_x + 4.0)
        ):
            platform_hits.append(_integer(move.get("frame")))

    categories = []
    evidence: dict[str, Any] = {}
    opening_type = sequence.get("openingType") or "unknown"
    if opening_type == "neutral-win":
        categories.append("neutral_opening_conversion")
    elif opening_type == "counter-attack":
        categories.append("counter_attack")
    elif opening_type == "trade":
        categories.append("trade_resolution")
        evidence["trade"] = sequence.get("trade_resolution_evidence")
    if sequence.get("is_reopening"):
        categories.append("reopening_extension")
    if len(moves) >= 2:
        categories.append("punish_sequence")
    if any(name in THROW_MOVES for name in move_names):
        categories.append("throw_route")
        evidence["throw_frames"] = [
            _integer(move.get("frame")) for move in moves if str(move.get("moveShortName") or "").lower() in THROW_MOVES
        ]
    tech_actions = sorted(defender_actions & TECH_ACTIONS)
    if tech_actions:
        categories.append("tech_chase")
        evidence["defender_tech_action_state_ids"] = tech_actions
    if platform_hits:
        categories.append("platform_juggle")
        evidence["platform_height_hit_frames"] = platform_hits
    if offstage_hits or ledge_runs:
        categories.append("edgeguard_recovery")
        evidence["offstage_hit_frames"] = offstage_hits
        evidence["defender_ledge_runs"] = [
            {"startFrame": run.get("startFrame"), "endFrame": run.get("endFrame"), "actionStateName": run.get("actionStateName")}
            for run in ledge_runs
        ]
    if sequence.get("didKill"):
        categories.append("stock_conversion")

    priority = [
        "edgeguard_recovery",
        "tech_chase",
        "platform_juggle",
        "throw_route",
        "trade_resolution",
        "reopening_extension",
        "neutral_opening_conversion",
        "counter_attack",
        "punish_sequence",
    ]
    primary = next((category for category in priority if category in categories), "advantage_extension")
    return {"primary_category": primary, "categories": categories or [primary], "evidence": evidence}


def commitment_info(run: dict[str, Any]) -> tuple[str, str] | None:
    action = _integer(run.get("actionStateId"), -1)
    if action in GROUND_ATTACK_NAMES:
        move = GROUND_ATTACK_NAMES[action]
        kind = "smash_attack_commitment" if "smash" in move else "grounded_attack_commitment"
        return kind, move
    if action in AERIAL_NAMES:
        return "aerial_attack_commitment", AERIAL_NAMES[action]
    if action in GRAB_THROW_NAMES:
        return "grab_throw_commitment", GRAB_THROW_NAMES[action]
    return None


def pseudo_move(frame: int, name: str, sequence: dict[str, Any], labels: dict[int, str]) -> dict[str, Any]:
    attacker = _integer(sequence.get("lastHitBy"), -1)
    return {
        "index": None,
        "frame": frame,
        "playerIndex": attacker,
        "playerLabel": labels.get(attacker, f"P{attacker + 1}"),
        "moveId": None,
        "moveName": name.replace("-", " ").title(),
        "moveShortName": name,
        "hitCount": 0,
        "damage": 0.0,
    }


def opportunity_points(
    sequence: dict[str, Any],
    *,
    labels: dict[int, str],
    dedupe_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    moves = sequence.get("moves") or []
    points: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    first_hit_frame = _integer(moves[0].get("frame"))
    if first_hit_frame - sequence["startFrame"] > dedupe_frames:
        points.append(
            {
                "frame": sequence["startFrame"],
                "role": "conversion_commitment",
                "branch_move_index": None,
                "followup_start_index": 0,
                "branch_move": pseudo_move(sequence["startFrame"], "conversion-start", sequence, labels),
                "point_evidence": {"first_hit_frame": first_hit_frame},
            }
        )

    for index, move in enumerate(moves):
        hit_frame = _integer(move.get("frame"))
        action_start = move.get("actionStartFrame")
        if index > 0 and action_start is not None:
            action_start = _integer(action_start)
            previous_hit = _integer(moves[index - 1].get("frame"))
            if action_start > previous_hit:
                points.append(
                    {
                        "frame": action_start,
                        "role": "continuation_commitment",
                        "branch_move_index": index,
                        "followup_start_index": index,
                        "branch_move": move,
                        "point_evidence": {"hit_frame": hit_frame, "action_start_frame": action_start},
                    }
                )
            else:
                exclusions.append(
                    {
                        "frame": action_start,
                        "role": "continuation_commitment",
                        "reason": "commitment_overlaps_previous_hit",
                        "move": move.get("moveShortName"),
                    }
                )

        if sequence.get("didKill") and index == len(moves) - 1:
            exclusions.append(
                {
                    "frame": hit_frame,
                    "role": "post_hit_extension",
                    "reason": "terminal_kill_hit_already_connected",
                    "move": move.get("moveShortName"),
                }
            )
            continue
        points.append(
            {
                "frame": hit_frame,
                "role": "opening_hit" if index == 0 else "post_hit_extension",
                "branch_move_index": index,
                "followup_start_index": index + 1,
                "branch_move": move,
                "point_evidence": {"connected_hit_frame": hit_frame},
            }
        )

    terminal_kill_frame = _integer(moves[-1].get("frame")) if sequence.get("didKill") and moves else None
    for run in sequence.get("attackerActionRuns") or []:
        info = commitment_info(run)
        if info is None:
            continue
        kind, action_name = info
        run_start = _integer(run.get("startFrame"))
        run_end = _integer(run.get("endFrame"), run_start)
        if terminal_kill_frame is not None and run_start > terminal_kill_frame:
            exclusions.append(
                {
                    "frame": run_start,
                    "role": "attacker_commitment",
                    "reason": "post_kill_action_not_an_advantage_decision",
                    "move": action_name,
                }
            )
            continue
        associated = next(
            (
                index
                for index, move in enumerate(moves)
                if (
                    move.get("actionStartFrame") is not None
                    and abs(_integer(move.get("actionStartFrame")) - run_start) <= 1
                )
                or run_start <= _integer(move.get("frame")) <= run_end
                or (
                    str(move.get("moveShortName") or "").lower() == action_name
                    and run_start <= _integer(move.get("frame")) <= run_end + 2
                )
            ),
            None,
        )
        if associated is None and kind == "grab_throw_commitment":
            associated = next(
                (
                    index
                    for index, move in enumerate(moves)
                    if _integer(move.get("frame")) >= run_start
                    and _integer(move.get("frame")) - run_start <= 90
                    and str(move.get("moveShortName") or "").lower() in THROW_MOVES | {"pummel"}
                ),
                None,
            )
        if associated == 0:
            continue
        next_index = associated
        if next_index is None:
            next_index = next(
                (index for index, move in enumerate(moves) if _integer(move.get("frame")) >= run_start),
                len(moves),
            )
        role = "continuation_commitment" if associated is not None else "failed_extension_commitment"
        points.append(
            {
                "frame": run_start,
                "role": role,
                "branch_move_index": associated,
                "followup_start_index": next_index,
                "branch_move": moves[associated] if associated is not None else pseudo_move(run_start, action_name, sequence, labels),
                "point_evidence": {
                    "commitment_kind": kind,
                    "action_state_id": run.get("actionStateId"),
                    "action_state_name": run.get("actionStateName"),
                    "commitment_end_frame": run_end,
                    "connected_move_index": None if associated is None else associated + 1,
                },
            }
        )

    return dedupe_points(points, dedupe_frames), exclusions


ROLE_PRIORITY = {
    "opening_hit": 100,
    "continuation_commitment": 95,
    "failed_extension_commitment": 90,
    "conversion_commitment": 85,
    "post_hit_extension": 75,
}


def dedupe_points(points: list[dict[str, Any]], dedupe_frames: int) -> list[dict[str, Any]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda point: (point["frame"], -ROLE_PRIORITY.get(point["role"], 0)))
    clusters: list[list[dict[str, Any]]] = []
    for point in ordered:
        if not clusters or point["frame"] - clusters[-1][-1]["frame"] > dedupe_frames:
            clusters.append([point])
        else:
            clusters[-1].append(point)
    result = []
    for cluster in clusters:
        selected = max(
            cluster,
            key=lambda point: (
                ROLE_PRIORITY.get(point["role"], 0),
                -point["frame"],
            ),
        )
        selected = dict(selected)
        selected["merged_injection_points"] = [
            {"frame": point["frame"], "role": point["role"]}
            for point in cluster
            if point is not selected and (point["frame"] != selected["frame"] or point["role"] != selected["role"])
        ]
        result.append(selected)
    return result


def candidate_from_point(
    sequence: dict[str, Any],
    point: dict[str, Any],
    context: dict[str, Any],
    missed_reason: str | None,
) -> dict[str, Any]:
    moves = sequence.get("moves") or []
    followup_start = max(0, min(len(moves), _integer(point.get("followup_start_index"))))
    current_followup_moves = moves[followup_start:]
    later_horizon_moves = sequence.get("baseline_later_moves") or []
    followup_moves = current_followup_moves + later_horizon_moves
    followup_damage = sum(_number(move.get("damage")) for move in followup_moves)
    followup_hits = sum(max(1, _integer(move.get("hitCount"), 1)) for move in followup_moves)
    followup_kill = bool(
        (sequence.get("didKill") and followup_start < len(moves))
        or sequence.get("baseline_later_kill")
    )
    branch_move = point.get("branch_move") or moves[0]
    branch_name = branch_move.get("moveShortName") or branch_move.get("moveName") or "option"
    role = point["role"]
    if role in {"conversion_commitment", "continuation_commitment"}:
        label = f"advantage choice before {branch_name} at f{point['frame']}"
    elif role == "failed_extension_commitment":
        label = f"failed extension commitment ({branch_name}) at f{point['frame']}"
    else:
        label = f"combo extension after {branch_name} at f{point['frame']}"
    categories = list(context["categories"])
    if role != "opening_hit" and "punish_continuation" not in categories:
        categories.append("punish_continuation")
    if role == "failed_extension_commitment" and "dropped_extension" not in categories:
        categories.append("dropped_extension")
    if len(sequence.get("baseline_horizon_sequence_ids") or []) > 1 and "merged_advantage_horizon" not in categories:
        categories.append("merged_advantage_horizon")
    branch_index = point.get("branch_move_index")
    merged_points = [
        {"sequence_id": sequence["sequence_id"], **merged}
        for merged in point.get("merged_injection_points") or []
    ]
    return {
        "frame": int(point["frame"]),
        "takeover_frame": int(point["frame"]),
        "sequence_id": sequence["sequence_id"],
        "covered_sequence_ids": [sequence["sequence_id"]],
        "segment_start_frame": int(sequence["startFrame"]),
        "segment_end_frame": int(sequence["endFrame"]),
        "presentation_segment": sequence.get("presentation"),
        "opening_type": sequence.get("openingType") or "unknown",
        "opportunity_role": role,
        "primary_category": context["primary_category"],
        "categories": categories,
        "category_evidence": context["evidence"],
        "point_evidence": point.get("point_evidence") or {},
        "branch_move_index": None if branch_index is None else int(branch_index) + 1,
        "branch_move": branch_move,
        "opening_move": branch_move,
        "sequence_opening_move": moves[0],
        "original_move_sequence": moves,
        "baseline_horizon_sequence_ids": sequence.get("baseline_horizon_sequence_ids") or [sequence["sequence_id"]],
        "baseline_horizon_end_frame": sequence.get("baseline_horizon_end_frame", sequence["endFrame"]),
        "baseline_horizon_damage": round(_number(sequence.get("baseline_horizon_damage"), sequence.get("damage")), 3),
        "baseline_horizon_hit_events": _integer(sequence.get("baseline_horizon_hit_events"), len(moves)),
        "replay_baseline_move_sequence": followup_moves,
        "remaining_move_sequence": followup_moves,
        "original_hits": len(moves),
        "original_total_hit_count": sum(max(1, _integer(move.get("hitCount"), 1)) for move in moves),
        "original_damage": round(_number(sequence.get("damage")), 3),
        "original_kill": bool(sequence.get("didKill")),
        "original_followup_damage": round(followup_damage, 3),
        "original_followup_hits": followup_hits,
        "original_followup_kill": followup_kill,
        "original_followup_score": round(followup_damage + 6.0 * followup_hits + 100.0 * int(followup_kill), 3),
        "previously_missed_by_legacy": bool(missed_reason or role != "opening_hit"),
        "legacy_miss_reason": missed_reason if missed_reason else (None if role == "opening_hit" else "legacy_only_emitted_opening_hit"),
        "merged_injection_points": merged_points,
        "resolution_requirements": {
            "general": "End only on stock loss, reversal, or stable neutral after both players are actionable.",
            "edgeguard": "Landing onstage does not end the lane while the recovering player remains in hitstun, special-fall, landing, or other forced lag.",
        },
        "label": label,
    }


def candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
    category_bonus = {
        "edgeguard_recovery": 45.0,
        "tech_chase": 35.0,
        "platform_juggle": 30.0,
        "throw_route": 25.0,
        "reopening_extension": 20.0,
    }.get(candidate.get("primary_category"), 10.0)
    role_bonus = ROLE_PRIORITY.get(candidate.get("opportunity_role"), 0) / 10.0
    return (
        category_bonus + role_bonus + _number(candidate.get("original_followup_score")),
        _number(candidate.get("original_followup_damage")),
        _number(candidate.get("original_damage")),
        -_integer(candidate.get("frame")),
    )


def dedupe_candidates(candidates: list[dict[str, Any]], dedupe_frames: int) -> list[dict[str, Any]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda candidate: candidate["frame"])
    clusters: list[list[dict[str, Any]]] = []
    for candidate in ordered:
        if not clusters or candidate["frame"] - clusters[-1][-1]["frame"] > dedupe_frames:
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    result = []
    for cluster in clusters:
        selected_source = max(cluster, key=candidate_rank)
        selected = dict(selected_source)
        covered = set(selected.get("covered_sequence_ids") or [])
        merged = list(selected.get("merged_injection_points") or [])
        for candidate in cluster:
            if candidate is selected_source:
                continue
            covered.update(candidate.get("covered_sequence_ids") or [])
            merged.append(
                {
                    "frame": candidate["frame"],
                    "role": candidate.get("opportunity_role"),
                    "sequence_id": candidate.get("sequence_id"),
                    "label": candidate.get("label"),
                }
            )
        selected["covered_sequence_ids"] = sorted(covered)
        selected["merged_injection_points"] = merged
        result.append(selected)
    return sorted(result, key=lambda candidate: candidate["frame"])


def cap_candidates(candidates: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates
    best_by_sequence: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for sequence_id in candidate.get("covered_sequence_ids") or [candidate["sequence_id"]]:
            current = best_by_sequence.get(sequence_id)
            if current is None or candidate_rank(candidate) > candidate_rank(current):
                best_by_sequence[sequence_id] = candidate
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for candidate in sorted(best_by_sequence.values(), key=candidate_rank, reverse=True):
        identity = id(candidate)
        if identity in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(identity)
        if len(selected) >= max_candidates:
            return sorted(selected, key=lambda item: item["frame"])
    for candidate in sorted(candidates, key=candidate_rank, reverse=True):
        if id(candidate) in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(id(candidate))
        if len(selected) >= max_candidates:
            break
    return sorted(selected, key=lambda item: item["frame"])


def build_coverage(
    *,
    sequences: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    legacy: list[dict[str, Any]],
    sequence_exclusions: list[dict[str, Any]],
    point_exclusions: list[dict[str, Any]],
    raw_available: bool,
) -> dict[str, Any]:
    eligible = [sequence for sequence in sequences if sequence.get("eligible")]
    covered_ids = {
        sequence_id
        for candidate in candidates
        for sequence_id in candidate.get("covered_sequence_ids") or [candidate.get("sequence_id")]
        if sequence_id
    }
    category_counts = Counter()
    primary_counts = Counter()
    role_counts = Counter()
    sequence_category_ids: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        category_counts.update(candidate.get("categories") or [])
        primary_counts.update([candidate.get("primary_category") or "unknown"])
        role_counts.update([candidate.get("opportunity_role") or "unknown"])
        for category in candidate.get("categories") or []:
            for sequence_id in candidate.get("covered_sequence_ids") or [candidate.get("sequence_id")]:
                if sequence_id:
                    sequence_category_ids[category].add(sequence_id)
    missed = [
        {
            "sequence_id": sequence["sequence_id"],
            "start_frame": sequence["startFrame"],
            "end_frame": sequence["endFrame"],
            "opening_frame": (sequence.get("moves") or [{}])[0].get("frame"),
            "opening_move": (sequence.get("moves") or [{}])[0].get("moveShortName"),
            "damage": sequence.get("damage"),
            "hits": len(sequence.get("moves") or []),
            "did_kill": sequence.get("didKill"),
            "reason": sequence.get("legacy_miss_reason"),
        }
        for sequence in sequences
        if sequence.get("legacy_miss_reason")
    ]
    merged_horizon_groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for sequence in sequences:
        horizon_ids = tuple(sequence.get("baseline_horizon_sequence_ids") or [])
        if len(horizon_ids) <= 1:
            continue
        full_group = tuple(
            item["sequence_id"]
            for item in sequences
            if (item.get("presentation") or {}).get("startFrame") == (sequence.get("presentation") or {}).get("startFrame")
            and (item.get("presentation") or {}).get("endFrame") == (sequence.get("presentation") or {}).get("endFrame")
        )
        merged_horizon_groups.setdefault(full_group, []).append(sequence)
    per_sequence = []
    for sequence in sequences:
        sequence_candidates = [
            candidate
            for candidate in candidates
            if sequence["sequence_id"] in (candidate.get("covered_sequence_ids") or [])
        ]
        represented_frames = []
        for candidate in sequence_candidates:
            represented_frames.append(candidate["frame"])
            for merged in candidate.get("merged_injection_points") or []:
                merged_sequence = merged.get("sequence_id")
                if merged_sequence in {None, sequence["sequence_id"]}:
                    represented_frames.append(_integer(merged.get("frame")))
        hit_event_coverage = []
        sequence_moves = sequence.get("moves") or []
        for move_index, move in enumerate(sequence_moves):
            hit_frame = _integer(move.get("frame"))
            terminal_kill = bool(sequence.get("didKill") and move_index == len(sequence_moves) - 1)
            action_start = move.get("actionStartFrame")
            hit_event_coverage.append(
                {
                    "move_index": move_index + 1,
                    "move": move.get("moveShortName"),
                    "hit_frame": hit_frame,
                    "action_start_frame": action_start,
                    "contact_branch_covered": terminal_kill or any(abs(frame - hit_frame) <= 3 for frame in represented_frames),
                    "commitment_branch_covered": move_index == 0
                    or action_start is None
                    or any(abs(frame - _integer(action_start)) <= 3 for frame in represented_frames),
                    "terminal_contact_excluded": terminal_kill,
                }
            )
        per_sequence.append(
            {
                "sequence_id": sequence["sequence_id"],
                "start_frame": sequence["startFrame"],
                "end_frame": sequence["endFrame"],
                "opening_type": sequence.get("openingType"),
                "damage": sequence.get("damage"),
                "hit_events": len(sequence.get("moves") or []),
                "did_kill": sequence.get("didKill"),
                "eligible": bool(sequence.get("eligible")),
                "ineligible_reason": sequence.get("ineligible_reason"),
                "legacy_miss_reason": sequence.get("legacy_miss_reason"),
                "presentation_segment": sequence.get("presentation"),
                "baseline_horizon_sequence_ids": sequence.get("baseline_horizon_sequence_ids"),
                "baseline_horizon_end_frame": sequence.get("baseline_horizon_end_frame"),
                "baseline_horizon_damage": sequence.get("baseline_horizon_damage"),
                "baseline_horizon_hit_events": sequence.get("baseline_horizon_hit_events"),
                "candidate_frames": [candidate["frame"] for candidate in sequence_candidates],
                "candidate_roles": [candidate.get("opportunity_role") for candidate in sequence_candidates],
                "hit_event_coverage": hit_event_coverage,
                "covered": sequence["sequence_id"] in covered_ids,
            }
        )
    return {
        "universe_definition": "Every raw slippi-js conversion whose lastHitBy is the analyzed player; phase segments are presentation context only.",
        "raw_slippi_conversions_available": raw_available,
        "total_advantage_sequences": len(sequences),
        "eligible_sequences": len(eligible),
        "ineligible_sequences": len(sequences) - len(eligible),
        "covered_eligible_sequences": sum(sequence["sequence_id"] in covered_ids for sequence in eligible),
        "uncovered_sequences": [
            {
                "sequence_id": sequence["sequence_id"],
                "start_frame": sequence["startFrame"],
                "reason": "candidate cap or deduplication removed every branch point",
            }
            for sequence in eligible
            if sequence["sequence_id"] not in covered_ids
        ],
        "legacy": {
            "candidate_count": len(legacy),
            "covered_sequence_count": len(sequences) - len(missed),
            "opening_frames": [candidate["frame"] for candidate in legacy],
        },
        "expanded": {
            "candidate_count": len(candidates),
            "candidate_frames": [candidate["frame"] for candidate in candidates],
            "candidates_by_primary_category": dict(sorted(primary_counts.items())),
            "candidates_by_all_categories": dict(sorted(category_counts.items())),
            "sequences_by_category": {
                category: len(sequence_ids) for category, sequence_ids in sorted(sequence_category_ids.items())
            },
            "candidates_by_role": dict(sorted(role_counts.items())),
        },
        "previously_missed_sequences": missed,
        "merged_baseline_horizons": [
            {
                "sequence_ids": list(group_ids),
                "start_frame": min(item["startFrame"] for item in group),
                "end_frame": max(_integer(item.get("baseline_horizon_end_frame"), item["endFrame"]) for item in group),
                "linked_earlier_sequences": len(group),
            }
            for group_ids, group in sorted(merged_horizon_groups.items())
        ],
        "excluded_sequence_reasons": dict(sorted(Counter(item["reason"] for item in sequence_exclusions).items())),
        "excluded_injection_point_reasons": dict(sorted(Counter(item["reason"] for item in point_exclusions).items())),
        "excluded_sequences": sequence_exclusions,
        "excluded_injection_points": point_exclusions,
        "per_sequence": per_sequence,
        "deterministic_data_limits": [
            "Raw Slippi identifies conversions, hit events, action states, positions, percent, stocks, and controller state, but not semantic intent.",
            "Character-specific special-state names may remain numeric, so whiffed special-move commitments are not exhaustively classified without character frame data.",
            "Fountain of Dreams moving platform collision heights are not reconstructed by the coarse stage geometry classifier.",
            "A recovery touching stage is not proof of neutral; rollout termination must separately require actionability after landing or special-fall lag.",
        ],
    }


def render_coverage_markdown(payload: dict[str, Any]) -> str:
    coverage = payload["coverage"]
    expanded = coverage["expanded"]
    lines = [
        "# Advantage Candidate Coverage",
        "",
        f"Replay: `{payload.get('replay') or 'unavailable'}`",
        f"Analyzed port: P{payload['analyzed_port']} (player index {payload['analyzed_player_index']})",
        "",
        "## Before / After",
        "",
        "| Metric | Legacy | Expanded |",
        "|---|---:|---:|",
        f"| Candidate injection points | {coverage['legacy']['candidate_count']} | {expanded['candidate_count']} |",
        f"| Covered eligible sequences | {coverage['legacy']['covered_sequence_count']} | {coverage['covered_eligible_sequences']} |",
        "",
        f"Universe: {coverage['total_advantage_sequences']} analyzed-player advantage sequences; "
        f"{coverage['eligible_sequences']} eligible and {coverage['ineligible_sequences']} ineligible.",
        "",
        "## Candidate Categories",
        "",
    ]
    for category, count in expanded["candidates_by_primary_category"].items():
        lines.append(f"- `{category}`: {count}")
    lines.extend(["", "## Previously Missed Sequences", ""])
    missed = coverage["previously_missed_sequences"]
    if missed:
        lines.extend(["| Range | Opening | Hits | Damage | Kill | Reason |", "|---|---|---:|---:|---|---|"])
        for item in missed:
            lines.append(
                f"| f{item['start_frame']}-{item['end_frame']} | {item.get('opening_move') or 'unknown'} "
                f"(f{item.get('opening_frame')}) | {item['hits']} | {item['damage']:.2f} | "
                f"{'yes' if item['did_kill'] else 'no'} | `{item['reason']}` |"
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Merged Replay Baseline Horizons", ""])
    if coverage["merged_baseline_horizons"]:
        lines.extend(["| Range | Raw sequence branches | Earlier baselines extended |", "|---|---:|---:|"])
        for item in coverage["merged_baseline_horizons"]:
            lines.append(
                f"| f{item['start_frame']}-{item['end_frame']} | {len(item['sequence_ids'])} | "
                f"{item['linked_earlier_sequences']} |"
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Exclusions", ""])
    for reason, count in coverage["excluded_sequence_reasons"].items():
        lines.append(f"- Sequence `{reason}`: {count}")
    for reason, count in coverage["excluded_injection_point_reasons"].items():
        lines.append(f"- Injection point `{reason}`: {count}")
    if not coverage["excluded_sequence_reasons"] and not coverage["excluded_injection_point_reasons"]:
        lines.append("None.")
    lines.extend(["", "## Uncovered Eligible Sequences", ""])
    if coverage["uncovered_sequences"]:
        for item in coverage["uncovered_sequences"]:
            lines.append(f"- `{item['sequence_id']}` at f{item['start_frame']}: {item['reason']}")
    else:
        lines.append("None.")
    lines.extend(["", "## Deterministic Limits", ""])
    lines.extend(f"- {item}" for item in coverage["deterministic_data_limits"])
    return "\n".join(lines) + "\n"


def build_candidate_payload(
    *,
    timeline: dict[str, Any],
    raw_events: dict[str, Any] | None,
    analyzed_port: int,
    min_damage: float = 1.0,
    dedupe_frames: int = 3,
    max_candidates: int = 0,
) -> dict[str, Any]:
    settings = (raw_events or {}).get("settings") or timeline.get("settings") or {}
    analyzed_index = player_index_for_port(settings, analyzed_port)
    labels = player_labels(settings)
    raw_available = raw_events is not None
    conversions = (
        list(raw_events.get("conversions") or [])
        if raw_available
        else presentation_fallback_conversions(timeline, labels)
    )
    sequences = make_sequences(
        conversions,
        analyzed_index=analyzed_index,
        analyzed_port=analyzed_port,
        labels=labels,
    )
    map_presentation_context(sequences, timeline, analyzed_index)
    link_baseline_horizons(sequences)
    legacy = legacy_candidates(timeline, analyzed_index, min_damage)
    legacy_frames = {candidate["frame"] for candidate in legacy}
    geometry = stage_geometry_for_settings(settings) if stage_geometry_for_settings else None

    sequence_exclusions = []
    point_exclusions = []
    candidates = []
    for sequence in sequences:
        sequence["legacy_miss_reason"] = find_missed_reason(sequence, timeline, legacy_frames)
        eligible, reason = sequence_eligibility(
            sequence,
            min_damage=min_damage,
            all_conversions=conversions,
        )
        sequence["eligible"] = eligible
        sequence["ineligible_reason"] = reason
        if not eligible:
            sequence_exclusions.append(
                {
                    "sequence_id": sequence["sequence_id"],
                    "start_frame": sequence["startFrame"],
                    "end_frame": sequence["endFrame"],
                    "reason": reason,
                }
            )
            continue
        context = classify_sequence(sequence, geometry)
        sequence["context"] = context
        points, excluded = opportunity_points(sequence, labels=labels, dedupe_frames=dedupe_frames)
        for item in excluded:
            point_exclusions.append({"sequence_id": sequence["sequence_id"], **item})
        for point in points:
            candidates.append(
                candidate_from_point(
                    sequence,
                    point,
                    context,
                    sequence.get("legacy_miss_reason"),
                )
            )

    candidates = dedupe_candidates(candidates, dedupe_frames)
    candidates = cap_candidates(candidates, max_candidates)
    coverage = build_coverage(
        sequences=sequences,
        candidates=candidates,
        legacy=legacy,
        sequence_exclusions=sequence_exclusions,
        point_exclusions=point_exclusions,
        raw_available=raw_available,
    )
    return {
        "source": "raw-slippi-advantage-opportunities",
        "replay": (raw_events or {}).get("replay") or timeline.get("replay"),
        "analyzed_port": analyzed_port,
        "analyzed_player_index": analyzed_index,
        "stage": geometry,
        "candidate_count": len(candidates),
        "frames": candidates,
        "coverage": coverage,
    }


def main() -> int:
    args = parse_args()
    if args.dedupe_frames < 0:
        raise ValueError("--dedupe-frames must be non-negative")
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.timeline:
        timeline_path = args.timeline.resolve()
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        replay_value = timeline.get("replay")
        replay = Path(replay_value).resolve() if replay_value else None
    else:
        replay = args.replay.resolve()
        timeline_path = out.with_name(f"{out.stem}_phase_timeline.json")
        timeline = build_timeline(replay, timeline_path)

    raw_events = None
    raw_events_path = args.raw_events_out.resolve() if args.raw_events_out else out.with_name(f"{out.stem}_slippi_opportunities.json")
    if replay is not None and replay.exists():
        raw_events = extract_raw_conversions(replay)
        raw_events_path.parent.mkdir(parents=True, exist_ok=True)
        raw_events_path.write_text(json.dumps(raw_events, indent=2) + "\n", encoding="utf-8")

    payload = build_candidate_payload(
        timeline=timeline,
        raw_events=raw_events,
        analyzed_port=int(args.analyzed_port),
        min_damage=float(args.min_damage),
        dedupe_frames=int(args.dedupe_frames),
        max_candidates=int(args.max_candidates),
    )
    payload["timeline"] = str(timeline_path)
    payload["raw_events"] = str(raw_events_path) if raw_events is not None else None
    coverage_report = args.coverage_report.resolve() if args.coverage_report else out.with_name(f"{out.stem}_coverage.md")
    payload["coverage_report"] = str(coverage_report)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    coverage_report.parent.mkdir(parents=True, exist_ok=True)
    coverage_report.write_text(render_coverage_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "coverage_report": str(coverage_report),
                "candidate_count": len(payload["frames"]),
                "eligible_sequences": payload["coverage"]["eligible_sequences"],
                "covered_sequences": payload["coverage"]["covered_eligible_sequences"],
                "legacy_candidates": payload["coverage"]["legacy"]["candidate_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
