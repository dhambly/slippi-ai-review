"""Build compact move-event lanes from MSLTRACE1 sparse frame data."""

from __future__ import annotations

from typing import Any


CHARACTER_NAMES = [
    "Mario", "Fox", "Captain Falcon", "Donkey Kong", "Kirby", "Bowser", "Link", "Sheik",
    "Ness", "Peach", "Popo", "Nana", "Pikachu", "Samus", "Yoshi", "Jigglypuff", "Mewtwo",
    "Luigi", "Marth", "Zelda", "Young Link", "Dr. Mario", "Falco", "Pichu", "Mr. Game & Watch",
    "Ganondorf", "Roy",
]

ACTION_NAMES = {
    44: "Jab", 45: "Jab 2", 46: "Jab 3",
    47: "Rapid jab", 48: "Rapid jab", 49: "Rapid jab", 50: "Dash attack",
    51: "F-tilt", 52: "F-tilt", 53: "F-tilt", 54: "F-tilt", 55: "F-tilt",
    56: "Up tilt", 57: "Down tilt", 58: "F-smash", 59: "F-smash", 60: "F-smash",
    61: "F-smash", 62: "F-smash", 63: "Up smash", 64: "Down smash", 65: "Nair",
    66: "Fair", 67: "Bair", 68: "Uair", 69: "Dair", 212: "Grab", 214: "Dash grab", 217: "Pummel",
    219: "Forward throw", 220: "Back throw", 221: "Up throw", 222: "Down throw",
    256: "Ledge attack", 257: "Ledge attack",
}

FOX_SPECIALS = {
    369: "Laser", 370: "Laser", 371: "Laser", 372: "Laser", 373: "Laser",
    374: "Air laser", 375: "Air laser", 376: "Air laser", 377: "Air laser", 378: "Air laser",
    379: "Side-B", 380: "Side-B", 381: "Up-B", 382: "Up-B",
    383: "Shine", 384: "Shine", 385: "Shine",
}

SHEIK_SPECIALS = {
    341: "Needles", 342: "Needles", 343: "Needles", 344: "Needles", 345: "Air needles",
    346: "Needles", 347: "Needles", 348: "Needles", 349: "Side-B", 350: "Side-B",
    351: "Side-B", 352: "Side-B", 353: "Side-B", 354: "Side-B", 355: "Up-B",
    356: "Up-B", 357: "Up-B", 358: "Up-B", 359: "Up-B", 360: "Up-B",
    361: "Down-B", 362: "Down-B", 363: "Down-B", 364: "Down-B",
}


def action_name(action_id: int, character_id: int) -> str | None:
    if character_id == 1 and action_id in FOX_SPECIALS:
        return FOX_SPECIALS[action_id]
    if character_id == 7 and action_id in SHEIK_SPECIALS:
        return SHEIK_SPECIALS[action_id]
    return ACTION_NAMES.get(action_id)


def action_kind(action_id: int) -> str:
    return "attack"


def _apply_player_row(state: list[Any] | None, row: Any, *, keyframe: bool, field_count: int) -> list[Any]:
    if keyframe:
        return list(row or [None] * field_count)
    next_state = list(state or [None] * field_count)
    for change in row or []:
        if not isinstance(change, list) or len(change) != 2:
            continue
        field, value = int(change[0]), change[1]
        if 0 <= field < field_count:
            next_state[field] = value
    return next_state


def trace_events(trace: dict[str, Any], *, frame_offset: int = 0) -> list[dict[str, Any]]:
    frames = trace.get("frames") or {}
    fields = list(frames.get("playerFields") or [])
    rows = list(frames.get("rows") or [])
    if "actionId" not in fields or "charId" not in fields:
        return []
    action_index = fields.index("actionId")
    character_index = fields.index("charId")
    states: list[list[Any] | None] = []
    active: list[dict[str, Any] | None] = []
    events: list[dict[str, Any]] = []
    last_frame = 0

    for row in rows:
        if not isinstance(row, list) or len(row) < 4:
            continue
        keyframe = int(row[0]) == 0
        trace_frame = int(row[1])
        last_frame = max(last_frame, trace_frame)
        for player, player_row in enumerate(row[3] or []):
            while len(states) <= player:
                states.append(None)
                active.append(None)
            states[player] = _apply_player_row(
                states[player], player_row, keyframe=keyframe, field_count=len(fields)
            )
            state = states[player]
            if state is None:
                continue
            action_id = int(state[action_index] or 0)
            character_id = int(state[character_index] or 0)
            prior = active[player]
            if prior and prior["actionId"] == action_id:
                continue
            if prior:
                prior["endFrame"] = frame_offset + trace_frame
            name = action_name(action_id, character_id)
            active[player] = None
            if name is None:
                continue
            event = {
                "frame": frame_offset + trace_frame,
                "endFrame": frame_offset + trace_frame + 1,
                "name": name,
                "player": player,
                "kind": action_kind(action_id),
                "character": CHARACTER_NAMES[character_id] if 0 <= character_id < len(CHARACTER_NAMES) else "",
                "actionId": action_id,
            }
            events.append(event)
            active[player] = event

    terminal = frame_offset + last_frame + 1
    for event in active:
        if event:
            event["endFrame"] = terminal
    return events


def timeline_payload(
    replay_trace: dict[str, Any], agent_trace: dict[str, Any], *, switch_frame: int
) -> dict[str, Any]:
    replay_events = trace_events(replay_trace)
    agent_events = trace_events(agent_trace, frame_offset=switch_frame)
    inherited = {
        (event["player"], event["actionId"])
        for event in replay_events
        if event["frame"] < switch_frame < event["endFrame"]
    }
    agent_events = [
        event
        for event in agent_events
        if not (
            event["frame"] == switch_frame
            and (event["player"], event["actionId"]) in inherited
        )
    ]
    return {
        "schemaVersion": 1,
        "lanes": {
            "replay": replay_events,
            "msl": agent_events,
        },
    }
