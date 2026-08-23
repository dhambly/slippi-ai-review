"""Encode decomp MSL observations as the review viewer's MSLTRACE1 format."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


KEYFRAME_INTERVAL = 60
INPUT_FIELDS = ["buttons", "mainX", "mainY", "cX", "cY", "l", "r"]
PLAYER_FIELDS = [
    "charId", "actionId", "actionFrame", "x", "y", "facing", "grounded",
    "percent", "shield", "stocks", "jumps", "hitlag", "hitstun", "hurtbox",
    "reflect", "fastfall", "shielding", "inHitstun", "powershield", "dead",
]


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return 0
        rounded = round(value, 6)
        return 0 if rounded == 0 else rounded
    return value


def _changes(previous: list[object] | None, current: list[object]) -> list[list[object]]:
    if previous is None:
        return [[index, value] for index, value in enumerate(current)]
    return [[index, value] for index, value in enumerate(current) if value != previous[index]]


def _source_slot(observation: np.void, source: int) -> np.void:
    for slot in observation["slots"]:
        if bool(slot["present"]) and int(slot["source_player"]) == source:
            return slot
    raise RuntimeError(f"trace observation is missing source player {source}")


def _input_row(player: np.void) -> list[object]:
    return [
        int(player["buttons"]),
        _json_value(float(np.float32(int(player["main_x"]) / 80.0))),
        _json_value(float(np.float32(int(player["main_y"]) / 80.0))),
        _json_value(float(np.float32(int(player["c_x"]) / 80.0))),
        _json_value(float(np.float32(int(player["c_y"]) / 80.0))),
        _json_value(float(np.float32(int(player["l"]) / 255.0))),
        _json_value(float(np.float32(int(player["r"]) / 255.0))),
    ]


def _player_row(slot: np.void) -> list[object]:
    hitstun = int(slot["hitstun"]) if "hitstun" in slot.dtype.names else 0
    return [
        int(slot["char_id"]),
        int(slot["action_id"]),
        _json_value(float(slot["action_frame"])),
        _json_value(float(slot["pos_x"])),
        _json_value(float(slot["pos_y"])),
        1 if bool(slot["facing"]) else -1,
        1 if bool(slot["on_ground"]) else 0,
        _json_value(float(slot["percent"])),
        _json_value(float(slot["shield_hp"])),
        int(slot["stocks"]),
        int(slot["jumps_left"]),
        int(slot["hitlag"]),
        hitstun,
        int(slot["hurtbox_state"]),
        0,
        0,
        0,
        1 if hitstun > 0 else 0,
        0,
        1 if int(slot["stocks"]) <= 0 else 0,
    ]


def encode_trace(
    observations: list[np.void],
    inputs: list[np.void],
    *,
    num_players: int,
    source: str,
    replay: str,
    sim_start_frame: int,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not observations or len(observations) != len(inputs):
        raise ValueError("trace observations and inputs must be non-empty and equal length")
    frame_rows: list[list[object]] = []
    input_streams: list[list[list[object]]] = [[] for _ in range(num_players)]
    previous_players: list[list[object]] | None = None
    previous_inputs: list[list[object] | None] = [None] * num_players
    death_frames: list[int | None] = [None] * num_players
    first = observations[0]
    player_metadata = []
    for source_player in range(num_players):
        slot = _source_slot(first, source_player)
        player_metadata.append({"port": source_player + 1, "charId": int(slot["char_id"]), "teamId": source_player})

    for trace_frame, (observation, frame_input) in enumerate(zip(observations, inputs, strict=True)):
        player_rows = []
        for source_player in range(num_players):
            slot = _source_slot(observation, source_player)
            player_rows.append(_player_row(slot))
            if death_frames[source_player] is None and int(slot["stocks"]) <= 0:
                death_frames[source_player] = trace_frame
        keyframe = trace_frame == 0 or trace_frame % KEYFRAME_INTERVAL == 0
        if keyframe or previous_players is None:
            frame_rows.append([0, trace_frame, None, player_rows])
        else:
            frame_rows.append([
                1,
                trace_frame,
                None,
                [_changes(previous_players[player], player_rows[player]) for player in range(num_players)],
            ])
        previous_players = player_rows
        for source_player in range(num_players):
            row = _input_row(frame_input["players"][source_player])
            changes = _changes(previous_inputs[source_player], row)
            if keyframe or previous_inputs[source_player] is None:
                input_streams[source_player].append([0, trace_frame, row])
            elif changes:
                input_streams[source_player].append([1, trace_frame, changes])
            previous_inputs[source_player] = row

    trace_metadata = {
        "source": source,
        "backend": "decomp",
        "replay": replay,
        "traceFrameToGameFrame": {"offset": sim_start_frame},
        "deathTraceFrames": death_frames,
        **(metadata or {}),
    }
    return {
        "format": "MSLTRACE1",
        "schemaVersion": 1,
        "producer": {"name": "slippi-ai-review-decomp", "version": None},
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "match": {
            "stageId": int(first["stage_id"]),
            "numPlayers": num_players,
            "isTeams": bool(first["is_teams"]),
            "players": player_metadata,
            "startFrame": 0,
            "start": {
                "mode": "replay-exact-decomp",
                "traceFrame": 0,
                "simFrameId": sim_start_frame,
                "dataset": replay,
            },
        },
        "inputs": {
            "encoding": "sparse-delta-v1",
            "keyframeInterval": KEYFRAME_INTERVAL,
            "fields": INPUT_FIELDS,
            "players": input_streams,
        },
        "frames": {
            "encoding": "sparse-delta-v1",
            "keyframeInterval": KEYFRAME_INTERVAL,
            "fields": ["frame", "randomSeed", "players"],
            "playerFields": PLAYER_FIELDS,
            "rows": frame_rows,
        },
        "items": {
            "encoding": "sparse-delta-v1",
            "keyframeInterval": KEYFRAME_INTERVAL,
            "fields": [
                "alive", "typeId", "state", "owner", "x", "y", "vx", "vy",
                "facing", "damage", "timer", "spawnId", "misc0", "misc1", "misc2",
            ],
            "rows": [],
        },
        "metadata": trace_metadata,
    }
