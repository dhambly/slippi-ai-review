"""Export an MSL representative lane as an MSLTRACE1 viewer trace.

This is for visualizing the simulator's actual rollout, not replaying an
artificial SLP through Dolphin. It uses the same queue/representative lane
metadata as ``render_msl_lane_target.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_settings
from .paths import PROJECT_DIR


WORK_DIR = PROJECT_DIR
DEFAULT_MSL_ROOT = load_settings().msl_root
KEYFRAME_INTERVAL = 60

INPUT_FIELDS = ["buttons", "mainX", "mainY", "cX", "cY", "l", "r"]
PLAYER_FIELDS = [
    "charId",
    "actionId",
    "actionFrame",
    "x",
    "y",
    "facing",
    "grounded",
    "percent",
    "shield",
    "stocks",
    "jumps",
    "hitlag",
    "hitstun",
    "hurtbox",
    "reflect",
    "fastfall",
    "shielding",
    "inHitstun",
    "powershield",
    "dead",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def host_path(raw_path: str) -> Path:
    if os.name == "nt" and raw_path.startswith("/mnt/") and len(raw_path) >= 7 and raw_path[6] == "/":
        drive = raw_path[5].upper()
        rest = raw_path[7:].replace("/", "\\")
        return Path(f"{drive}:\\{rest}")
    return Path(raw_path)


def setup_msl(msl_root: Path) -> None:
    root = msl_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"MSL root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)


def target_from_queue(queue_path: Path, target_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    queue = load_json(queue_path)
    targets = queue.get("targets") or []
    if target_index < 1 or target_index > len(targets):
        raise SystemExit(f"--target-index {target_index} outside target range 1..{len(targets)}")
    return queue, targets[target_index - 1]


def full_lane_from_queue(queue: dict[str, Any], compact_lane: dict[str, Any]) -> dict[str, Any]:
    if compact_lane.get("startRecord") is not None:
        return compact_lane
    lane_id = compact_lane.get("laneId")
    lanes_path = queue.get("lanes_jsonl")
    if lane_id is None or not lanes_path:
        return compact_lane
    path = Path(lanes_path)
    if not path.exists():
        path = host_path(str(lanes_path))
    if not path.exists():
        return compact_lane
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if int(row.get("laneId") or -1) == int(lane_id):
                return row
    return compact_lane


def u8_rows(arr: np.ndarray) -> np.ndarray:
    return arr.view(np.uint8).reshape(arr.shape[0], arr.dtype.itemsize)


def input_for_source(frame_input: np.void, source_player: int) -> np.void:
    return frame_input["p"][source_player]


def slot_for_source(frame: np.ndarray, env: int, source_player: int) -> np.void | None:
    slots = frame["slots"][env]
    for slot in slots:
        if bool(slot["present"]) and int(slot["source_player"]) == int(source_player):
            return slot
    return None


def json_value(value: object) -> object:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if value != value:
            return 0
        rounded = round(value, 6)
        return 0 if rounded == 0 else rounded
    return value


def changed_fields(previous: list[object] | None, current: list[object]) -> list[list[object]]:
    if previous is None:
        return [[idx, value] for idx, value in enumerate(current)]
    return [[idx, value] for idx, value in enumerate(current) if value != previous[idx]]


def input_row(inp: np.void | None) -> list[object]:
    if inp is None:
        return [0, 0, 0, 0, 0, 0, 0]
    return [
        int(inp["buttons"]),
        json_value(float(np.float32(int(inp["main_x"]) / 80.0))),
        json_value(float(np.float32(int(inp["main_y"]) / 80.0))),
        json_value(float(np.float32(int(inp["c_x"]) / 80.0))),
        json_value(float(np.float32(int(inp["c_y"]) / 80.0))),
        json_value(float(np.float32(int(inp["l"]) / 255.0))),
        json_value(float(np.float32(int(inp["r"]) / 255.0))),
    ]


def player_row(slot: np.void) -> list[object]:
    # Some optional flags are not exposed by the validation dtype used here.
    hitstun = 1 if int(slot["action_id"]) in range(75, 92) else 0
    return [
        int(slot["char_id"]),
        int(slot["action_id"]),
        json_value(float(np.float32(slot["action_frame"]))),
        json_value(float(np.float32(slot["pos_x"]))),
        json_value(float(np.float32(slot["pos_y"]))),
        1 if int(slot["facing"]) else -1,
        1 if int(slot["on_ground"]) else 0,
        json_value(float(np.float32(slot["percent"]))),
        json_value(float(np.float32(slot["shield_hp"]))),
        int(slot["stocks"]),
        int(slot["jumps_left"]),
        int(slot["hitlag"]),
        hitstun,
        int(slot["hurtbox_state"]),
        0,
        0,
        0,
        hitstun,
        0,
        1 if int(slot["stocks"]) <= 0 else 0,
    ]


def build_trace(
    *,
    queue: dict[str, Any],
    lane: dict[str, Any],
    stream: dict[str, Any],
    stream_path: Path,
    env: int,
    frame_limit: int,
    msl_root: Path,
) -> dict[str, Any]:
    replay = Path(queue["replay"]).resolve()
    start_record = int(lane.get("startRecord"))
    takeover_frame = int(stream["takeoverFrame"])
    input_data = np.load(stream_path, allow_pickle=False)
    inputs = input_data[str(stream.get("array") or "inputs")]
    frame_limit = max(
        0,
        min(
            frame_limit,
            int(inputs.shape[0]),
            int(stream.get("frames") or inputs.shape[0]),
        ),
    )

    setup_msl(msl_root)
    import melee_sim.dtypes as msl_dtypes  # type: ignore
    import msl_binding  # type: ignore
    from tools.eval.validation_dtypes import INPUT_DTYPE  # type: ignore
    from tools.slippi.validation_buffer_builder import build_validation_buffers_from_slp  # type: ignore

    buffers = build_validation_buffers_from_slp(slp_path=str(replay))
    num_players = int(buffers.num_players)
    gamestate_dtype = msl_dtypes.gamestate_dtype()
    handle = msl_binding.init(1, num_players)
    viewpoint = np.zeros(1, dtype=np.uint8)
    gamestate_raw = np.zeros((1, gamestate_dtype.itemsize), dtype=np.uint8)
    gamestate = gamestate_raw.view(gamestate_dtype).reshape(1)
    prev_input = buffers.prev_input_t[[start_record]].copy()

    frame_rows: list[list[object]] = []
    input_streams: list[list[list[object]]] = [[] for _ in range(num_players)]
    prev_players: list[list[object]] | None = None
    prev_inputs: list[list[object] | None] = [None for _ in range(num_players)]
    first_players_meta: list[dict[str, int]] | None = None
    death_trace_frames: list[int | None] = [None for _ in range(num_players)]
    stage_id: int | None = None

    def append_current(trace_frame: int, frame_input: np.void | None) -> None:
        nonlocal prev_players, first_players_meta, stage_id
        if stage_id is None:
            stage_id = int(gamestate["stage_id"][0])
        player_rows = []
        player_meta = []
        for source in range(num_players):
            slot = slot_for_source(gamestate, 0, source)
            if slot is None:
                raise RuntimeError(f"missing source player {source} in gamestate")
            player_rows.append(player_row(slot))
            player_meta.append({"port": source + 1, "charId": int(slot["char_id"]), "teamId": source})
            if death_trace_frames[source] is None and 0 <= int(slot["action_id"]) <= 10:
                death_trace_frames[source] = trace_frame

        if first_players_meta is None:
            first_players_meta = player_meta

        is_keyframe = trace_frame == 0 or trace_frame % KEYFRAME_INTERVAL == 0
        if is_keyframe or prev_players is None:
            frame_rows.append([0, trace_frame, None, player_rows])
        else:
            frame_rows.append(
                [
                    1,
                    trace_frame,
                    None,
                    [changed_fields(prev_players[idx], player_rows[idx]) for idx in range(num_players)],
                ]
            )
        prev_players = player_rows

        for source in range(num_players):
            row = input_row(input_for_source(frame_input, source) if frame_input is not None else None)
            previous = prev_inputs[source]
            changes = changed_fields(previous, row)
            if is_keyframe or previous is None:
                input_streams[source].append([0, trace_frame, row])
            elif changes:
                input_streams[source].append([1, trace_frame, changes])
            prev_inputs[source] = row

    try:
        msl_binding.reseed_seed_rollout(handle, u8_rows(buffers.seed_t[[start_record]]))
        msl_binding.write_gamestate(handle, viewpoint, gamestate_raw)
        append_current(0, inputs[0, env] if frame_limit > 0 else None)
        for step in range(frame_limit):
            current_input = np.zeros(1, dtype=INPUT_DTYPE)
            current_input[0] = inputs[step, env]
            replay_record = min(start_record + step, int(buffers.num_records) - 1)
            msl_binding.step_input_replay_frame_rng(
                handle,
                u8_rows(buffers.seed_t[[replay_record]]),
                u8_rows(prev_input),
                u8_rows(current_input),
            )
            msl_binding.write_gamestate(handle, viewpoint, gamestate_raw)
            next_input = inputs[min(step + 1, frame_limit - 1), env] if frame_limit > 0 else None
            append_current(step + 1, next_input)
            prev_input = current_input.copy()
    finally:
        msl_binding.destroy(handle)
        os.chdir(WORK_DIR)

    if first_players_meta is None or stage_id is None:
        raise RuntimeError("trace has no frames")

    return {
        "format": "MSLTRACE1",
        "schemaVersion": 1,
        "producer": {"name": "codex-msl-lane-export", "version": None},
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "match": {
            "stageId": stage_id,
            "numPlayers": num_players,
            "isTeams": False,
            "players": first_players_meta,
            "startFrame": 0,
            "start": {
                "mode": "replay-reseed-msl-lane",
                "traceFrame": 0,
                "simFrameId": takeover_frame,
                "dataset": str(replay),
                "startRecord": start_record,
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
                "alive",
                "typeId",
                "state",
                "owner",
                "x",
                "y",
                "vx",
                "vy",
                "facing",
                "damage",
                "timer",
                "spawnId",
                "misc0",
                "misc1",
                "misc2",
            ],
            "rows": [],
        },
        "metadata": {
            "source": "msl_representative_lane",
            "queue": str(queue.get("source_run_dir") or ""),
            "replay": str(replay),
            "targetIndex": int(stream.get("targetIndex") or 0),
            "laneId": int(lane.get("laneId")),
            "takeoverFrame": takeover_frame,
            "defenderTakeoverFrame": int(stream.get("defenderTakeoverFrame") or -1),
            "traceFrameToGameFrame": {"offset": takeover_frame},
            "deathTraceFrames": death_trace_frames,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-json", required=True, type=Path)
    parser.add_argument("--target-index", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--msl-root", type=Path, default=DEFAULT_MSL_ROOT)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    queue, target = target_from_queue(args.queue_json.resolve(), args.target_index)
    lane = full_lane_from_queue(queue, target.get("representative_lane") or {})
    stream = lane.get("controllerStream") or {}
    if not stream:
        raise SystemExit("target has no controllerStream")
    stream_path = host_path(str(stream["path"]))
    if not stream_path.exists():
        raise SystemExit(f"controller stream missing: {stream_path}")
    stream_data = np.load(stream_path, allow_pickle=False)
    inputs = stream_data[str(stream.get("array") or "inputs")]
    frame_limit = int(args.max_frames if args.max_frames is not None else inputs.shape[0])
    trace = build_trace(
        queue=queue,
        lane=lane,
        stream=stream,
        stream_path=stream_path,
        env=int(stream["env"]),
        frame_limit=frame_limit,
        msl_root=args.msl_root,
    )
    trace["metadata"]["targetIndex"] = int(args.target_index)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "frames": len(trace["frames"]["rows"])}, indent=2))


if __name__ == "__main__":
    main()
