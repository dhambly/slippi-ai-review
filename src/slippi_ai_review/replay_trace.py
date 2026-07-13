"""Export an original Slippi replay window as an MSLTRACE1 viewer trace."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .lane_trace import (
    INPUT_FIELDS,
    KEYFRAME_INTERVAL,
    PLAYER_FIELDS,
    WORK_DIR,
    changed_fields,
    input_for_source,
    input_row,
    player_row,
    setup_msl,
    slot_for_source,
    u8_rows,
)
from .config import load_settings


DEFAULT_MSL_ROOT = load_settings().msl_root


def record_for_frame(frame_ids: np.ndarray, frame: int) -> int:
    return int(np.argmin(np.abs(frame_ids.astype(np.int64) - int(frame))))


def build_trace(*, replay: Path, start_frame: int, end_frame: int, msl_root: Path) -> dict[str, Any]:
    setup_msl(msl_root)
    import melee_sim.dtypes as msl_dtypes  # type: ignore
    import msl_binding  # type: ignore
    from tools.slippi.validation_buffer_builder import build_validation_buffers_from_slp  # type: ignore

    buffers = build_validation_buffers_from_slp(slp_path=str(replay))
    frame_ids = np.asarray(buffers.seed_t["frame_id"], dtype=np.int32)
    start_record = record_for_frame(frame_ids, start_frame)
    end_record = record_for_frame(frame_ids, end_frame)
    if end_record < start_record:
        raise ValueError(f"end frame {end_frame} precedes start frame {start_frame}")

    num_players = int(buffers.num_players)
    gamestate_dtype = msl_dtypes.gamestate_dtype()
    handle = msl_binding.init(1, num_players)
    viewpoint = np.zeros(1, dtype=np.uint8)
    gamestate_raw = np.zeros((1, gamestate_dtype.itemsize), dtype=np.uint8)
    gamestate = gamestate_raw.view(gamestate_dtype).reshape(1)
    frame_rows: list[list[object]] = []
    input_streams: list[list[list[object]]] = [[] for _ in range(num_players)]
    previous_players: list[list[object]] | None = None
    previous_inputs: list[list[object] | None] = [None for _ in range(num_players)]
    players_meta: list[dict[str, int]] | None = None
    stage_id: int | None = None

    try:
        for trace_frame, record in enumerate(range(start_record, end_record + 1)):
            msl_binding.reseed_seed_rollout(handle, u8_rows(buffers.seed_t[[record]]))
            msl_binding.write_gamestate(handle, viewpoint, gamestate_raw)
            if stage_id is None:
                stage_id = int(gamestate["stage_id"][0])
            current_players = []
            current_meta = []
            for source in range(num_players):
                slot = slot_for_source(gamestate, 0, source)
                if slot is None:
                    raise RuntimeError(f"missing source player {source} at record {record}")
                current_players.append(player_row(slot))
                current_meta.append({"port": source + 1, "charId": int(slot["char_id"]), "teamId": source})
            if players_meta is None:
                players_meta = current_meta

            keyframe = trace_frame == 0 or trace_frame % KEYFRAME_INTERVAL == 0
            if keyframe or previous_players is None:
                frame_rows.append([0, trace_frame, None, current_players])
            else:
                frame_rows.append([
                    1,
                    trace_frame,
                    None,
                    [changed_fields(previous_players[idx], current_players[idx]) for idx in range(num_players)],
                ])
            previous_players = current_players

            frame_input = buffers.input_t[record]
            for source in range(num_players):
                current_input = input_row(input_for_source(frame_input, source))
                changes = changed_fields(previous_inputs[source], current_input)
                if keyframe or previous_inputs[source] is None:
                    input_streams[source].append([0, trace_frame, current_input])
                elif changes:
                    input_streams[source].append([1, trace_frame, changes])
                previous_inputs[source] = current_input
    finally:
        msl_binding.destroy(handle)
        os.chdir(WORK_DIR)

    if stage_id is None or players_meta is None:
        raise RuntimeError("replay window produced no trace frames")
    actual_start_frame = int(frame_ids[start_record])
    return {
        "format": "MSLTRACE1",
        "schemaVersion": 1,
        "producer": {"name": "codex-slp-window-export", "version": None},
        "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "match": {
            "stageId": stage_id,
            "numPlayers": num_players,
            "isTeams": False,
            "players": players_meta,
            "startFrame": 0,
            "start": {
                "mode": "slippi-replay-window",
                "traceFrame": 0,
                "simFrameId": actual_start_frame,
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
                "alive", "typeId", "state", "owner", "x", "y", "vx", "vy",
                "facing", "damage", "timer", "spawnId", "misc0", "misc1", "misc2",
            ],
            "rows": [],
        },
        "metadata": {
            "source": "slippi_replay_window",
            "replay": str(replay),
            "startFrame": actual_start_frame,
            "endFrame": int(frame_ids[end_record]),
            "traceFrameToGameFrame": {"offset": actual_start_frame},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--msl-root", type=Path, default=DEFAULT_MSL_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    replay = args.replay.resolve()
    trace = build_trace(
        replay=replay,
        start_frame=int(args.start_frame),
        end_frame=int(args.end_frame),
        msl_root=args.msl_root,
    )
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(trace, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "frames": len(trace["frames"]["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
