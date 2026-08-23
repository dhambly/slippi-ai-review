"""Export one selected decomp controller stream as an MSLTRACE1 trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .decomp_trace_common import encode_trace
from .lane_trace import full_lane_from_queue, host_path, target_from_queue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", type=Path, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--msl-root", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    import melee_sim as msl
    from tools.migration.replay_checkpoint import extract_replay_authority, warm_replay_checkpoint

    queue, target = target_from_queue(args.queue_json.resolve(), args.target_index)
    lane = full_lane_from_queue(queue, target.get("representative_lane") or {})
    stream = lane.get("controllerStream") or {}
    if stream.get("format") != "decomp-replay-input-v1":
        raise SystemExit("selected route is not a decomp controller stream")
    stream_path = host_path(str(stream["path"]))
    data = np.load(stream_path, allow_pickle=False)
    stored = data[str(stream.get("array") or "inputs")]
    env_index = int(stream["env"])
    frame_limit = min(
        int(stream.get("frames") or stored.shape[0]),
        int(args.max_frames if args.max_frames is not None else stored.shape[0]),
        int(stored.shape[0]),
    )
    replay = Path(str(queue["replay"])).resolve()
    authority = extract_replay_authority(replay)
    start_record = int(lane.get("startRecord") if lane.get("startRecord") is not None else authority.frame_index(int(stream["takeoverFrame"])))
    checkpoint = warm_replay_checkpoint(authority, int(authority.frame_ids[start_record]), data_dir=args.msl_root / "data")
    num_players = int(authority.config[0]["match"]["num_players"])
    observations: list[np.void] = []
    inputs: list[np.void] = []
    with msl.EnvBatch(batch_size=1, length=256, data_dir=args.msl_root / "data", action_format="raw") as env:
        env.reset_replay(authority.config, authority.previous_input)
        env.restore(0, checkpoint.state)
        env.observe_viewpoints(np.zeros(1, dtype=np.uint8))
        observations.append(env.current_frame[0].copy())
        inputs.append(stored[0, env_index].copy())
        for step in range(frame_limit):
            record = min(start_record + step + 1, len(authority.frames) - 1)
            replay_frame = authority.frames[record : record + 1].copy()
            replay_frame["input"]["players"][0] = stored[step, env_index]["players"]
            if env.t == env.length:
                env.reset_cursor()
            env.step_replay(replay_frame)
            if env.t == env.length:
                env.reset_cursor()
            env.observe_viewpoints(np.zeros(1, dtype=np.uint8))
            observations.append(env.current_frame[0].copy())
            inputs.append(stored[min(step + 1, frame_limit - 1), env_index].copy())
    trace = encode_trace(
        observations,
        inputs,
        num_players=num_players,
        source="decomp_representative_lane",
        replay=str(replay),
        sim_start_frame=int(stream["takeoverFrame"]),
        metadata={
            "queue": str(queue.get("source_run_dir") or ""),
            "targetIndex": args.target_index,
            "laneId": int(lane["laneId"]),
            "takeoverFrame": int(stream["takeoverFrame"]),
            "modelControlFrame": int(stream.get("modelControlFrame") or stream["takeoverFrame"]),
            "defenderTakeoverFrame": int(stream.get("defenderTakeoverFrame") or -1),
        },
    )
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(trace, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "frames": len(observations), "backend": "decomp"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
