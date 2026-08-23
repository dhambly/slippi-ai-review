"""Export an exact replay window through the decomp MSL backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .decomp_trace_common import encode_trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--msl-root", type=Path, required=True)
    args = parser.parse_args()
    if args.end_frame < args.start_frame:
        raise SystemExit("--end-frame must be >= --start-frame")

    import melee_sim as msl
    from tools.migration.replay_checkpoint import extract_replay_authority, warm_replay_checkpoint

    replay = args.replay.resolve()
    authority = extract_replay_authority(replay)
    start_record = authority.frame_index(args.start_frame)
    end_record = authority.frame_index(args.end_frame)
    checkpoint = warm_replay_checkpoint(authority, args.start_frame, data_dir=args.msl_root / "data")
    num_players = int(authority.config[0]["match"]["num_players"])
    observations: list[np.void] = []
    inputs: list[np.void] = []
    with msl.EnvBatch(batch_size=1, length=256, data_dir=args.msl_root / "data", action_format="raw") as env:
        env.reset_replay(authority.config, authority.previous_input)
        env.restore(0, checkpoint.state)
        env.observe_viewpoints(np.zeros(1, dtype=np.uint8))
        observations.append(env.current_frame[0].copy())
        inputs.append(authority.frames[start_record]["input"].copy())
        for record in range(start_record + 1, end_record + 1):
            if env.t == env.length:
                env.reset_cursor()
            env.step_replay(authority.frames[record : record + 1])
            if env.t == env.length:
                env.reset_cursor()
            env.observe_viewpoints(np.zeros(1, dtype=np.uint8))
            observations.append(env.current_frame[0].copy())
            inputs.append(authority.frames[record]["input"].copy())
    trace = encode_trace(
        observations,
        inputs,
        num_players=num_players,
        source="decomp_exact_replay",
        replay=str(replay),
        sim_start_frame=args.start_frame,
        metadata={"startFrame": args.start_frame, "endFrame": args.end_frame},
    )
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(json.dumps(trace, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "frames": len(observations), "backend": "decomp"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
