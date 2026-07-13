"""Run many replay-seeded MSL Phillip takeover rollouts in fused batches.

This is the high-volume counterpart to ``run_msl_replay_takeover.py``. It fans
out a grid of injection points and stochastic samples from one Slippi replay,
then runs the lanes through one fused Phillip batch per chunk.

The intended punish-analysis shape is:

* analyzed port: Phillip controls from the lane's takeover frame.
* defender port: replay inputs continue for ``--defender-delay-frames`` frames,
  then Phillip takes over too.
* both ports are fed model observations from the start so delayed outputs are
  already queued when control switches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from .config import load_settings
except ImportError:
    load_settings = None


_SETTINGS = load_settings() if load_settings else None
DEFAULT_MSL_ROOT = _SETTINGS.msl_root if _SETTINGS else None
DEFAULT_SLIPPI_AI_ROOT = _SETTINGS.slippi_ai_root if _SETTINGS else None
DEFAULT_MODEL = _SETTINGS.model if _SETTINGS else None


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class LaneSpec:
    lane_id: int
    base_frame: int
    offset: int
    sample_index: int
    start_record: int
    takeover_frame: int
    defender_takeover_frame: int


def parse_csv_ints(value: str, *, name: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be comma-separated ints, got {value!r}") from exc
    if not out:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one int")
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a batched replay-seeded Phillip takeover grid through MSL.")
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--analyzed-port", type=int, choices=(1, 2), required=True)
    ap.add_argument(
        "--takeover-frame",
        type=int,
        action="append",
        default=[],
        help="Base replay frame to test. Repeat for multiple frames.",
    )
    ap.add_argument(
        "--takeover-frames-json",
        type=Path,
        default=None,
        help="Optional JSON list, or object with a 'frames' list, of base replay frames.",
    )
    ap.add_argument("--offsets", default="-12,0,12,24", help="Comma-separated frame offsets around each base frame.")
    ap.add_argument("--samples-per-point", type=int, default=64)
    ap.add_argument("--max-batch-lanes", type=int, default=1024)
    ap.add_argument("--rollout-frames", type=int, default=180)
    ap.add_argument("--warmup-frames", type=int, default=90)
    ap.add_argument("--defender-delay-frames", type=int, default=60)
    ap.add_argument("--history-mode", choices=("teacher-forced", "dummy"), default="teacher-forced")
    ap.add_argument("--opponent-mode", choices=("replay", "neutral"), default="replay")
    ap.add_argument("--rng-mode", choices=("replay", "native"), default="replay")
    ap.add_argument("--msl-root", type=Path, default=DEFAULT_MSL_ROOT)
    ap.add_argument("--slippi-ai-root", type=Path, default=DEFAULT_SLIPPI_AI_ROOT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--enable-gpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs") / "msl_takeover_grid" / f"run_{_timestamp()}")
    return ap.parse_args()


def setup_msl(msl_root: Path) -> None:
    root = msl_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"MSL root does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)


def _u8_rows(arr: np.ndarray) -> np.ndarray:
    return arr.view(np.uint8).reshape(arr.shape[0], arr.dtype.itemsize)


def _record_for_frame(frame_ids: np.ndarray, frame: int) -> int:
    matches = np.flatnonzero(np.asarray(frame_ids, dtype=np.int32) == np.int32(frame))
    if matches.size:
        return int(matches[0])
    insertion = int(np.searchsorted(frame_ids, frame))
    if insertion <= 0 or insertion >= len(frame_ids):
        raise ValueError(
            f"frame {frame} is outside replay seed range {int(frame_ids[0])}..{int(frame_ids[-1])}"
        )
    before = int(frame_ids[insertion - 1])
    after = int(frame_ids[insertion])
    return int(insertion - 1 if abs(frame - before) <= abs(after - frame) else insertion)


def _load_base_frames(args: argparse.Namespace) -> list[int]:
    frames = list(args.takeover_frame)
    if args.takeover_frames_json is not None:
        payload = json.loads(args.takeover_frames_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("frames", "candidates", "attempted_candidates", "successful_candidates", "groups", "targets"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
            else:
                payload = []
        if not isinstance(payload, list):
            raise ValueError("--takeover-frames-json must be a JSON list or an object containing a frame list")
        for item in payload:
            if isinstance(item, int):
                frames.append(int(item))
                continue
            if isinstance(item, dict):
                for key in ("takeover_frame", "takeoverFrame", "start_frame", "startFrame", "frame"):
                    if item.get(key) is not None:
                        frames.append(int(item[key]))
                        break
                continue
            raise ValueError(f"unsupported frame entry in --takeover-frames-json: {item!r}")
    frames = sorted(dict.fromkeys(frames))
    if not frames:
        raise ValueError("provide at least one --takeover-frame or --takeover-frames-json")
    return frames


def _chunked(values: list[LaneSpec], size: int) -> Iterable[list[LaneSpec]]:
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _controller_to_batched_sa(controllers: list[Any]):
    from slippi_ai import types as sa_types

    def arr(attr: str, dtype: Any = np.float32) -> np.ndarray:
        return np.asarray([getattr(controller, attr) for controller in controllers], dtype=dtype)

    buttons = controllers[0].buttons
    return sa_types.Controller(
        main_stick=sa_types.Stick(
            np.asarray([float(c.main_stick.x) for c in controllers], dtype=np.float32),
            np.asarray([float(c.main_stick.y) for c in controllers], dtype=np.float32),
        ),
        c_stick=sa_types.Stick(
            np.asarray([float(c.c_stick.x) for c in controllers], dtype=np.float32),
            np.asarray([float(c.c_stick.y) for c in controllers], dtype=np.float32),
        ),
        shoulder=arr("shoulder"),
        buttons=sa_types.Buttons(
            **{
                name: np.asarray([bool(getattr(c.buttons, name)) for c in controllers], dtype=np.bool_)
                for name in buttons._fields
            }
        ),
    )


def _map_nt(func, value):
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_map_nt(func, item) for item in value))
    return func(value)


def _map2_nt(func, left, right):
    if isinstance(left, tuple) and hasattr(left, "_fields"):
        return type(left)(*(_map2_nt(func, l_item, r_item) for l_item, r_item in zip(left, right)))
    return func(left, right)


def _force_prev_controller(delayed_agent: Any, controllers: list[Any], force_mask: np.ndarray) -> bool:
    basic_agent = getattr(delayed_agent, "_agent", None)
    policy = getattr(delayed_agent, "policy", None)
    if basic_agent is None or policy is None or not hasattr(basic_agent, "_prev_controller"):
        return False
    force_mask = np.asarray(force_mask, dtype=np.bool_)
    if not bool(np.any(force_mask)):
        return True
    embedding = policy.controller_head.controller_embedding
    forced = embedding.from_state(_controller_to_batched_sa(controllers))
    current = basic_agent._prev_controller

    def merge(cur, forced_leaf):
        cur_arr = np.asarray(cur)
        forced_arr = np.asarray(forced_leaf)
        shape = (force_mask.shape[0],) + (1,) * max(0, forced_arr.ndim - 1)
        return np.where(force_mask.reshape(shape), forced_arr, cur_arr)

    basic_agent._prev_controller = _map2_nt(merge, current, forced)
    return True


def _state_summary(state: Any, *, num_players: int) -> dict[str, Any]:
    return {
        "frame": int(state.frame_id),
        "actions": [int(x) for x in state.action_id[:num_players]],
        "actionFrames": [float(x) for x in state.action_frame[:num_players]],
        "percent": [round(float(x), 3) for x in state.percent[:num_players]],
        "stocks": [int(x) for x in state.stocks[:num_players]],
        "x": [round(float(x), 3) for x in state.pos_x[:num_players]],
        "y": [round(float(x), 3) for x in state.pos_y[:num_players]],
        "hitlag": [int(x) for x in state.hitlag[:num_players]],
        "hitstun": [int(x) for x in state.hitstun[:num_players]],
    }


def _score_lane(start_state: Any, end_state: Any, *, analyzed_idx: int, defender_idx: int) -> dict[str, float | int]:
    start_percent = [float(x) for x in start_state.percent[: start_state.num_players]]
    end_percent = [float(x) for x in end_state.percent[: end_state.num_players]]
    start_stocks = [int(x) for x in start_state.stocks[: start_state.num_players]]
    end_stocks = [int(x) for x in end_state.stocks[: end_state.num_players]]
    dealt = max(0.0, end_percent[defender_idx] - start_percent[defender_idx])
    taken = max(0.0, end_percent[analyzed_idx] - start_percent[analyzed_idx])
    defender_stocks_lost = max(0, start_stocks[defender_idx] - end_stocks[defender_idx])
    analyzed_stocks_lost = max(0, start_stocks[analyzed_idx] - end_stocks[analyzed_idx])
    score = dealt - taken + 80.0 * float(defender_stocks_lost - analyzed_stocks_lost)
    return {
        "damageDealt": round(dealt, 3),
        "damageTaken": round(taken, 3),
        "defenderStocksLost": defender_stocks_lost,
        "analyzedStocksLost": analyzed_stocks_lost,
        "score": round(score, 3),
    }


def main() -> int:
    args = parse_args()
    if args.samples_per_point <= 0:
        raise ValueError("--samples-per-point must be positive")
    if args.rollout_frames <= 0:
        raise ValueError("--rollout-frames must be positive")

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    setup_msl(args.msl_root)

    import msl_binding  # type: ignore
    from tools.eval.validation_dtypes import COMPARE_DTYPE, INPUT_DTYPE
    from tools.modelplay.slippi_ai_bridge import build_fused_batch_model_agent, stop_model_agent
    from tools.modelplay.state_adapter import (
        STAGE_DEBUG_DTYPE,
        build_slippi_ai_game,
        controller_to_input_player,
        empty_controller,
        frame_state_from_compare,
        frame_state_from_seed,
        frame_state_with_timebase,
        input_array_to_controllers,
    )
    from tools.slippi.validation_buffer_builder import build_validation_buffers_from_slp

    replay = args.replay.resolve()
    timings: dict[str, float] = {}
    start = time.perf_counter()
    buffers = build_validation_buffers_from_slp(slp_path=str(replay))
    timings["build_validation_buffers_s"] = time.perf_counter() - start

    frame_ids = np.asarray(buffers.seed_t["frame_id"], dtype=np.int32)
    offsets = parse_csv_ints(args.offsets, name="--offsets")
    base_frames = _load_base_frames(args)
    num_players = int(buffers.num_players)
    analyzed_idx = int(args.analyzed_port) - 1
    defender_idx = 1 if analyzed_idx == 0 else 0
    if analyzed_idx >= num_players or defender_idx >= num_players:
        raise ValueError(f"ports require 2 players, replay has num_players={num_players}")

    lane_specs: list[LaneSpec] = []
    lane_id = 0
    for base in base_frames:
        for offset in offsets:
            requested = int(base) + int(offset)
            start_record = _record_for_frame(frame_ids, requested)
            takeover_frame = int(frame_ids[start_record])
            for sample_index in range(args.samples_per_point):
                lane_specs.append(
                    LaneSpec(
                        lane_id=lane_id,
                        base_frame=int(base),
                        offset=int(offset),
                        sample_index=int(sample_index),
                        start_record=int(start_record),
                        takeover_frame=takeover_frame,
                        defender_takeover_frame=takeover_frame + int(args.defender_delay_frames),
                    )
                )
                lane_id += 1

    all_rows: list[dict[str, Any]] = []
    chunks = list(_chunked(lane_specs, args.max_batch_lanes))
    timings["lane_count"] = float(len(lane_specs))
    timings["chunk_count"] = float(len(chunks))

    for chunk_index, chunk in enumerate(chunks):
        chunk_start = time.perf_counter()
        chunk_timings = {
            "build_agent_s": 0.0,
            "warmup_game_build_s": 0.0,
            "warmup_force_s": 0.0,
            "warmup_agent_step_s": 0.0,
            "seed_s": 0.0,
            "rollout_game_build_s": 0.0,
            "rollout_force_s": 0.0,
            "rollout_agent_step_s": 0.0,
            "rollout_input_build_s": 0.0,
            "rollout_sim_step_s": 0.0,
            "rollout_state_refresh_s": 0.0,
            "summarize_s": 0.0,
        }
        batch_size = len(chunk)
        port_batch_size = batch_size * 2
        handle = msl_binding.init(batch_size, num_players)
        compare = np.zeros(batch_size, dtype=COMPARE_DTYPE)
        compare_u8 = _u8_rows(compare)
        stage_debug = np.zeros(batch_size, dtype=STAGE_DEBUG_DTYPE)
        stage_debug_u8 = _u8_rows(stage_debug)
        processed_input = np.zeros(batch_size, dtype=INPUT_DTYPE)
        processed_input_u8 = _u8_rows(processed_input)
        agent = None
        try:
            phase = time.perf_counter()
            agent = build_fused_batch_model_agent(
                slippi_ai_root=args.slippi_ai_root.resolve(),
                model_path=args.model.resolve(),
                batch_size=port_batch_size,
                name=None,
                disable_gpus=not args.enable_gpu,
            )
            chunk_timings["build_agent_s"] += time.perf_counter() - phase

            if args.history_mode == "teacher-forced":
                needs_reset = np.ones(port_batch_size, dtype=np.bool_)
                min_warmup_start = min(max(0, lane.start_record - args.warmup_frames) for lane in chunk)
                max_warmup_steps = max(lane.start_record - min_warmup_start for lane in chunk)
                for warm_step in range(max_warmup_steps):
                    phase = time.perf_counter()
                    games: list[Any] = []
                    prev_controllers_for_force: list[Any] = []
                    force_mask = np.ones(port_batch_size, dtype=np.bool_)
                    for port in (1, 2):
                        for lane in chunk:
                            record = max(0, lane.start_record - max_warmup_steps + warm_step)
                            record = min(record, lane.start_record - 1)
                            state = frame_state_from_seed(buffers.seed_t[record])
                            controllers = input_array_to_controllers(
                                buffers.input_t[record : record + 1],
                                num_players=num_players,
                            )
                            prev_controllers = input_array_to_controllers(
                                buffers.prev_input_t[record : record + 1],
                                num_players=num_players,
                            )
                            games.append(build_slippi_ai_game(state, controllers, viewpoint_port=port))
                            prev_controllers_for_force.append(prev_controllers.get(port, empty_controller()))
                    chunk_timings["warmup_game_build_s"] += time.perf_counter() - phase
                    phase = time.perf_counter()
                    _force_prev_controller(agent.delayed_agent, prev_controllers_for_force, force_mask)
                    chunk_timings["warmup_force_s"] += time.perf_counter() - phase
                    phase = time.perf_counter()
                    agent.step(games, needs_reset=needs_reset)
                    chunk_timings["warmup_agent_step_s"] += time.perf_counter() - phase
                    needs_reset[:] = False

            phase = time.perf_counter()
            start_records = np.asarray([lane.start_record for lane in chunk], dtype=np.int64)
            seed_rows = buffers.seed_t[start_records]
            msl_binding.reseed_seed_rollout(handle, _u8_rows(seed_rows))
            msl_binding.write_compare(handle, compare_u8)
            msl_binding.debug_write_stage_state(handle, stage_debug_u8)
            msl_binding.debug_write_processed_input(handle, processed_input_u8)
            states = [
                frame_state_with_timebase(
                    frame_state_from_compare(compare[env], stage_debug[env]),
                    msl_binding.debug_timebase(handle, env),
                )
                for env in range(batch_size)
            ]
            start_states = list(states)
            last_controllers = [
                input_array_to_controllers(
                    buffers.prev_input_t[record : record + 1],
                    num_players=num_players,
                )
                for record in start_records
            ]
            prev_input = buffers.prev_input_t[start_records].copy()
            needs_reset = np.zeros(port_batch_size, dtype=np.bool_)
            if args.history_mode == "dummy":
                needs_reset[:] = True
            chunk_timings["seed_s"] += time.perf_counter() - phase

            for step in range(args.rollout_frames):
                phase = time.perf_counter()
                games = []
                force_controllers = []
                force_mask = np.zeros(port_batch_size, dtype=np.bool_)
                for port in (1, 2):
                    for env, lane in enumerate(chunk):
                        games.append(build_slippi_ai_game(states[env], last_controllers[env], viewpoint_port=port))
                        record = min(lane.start_record + step, buffers.num_records - 1)
                        prev_controllers = input_array_to_controllers(
                            buffers.prev_input_t[record : record + 1],
                            num_players=num_players,
                        )
                        force_controllers.append(prev_controllers.get(port, empty_controller()))
                        if args.history_mode == "teacher-forced":
                            flat_index = (0 if port == 1 else batch_size) + env
                            if port == args.analyzed_port:
                                force_mask[flat_index] = step == 0
                            else:
                                force_mask[flat_index] = step < args.defender_delay_frames
                chunk_timings["rollout_game_build_s"] += time.perf_counter() - phase
                phase = time.perf_counter()
                _force_prev_controller(agent.delayed_agent, force_controllers, force_mask)
                chunk_timings["rollout_force_s"] += time.perf_counter() - phase
                phase = time.perf_counter()
                flat_model_controllers = agent.step(games, needs_reset=needs_reset)
                chunk_timings["rollout_agent_step_s"] += time.perf_counter() - phase
                needs_reset[:] = False

                phase = time.perf_counter()
                current_input = np.zeros(batch_size, dtype=INPUT_DTYPE)
                if args.opponent_mode == "replay":
                    records = np.minimum(start_records + step, buffers.num_records - 1)
                    current_input[...] = buffers.input_t[records]

                for env, lane in enumerate(chunk):
                    current_input["p"][env, analyzed_idx] = controller_to_input_player(
                        flat_model_controllers[env if args.analyzed_port == 1 else batch_size + env]
                    )
                    if step >= args.defender_delay_frames:
                        defender_flat = env if defender_idx == 0 else batch_size + env
                        current_input["p"][env, defender_idx] = controller_to_input_player(
                            flat_model_controllers[defender_flat]
                        )
                chunk_timings["rollout_input_build_s"] += time.perf_counter() - phase

                records = np.minimum(start_records + step, buffers.num_records - 1)
                phase = time.perf_counter()
                if args.rng_mode == "replay":
                    msl_binding.step_input_replay_frame_rng(
                        handle,
                        _u8_rows(buffers.seed_t[records]),
                        _u8_rows(prev_input),
                        _u8_rows(current_input),
                    )
                else:
                    msl_binding.step_input(handle, _u8_rows(prev_input), _u8_rows(current_input))
                chunk_timings["rollout_sim_step_s"] += time.perf_counter() - phase

                phase = time.perf_counter()
                msl_binding.write_compare(handle, compare_u8)
                msl_binding.debug_write_stage_state(handle, stage_debug_u8)
                msl_binding.debug_write_processed_input(handle, processed_input_u8)
                states = [
                    frame_state_with_timebase(
                        frame_state_from_compare(compare[env], stage_debug[env]),
                        msl_binding.debug_timebase(handle, env),
                    )
                    for env in range(batch_size)
                ]
                last_controllers = [
                    input_array_to_controllers(processed_input[env : env + 1], num_players=num_players)
                    for env in range(batch_size)
                ]
                prev_input = current_input.copy()
                chunk_timings["rollout_state_refresh_s"] += time.perf_counter() - phase

            phase = time.perf_counter()
            for env, lane in enumerate(chunk):
                score = _score_lane(
                    start_states[env],
                    states[env],
                    analyzed_idx=analyzed_idx,
                    defender_idx=defender_idx,
                )
                all_rows.append(
                    {
                        "laneId": int(lane.lane_id),
                        "chunkIndex": int(chunk_index),
                        "baseFrame": int(lane.base_frame),
                        "offset": int(lane.offset),
                        "sampleIndex": int(lane.sample_index),
                        "startRecord": int(lane.start_record),
                        "takeoverFrame": int(lane.takeover_frame),
                        "defenderTakeoverFrame": int(lane.defender_takeover_frame),
                        "startState": _state_summary(start_states[env], num_players=num_players),
                        "endState": _state_summary(states[env], num_players=num_players),
                        **score,
                    }
                )
            chunk_timings["summarize_s"] += time.perf_counter() - phase
        finally:
            if agent is not None:
                stop_model_agent(agent)
            msl_binding.destroy(handle)
        chunk_timings["total_s"] = time.perf_counter() - chunk_start
        timings[f"chunk_{chunk_index}"] = chunk_timings
        print(
            json.dumps(
                {
                    "chunk": chunk_index,
                    "lanes": batch_size,
                    "seconds": round(chunk_timings["total_s"], 3),
                    "timings": {key: round(value, 3) for key, value in chunk_timings.items()},
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    rows_path = out_dir / "lanes.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")

    grouped: dict[str, dict[str, Any]] = {}
    for row in all_rows:
        key = f"{row['baseFrame']}:{row['offset']}"
        bucket = grouped.setdefault(
            key,
            {
                "baseFrame": row["baseFrame"],
                "offset": row["offset"],
                "takeoverFrame": row["takeoverFrame"],
                "samples": 0,
                "scores": [],
                "damageDealt": [],
                "damageTaken": [],
                "defenderStocksLost": 0,
                "analyzedStocksLost": 0,
            },
        )
        bucket["samples"] += 1
        bucket["scores"].append(float(row["score"]))
        bucket["damageDealt"].append(float(row["damageDealt"]))
        bucket["damageTaken"].append(float(row["damageTaken"]))
        bucket["defenderStocksLost"] += int(row["defenderStocksLost"])
        bucket["analyzedStocksLost"] += int(row["analyzedStocksLost"])

    groups = []
    for bucket in grouped.values():
        scores = np.asarray(bucket.pop("scores"), dtype=np.float32)
        dealt = np.asarray(bucket.pop("damageDealt"), dtype=np.float32)
        taken = np.asarray(bucket.pop("damageTaken"), dtype=np.float32)
        groups.append(
            {
                **bucket,
                "meanScore": round(float(np.mean(scores)), 3),
                "p25Score": round(float(np.percentile(scores, 25)), 3),
                "p75Score": round(float(np.percentile(scores, 75)), 3),
                "bestScore": round(float(np.max(scores)), 3),
                "worstScore": round(float(np.min(scores)), 3),
                "meanDamageDealt": round(float(np.mean(dealt)), 3),
                "meanDamageTaken": round(float(np.mean(taken)), 3),
                "killRate": round(float(bucket["defenderStocksLost"]) / max(1, int(bucket["samples"])), 4),
                "selfDeathRate": round(float(bucket["analyzedStocksLost"]) / max(1, int(bucket["samples"])), 4),
            }
        )
    groups.sort(key=lambda item: (float(item["meanScore"]), float(item["bestScore"])), reverse=True)
    timings["total_s"] = time.perf_counter() - t0
    summary = {
        "runner": "work/run_msl_replay_takeover_grid.py",
        "replay": str(replay),
        "model": str(args.model.resolve()),
        "mslRoot": str(args.msl_root.resolve()),
        "slippiAiRoot": str(args.slippi_ai_root.resolve()),
        "analyzedPort": int(args.analyzed_port),
        "defenderPort": defender_idx + 1,
        "baseFrames": base_frames,
        "offsets": offsets,
        "samplesPerPoint": int(args.samples_per_point),
        "laneCount": len(lane_specs),
        "maxBatchLanes": int(args.max_batch_lanes),
        "rolloutFrames": int(args.rollout_frames),
        "warmupFrames": int(args.warmup_frames),
        "defenderDelayFrames": int(args.defender_delay_frames),
        "historyMode": args.history_mode,
        "opponentMode": args.opponent_mode,
        "rngMode": args.rng_mode,
        "enableGpu": bool(args.enable_gpu),
        "lanesJsonl": str(rows_path),
        "groups": groups,
        "timings": timings,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
