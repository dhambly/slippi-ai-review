"""Run replay-exact Phillip rollouts with the decomp melee-sim-light backend.

The JSON contract intentionally mirrors :mod:`slippi_ai_review.simulation` so
selection and reporting can compare legacy and decomp runs without knowing how
the simulator was initialized.  Unlike the legacy seed path, every lane starts
from a replay-warmed native checkpoint and keeps replay RNG/stage authority while
only replacing controller inputs owned by Phillip.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .simulation import (
    _LogicalHitTracker,
    _action_name,
    _combo_outperformed_replay,
    _groups,
    _input_token,
    _load_frame_metadata,
    _move_ids_for_slots,
    _option_summary,
    _recorded_contact_frame,
    _replay_baseline_after_anchor,
)
from .simulation_support import (
    _acquire_gpu_lock,
    _configure_gpu_budget,
    _load_base_frames,
    _record_for_frame,
    _throttle_gpu_inference,
    parse_csv_ints,
)


@dataclass(frozen=True, slots=True)
class LaneSpec:
    lane_id: int
    base_frame: int
    offset: int
    sample_index: int
    start_record: int
    takeover_frame: int
    defender_takeover_frame: int
    warm_start_record: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--analyzed-port", type=int, choices=(1, 2), required=True)
    parser.add_argument("--takeover-frame", type=int, action="append", default=[])
    parser.add_argument("--takeover-frames-json", type=Path)
    parser.add_argument("--only-base-frame", type=int, action="append", default=[])
    parser.add_argument("--offsets", default="-12,0,12,24")
    parser.add_argument("--samples-per-point", type=int, default=64)
    parser.add_argument("--max-batch-lanes", type=int, default=1024)
    parser.add_argument("--objective", choices=("general", "combo-extension"), default="general")
    parser.add_argument("--rollout-frames", type=int, default=180)
    parser.add_argument("--defense-resolution-extra-frames", type=int, default=180)
    parser.add_argument("--combo-max-rollout-frames", type=int, default=420)
    parser.add_argument("--defense-resolution-min-frames", type=int)
    parser.add_argument("--defense-neutral-stable-frames", type=int, default=20)
    parser.add_argument("--combo-escape-stable-frames", type=int, default=20)
    parser.add_argument("--combo-escape-min-separation", type=float, default=12.0)
    parser.add_argument("--combo-escape-pressure-range", type=float, default=30.0)
    parser.add_argument("--combo-escape-max-air-height", type=float, default=25.0)
    parser.add_argument("--combo-edge-x-threshold", type=float, default=55.0)
    parser.add_argument("--unresolved-defense-penalty", type=float, default=60.0)
    parser.add_argument("--warmup-frames", type=int, default=90)
    parser.add_argument("--defender-delay-frames", type=int, default=60)
    parser.add_argument(
        "--defender-takeover-mode",
        choices=(
            "fixed-delay",
            "observed-followup",
            "observed-opponent-followup",
            "observed-phase-followup",
        ),
        default="fixed-delay",
    )
    parser.add_argument("--anchor-recorded-contact", action="store_true")
    parser.add_argument("--history-mode", choices=("teacher-forced", "dummy"), default="teacher-forced")
    parser.add_argument("--replay-bridge-policy-delay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--analyzed-mode", choices=("model", "replay"), default="model")
    parser.add_argument("--opponent-mode", choices=("replay", "neutral"), default="replay")
    parser.add_argument("--rng-mode", choices=("replay", "native"), default="replay")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--option-horizon-frames", type=int, default=90)
    parser.add_argument("--option-max-action-segments", type=int, default=6)
    parser.add_argument("--option-max-input-segments", type=int, default=8)
    parser.add_argument("--dump-controller-streams", action="store_true")
    parser.add_argument("--msl-root", type=Path, required=True)
    parser.add_argument("--slippi-ai-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--enable-gpu", action="store_true")
    parser.add_argument("--gpu-duty-cycle", type=float, default=0.20)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _chunked(values: list[LaneSpec], size: int) -> Iterable[list[LaneSpec]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _slot(observation: np.ndarray | np.void, source: int) -> np.ndarray | np.void:
    slots = observation["slots"]
    if slots.ndim == 1:
        for row in slots:
            if bool(row["present"]) and int(row["source_player"]) == source:
                return row
    else:
        for column in range(slots.shape[1]):
            rows = slots[:, column]
            present = rows["present"].astype(np.bool_)
            if np.any(present) and np.all((~present) | (rows["source_player"] == source)):
                return rows
    raise RuntimeError(f"missing source player {source}")


def _state_summary(observation: np.ndarray, env: int, num_players: int) -> dict[str, Any]:
    row = observation[env]
    players = [_slot(row, source) for source in range(num_players)]
    return {
        "frame": int(row["frame_id"]),
        "actions": [int(player["action_id"]) for player in players],
        "actionFrames": [float(player["action_frame"]) for player in players],
        "percent": [round(float(player["percent"]), 3) for player in players],
        "stocks": [int(player["stocks"]) for player in players],
        "x": [round(float(player["pos_x"]), 3) for player in players],
        "y": [round(float(player["pos_y"]), 3) for player in players],
        "hitlag": [int(player["hitlag"]) for player in players],
        "hitstun": [int(player["hitstun"]) for player in players],
    }


def _controller_map(policy_row: np.void, num_players: int) -> dict[int, object]:
    from tools.modelplay.phillip_adapter import controllers_from_policy_row

    return controllers_from_policy_row(policy_row, num_players=num_players)


def _batch_games(
    env: Any,
    controller_rows: list[dict[int, object]],
    num_players: int,
    game_buffers: Any,
    needs_reset: np.ndarray,
) -> Any:
    from tools.modelplay.phillip_agent import batch_controllers

    if env.t == env.length:
        env.reset_cursor()
    env.observe_viewpoints(np.zeros(env.batch_size, dtype=np.uint8))
    controllers = {
        player + 1: batch_controllers(
            [controller_rows[lane][player + 1] for lane in range(env.batch_size)]
        )
        for player in range(num_players)
    }
    game_buffers.fill(env.current_frame, needs_reset, controllers=controllers)
    return game_buffers.game


def _policy_batch(
    authority: Any,
    records: np.ndarray,
    *,
    delay: int,
    num_players: int,
) -> tuple[list[object], np.ndarray]:
    rows = []
    policy_records = []
    for record in records:
        policy_record = min(int(record) + delay, len(authority.frames) - 1)
        rows.append(_controller_map(authority.policy_controllers[policy_record], num_players))
        policy_records.append(policy_record)
    controllers = [
        rows[lane][player + 1]
        for player in range(num_players)
        for lane in range(len(records))
    ]
    frame_ids = [
        int(authority.frame_ids[policy_records[lane]])
        for _player in range(num_players)
        for lane in range(len(records))
    ]
    return controllers, np.asarray(frame_ids, dtype=np.int32)


def _active_controller_rows(
    authority: Any,
    records: np.ndarray,
    decoded: list[object],
    analyzed_active: np.ndarray,
    defender_active: np.ndarray,
    *,
    analyzed: int,
    defender: int,
    num_players: int,
) -> tuple[list[dict[int, object]], np.ndarray]:
    from melee_sim import dtypes
    from tools.modelplay.phillip_adapter import controller_to_input_player

    inputs = np.zeros(len(records), dtype=dtypes.input_dtype())
    rows: list[dict[int, object]] = []
    for lane, record in enumerate(records):
        replay_frame = authority.frames[int(record)]
        inputs["players"][lane] = replay_frame["input"]["players"]
        controllers = _controller_map(authority.policy_controllers[int(record)], num_players)
        for player, active in ((analyzed, analyzed_active[lane]), (defender, defender_active[lane])):
            if active:
                controller = decoded[player * len(records) + lane]
                controllers[player + 1] = controller
                inputs["players"][lane, player] = controller_to_input_player(controller)
        rows.append(controllers)
    return rows, inputs


def _canonical_scan(authority: Any, requested_records: set[int], data_dir: Path) -> tuple[dict[int, bytes], dict[int, np.void]]:
    import melee_sim as msl

    states: dict[int, bytes] = {}
    observations: dict[int, np.void] = {}
    if not requested_records:
        return states, observations
    requested = set(int(record) for record in requested_records)
    with msl.EnvBatch(batch_size=1, length=256, data_dir=data_dir, action_format="raw") as env:
        env.reset_replay(authority.config, authority.previous_input)
        for record in range(max(requested) + 1):
            if env.t == env.length:
                env.reset_cursor()
            env.step_replay(authority.frames[record : record + 1])
            if record in requested:
                env.observe_viewpoints(np.zeros(1, dtype=np.uint8))
                observations[record] = env.current_frame[0].copy()
                states[record] = env.save(0)
    missing = requested - states.keys()
    if missing:
        raise RuntimeError(f"failed to build replay checkpoints for records {sorted(missing)[:5]}")
    return states, observations


def _contact_anchor_record(
    authority: Any,
    canonical: dict[int, np.void],
    metadata: dict[str, Any],
    *,
    analyzed: int,
    defender: int,
) -> int | None:
    contact_frame = _recorded_contact_frame(metadata)
    if contact_frame is None:
        return None
    contact = _record_for_frame(authority.frame_ids, contact_frame)
    last_hitlag: int | None = None
    upper = min(len(authority.frames), contact + 30)
    for record in range(max(0, contact - 1), upper):
        observation = canonical.get(record)
        if observation is None:
            continue
        players = (_slot(observation, analyzed), _slot(observation, defender))
        if any(int(player["hitlag"]) > 0 for player in players):
            last_hitlag = record
            continue
        if last_hitlag is not None and record > last_hitlag:
            return record
    return None


def _is_constrained(player: np.void) -> bool:
    action = _action_name(int(player["action_id"]))
    return int(player["hitlag"]) > 0 or int(player["hitstun"]) > 0 or action.startswith("DAMAGE_") or action == "TUMBLING"


def _escape_eligible(
    observation: np.void,
    *,
    analyzed: int,
    defender: int,
    min_separation: float,
    pressure_range: float,
    max_air_height: float,
    edge_x: float,
) -> tuple[bool, dict[str, Any]]:
    attacker = _slot(observation, analyzed)
    target = _slot(observation, defender)
    target_action = _action_name(int(target["action_id"]))
    attacker_action = _action_name(int(attacker["action_id"]))
    separation = float(np.hypot(float(target["pos_x"] - attacker["pos_x"]), float(target["pos_y"] - attacker["pos_y"])))
    in_hit = _is_constrained(target)
    offstage = (not bool(target["on_ground"])) and (
        abs(float(target["pos_x"])) > edge_x
        or float(target["pos_y"]) < 0.0
        or float(target["pos_y"]) > max_air_height
    )
    committed_roots = (
        "JAB", "DASH_ATTACK", "FTILT", "UPTILT", "DOWNTILT", "FSMASH", "UPSMASH",
        "DOWNSMASH", "NAIR", "FAIR", "BAIR", "UAIR", "DAIR", "GRAB", "CATCH",
        "THROW", "SHINE", "DOWN_B", "UP_B", "SIDE_B", "NEUTRAL_B",
    )
    attacker_committed = attacker_action.startswith(committed_roots)
    attacker_pressuring = attacker_committed and separation < pressure_range
    actionable = bool(target["on_ground"]) and not (
        target_action.startswith("DAMAGE_")
        or target_action.startswith("DOWN_")
        or target_action in {"TUMBLING", "TECH_MISS_UP", "TECH_MISS_DOWN"}
    )
    eligible = not in_hit and not offstage and actionable and separation >= min_separation and not attacker_pressuring
    reason = (
        "in_hit" if in_hit else "offstage_or_juggled" if offstage else "not_actionable"
        if not actionable else "attacker_pressuring" if attacker_pressuring else "too_close"
        if separation < min_separation else "eligible"
    )
    return eligible, {
        "defenderActionId": int(target["action_id"]),
        "defenderAction": target_action,
        "defenderActionFrame": int(target["action_frame"]),
        "defenderGrounded": bool(target["on_ground"]),
        "defenderPosition": {"x": round(float(target["pos_x"]), 3), "y": round(float(target["pos_y"]), 3)},
        "attackerActionId": int(attacker["action_id"]),
        "attackerAction": attacker_action,
        "attackerActionFrame": int(attacker["action_frame"]),
        "playerSeparation": round(separation, 3),
        "defenderInHit": in_hit,
        "defenderInCommittedState": not actionable,
        "defenderActionableLocomotion": actionable,
        "offstageOrJuggled": offstage,
        "attackerPressuring": attacker_pressuring,
        "attackerTooClose": separation < min_separation,
        "attackerActivePressure": attacker_committed,
        "escapeEligible": eligible,
        "escapeBlockReason": reason,
    }


def _stream_ref(
    path: Path | None,
    env: int,
    lane: LaneSpec,
    frames: int,
    *,
    analyzed_port: int,
    defender_port: int,
    defender_takeover_frame: int,
    model_control_frame: int,
) -> dict[str, Any] | None:
    if path is None:
        return None
    return {
        "format": "decomp-replay-input-v1",
        "backend": "decomp",
        "path": str(path.resolve()),
        "array": "inputs",
        "env": env,
        "laneId": lane.lane_id,
        "takeoverFrame": lane.takeover_frame,
        "modelControlFrame": model_control_frame,
        "defenderTakeoverFrame": defender_takeover_frame,
        "endFrame": lane.takeover_frame + frames - 1,
        "frames": frames,
        "analyzedPort": analyzed_port,
        "defenderPort": defender_port,
    }


def _max_rollout(args: argparse.Namespace) -> int:
    if args.objective == "combo-extension":
        return max(args.rollout_frames + args.defense_resolution_extra_frames, args.combo_max_rollout_frames)
    return args.rollout_frames + args.defense_resolution_extra_frames


def main() -> int:
    args = parse_args()
    if args.samples_per_point < 1 or args.max_batch_lanes < 1:
        raise SystemExit("sample and batch counts must be positive")
    if args.warmup_frames < 1 or args.rollout_frames < 1:
        raise SystemExit("warmup and rollout frames must be positive")
    if not 0 < args.gpu_duty_cycle <= 1:
        raise SystemExit("--gpu-duty-cycle must be in (0, 1]")
    if args.defense_resolution_min_frames is None:
        args.defense_resolution_min_frames = args.rollout_frames

    started = time.perf_counter()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    streams_dir = out / "controller_streams"
    if args.dump_controller_streams:
        streams_dir.mkdir(parents=True, exist_ok=True)
    _gpu_lock = _acquire_gpu_lock(args.enable_gpu)
    _configure_gpu_budget(enable_gpu=args.enable_gpu, duty_cycle=args.gpu_duty_cycle)

    import melee_sim as msl
    from melee_sim import dtypes
    from tools.migration.replay_checkpoint import extract_replay_authority
    from tools.modelplay.phillip_agent import (
        AnimationFilter,
        build_agent,
        decoded_controllers,
        force_previous_controller,
        import_eval_lib,
        load_state,
    )
    from slippi_ai.sim_env.observations import GameBatchBuffers
    from tools.slippi.action_state_tables import load_action_state_tables

    timings: dict[str, Any] = {}
    phase = time.perf_counter()
    authority = extract_replay_authority(args.replay.resolve())
    action_state_tables = load_action_state_tables(str(args.msl_root.resolve() / "data"))
    timings["extract_replay_authority_s"] = time.perf_counter() - phase
    num_players = int(authority.config[0]["match"]["num_players"])
    if num_players != 2:
        raise SystemExit(f"decomp analyzer currently requires singles; replay has {num_players} players")
    analyzed = args.analyzed_port - 1
    defender = 1 - analyzed
    frame_metadata = _load_frame_metadata(args.takeover_frames_json)
    base_frames = _load_base_frames(args)
    if args.only_base_frame:
        allowed = set(args.only_base_frame)
        base_frames = [frame for frame in base_frames if frame in allowed]
    offsets = parse_csv_ints(args.offsets, name="--offsets")

    # Contact anchoring needs only a tiny canonical neighborhood. Checkpoints
    # are then collected in the same pass after final branch records are known.
    contact_records: set[int] = set()
    if args.anchor_recorded_contact:
        for base in base_frames:
            contact = _recorded_contact_frame(frame_metadata.get(base, {}))
            if contact is None:
                continue
            record = _record_for_frame(authority.frame_ids, contact)
            contact_records.update(range(max(0, record - 1), min(len(authority.frames), record + 30)))
    phase = time.perf_counter()
    _, contact_observations = _canonical_scan(authority, contact_records, args.msl_root / "data")
    contact_anchor_frames: dict[int, int] = {}
    if args.anchor_recorded_contact:
        for base in base_frames:
            anchor = _contact_anchor_record(
                authority,
                contact_observations,
                frame_metadata.get(base, {}),
                analyzed=analyzed,
                defender=defender,
            )
            if anchor is not None:
                contact_anchor_frames[base] = int(authority.frame_ids[anchor])
    timings["contact_anchor_scan_s"] = time.perf_counter() - phase

    lanes: list[LaneSpec] = []
    for base in base_frames:
        branch = contact_anchor_frames.get(base, base)
        for offset in offsets:
            record = _record_for_frame(authority.frame_ids, branch + offset)
            takeover = int(authority.frame_ids[record])
            # Match the established Phillip lane: ``warmup_frames`` replay
            # observations precede the branch observation, which is then the
            # policy call that generates the first physically applied output.
            warm_start = max(0, record - args.warmup_frames)
            for sample in range(args.samples_per_point):
                lanes.append(
                    LaneSpec(
                        lane_id=len(lanes),
                        base_frame=base,
                        offset=offset,
                        sample_index=sample,
                        start_record=record,
                        takeover_frame=takeover,
                        defender_takeover_frame=takeover + args.defender_delay_frames,
                        warm_start_record=warm_start,
                    )
                )
    if not lanes:
        raise SystemExit("no rollout lanes were created")

    eval_lib = import_eval_lib(args.slippi_ai_root, disable_gpus=not args.enable_gpu)
    model_state = load_state(eval_lib, args.model)
    max_steps = _max_rollout(args)
    all_rows: list[dict[str, Any]] = []
    # Every policy lane in a fused batch must receive the same count of replay
    # history observations. Early-game branches have shorter histories, so keep
    # them in separate chunks instead of padding with repeated states.
    by_warm_length: dict[int, list[LaneSpec]] = {}
    for lane in lanes:
        warm_length = lane.start_record - lane.warm_start_record + 1
        by_warm_length.setdefault(warm_length, []).append(lane)
    chunks = [
        chunk
        for warm_length in sorted(by_warm_length)
        for chunk in _chunked(by_warm_length[warm_length], args.max_batch_lanes)
    ]
    policy_delay = 0

    for chunk_index, chunk in enumerate(chunks):
        chunk_started = time.perf_counter()
        batch = len(chunk)
        agent = build_agent(
            eval_lib,
            state=model_state,
            batch_size=batch * num_players,
            sample_temperature=args.sample_temperature,
        )
        agent.start()
        policy_delay = max(0, int(agent.delay))
        replay_bridge = policy_delay if args.replay_bridge_policy_delay else 0
        start_records = np.asarray([lane.start_record for lane in chunk], dtype=np.int32)
        warm_records = np.asarray([lane.warm_start_record for lane in chunk], dtype=np.int32)
        takeover_frames = np.asarray([lane.takeover_frame for lane in chunk], dtype=np.int32)
        analyzed_control_frames = takeover_frames + replay_bridge
        if args.analyzed_mode == "replay":
            analyzed_control_frames[:] = np.iinfo(np.int32).max
        defender_control_frames = np.asarray(
            [lane.defender_takeover_frame for lane in chunk], dtype=np.int32
        )
        if args.defender_takeover_mode != "fixed-delay":
            defender_control_frames[:] = np.iinfo(np.int32).max
        actual_defender_frames = defender_control_frames.copy()
        defender_reasons = np.asarray(["fixed_delay"] * batch, dtype=object)
        detected_frames = np.full(batch, -1, dtype=np.int32)
        phase_names = [
            str(((frame_metadata.get(lane.base_frame, {}).get("point_evidence") or {}).get("phase") or "")).lower()
            for lane in chunk
        ]

        stream_history = np.zeros((max_steps, batch), dtype=dtypes.input_dtype()) if args.dump_controller_streams else None
        action_history = np.zeros((min(max_steps, args.option_horizon_frames), batch), dtype=np.uint16)
        input_tokens: list[list[str]] = [[] for _ in chunk]
        resolved = np.zeros(batch, dtype=np.bool_)
        resolved_step = np.full(batch, max_steps, dtype=np.int32)
        reason = np.asarray(["combo_timeout" if args.objective == "combo-extension" else "nominal_non_defense"] * batch, dtype=object)
        stable_counts = np.zeros(batch, dtype=np.int16)
        defense_resolution_active = np.zeros(batch, dtype=np.bool_)
        unresolved_penalty = np.zeros(batch, dtype=np.float32)
        first_dealt = np.full(batch, -1, dtype=np.int32)
        first_taken = np.full(batch, -1, dtype=np.int32)
        followup_damage = np.zeros(batch, dtype=np.float32)
        followup_hits = np.zeros(batch, dtype=np.int16)
        true_followup_hits = np.zeros(batch, dtype=np.int16)
        first_followup = np.full(batch, -1, dtype=np.int32)
        last_followup = np.full(batch, -1, dtype=np.int32)

        try:
            with msl.EnvBatch(
                batch_size=batch,
                length=max(max_steps + 2, args.warmup_frames + 2),
                data_dir=args.msl_root / "data",
                action_format="raw",
            ) as env:
                env.reset_replay(
                    np.repeat(authority.config, batch),
                    np.repeat(authority.previous_input, batch),
                )
                replay_warm_started = time.perf_counter()
                for replay_record in range(int(np.max(warm_records)) + 1):
                    if env.t == env.length:
                        env.reset_cursor()
                    replay_frames = np.repeat(
                        authority.frames[replay_record : replay_record + 1],
                        batch,
                    )
                    env.step_replay(
                        replay_frames,
                        (replay_record <= warm_records).astype(np.uint8),
                    )
                timings.setdefault("canonical_replay_warm_s", 0.0)
                timings["canonical_replay_warm_s"] += time.perf_counter() - replay_warm_started

                needs_reset = np.ones(batch * num_players, dtype=np.bool_)
                lane_needs_reset = np.ones(batch, dtype=np.bool_)
                game_buffers = GameBatchBuffers(batch)
                animation_filter = AnimationFilter(
                    batch * num_players,
                    enabled=bool(agent.observation_config.animation.mask),
                )
                outputs = None
                max_warm = max(lane.start_record - lane.warm_start_record + 1 for lane in chunk)
                for warm_step in range(max_warm):
                    if warm_step:
                        if env.t == env.length:
                            env.reset_cursor()
                        records = np.minimum(warm_records + warm_step, start_records)
                        replay_frames = authority.frames[records].copy()
                        active = (warm_records + warm_step <= start_records).astype(np.uint8)
                        env.step_replay(replay_frames, active)
                    else:
                        records = warm_records.copy()
                    controller_rows = [
                        _controller_map(authority.policy_controllers[int(record)], num_players)
                        for record in records
                    ]
                    games = _batch_games(
                        env,
                        controller_rows,
                        num_players,
                        game_buffers,
                        lane_needs_reset,
                    )
                    policy_controllers, policy_frames = _policy_batch(
                        authority, records, delay=policy_delay, num_players=num_players
                    )
                    if args.history_mode == "teacher-forced":
                        warm_force_mask = np.zeros(batch * num_players, dtype=np.bool_)
                        for lane_index in range(batch):
                            for player in range(num_players):
                                boundary = (
                                    analyzed_control_frames[lane_index]
                                    if player == analyzed
                                    else defender_control_frames[lane_index]
                                )
                                policy_index = player * batch + lane_index
                                warm_force_mask[policy_index] = policy_frames[policy_index] < boundary
                        force_previous_controller(
                            agent,
                            policy_controllers,
                            warm_force_mask,
                        )
                    animation_filter.filter(games, needs_reset)
                    inference_started = time.perf_counter()
                    outputs = agent.step(games, needs_reset)
                    _throttle_gpu_inference(
                        elapsed_s=time.perf_counter() - inference_started,
                        duty_cycle=args.gpu_duty_cycle if args.enable_gpu else 1.0,
                    )
                    needs_reset[:] = False
                    lane_needs_reset[:] = False
                if outputs is None:
                    raise RuntimeError("teacher-forced warmup produced no output")

                if env.t == env.length:
                    env.reset_cursor()
                env.observe_viewpoints(np.zeros(batch, dtype=np.uint8))
                start_state = env.current_frame.copy()
                current_records = start_records.copy()
                previous_state = start_state.copy()
                initial_analyzed_slots = [_slot(start_state[lane], analyzed) for lane in range(batch)]
                initial_defender_slots = [_slot(start_state[lane], defender) for lane in range(batch)]
                initial_actions = np.asarray(
                    [int(slot["action_id"]) for slot in initial_analyzed_slots], dtype=np.uint16
                )
                initial_constrained = np.asarray(
                    [_is_constrained(slot) for slot in initial_defender_slots],
                    dtype=np.bool_,
                )
                initial_analyzed_constrained = np.asarray(
                    [_is_constrained(slot) for slot in initial_analyzed_slots], dtype=np.bool_
                )
                analyzed_hit_tracker = _LogicalHitTracker(
                    move_ids=_move_ids_for_slots(initial_analyzed_slots, action_state_tables),
                    action_ids=np.asarray([int(slot["action_id"]) for slot in initial_analyzed_slots]),
                    action_frames=np.asarray([float(slot["action_frame"]) for slot in initial_analyzed_slots]),
                    already_counted=initial_constrained,
                )
                opponent_hit_tracker = _LogicalHitTracker(
                    move_ids=_move_ids_for_slots(initial_defender_slots, action_state_tables),
                    action_ids=np.asarray([int(slot["action_id"]) for slot in initial_defender_slots]),
                    action_frames=np.asarray([float(slot["action_frame"]) for slot in initial_defender_slots]),
                    already_counted=initial_analyzed_constrained,
                )
                end_state = start_state.copy()
                terminal_evidence: list[dict[str, Any]] = [{} for _ in chunk]

                for step in range(max_steps):
                    decoded = decoded_controllers(agent, outputs)
                    next_records = np.minimum(current_records + 1, len(authority.frames) - 1)
                    next_frames = authority.frame_ids[next_records].astype(np.int32)
                    analyzed_active = next_frames >= analyzed_control_frames
                    defender_active = next_frames >= defender_control_frames
                    controller_rows, inputs = _active_controller_rows(
                        authority,
                        next_records,
                        decoded,
                        analyzed_active,
                        defender_active,
                        analyzed=analyzed,
                        defender=defender,
                        num_players=num_players,
                    )
                    if args.opponent_mode == "neutral":
                        inputs["players"][:, defender] = np.zeros((), dtype=dtypes.input_player_dtype())
                    if stream_history is not None:
                        stream_history[step] = inputs
                    replay_frames = authority.frames[next_records].copy()
                    replay_frames["input"]["players"] = inputs["players"]
                    if env.t == env.length:
                        env.reset_cursor()
                    if args.rng_mode == "replay":
                        env.step_replay(replay_frames, (~resolved).astype(np.uint8))
                    else:
                        env.raw_action_view[env.t] = inputs
                        env.step((~resolved).astype(np.uint8))
                    if env.t == env.length:
                        env.reset_cursor()
                    env.observe_viewpoints(np.zeros(batch, dtype=np.uint8))
                    state = env.current_frame.copy()

                    analyzed_slots = [_slot(state[lane], analyzed) for lane in range(batch)]
                    defender_slots = [_slot(state[lane], defender) for lane in range(batch)]
                    previous_analyzed_slots = [_slot(previous_state[lane], analyzed) for lane in range(batch)]
                    previous_defender_slots = [_slot(previous_state[lane], defender) for lane in range(batch)]
                    dealt_events = np.asarray([
                        float(defender_slots[lane]["percent"])
                        > float(previous_defender_slots[lane]["percent"]) + 0.01
                        for lane in range(batch)
                    ]) & (~resolved)
                    taken_events = np.asarray([
                        float(analyzed_slots[lane]["percent"])
                        > float(previous_analyzed_slots[lane]["percent"]) + 0.01
                        for lane in range(batch)
                    ]) & (~resolved)
                    logical_dealt_events = analyzed_hit_tracker.observe(
                        hit_event=dealt_events,
                        move_ids=_move_ids_for_slots(analyzed_slots, action_state_tables),
                        action_ids=np.asarray([int(slot["action_id"]) for slot in analyzed_slots]),
                        action_frames=np.asarray([float(slot["action_frame"]) for slot in analyzed_slots]),
                    )
                    logical_taken_events = opponent_hit_tracker.observe(
                        hit_event=taken_events,
                        move_ids=_move_ids_for_slots(defender_slots, action_state_tables),
                        action_ids=np.asarray([int(slot["action_id"]) for slot in defender_slots]),
                        action_frames=np.asarray([float(slot["action_frame"]) for slot in defender_slots]),
                    )

                    for lane_index in range(batch):
                        if resolved[lane_index]:
                            continue
                        previous_attacker = _slot(previous_state[lane_index], analyzed)
                        previous_defender = _slot(previous_state[lane_index], defender)
                        attacker = _slot(state[lane_index], analyzed)
                        target = _slot(state[lane_index], defender)
                        dealt_delta = max(0.0, float(target["percent"]) - float(previous_defender["percent"]))
                        taken_delta = max(0.0, float(attacker["percent"]) - float(previous_attacker["percent"]))
                        if dealt_delta > 0:
                            if first_dealt[lane_index] < 0:
                                first_dealt[lane_index] = step + 1
                            followup_damage[lane_index] += dealt_delta
                            followup_hits[lane_index] += 1
                            if _is_constrained(previous_defender):
                                true_followup_hits[lane_index] += 1
                            if first_followup[lane_index] < 0:
                                first_followup[lane_index] = step + 1
                            last_followup[lane_index] = step + 1
                        if taken_delta > 0 and first_taken[lane_index] < 0:
                            first_taken[lane_index] = step + 1

                        if args.defender_takeover_mode != "fixed-delay" and actual_defender_frames[lane_index] == np.iinfo(np.int32).max:
                            phase_is_disadvantage = phase_names[lane_index] == "disadvantage"
                            observed = (
                                bool(logical_taken_events[lane_index])
                                if args.defender_takeover_mode == "observed-opponent-followup"
                                or (args.defender_takeover_mode == "observed-phase-followup" and phase_is_disadvantage)
                                else bool(logical_dealt_events[lane_index])
                            )
                            if observed:
                                actual_defender_frames[lane_index] = int(next_frames[lane_index] + 1)
                                defender_control_frames[lane_index] = actual_defender_frames[lane_index]
                                detected_frames[lane_index] = int(next_frames[lane_index])
                                defender_reasons[lane_index] = (
                                    "observed_opponent_followup" if taken_delta > 0 else "observed_followup"
                                )

                        if step < action_history.shape[0]:
                            action_history[step, lane_index] = int(attacker["action_id"])
                            input_tokens[lane_index].append(_input_token(inputs["players"][lane_index, analyzed]))

                        defender_stock_lost = int(target["stocks"]) < int(_slot(start_state[lane_index], defender)["stocks"])
                        analyzed_stock_lost = int(attacker["stocks"]) < int(_slot(start_state[lane_index], analyzed)["stocks"])
                        eligible, evidence = _escape_eligible(
                            state[lane_index],
                            analyzed=analyzed,
                            defender=defender,
                            min_separation=args.combo_escape_min_separation,
                            pressure_range=args.combo_escape_pressure_range,
                            max_air_height=args.combo_escape_max_air_height,
                            edge_x=args.combo_edge_x_threshold,
                        )
                        terminal_evidence[lane_index] = evidence
                        stable_counts[lane_index] = stable_counts[lane_index] + 1 if eligible else 0
                        minimum = int(args.defense_resolution_min_frames)
                        if step + 1 >= minimum and (
                            bool(evidence["defenderInHit"])
                            or bool(evidence["offstageOrJuggled"])
                            or defender_stock_lost
                        ):
                            defense_resolution_active[lane_index] = True
                        if defender_stock_lost:
                            resolved[lane_index] = True
                            reason[lane_index] = "combo_stock_lost" if args.objective == "combo-extension" else "stock_lost"
                        elif analyzed_stock_lost or (taken_delta > 0 and args.objective == "combo-extension"):
                            resolved[lane_index] = True
                            reason[lane_index] = "combo_reversal" if args.objective == "combo-extension" else "nominal_non_defense"
                        elif args.objective == "combo-extension" and step + 1 >= minimum and stable_counts[lane_index] >= args.combo_escape_stable_frames:
                            resolved[lane_index] = True
                            reason[lane_index] = "combo_escape"
                        elif args.objective == "general" and step + 1 >= minimum and stable_counts[lane_index] >= args.defense_neutral_stable_frames:
                            resolved[lane_index] = True
                            reason[lane_index] = "stable_neutral"
                        if resolved[lane_index]:
                            resolved_step[lane_index] = step + 1
                            end_state[lane_index] = state[lane_index]

                    current_records = next_records
                    previous_state = state
                    if bool(np.all(resolved)):
                        break
                    games = _batch_games(
                        env,
                        controller_rows,
                        num_players,
                        game_buffers,
                        np.zeros(batch, dtype=np.bool_),
                    )
                    policy_controllers, policy_frames = _policy_batch(
                        authority, current_records, delay=policy_delay, num_players=num_players
                    )
                    force_mask = np.zeros(batch * num_players, dtype=np.bool_)
                    for lane_index in range(batch):
                        for player in range(num_players):
                            boundary = analyzed_control_frames[lane_index] if player == analyzed else defender_control_frames[lane_index]
                            policy_index = player * batch + lane_index
                            force_mask[policy_index] = policy_frames[policy_index] < boundary
                    if args.history_mode == "teacher-forced":
                        force_previous_controller(agent, policy_controllers, force_mask)
                    animation_filter.filter(games, np.zeros(batch * num_players, dtype=np.bool_))
                    inference_started = time.perf_counter()
                    outputs = agent.step(games, np.zeros(batch * num_players, dtype=np.bool_))
                    _throttle_gpu_inference(
                        elapsed_s=time.perf_counter() - inference_started,
                        duty_cycle=args.gpu_duty_cycle if args.enable_gpu else 1.0,
                    )

                unresolved = ~resolved
                if np.any(unresolved):
                    end_state[unresolved] = previous_state[unresolved]
                    resolved_step[unresolved] = min(max_steps, step + 1)
                    for lane_index in np.flatnonzero(unresolved):
                        _, terminal_evidence[lane_index] = _escape_eligible(
                            end_state[lane_index],
                            analyzed=analyzed,
                            defender=defender,
                            min_separation=args.combo_escape_min_separation,
                            pressure_range=args.combo_escape_pressure_range,
                            max_air_height=args.combo_escape_max_air_height,
                            edge_x=args.combo_edge_x_threshold,
                        )
                        if args.objective == "combo-extension":
                            reason[lane_index] = "combo_timeout"
                            unresolved_penalty[lane_index] = args.unresolved_defense_penalty
                        elif defense_resolution_active[lane_index]:
                            reason[lane_index] = "timeout_unresolved_defense"
                            unresolved_penalty[lane_index] = args.unresolved_defense_penalty
                        else:
                            reason[lane_index] = "nominal_non_defense"

                stream_path: Path | None = None
                if stream_history is not None:
                    stored = max(1, int(np.max(resolved_step)))
                    stream_path = streams_dir / f"chunk_{chunk_index:04d}_inputs.npz"
                    np.savez_compressed(
                        stream_path,
                        inputs=stream_history[:stored],
                        lane_ids=np.asarray([lane.lane_id for lane in chunk], dtype=np.int32),
                        base_frames=np.asarray([lane.base_frame for lane in chunk], dtype=np.int32),
                        takeover_frames=takeover_frames,
                    )

                for lane_index, lane in enumerate(chunk):
                    start_attacker = _slot(start_state[lane_index], analyzed)
                    start_defender = _slot(start_state[lane_index], defender)
                    final_attacker = _slot(end_state[lane_index], analyzed)
                    final_defender = _slot(end_state[lane_index], defender)
                    dealt = max(0.0, float(final_defender["percent"]) - float(start_defender["percent"]))
                    taken = max(0.0, float(final_attacker["percent"]) - float(start_attacker["percent"]))
                    defender_stocks_lost = max(0, int(start_defender["stocks"]) - int(final_defender["stocks"]))
                    analyzed_stocks_lost = max(0, int(start_attacker["stocks"]) - int(final_attacker["stocks"]))
                    combo_reversed = reason[lane_index] == "combo_reversal"
                    if args.objective == "combo-extension":
                        score = (
                            float(followup_damage[lane_index])
                            + 6.0 * int(followup_hits[lane_index])
                            + 4.0 * int(true_followup_hits[lane_index])
                            + 100.0 * defender_stocks_lost
                            - 1.25 * taken
                            - 100.0 * analyzed_stocks_lost
                            - 25.0 * int(combo_reversed)
                        )
                    else:
                        score = dealt - taken + 80.0 * (defender_stocks_lost - analyzed_stocks_lost)
                    score -= float(unresolved_penalty[lane_index])
                    baseline = frame_metadata.get(lane.base_frame, {})
                    anchor = contact_anchor_frames.get(lane.base_frame)
                    if anchor is not None:
                        baseline_damage, baseline_hits, baseline_kill, baseline_score = _replay_baseline_after_anchor(baseline, anchor)
                    else:
                        baseline_damage = float(baseline.get("original_followup_damage") or 0.0)
                        baseline_hits = int(baseline.get("original_followup_hits") or 0)
                        baseline_kill = bool(baseline.get("original_followup_kill"))
                        baseline_score = float(baseline.get("original_followup_score") or baseline_damage + 6 * baseline_hits + 100 * int(baseline_kill))
                    damage_delta = float(followup_damage[lane_index]) - baseline_damage
                    score_delta = score - baseline_score
                    reason_code = 7 if reason[lane_index] == "combo_timeout" else 8 if combo_reversed else 0
                    outperformed = _combo_outperformed_replay(
                        combo_reversed=combo_reversed,
                        resolution_reason_code=reason_code,
                        model_killed=defender_stocks_lost > 0,
                        replay_killed=baseline_kill,
                        damage_delta=damage_delta,
                    )
                    option_steps = min(int(resolved_step[lane_index]), action_history.shape[0])
                    option = _option_summary(
                        action_ids=[int(value) for value in action_history[:option_steps, lane_index]],
                        input_tokens=input_tokens[lane_index][:option_steps],
                        initial_action_id=int(initial_actions[lane_index]),
                        max_action_segments=args.option_max_action_segments,
                        max_input_segments=args.option_max_input_segments,
                    )
                    dealt_step = int(first_dealt[lane_index])
                    taken_step = int(first_taken[lane_index])
                    if dealt_step < 0 and taken_step < 0:
                        contact_by = None
                        contact_step = None
                    elif dealt_step >= 0 and (taken_step < 0 or dealt_step < taken_step):
                        contact_by, contact_step = "analyzed", dealt_step
                    elif taken_step >= 0 and (dealt_step < 0 or taken_step < dealt_step):
                        contact_by, contact_step = "opponent", taken_step
                    else:
                        contact_by, contact_step = "trade", dealt_step
                    defender_frame = int(actual_defender_frames[lane_index])
                    all_rows.append({
                        "laneId": lane.lane_id,
                        "chunkIndex": chunk_index,
                        "baseFrame": lane.base_frame,
                        "offset": lane.offset,
                        "sampleIndex": lane.sample_index,
                        "startRecord": lane.start_record,
                        "takeoverFrame": lane.takeover_frame,
                        "modelControlFrame": int(analyzed_control_frames[lane_index]),
                        "replayBridgeFrames": int(replay_bridge),
                        "recordedContactAnchorFrame": anchor,
                        "defenderTakeoverFrame": defender_frame,
                        "defenderTakeover": {
                            "reason": str(defender_reasons[lane_index]),
                            "expectedReplayThrowActionId": None,
                            "expectedReplayThrowAction": None,
                            "modelThrowActionId": None,
                            "modelThrowAction": None,
                            "detectedFrame": int(detected_frames[lane_index]) if detected_frames[lane_index] >= 0 else None,
                        },
                        "startState": _state_summary(start_state, lane_index, num_players),
                        "endState": _state_summary(end_state, lane_index, num_players),
                        "resolution": {
                            "step": int(resolved_step[lane_index]),
                            "frame": lane.takeover_frame + max(0, int(resolved_step[lane_index]) - 1),
                            "reason": str(reason[lane_index]),
                            "nominalRolloutFrames": args.rollout_frames,
                            "maxRolloutFrames": max_steps,
                            "unresolvedDefensePenalty": round(float(unresolved_penalty[lane_index]), 3),
                            "stableNeutralRequiredFrames": args.defense_neutral_stable_frames,
                            "escapeStableRequiredFrames": args.combo_escape_stable_frames,
                            "escapeStableObservedFrames": int(stable_counts[lane_index]),
                            "defenderKoConfirmed": defender_stocks_lost > 0,
                            "analyzedKoConfirmed": analyzed_stocks_lost > 0,
                            "terminalEvidence": terminal_evidence[lane_index],
                        },
                        "damageDealt": round(dealt, 3),
                        "damageTaken": round(taken, 3),
                        "neutralContact": {
                            "firstContactBy": contact_by,
                            "firstContactStep": contact_step,
                            "firstContactFrame": lane.takeover_frame + contact_step - 1 if contact_step else None,
                            "firstDamageDealtStep": dealt_step if dealt_step >= 0 else None,
                            "firstDamageDealtFrame": lane.takeover_frame + dealt_step - 1 if dealt_step >= 0 else None,
                            "firstDamageTakenStep": taken_step if taken_step >= 0 else None,
                            "firstDamageTakenFrame": lane.takeover_frame + taken_step - 1 if taken_step >= 0 else None,
                        },
                        "followupDamage": round(float(followup_damage[lane_index]), 3),
                        "followupHits": int(followup_hits[lane_index]),
                        "trueFollowupHits": int(true_followup_hits[lane_index]),
                        "extendedCombo": bool(followup_hits[lane_index] > 0),
                        "trueExtension": bool(true_followup_hits[lane_index] > 0),
                        "comboReversed": combo_reversed,
                        "replayBaseline": {
                            "followupDamage": round(baseline_damage, 3),
                            "followupHits": baseline_hits,
                            "killed": baseline_kill,
                            "score": round(baseline_score, 3),
                        },
                        "damageDeltaVsReplay": round(damage_delta, 3),
                        "scoreDeltaVsReplay": round(score_delta, 3),
                        "outperformedReplay": bool(outperformed),
                        "firstFollowupStep": int(first_followup[lane_index]) if first_followup[lane_index] >= 0 else None,
                        "lastFollowupStep": int(last_followup[lane_index]) if last_followup[lane_index] >= 0 else None,
                        "comboDurationFrames": int(last_followup[lane_index]) if last_followup[lane_index] >= 0 else 0,
                        "initialDefenderConstrained": bool(initial_constrained[lane_index]),
                        "defenderStocksLost": defender_stocks_lost,
                        "analyzedStocksLost": analyzed_stocks_lost,
                        "score": round(score, 3),
                        "optionSignature": option["optionSignature"],
                        "comboOptionSignature": option["comboOptionSignature"],
                        "optionDetailedSignature": option["optionDetailedSignature"],
                        "inputSignature": option["inputSignature"],
                        "option": option,
                        "controllerStream": _stream_ref(
                            stream_path,
                            lane_index,
                            lane,
                            max(1, int(resolved_step[lane_index])),
                            analyzed_port=args.analyzed_port,
                            defender_port=defender + 1,
                            defender_takeover_frame=defender_frame,
                            model_control_frame=int(analyzed_control_frames[lane_index]),
                        ),
                    })
        finally:
            agent.stop()
        chunk_seconds = time.perf_counter() - chunk_started
        timings[f"chunk_{chunk_index}"] = {"total_s": chunk_seconds, "lanes": batch}
        print(json.dumps({"chunk": chunk_index, "lanes": batch, "seconds": round(chunk_seconds, 3)}), flush=True)

    rows_path = out / "lanes.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    timings["total_s"] = time.perf_counter() - started
    summary = {
        "runner": "slippi_ai_review.decomp_simulation",
        "simulationBackend": "decomp",
        "replayAuthority": "peppi-native-exact",
        "replay": str(args.replay.resolve()),
        "model": str(args.model.resolve()),
        "mslRoot": str(args.msl_root.resolve()),
        "slippiAiRoot": str(args.slippi_ai_root.resolve()),
        "analyzedPort": args.analyzed_port,
        "defenderPort": defender + 1,
        "objective": args.objective,
        "baseFrames": base_frames,
        "anchorRecordedContact": args.anchor_recorded_contact,
        "recordedContactAnchorFrames": contact_anchor_frames,
        "offsets": offsets,
        "samplesPerPoint": args.samples_per_point,
        "laneCount": len(lanes),
        "maxBatchLanes": args.max_batch_lanes,
        "rolloutFrames": args.rollout_frames,
        "maxRolloutFrames": max_steps,
        "warmupFrames": args.warmup_frames,
        "policyDelay": policy_delay,
        "defenderDelayFrames": args.defender_delay_frames,
        "defenderTakeoverMode": args.defender_takeover_mode,
        "historyMode": args.history_mode,
        "replayBridgePolicyDelay": args.replay_bridge_policy_delay,
        "replayBridgeFrames": policy_delay if args.replay_bridge_policy_delay else 0,
        "opponentMode": args.opponent_mode,
        "rngMode": args.rng_mode,
        "sampleTemperature": args.sample_temperature,
        "dumpControllerStreams": args.dump_controller_streams,
        "controllerStreamsDir": str(streams_dir) if args.dump_controller_streams else None,
        "enableGpu": args.enable_gpu,
        "lanesJsonl": str(rows_path),
        "groups": _groups(all_rows, objective=args.objective),
        "timings": timings,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "groups"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
