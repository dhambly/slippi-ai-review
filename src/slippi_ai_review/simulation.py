"""Fast replay-seeded MSL takeover grid using vectorized GameBatch observations."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .simulation_support import (
        DEFAULT_MODEL,
        DEFAULT_MSL_ROOT,
        DEFAULT_SLIPPI_AI_ROOT,
        LaneSpec,
        _chunked,
        _load_base_frames,
        _record_for_frame,
        _timestamp,
        _u8_rows,
        parse_csv_ints,
        setup_msl,
    )
except ImportError:
    from simulation_support import (  # noqa: E402
        DEFAULT_MODEL,
        DEFAULT_MSL_ROOT,
        DEFAULT_SLIPPI_AI_ROOT,
        LaneSpec,
        _chunked,
        _load_base_frames,
        _record_for_frame,
        _timestamp,
        _u8_rows,
        parse_csv_ints,
        setup_msl,
    )


BUTTON_BITS = {
    "A": 0x0100,
    "B": 0x0200,
    "X": 0x0400,
    "Y": 0x0800,
    "Z": 0x0010,
    "L": 0x0040,
    "R": 0x0020,
    "D_UP": 0x0008,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a fast batched replay-seeded Phillip takeover grid through MSL.")
    ap.add_argument("--replay", type=Path, required=True)
    ap.add_argument("--analyzed-port", type=int, choices=(1, 2), required=True)
    ap.add_argument("--takeover-frame", type=int, action="append", default=[])
    ap.add_argument("--takeover-frames-json", type=Path, default=None)
    ap.add_argument(
        "--only-base-frame",
        type=int,
        action="append",
        default=[],
        help="Restrict a metadata JSON to selected base frames for focused diagnostics.",
    )
    ap.add_argument("--offsets", default="-12,0,12,24")
    ap.add_argument("--samples-per-point", type=int, default=64)
    ap.add_argument("--max-batch-lanes", type=int, default=1024)
    ap.add_argument("--objective", choices=("general", "combo-extension"), default="general")
    ap.add_argument("--rollout-frames", type=int, default=180)
    ap.add_argument("--defense-resolution-extra-frames", type=int, default=180)
    ap.add_argument(
        "--combo-max-rollout-frames",
        type=int,
        default=420,
        help=(
            "Safety cap for combo/edgeguard resolution. Combo lanes use at least this many "
            "frames so late recoveries and star KOs are not cut at the nominal rollout."
        ),
    )
    ap.add_argument("--defense-resolution-min-frames", type=int, default=None)
    ap.add_argument("--defense-neutral-stable-frames", type=int, default=20)
    ap.add_argument(
        "--combo-escape-stable-frames",
        type=int,
        default=20,
        help=(
            "Consecutive frames the defender must remain in a controllable, onstage, "
            "unpressured state before a combo/edgeguard lane resolves as an escape."
        ),
    )
    ap.add_argument(
        "--combo-escape-min-separation",
        type=float,
        default=12.0,
        help="Minimum player separation for an escape to count even when the attacker is not attacking.",
    )
    ap.add_argument(
        "--combo-escape-pressure-range",
        type=float,
        default=30.0,
        help="Block escape while an attacking/committed attacker is within this distance.",
    )
    ap.add_argument(
        "--combo-escape-max-air-height",
        type=float,
        default=25.0,
        help="Airborne defenders above this height are still treated as being juggled.",
    )
    ap.add_argument(
        "--combo-edge-x-threshold",
        type=float,
        default=55.0,
        help="Conservative horizontal edge threshold for airborne edgeguard resolution.",
    )
    ap.add_argument("--unresolved-defense-penalty", type=float, default=60.0)
    ap.add_argument("--warmup-frames", type=int, default=90)
    ap.add_argument("--defender-delay-frames", type=int, default=60)
    ap.add_argument(
        "--anchor-recorded-contact",
        action="store_true",
        help="For connected punish points, branch after the replay's opening hitlag instead of re-simulating the contact.",
    )
    ap.add_argument("--history-mode", choices=("teacher-forced", "dummy"), default="teacher-forced")
    ap.add_argument("--opponent-mode", choices=("replay", "neutral"), default="replay")
    ap.add_argument("--rng-mode", choices=("replay", "native"), default="replay")
    ap.add_argument("--sample-temperature", type=float, default=1.0)
    ap.add_argument("--option-horizon-frames", type=int, default=90)
    ap.add_argument("--option-max-action-segments", type=int, default=6)
    ap.add_argument("--option-max-input-segments", type=int, default=8)
    ap.add_argument(
        "--dump-controller-streams",
        action="store_true",
        help=(
            "Write compact per-frame INPUT_DTYPE streams for every lane. "
            "This is required to render the exact sampled MSL lane later."
        ),
    )
    ap.add_argument("--msl-root", type=Path, default=DEFAULT_MSL_ROOT)
    ap.add_argument("--slippi-ai-root", type=Path, default=DEFAULT_SLIPPI_AI_ROOT)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--enable-gpu", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("outputs") / "msl_takeover_grid_fast" / f"run_{_timestamp()}")
    return ap.parse_args()


def _map2_nt(func, left, right):
    if isinstance(left, tuple) and hasattr(left, "_fields"):
        return type(left)(*(_map2_nt(func, l_item, r_item) for l_item, r_item in zip(left, right)))
    return func(left, right)


def _disable_gpus_if_needed(enable_gpu: bool) -> None:
    if enable_gpu:
        return
    from slippi_ai import eval_lib  # type: ignore

    eval_lib.disable_gpus()


def _max_rollout_steps(args: argparse.Namespace) -> int:
    requested = int(args.rollout_frames) + int(args.defense_resolution_extra_frames)
    if args.objective == "combo-extension":
        return max(requested, int(args.combo_max_rollout_frames))
    return requested


def _load_state(eval_lib, model_path: Path) -> dict[str, Any]:
    try:
        return eval_lib.load_state(path=str(model_path))
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return eval_lib.load_state({"path": str(model_path)})


def _build_agent(eval_lib, *, state: dict[str, Any], batch_size: int, sample_temperature: float):
    return eval_lib.build_delayed_agent(
        state=state,
        console_delay=0,
        batch_size=batch_size,
        async_inference=False,
        name=None,
        sample_temperature=sample_temperature,
    )


def _input_to_controller_batch(input_arr: np.ndarray, *, player_index: int):
    from slippi_ai import types as sa_types  # type: ignore

    player = input_arr["p"][:, int(player_index)]
    buttons = player["buttons"].astype(np.uint16)
    return sa_types.Controller(
        main_stick=sa_types.Stick(
            (player["main_x"].astype(np.float32) + 80.0) / 160.0,
            (player["main_y"].astype(np.float32) + 80.0) / 160.0,
        ),
        c_stick=sa_types.Stick(
            (player["c_x"].astype(np.float32) + 80.0) / 160.0,
            (player["c_y"].astype(np.float32) + 80.0) / 160.0,
        ),
        shoulder=player["l"].astype(np.float32) / 255.0,
        buttons=sa_types.Buttons(
            **{name: (buttons & np.uint16(bit)) != 0 for name, bit in BUTTON_BITS.items()}
        ),
    )


def _controllers_from_input(input_arr: np.ndarray, *, num_players: int) -> dict[int, Any]:
    return {
        port + 1: _input_to_controller_batch(input_arr, player_index=port)
        for port in range(num_players)
    }


def _stream_ref(
    *,
    stream_path: Path | None,
    env: int,
    lane: LaneSpec,
    stream_frames: int,
    analyzed_port: int,
    defender_port: int,
    defender_takeover_frame: int,
) -> dict[str, Any] | None:
    if stream_path is None:
        return None
    return {
        "format": "npz.input_dtype.v1",
        "path": str(stream_path.resolve()),
        "array": "inputs",
        "env": int(env),
        "laneId": int(lane.lane_id),
        "takeoverFrame": int(lane.takeover_frame),
        "endFrame": int(lane.takeover_frame + stream_frames - 1),
        "frames": int(stream_frames),
        "analyzedPort": int(analyzed_port),
        "defenderPort": int(defender_port),
        "defenderTakeoverFrame": int(defender_takeover_frame),
        "notes": (
            "Both ports are stored from takeoverFrame. Before defenderTakeoverFrame, "
            "the defender stream contains replay inputs; afterward it contains model inputs. "
            "A divergent model throw may advance defenderTakeoverFrame."
        ),
    }


def _concat_port_controllers(controllers: dict[int, Any], *, batch_size: int):
    from slippi_ai import types as sa_types  # type: ignore

    first = controllers[1]
    second = controllers[2]
    return sa_types.Controller(
        main_stick=sa_types.Stick(
            np.concatenate([first.main_stick.x, second.main_stick.x]),
            np.concatenate([first.main_stick.y, second.main_stick.y]),
        ),
        c_stick=sa_types.Stick(
            np.concatenate([first.c_stick.x, second.c_stick.x]),
            np.concatenate([first.c_stick.y, second.c_stick.y]),
        ),
        shoulder=np.concatenate([first.shoulder, second.shoulder]),
        buttons=sa_types.Buttons(
            **{
                name: np.concatenate([getattr(first.buttons, name), getattr(second.buttons, name)])
                for name in first.buttons._fields
            }
        ),
    )


def _force_prev_controller(delayed_agent: Any, forced_controller: Any, force_mask: np.ndarray) -> bool:
    basic_agent = getattr(delayed_agent, "_agent", None)
    policy = getattr(delayed_agent, "policy", None)
    if basic_agent is None or policy is None or not hasattr(basic_agent, "_prev_controller"):
        return False
    force_mask = np.asarray(force_mask, dtype=np.bool_)
    if not bool(np.any(force_mask)):
        return True
    embedding = policy.controller_head.controller_embedding
    forced = embedding.from_state(forced_controller)
    current = basic_agent._prev_controller

    def merge(cur, forced_leaf):
        cur_arr = np.asarray(cur)
        forced_arr = np.asarray(forced_leaf)
        shape = (force_mask.shape[0],) + (1,) * max(0, forced_arr.ndim - 1)
        return np.where(force_mask.reshape(shape), forced_arr, cur_arr)

    basic_agent._prev_controller = _map2_nt(merge, current, forced)
    return True


def _write_encoded_to_input(
    input_arr: np.ndarray,
    *,
    player_index: int,
    encoded_controller: Any,
    source_slice: slice,
    axis_spacing: int,
    shoulder_spacing: int,
) -> None:
    player = input_arr["p"][:, int(player_index)]
    buttons = np.zeros(player.shape[0], dtype=np.uint16)
    for name, bit in BUTTON_BITS.items():
        buttons |= np.asarray(getattr(encoded_controller.buttons, name))[source_slice].astype(np.uint16) * np.uint16(bit)
    player["buttons"] = buttons

    scale_axis = np.float32(160.0 / float(axis_spacing))
    player["main_x"] = np.clip(
        np.rint(np.asarray(encoded_controller.main_stick.x)[source_slice] * scale_axis - 80.0),
        -80,
        80,
    ).astype(np.int8)
    player["main_y"] = np.clip(
        np.rint(np.asarray(encoded_controller.main_stick.y)[source_slice] * scale_axis - 80.0),
        -80,
        80,
    ).astype(np.int8)
    player["c_x"] = np.clip(
        np.rint(np.asarray(encoded_controller.c_stick.x)[source_slice] * scale_axis - 80.0),
        -80,
        80,
    ).astype(np.int8)
    player["c_y"] = np.clip(
        np.rint(np.asarray(encoded_controller.c_stick.y)[source_slice] * scale_axis - 80.0),
        -80,
        80,
    ).astype(np.int8)
    player["l"] = np.clip(
        np.rint(np.asarray(encoded_controller.shoulder)[source_slice] * (255.0 / float(shoulder_spacing))),
        0,
        255,
    ).astype(np.uint8)
    player["r"] = np.uint8(0)


_ACTION_NAME_CACHE: dict[int, str] = {}
_ACTION_SETS_CACHE: dict[str, set[int]] = {}


def _action_name(action_id: int) -> str:
    action_id = int(action_id)
    if action_id not in _ACTION_NAME_CACHE:
        try:
            from melee.enums import Action  # type: ignore

            _ACTION_NAME_CACHE[action_id] = Action(action_id).name
        except Exception:
            _ACTION_NAME_CACHE[action_id] = f"ACTION_{action_id}"
    return _ACTION_NAME_CACHE[action_id]


def _action_set(*, prefixes: tuple[str, ...] = (), names: tuple[str, ...] = ()) -> set[int]:
    key = repr((prefixes, names))
    if key in _ACTION_SETS_CACHE:
        return _ACTION_SETS_CACHE[key]
    out: set[int] = set()
    try:
        from melee.enums import Action  # type: ignore

        for action in Action:
            name = action.name
            if name in names or any(name.startswith(prefix) for prefix in prefixes):
                out.add(int(action.value))
    except Exception:
        pass
    _ACTION_SETS_CACHE[key] = out
    return out


def _throw_action_ids() -> set[int]:
    return _action_set(names=("THROW_FORWARD", "THROW_BACK", "THROW_UP", "THROW_DOWN")) | {219, 220, 221, 222}


def _expected_replay_throw_action(metadata: dict[str, Any]) -> int | None:
    short_names = {
        "fthrow": 219,
        "forward_throw": 219,
        "bthrow": 220,
        "back_throw": 220,
        "uthrow": 221,
        "up_throw": 221,
        "dthrow": 222,
        "down_throw": 222,
    }

    def action_for(move: Any) -> int | None:
        if not isinstance(move, dict):
            return None
        attacker = move.get("attacker") or {}
        action_id = attacker.get("actionStateId")
        if action_id is not None and int(action_id) in _throw_action_ids():
            return int(action_id)
        short_name = str(move.get("moveShortName") or move.get("moveName") or "").strip().lower()
        normalized = short_name.replace("-", "_").replace(" ", "_")
        return short_names.get(normalized)

    for key in ("replay_baseline_move_sequence", "remaining_move_sequence"):
        for move in metadata.get(key) or []:
            action_id = action_for(move)
            if action_id is not None:
                return action_id
    for key in ("sequence_opening_move", "opening_move", "branch_move"):
        action_id = action_for(metadata.get(key))
        if action_id is not None:
            return action_id
    return None


def _divergent_throw_takeover_mask(
    *,
    expected_actions: np.ndarray,
    observed_actions: np.ndarray,
    takeover_steps: np.ndarray,
    step: int,
    resolved_mask: np.ndarray,
) -> np.ndarray:
    return (
        (~np.asarray(resolved_mask, dtype=np.bool_))
        & (expected_actions >= 0)
        & np.isin(observed_actions, np.asarray(sorted(_throw_action_ids()), dtype=np.int16))
        & (observed_actions != expected_actions)
        & (int(step) + 1 < takeover_steps)
    )


def _damage_action_ids() -> set[int]:
    return _action_set(
        prefixes=("DAMAGE_", "DEAD_", "REBIRTH", "ENTRY"),
        names=("TUMBLING",),
    )


def _dead_action_ids() -> set[int]:
    # GALE01's actual death animations are 0..10. Do not use libmelee's
    # DEAD_FALL name here: action 35 is the live helpless-fall state in MSL.
    return set(range(0, 11))


def _neutral_action_ids() -> set[int]:
    return _action_set(
        prefixes=("WALK_", "JUMPING_", "LANDING"),
        names=(
            "STANDING",
            "TURNING",
            "TURNING_RUN",
            "DASHING",
            "RUNNING",
            "RUN_BRAKE",
            "KNEE_BEND",
            "FALLING",
            "EDGE_CATCHING",
            "EDGE_HANGING",
        ),
    )


def _ledge_stable_action_ids() -> set[int]:
    return _action_set(names=("EDGE_CATCHING", "EDGE_HANGING"))


def _combo_lockout_action_ids() -> set[int]:
    """Actions that cannot prove a defender escaped an advantage state.

    This intentionally includes the whole landing, ledge, knockdown, tech, and
    roll animations. The simulator does not expose a universal per-character
    IASA flag, so the robust signal is transition into a player-controlled
    locomotion state, not merely transition out of hitstun.
    """

    return _action_set(
        prefixes=(
            "DAMAGE_",
            "DEAD_",
            "ON_HALO_",
            "ENTRY",
            "CAPTURE_",
            "THROWN_",
            "EDGE_",
            "SHIELD_BREAK_",
            "TECH_",
            "LYING_GROUND_",
            "GROUND_",
            "LANDING",
            "SPECIAL_FALL",
            "SHOULDERED_",
            "BURY",
        ),
        names=(
            "TUMBLING",
            "DEAD_FALL",
            "AIRDODGE",
            "SHIELD_START",
            "SHIELD",
            "SHIELD_RELEASE",
            "SHIELD_STUN",
            "SHIELD_REFLECT",
            "TECH_MISS_UP",
            "LYING_GROUND_UP",
            "LYING_GROUND_UP_HIT",
            "TECH_MISS_DOWN",
            "LYING_GROUND_DOWN",
            "DAMAGE_GROUND",
            "NEUTRAL_TECH",
            "FORWARD_TECH",
            "BACKWARD_TECH",
            "WALL_TECH",
            "WALL_TECH_JUMP",
            "CEILING_TECH",
            "NEUTRAL_GETUP",
            "GETUP_ATTACK",
            "ROLL_FORWARD",
            "ROLL_BACKWARD",
            "SPOTDODGE",
            "SLIDING_OFF_EDGE",
            "BOUNCE_WALL",
            "BOUNCE_CEILING",
            "BUMP_WALL",
            "BUMP_CIELING",
            "REBOUND_STOP",
            "REBOUND",
            "GRABBED",
            "GRABBED_WAIT_HIGH",
            "PUMMELED_HIGH",
            "GRAB_PUMMELED",
            "GRAB_ESCAPE",
            "DAMAGE_SONG",
            "DAMAGE_SONG_WAIT",
            "DAMAGE_SONG_RV",
            "DAMAGE_BIND",
        ),
    )


def _combo_actionable_action_ids() -> set[int]:
    """Player-controlled locomotion states that can establish a neutral reset."""

    return _action_set(
        prefixes=("WALK_", "JUMPING_", "FALLING"),
        names=(
            "STANDING",
            "TURNING",
            "TURNING_RUN",
            "DASHING",
            "RUNNING",
            "RUN_DIRECT",
            "RUN_BRAKE",
            "CROUCH_START",
            "CROUCHING",
            "CROUCH_END",
        ),
    )


def _attacker_pressure_action_ids() -> set[int]:
    return _action_set(
        prefixes=(
            "NEUTRAL_ATTACK_",
            "LOOPING_ATTACK_",
            "FTILT_",
            "FSMASH_",
            "ITEM_THROW_",
        ),
        names=(
            "DASH_ATTACK",
            "UPTILT",
            "DOWNTILT",
            "UPSMASH",
            "DOWNSMASH",
            "NAIR",
            "FAIR",
            "BAIR",
            "UAIR",
            "DAIR",
            "GRAB",
            "GRAB_RUNNING",
            "GRAB_PUMMEL",
            "THROW_FORWARD",
            "THROW_BACK",
            "THROW_UP",
            "THROW_DOWN",
        ),
    )


COMBO_ESCAPE_BLOCK_REASONS = {
    0: "eligible",
    1: "hitlag_or_hitstun",
    2: "committed_defensive_state",
    3: "offstage_or_juggled",
    4: "not_actionable_locomotion",
    5: "attacker_too_close",
    6: "attacker_active_pressure",
}


def _combo_escape_masks(
    frame: np.ndarray,
    *,
    analyzed_idx: int,
    defender_idx: int,
    min_separation: float,
    pressure_range: float,
    max_air_height: float,
    edge_x_threshold: float,
) -> dict[str, np.ndarray]:
    """Classify whether the defender has genuinely escaped advantage this frame."""

    defender = _slot_for_source(frame, defender_idx)
    attacker = _slot_for_source(frame, analyzed_idx)
    lockout_action = _slot_action_mask(defender, _combo_lockout_action_ids())
    actionable_action = _slot_action_mask(defender, _combo_actionable_action_ids())
    in_hit = (defender["hitlag"] > 0) | (defender["hitstun"] > 0)
    on_ground = defender["on_ground"].astype(np.bool_)
    defender_x = defender["pos_x"].astype(np.float32)
    defender_y = defender["pos_y"].astype(np.float32)
    attacker_x = attacker["pos_x"].astype(np.float32)
    attacker_y = attacker["pos_y"].astype(np.float32)
    offstage_like = (~on_ground) & (
        (np.abs(defender_x) > float(edge_x_threshold))
        | (defender_y < 0.0)
        | (defender_y > float(max_air_height))
    )
    separation = np.hypot(defender_x - attacker_x, defender_y - attacker_y).astype(np.float32)
    attacker_action = attacker["action_id"].astype(np.uint16)
    common_pressure = _slot_action_mask(attacker, _attacker_pressure_action_ids())
    # Character-specific attacks and specials live above the common action range.
    character_pressure = attacker_action >= np.uint16(341)
    attacker_pressuring = common_pressure | character_pressure
    too_close = separation < float(min_separation)
    active_pressure = attacker_pressuring & (separation < float(pressure_range))
    escape_eligible = (
        (~in_hit)
        & (~lockout_action)
        & (~offstage_like)
        & actionable_action
        & (~too_close)
        & (~active_pressure)
    )

    block_reason = np.zeros(defender.shape[0], dtype=np.uint8)
    block_reason = np.where(active_pressure, 6, block_reason)
    block_reason = np.where(too_close, 5, block_reason)
    block_reason = np.where(~actionable_action, 4, block_reason)
    block_reason = np.where(offstage_like, 3, block_reason)
    block_reason = np.where(lockout_action, 2, block_reason)
    block_reason = np.where(in_hit, 1, block_reason)
    return {
        "escape_eligible": escape_eligible.astype(np.bool_),
        "in_hit": in_hit.astype(np.bool_),
        "lockout_action": lockout_action.astype(np.bool_),
        "actionable_action": actionable_action.astype(np.bool_),
        "offstage_like": offstage_like.astype(np.bool_),
        "attacker_pressuring": attacker_pressuring.astype(np.bool_),
        "too_close": too_close.astype(np.bool_),
        "active_pressure": active_pressure.astype(np.bool_),
        "separation": separation,
        "block_reason": block_reason.astype(np.uint8),
    }


def _combo_constraint_masks(frame: np.ndarray, *, defender_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """Legacy combo-start signal; terminal resolution uses _combo_escape_masks."""

    defender = _slot_for_source(frame, defender_idx)
    constrained_action = _slot_action_mask(defender, _combo_lockout_action_ids())
    in_hit = (defender["hitlag"] > 0) | (defender["hitstun"] > 0)
    on_ground = defender["on_ground"].astype(np.bool_)
    offstage_like = (~on_ground) & (
        (np.abs(defender["pos_x"].astype(np.float32)) > 55.0)
        | (defender["pos_y"].astype(np.float32) < 0.0)
    )
    constrained = in_hit | constrained_action | offstage_like
    return constrained.astype(np.bool_), in_hit.astype(np.bool_)


def _slot_action_mask(slot: np.ndarray, action_ids: set[int]) -> np.ndarray:
    if not action_ids:
        return np.zeros(slot.shape[0], dtype=np.bool_)
    return np.isin(slot["action_id"].astype(np.uint16), np.asarray(sorted(action_ids), dtype=np.uint16))


def _defense_resolution_masks(
    frame: np.ndarray,
    *,
    start_frame: np.ndarray,
    analyzed_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    analyzed = _slot_for_source(frame, analyzed_idx)
    start_analyzed = _slot_for_source(start_frame, analyzed_idx)
    damage_or_dead = _slot_action_mask(analyzed, _damage_action_ids())
    ledge_stable = _slot_action_mask(analyzed, _ledge_stable_action_ids())
    neutralish = _slot_action_mask(analyzed, _neutral_action_ids())
    stocks_lost = start_analyzed["stocks"].astype(np.int16) > analyzed["stocks"].astype(np.int16)
    in_hit = (analyzed["hitlag"] > 0) | (analyzed["hitstun"] > 0) | damage_or_dead
    on_ground = analyzed["on_ground"].astype(np.bool_)
    # This is intentionally conservative across stages. If the player is airborne
    # and far from center or below stage height, do not call the defense resolved.
    offstage_like = (~on_ground) & ((np.abs(analyzed["pos_x"].astype(np.float32)) > 55.0) | (analyzed["pos_y"].astype(np.float32) < 0.0))
    stable_neutral = (~in_hit) & (ledge_stable | (on_ground & neutralish))
    unresolved_defense = in_hit | offstage_like
    return stocks_lost.astype(np.bool_), stable_neutral.astype(np.bool_), unresolved_defense.astype(np.bool_)


def _axis_dir(x: int, y: int, *, threshold: int = 35) -> str:
    parts = []
    if y >= threshold:
        parts.append("up")
    elif y <= -threshold:
        parts.append("down")
    if x >= threshold:
        parts.append("right")
    elif x <= -threshold:
        parts.append("left")
    return "+".join(parts) if parts else "neutral"


def _input_token(player: np.void) -> str:
    buttons = int(player["buttons"])
    names = [name for name, bit in BUTTON_BITS.items() if buttons & int(bit)]
    button_text = "+".join(names) if names else "no_button"
    main = _axis_dir(int(player["main_x"]), int(player["main_y"]))
    c_stick = _axis_dir(int(player["c_x"]), int(player["c_y"]))
    shield = "shield" if int(player["l"]) >= 40 or int(player["r"]) >= 40 else ""
    parts = [button_text, f"main_{main}"]
    if c_stick != "neutral":
        parts.append(f"c_{c_stick}")
    if shield:
        parts.append(shield)
    return "+".join(parts)


def _compress_values(values: list[Any], *, max_segments: int) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for idx, value in enumerate(values):
        if segments and segments[-1]["value"] == value:
            segments[-1]["endStep"] = idx
            segments[-1]["frames"] += 1
        else:
            segments.append({"value": value, "startStep": idx, "endStep": idx, "frames": 1})
    return segments[:max_segments]


def _option_summary(
    *,
    action_ids: list[int],
    input_tokens: list[str],
    initial_action_id: int,
    max_action_segments: int,
    max_input_segments: int,
) -> dict[str, Any]:
    action_segments = _compress_values([int(x) for x in action_ids], max_segments=max(1, len(action_ids)))
    all_named_action_segments = [
        {
            "actionId": int(segment["value"]),
            "actionName": _action_name(int(segment["value"])),
            "startStep": int(segment["startStep"]),
            "endStep": int(segment["endStep"]),
            "frames": int(segment["frames"]),
        }
        for segment in action_segments
    ]
    named_action_segments = all_named_action_segments[: max_action_segments + 1]
    decision_segments = named_action_segments
    if len(decision_segments) > 1 and int(decision_segments[0]["actionId"]) == int(initial_action_id):
        decision_segments = decision_segments[1:]
    decision_segments = decision_segments[:max_action_segments] or named_action_segments[:max_action_segments]
    action_signature = ">".join(segment["actionName"] for segment in decision_segments) or _action_name(initial_action_id)
    detailed_action_signature = ">".join(
        f"{segment['actionName']}:{segment['frames']}f" for segment in decision_segments
    ) or f"{_action_name(initial_action_id)}:0f"

    input_segments = _compress_values(input_tokens, max_segments=max_input_segments)
    input_signature = ">".join(str(segment["value"]) for segment in input_segments) if input_segments else "no_input"
    detailed_input_signature = ">".join(
        f"{segment['value']}:{segment['frames']}f" for segment in input_segments
    ) if input_segments else "no_input"
    return {
        "optionSignature": action_signature,
        "optionDetailedSignature": detailed_action_signature,
        "inputSignature": input_signature,
        "inputDetailedSignature": detailed_input_signature,
        "initialActionId": int(initial_action_id),
        "initialActionName": _action_name(int(initial_action_id)),
        "actionSegments": decision_segments,
        "rawActionSegments": named_action_segments[:max_action_segments],
        "inputSegments": [
            {
                "token": str(segment["value"]),
                "startStep": int(segment["startStep"]),
                "endStep": int(segment["endStep"]),
                "frames": int(segment["frames"]),
            }
            for segment in input_segments
        ],
        "comboOptionSignature": _combo_option_signature(all_named_action_segments, initial_action_id=initial_action_id),
    }


def _combo_option_signature(action_segments: list[dict[str, Any]], *, initial_action_id: int) -> str:
    passive_names = {
        "STANDING",
        "TURNING",
        "TURNING_RUN",
        "DASHING",
        "RUNNING",
        "RUN_BRAKE",
        "CROUCH_START",
        "CROUCHING",
        "CROUCH_END",
        "KNEE_BEND",
        "FALLING",
        "FALLING_AERIAL",
    }
    passive_prefixes = ("WALK_", "JUMPING_", "LANDING")
    current_action = int(initial_action_id)
    initial_name = _action_name(current_action)
    if "GRAB" in initial_name or initial_name.startswith("CATCH"):
        for segment in action_segments:
            name = str(segment["actionName"])
            if name.startswith("THROW_"):
                return name
    for index, segment in enumerate(action_segments):
        action_id = int(segment["actionId"])
        name = str(segment["actionName"])
        if (
            (index == 0 and action_id == current_action)
            or name.endswith("_LANDING")
            or name in passive_names
            or name.startswith(passive_prefixes)
        ):
            continue
        if name.startswith("DAMAGE_") or name == "TUMBLING":
            return "REVERSAL"
        if name.startswith("SHIELD"):
            return "SHIELD"
        if name.startswith("SHINE"):
            return "DOWN_B"
        if "GRAB" in name or name.startswith("CATCH"):
            return "GRAB"
        for root in (
            "JAB",
            "DASH_ATTACK",
            "FTILT",
            "UPTILT",
            "DOWNTILT",
            "FSMASH",
            "UPSMASH",
            "DOWNSMASH",
            "NAIR",
            "FAIR",
            "BAIR",
            "UAIR",
            "DAIR",
            "NEUTRAL_B",
            "SIDE_B",
            "UP_B",
            "DOWN_B",
        ):
            if name.startswith(root):
                return root
        return name
    return "NO_COMMITMENT"


def _load_frame_metadata(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("frames", "candidates", "attempted_candidates", "successful_candidates", "groups", "targets"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return {}
    out: dict[int, dict[str, Any]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        for key in ("base_frame", "baseFrame", "takeover_frame", "takeoverFrame", "start_frame", "startFrame", "frame"):
            if item.get(key) is not None:
                baseline = item.get("replay_baseline")
                out[int(item[key])] = {**baseline, **item} if isinstance(baseline, dict) else item
                break
    return out


def _recorded_contact_frame(metadata: dict[str, Any]) -> int | None:
    evidence = metadata.get("point_evidence") or {}
    for key in ("hit_frame", "connected_hit_frame", "first_hit_frame"):
        if evidence.get(key) is not None:
            return int(evidence[key])
    move = metadata.get("branch_move") or metadata.get("opening_move") or {}
    return int(move["frame"]) if move.get("frame") is not None else None


def _post_contact_anchor_record(
    buffers: Any,
    frame_ids: np.ndarray,
    metadata: dict[str, Any],
    *,
    analyzed_idx: int,
    defender_idx: int,
) -> int | None:
    """Return the first recorded physics frame after opening hitlag resolves.

    MSL is approximate enough that re-simulating an already-recorded contact can
    move the hit by a frame and therefore change DI. Punish review should hold
    the recorded opening and its defensive result fixed, then branch.
    """
    contact_frame = _recorded_contact_frame(metadata)
    if contact_frame is None:
        return None
    contact_record = _record_for_frame(frame_ids, contact_frame)
    last_hitlag_record: int | None = None
    upper = min(int(buffers.num_records), contact_record + 30)
    for record in range(max(0, contact_record - 1), upper):
        hitlag = buffers.seed_t["hitlag"][record]
        if int(hitlag[analyzed_idx]) > 0 or int(hitlag[defender_idx]) > 0:
            last_hitlag_record = record
            continue
        if last_hitlag_record is None:
            continue
        prior = buffers.seed_t[last_hitlag_record]
        current = buffers.seed_t[record]
        progressed = any(
            int(current["action_frame"][player]) != int(prior["action_frame"][player])
            or float(current["pos_x"][player]) != float(prior["pos_x"][player])
            or float(current["pos_y"][player]) != float(prior["pos_y"][player])
            for player in (analyzed_idx, defender_idx)
        )
        if progressed:
            return record
    return None


def _replay_baseline_after_anchor(metadata: dict[str, Any], anchor_frame: int) -> tuple[float, int, bool, float]:
    moves = metadata.get("remaining_move_sequence") or metadata.get("replay_baseline_move_sequence") or []
    remaining = [move for move in moves if int(move.get("frame") or -10**9) > int(anchor_frame)]
    if not remaining:
        return 0.0, 0, False, 0.0
    damage = float(sum(float(move.get("damage") or 0.0) for move in remaining))
    hits = int(sum(max(1, int(move.get("hitCount") or 1)) for move in remaining))
    original_kill = bool(metadata.get("original_followup_kill"))
    original_moves = metadata.get("remaining_move_sequence") or []
    kill = original_kill and bool(original_moves) and remaining[-1] is original_moves[-1]
    score = damage + 6.0 * hits + 100.0 * int(kill)
    return damage, hits, kill, score


def _slot_for_source(frame: np.ndarray, source_player: int) -> np.ndarray:
    slots = frame["slots"]
    for slot_idx in range(slots.shape[1]):
        slot = slots[:, slot_idx]
        present = slot["present"].astype(np.bool_)
        source = slot["source_player"]
        if np.all((~present) | (source == int(source_player))) and np.any(present):
            return slot
    raise RuntimeError(f"gamestate missing source player {source_player}")


def _score_batch(
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    *,
    analyzed_idx: int,
    defender_idx: int,
    unresolved_penalty: np.ndarray | None = None,
):
    start_analyzed = _slot_for_source(start_frame, analyzed_idx)
    start_defender = _slot_for_source(start_frame, defender_idx)
    end_analyzed = _slot_for_source(end_frame, analyzed_idx)
    end_defender = _slot_for_source(end_frame, defender_idx)
    dealt = np.maximum(0.0, end_defender["percent"] - start_defender["percent"])
    taken = np.maximum(0.0, end_analyzed["percent"] - start_analyzed["percent"])
    defender_stocks_lost = np.maximum(0, start_defender["stocks"].astype(np.int16) - end_defender["stocks"].astype(np.int16))
    analyzed_stocks_lost = np.maximum(0, start_analyzed["stocks"].astype(np.int16) - end_analyzed["stocks"].astype(np.int16))
    score = dealt - taken + 80.0 * (defender_stocks_lost - analyzed_stocks_lost)
    if unresolved_penalty is not None:
        score = score - np.asarray(unresolved_penalty, dtype=np.float32)
    return dealt, taken, defender_stocks_lost, analyzed_stocks_lost, score


def _combo_score_batch(
    *,
    followup_damage: np.ndarray,
    followup_hits: np.ndarray,
    true_followup_hits: np.ndarray,
    damage_taken: np.ndarray,
    defender_stocks_lost: np.ndarray,
    analyzed_stocks_lost: np.ndarray,
    reversed_combo: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(followup_damage, dtype=np.float32)
        + 6.0 * np.asarray(followup_hits, dtype=np.float32)
        + 4.0 * np.asarray(true_followup_hits, dtype=np.float32)
        + 100.0 * np.asarray(defender_stocks_lost, dtype=np.float32)
        - 1.25 * np.asarray(damage_taken, dtype=np.float32)
        - 100.0 * np.asarray(analyzed_stocks_lost, dtype=np.float32)
        - 25.0 * np.asarray(reversed_combo, dtype=np.float32)
    )


def _combo_outperformed_replay(
    *,
    combo_reversed: bool,
    resolution_reason_code: int,
    model_killed: bool,
    replay_killed: bool,
    damage_delta: float,
) -> bool:
    if combo_reversed or int(resolution_reason_code) == 7:
        return False
    if model_killed and not replay_killed:
        return True
    if replay_killed:
        return False
    return not model_killed and float(damage_delta) >= 3.0


def _state_summary(frame: np.ndarray, env: int, *, num_players: int) -> dict[str, Any]:
    slots = [_slot_for_source(frame, source)[env] for source in range(num_players)]
    return {
        "frame": int(frame["frame_id"][env]),
        "actions": [int(slot["action_id"]) for slot in slots],
        "actionFrames": [float(slot["action_frame"]) for slot in slots],
        "percent": [round(float(slot["percent"]), 3) for slot in slots],
        "stocks": [int(slot["stocks"]) for slot in slots],
        "x": [round(float(slot["pos_x"]), 3) for slot in slots],
        "y": [round(float(slot["pos_y"]), 3) for slot in slots],
        "hitlag": [int(slot["hitlag"]) for slot in slots],
        "hitstun": [int(slot["hitstun"]) for slot in slots],
    }


def _groups(rows: list[dict[str, Any]], *, objective: str = "general") -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
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
                "followupDamage": [],
                "followupHits": [],
                "trueFollowupHits": [],
                "comboDurationFrames": [],
                "extensions": 0,
                "trueExtensions": 0,
                "reversals": 0,
                "outperformedReplay": 0,
                "scoreDeltaVsReplay": [],
                "damageDeltaVsReplay": [],
                "optionsBySignature": {},
            },
        )
        bucket["samples"] += 1
        bucket["scores"].append(float(row["score"]))
        bucket["damageDealt"].append(float(row["damageDealt"]))
        bucket["damageTaken"].append(float(row["damageTaken"]))
        bucket["defenderStocksLost"] += int(row["defenderStocksLost"])
        bucket["analyzedStocksLost"] += int(row["analyzedStocksLost"])
        bucket["followupDamage"].append(float(row.get("followupDamage") or 0.0))
        bucket["followupHits"].append(int(row.get("followupHits") or 0))
        bucket["trueFollowupHits"].append(int(row.get("trueFollowupHits") or 0))
        bucket["comboDurationFrames"].append(int(row.get("comboDurationFrames") or 0))
        bucket["extensions"] += int((row.get("followupHits") or 0) > 0)
        bucket["trueExtensions"] += int((row.get("trueFollowupHits") or 0) > 0)
        bucket["reversals"] += int(bool(row.get("comboReversed")))
        bucket["outperformedReplay"] += int(bool(row.get("outperformedReplay")))
        bucket["scoreDeltaVsReplay"].append(float(row.get("scoreDeltaVsReplay") or 0.0))
        bucket["damageDeltaVsReplay"].append(float(row.get("damageDeltaVsReplay") or 0.0))
        signature_key = "comboOptionSignature" if objective == "combo-extension" else "optionSignature"
        signature = str(row.get(signature_key) or row.get("optionSignature") or "unknown")
        option_bucket = bucket["optionsBySignature"].setdefault(
            signature,
            {
                "optionSignature": signature,
                "samples": 0,
                "scores": [],
                "damageDealt": [],
                "damageTaken": [],
                "defenderStocksLost": 0,
                "analyzedStocksLost": 0,
                "followupDamage": [],
                "followupHits": [],
                "trueFollowupHits": [],
                "comboDurationFrames": [],
                "extensions": 0,
                "trueExtensions": 0,
                "reversals": 0,
                "outperformedReplay": 0,
                "scoreDeltaVsReplay": [],
                "damageDeltaVsReplay": [],
                "example": row.get("option") or {},
                "inputSignatures": {},
            },
        )
        option_bucket["samples"] += 1
        option_bucket["scores"].append(float(row["score"]))
        option_bucket["damageDealt"].append(float(row["damageDealt"]))
        option_bucket["damageTaken"].append(float(row["damageTaken"]))
        option_bucket["defenderStocksLost"] += int(row["defenderStocksLost"])
        option_bucket["analyzedStocksLost"] += int(row["analyzedStocksLost"])
        option_bucket["followupDamage"].append(float(row.get("followupDamage") or 0.0))
        option_bucket["followupHits"].append(int(row.get("followupHits") or 0))
        option_bucket["trueFollowupHits"].append(int(row.get("trueFollowupHits") or 0))
        option_bucket["comboDurationFrames"].append(int(row.get("comboDurationFrames") or 0))
        option_bucket["extensions"] += int((row.get("followupHits") or 0) > 0)
        option_bucket["trueExtensions"] += int((row.get("trueFollowupHits") or 0) > 0)
        option_bucket["reversals"] += int(bool(row.get("comboReversed")))
        option_bucket["outperformedReplay"] += int(bool(row.get("outperformedReplay")))
        option_bucket["scoreDeltaVsReplay"].append(float(row.get("scoreDeltaVsReplay") or 0.0))
        option_bucket["damageDeltaVsReplay"].append(float(row.get("damageDeltaVsReplay") or 0.0))
        input_signature = str(row.get("inputSignature") or "unknown")
        option_bucket["inputSignatures"][input_signature] = int(option_bucket["inputSignatures"].get(input_signature, 0)) + 1
    out = []
    for bucket in grouped.values():
        scores = np.asarray(bucket.pop("scores"), dtype=np.float32)
        dealt = np.asarray(bucket.pop("damageDealt"), dtype=np.float32)
        taken = np.asarray(bucket.pop("damageTaken"), dtype=np.float32)
        followup_damage = np.asarray(bucket.pop("followupDamage"), dtype=np.float32)
        followup_hits = np.asarray(bucket.pop("followupHits"), dtype=np.float32)
        true_followup_hits = np.asarray(bucket.pop("trueFollowupHits"), dtype=np.float32)
        combo_duration = np.asarray(bucket.pop("comboDurationFrames"), dtype=np.float32)
        score_delta = np.asarray(bucket.pop("scoreDeltaVsReplay"), dtype=np.float32)
        damage_delta = np.asarray(bucket.pop("damageDeltaVsReplay"), dtype=np.float32)
        option_buckets = []
        raw_options = bucket.pop("optionsBySignature")
        for option in raw_options.values():
            option_scores = np.asarray(option.pop("scores"), dtype=np.float32)
            option_dealt = np.asarray(option.pop("damageDealt"), dtype=np.float32)
            option_taken = np.asarray(option.pop("damageTaken"), dtype=np.float32)
            option_followup_damage = np.asarray(option.pop("followupDamage"), dtype=np.float32)
            option_followup_hits = np.asarray(option.pop("followupHits"), dtype=np.float32)
            option_true_followup_hits = np.asarray(option.pop("trueFollowupHits"), dtype=np.float32)
            option_combo_duration = np.asarray(option.pop("comboDurationFrames"), dtype=np.float32)
            option_score_delta = np.asarray(option.pop("scoreDeltaVsReplay"), dtype=np.float32)
            option_damage_delta = np.asarray(option.pop("damageDeltaVsReplay"), dtype=np.float32)
            input_counts = option.pop("inputSignatures")
            option_samples = int(option["samples"])
            option_buckets.append(
                {
                    **option,
                    "share": round(float(option_samples) / max(1, int(bucket["samples"])), 4),
                    "meanScore": round(float(np.mean(option_scores)), 3),
                    "stdScore": round(float(np.std(option_scores)), 3),
                    "p25Score": round(float(np.percentile(option_scores, 25)), 3),
                    "p75Score": round(float(np.percentile(option_scores, 75)), 3),
                    "bestScore": round(float(np.max(option_scores)), 3),
                    "worstScore": round(float(np.min(option_scores)), 3),
                    "meanDamageDealt": round(float(np.mean(option_dealt)), 3),
                    "meanDamageTaken": round(float(np.mean(option_taken)), 3),
                    "killRate": round(float(option["defenderStocksLost"]) / max(1, option_samples), 4),
                    "selfDeathRate": round(float(option["analyzedStocksLost"]) / max(1, option_samples), 4),
                    "extensionRate": round(float(option["extensions"]) / max(1, option_samples), 4),
                    "trueExtensionRate": round(float(option["trueExtensions"]) / max(1, option_samples), 4),
                    "reversalRate": round(float(option["reversals"]) / max(1, option_samples), 4),
                    "meanFollowupDamage": round(float(np.mean(option_followup_damage)), 3),
                    "meanFollowupHits": round(float(np.mean(option_followup_hits)), 3),
                    "meanTrueFollowupHits": round(float(np.mean(option_true_followup_hits)), 3),
                    "meanComboDurationFrames": round(float(np.mean(option_combo_duration)), 3),
                    "improvementRate": round(float(option["outperformedReplay"]) / max(1, option_samples), 4),
                    "meanScoreDeltaVsReplay": round(float(np.mean(option_score_delta)), 3),
                    "p25ScoreDeltaVsReplay": round(float(np.percentile(option_score_delta, 25)), 3),
                    "meanDamageDeltaVsReplay": round(float(np.mean(option_damage_delta)), 3),
                    "topInputSignatures": [
                        {"inputSignature": sig, "samples": count, "share": round(float(count) / max(1, option_samples), 4)}
                        for sig, count in sorted(input_counts.items(), key=lambda item: item[1], reverse=True)[:5]
                    ],
                }
            )
        if objective == "combo-extension":
            option_buckets.sort(
                key=lambda item: (
                    float(item["improvementRate"]),
                    float(item["p25ScoreDeltaVsReplay"]),
                    float(item["p25Score"]),
                    float(item["extensionRate"]),
                    float(item["meanScore"]),
                    int(item["samples"]),
                ),
                reverse=True,
            )
        else:
            option_buckets.sort(
                key=lambda item: (
                    float(item["meanScore"]),
                    int(item["samples"]),
                    float(item["bestScore"]),
                ),
                reverse=True,
            )
        common_option_buckets = sorted(
            option_buckets,
            key=lambda item: (
                int(item["samples"]),
                float(item["meanScore"]),
                float(item["bestScore"]),
            ),
            reverse=True,
        )
        top_option = max(option_buckets, key=lambda item: int(item["samples"])) if option_buckets else None
        best_mean_option = option_buckets[0] if option_buckets else None
        out.append(
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
                "extensionRate": round(float(bucket["extensions"]) / max(1, int(bucket["samples"])), 4),
                "trueExtensionRate": round(float(bucket["trueExtensions"]) / max(1, int(bucket["samples"])), 4),
                "reversalRate": round(float(bucket["reversals"]) / max(1, int(bucket["samples"])), 4),
                "meanFollowupDamage": round(float(np.mean(followup_damage)), 3),
                "meanFollowupHits": round(float(np.mean(followup_hits)), 3),
                "meanTrueFollowupHits": round(float(np.mean(true_followup_hits)), 3),
                "meanComboDurationFrames": round(float(np.mean(combo_duration)), 3),
                "improvementRate": round(float(bucket["outperformedReplay"]) / max(1, int(bucket["samples"])), 4),
                "meanScoreDeltaVsReplay": round(float(np.mean(score_delta)), 3),
                "p25ScoreDeltaVsReplay": round(float(np.percentile(score_delta, 25)), 3),
                "meanDamageDeltaVsReplay": round(float(np.mean(damage_delta)), 3),
                "optionCount": len(option_buckets),
                "topOptionShare": round(float(top_option["samples"]) / max(1, int(bucket["samples"])), 4) if top_option else 0.0,
                "topOptionSignature": top_option["optionSignature"] if top_option else None,
                "bestMeanOptionSignature": best_mean_option["optionSignature"] if best_mean_option else None,
                "bestMeanOptionScore": best_mean_option["meanScore"] if best_mean_option else None,
                "mostCommonOptions": common_option_buckets[:10],
                "options": option_buckets[:25],
            }
        )
    if objective == "combo-extension":
        out.sort(
            key=lambda item: (
                float(item["improvementRate"]),
                float(item["p25ScoreDeltaVsReplay"]),
                float(item["p25Score"]),
                float(item["extensionRate"]),
                float(item["meanScore"]),
            ),
            reverse=True,
        )
    else:
        out.sort(key=lambda item: (float(item["meanScore"]), float(item["bestScore"])), reverse=True)
    return out


def main() -> int:
    args = parse_args()
    if args.samples_per_point <= 0:
        raise ValueError("--samples-per-point must be positive")
    if args.option_horizon_frames < 0:
        raise ValueError("--option-horizon-frames must be non-negative")
    if args.defense_resolution_extra_frames < 0:
        raise ValueError("--defense-resolution-extra-frames must be non-negative")
    if args.combo_max_rollout_frames < args.rollout_frames:
        raise ValueError("--combo-max-rollout-frames must be >= --rollout-frames")
    if args.defense_neutral_stable_frames < 1:
        raise ValueError("--defense-neutral-stable-frames must be at least 1")
    if args.combo_escape_stable_frames < 1:
        raise ValueError("--combo-escape-stable-frames must be at least 1")
    if args.combo_escape_min_separation < 0:
        raise ValueError("--combo-escape-min-separation must be non-negative")
    if args.combo_escape_pressure_range < args.combo_escape_min_separation:
        raise ValueError("--combo-escape-pressure-range must be >= --combo-escape-min-separation")
    if args.combo_escape_max_air_height < 0:
        raise ValueError("--combo-escape-max-air-height must be non-negative")
    if args.combo_edge_x_threshold <= 0:
        raise ValueError("--combo-edge-x-threshold must be positive")
    if args.defense_resolution_min_frames is None:
        args.defense_resolution_min_frames = args.rollout_frames
    if args.defense_resolution_min_frames < 1:
        raise ValueError("--defense-resolution-min-frames must be at least 1")
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    setup_msl(args.msl_root)
    if str(args.slippi_ai_root.resolve()) not in sys.path:
        sys.path.insert(0, str(args.slippi_ai_root.resolve()))

    import melee_sim.dtypes as msl_dtypes  # type: ignore
    import msl_binding  # type: ignore
    from slippi_ai import eval_lib  # type: ignore
    from slippi_ai.sim_env.observations import GameBatchBuffers  # type: ignore
    from tools.eval.validation_dtypes import INPUT_DTYPE  # type: ignore
    from tools.slippi.validation_buffer_builder import build_validation_buffers_from_slp  # type: ignore

    _disable_gpus_if_needed(args.enable_gpu)

    replay = args.replay.resolve()
    timings: dict[str, Any] = {}
    phase = time.perf_counter()
    buffers = build_validation_buffers_from_slp(slp_path=str(replay))
    timings["build_validation_buffers_s"] = time.perf_counter() - phase

    phase = time.perf_counter()
    state = _load_state(eval_lib, args.model.resolve())
    controller_cfg = state.get("config", {}).get("embed", {}).get("controller", {})
    axis_spacing = int(controller_cfg.get("axis_spacing", 32))
    shoulder_spacing = int(controller_cfg.get("shoulder_spacing", 10))
    timings["load_state_s"] = time.perf_counter() - phase

    frame_ids = np.asarray(buffers.seed_t["frame_id"], dtype=np.int32)
    offsets = parse_csv_ints(args.offsets, name="--offsets")
    base_frames = _load_base_frames(args)
    if args.only_base_frame:
        requested_frames = set(int(frame) for frame in args.only_base_frame)
        base_frames = [frame for frame in base_frames if frame in requested_frames]
        missing_requested = sorted(requested_frames - set(base_frames))
        if missing_requested:
            raise ValueError(f"--only-base-frame values are absent from takeover metadata: {missing_requested}")
    frame_metadata = _load_frame_metadata(args.takeover_frames_json)
    num_players = int(buffers.num_players)
    analyzed_idx = int(args.analyzed_port) - 1
    defender_idx = 1 if analyzed_idx == 0 else 0
    if num_players < 2:
        raise ValueError(f"expected a singles replay, got num_players={num_players}")

    contact_anchor_frames: dict[int, int] = {}
    if args.anchor_recorded_contact:
        for base in base_frames:
            anchor_record = _post_contact_anchor_record(
                buffers,
                frame_ids,
                frame_metadata.get(int(base), {}),
                analyzed_idx=analyzed_idx,
                defender_idx=defender_idx,
            )
            if anchor_record is not None:
                contact_anchor_frames[int(base)] = int(frame_ids[anchor_record])

    lane_specs: list[LaneSpec] = []
    lane_id = 0
    for base in base_frames:
        branch_base = contact_anchor_frames.get(int(base), int(base))
        for offset in offsets:
            start_record = _record_for_frame(frame_ids, int(branch_base) + int(offset))
            takeover_frame = int(frame_ids[start_record])
            for sample_index in range(args.samples_per_point):
                lane_specs.append(
                    LaneSpec(
                        lane_id=lane_id,
                        base_frame=int(base),
                        offset=int(offset),
                        sample_index=int(sample_index),
                        start_record=start_record,
                        takeover_frame=takeover_frame,
                        defender_takeover_frame=takeover_frame + int(args.defender_delay_frames),
                    )
                )
                lane_id += 1

    all_rows: list[dict[str, Any]] = []
    chunks = list(_chunked(lane_specs, args.max_batch_lanes))
    gamestate_dtype = msl_dtypes.gamestate_dtype()
    streams_dir = out_dir / "controller_streams"
    if args.dump_controller_streams:
        streams_dir.mkdir(parents=True, exist_ok=True)

    for chunk_index, chunk in enumerate(chunks):
        chunk_t0 = time.perf_counter()
        chunk_timings = {
            "build_agent_s": 0.0,
            "warmup_s": 0.0,
            "seed_s": 0.0,
            "observation_fill_s": 0.0,
            "force_s": 0.0,
            "agent_step_s": 0.0,
            "input_write_s": 0.0,
            "sim_step_s": 0.0,
            "summarize_s": 0.0,
        }
        batch_size = len(chunk)
        port_batch_size = batch_size * 2
        handle = msl_binding.init(batch_size, num_players)
        processed_input = np.zeros(batch_size, dtype=INPUT_DTYPE)
        processed_input_u8 = _u8_rows(processed_input)
        agent = None
        try:
            phase = time.perf_counter()
            agent = _build_agent(
                eval_lib,
                state=state,
                batch_size=port_batch_size,
                sample_temperature=float(args.sample_temperature),
            )
            agent.start()
            game_buffers = GameBatchBuffers(batch_size)
            chunk_timings["build_agent_s"] += time.perf_counter() - phase

            viewpoint = np.zeros(batch_size, dtype=np.uint8)
            gamestate_raw = np.zeros((batch_size, gamestate_dtype.itemsize), dtype=np.uint8)
            gamestate = gamestate_raw.view(gamestate_dtype).reshape(batch_size)
            start_records = np.asarray([lane.start_record for lane in chunk], dtype=np.int64)

            if args.history_mode == "teacher-forced":
                phase_warm = time.perf_counter()
                needs_reset = np.ones(batch_size, dtype=np.bool_)
                max_warmup_steps = max(min(args.warmup_frames, lane.start_record) for lane in chunk)
                for warm_step in range(max_warmup_steps):
                    warm_records = np.maximum(0, start_records - max_warmup_steps + warm_step)
                    warm_records = np.minimum(warm_records, np.maximum(0, start_records - 1))
                    msl_binding.reseed_seed_rollout(handle, _u8_rows(buffers.seed_t[warm_records]))
                    msl_binding.write_gamestate(handle, viewpoint, gamestate_raw)
                    current_controllers = _controllers_from_input(
                        buffers.input_t[warm_records],
                        num_players=num_players,
                    )
                    prev_controllers = _controllers_from_input(
                        buffers.prev_input_t[warm_records],
                        num_players=num_players,
                    )
                    game_buffers.fill(gamestate, needs_reset, controllers=current_controllers)
                    _force_prev_controller(
                        agent,
                        _concat_port_controllers(prev_controllers, batch_size=batch_size),
                        np.ones(port_batch_size, dtype=np.bool_),
                    )
                    agent.step(game_buffers.game, game_buffers.needs_reset)
                    needs_reset[:] = False
                chunk_timings["warmup_s"] += time.perf_counter() - phase_warm

            phase = time.perf_counter()
            msl_binding.reseed_seed_rollout(handle, _u8_rows(buffers.seed_t[start_records]))
            msl_binding.write_gamestate(handle, viewpoint, gamestate_raw)
            msl_binding.debug_write_processed_input(handle, processed_input_u8)
            start_gamestate = gamestate.copy()
            max_rollout_steps = _max_rollout_steps(args)
            option_steps = min(int(args.option_horizon_frames), max_rollout_steps)
            initial_actions = _slot_for_source(start_gamestate, analyzed_idx)["action_id"].astype(np.uint16, copy=True)
            expected_throw_values = [
                _expected_replay_throw_action(frame_metadata.get(int(lane.base_frame), {}))
                for lane in chunk
            ]
            expected_replay_throw_actions = np.asarray(
                [value if value is not None else -1 for value in expected_throw_values],
                dtype=np.int16,
            )
            defender_takeover_steps = np.full(batch_size, int(args.defender_delay_frames), dtype=np.int16)
            defender_takeover_reason = np.zeros(batch_size, dtype=np.uint8)
            divergent_throw_action = np.full(batch_size, -1, dtype=np.int16)
            divergent_throw_detected_frame = np.full(batch_size, -1, dtype=np.int32)
            action_history = np.zeros((option_steps, batch_size), dtype=np.uint16)
            input_history = np.zeros((option_steps, batch_size), dtype=INPUT_DTYPE)
            stream_history = (
                np.zeros((max_rollout_steps, batch_size), dtype=INPUT_DTYPE)
                if args.dump_controller_streams
                else None
            )
            resolved_gamestate = start_gamestate.copy()
            resolved_mask = np.zeros(batch_size, dtype=np.bool_)
            resolved_step = np.full(batch_size, -1, dtype=np.int32)
            resolved_reason_code = np.zeros(batch_size, dtype=np.uint8)
            stable_neutral_counts = np.zeros(batch_size, dtype=np.int16)
            unresolved_timeout_penalty = np.zeros(batch_size, dtype=np.float32)
            defense_resolution_active = np.zeros(batch_size, dtype=np.bool_)
            combo_escape_counts = np.zeros(batch_size, dtype=np.int16)
            resolved_combo_escape_counts = np.zeros(batch_size, dtype=np.int16)
            followup_damage = np.zeros(batch_size, dtype=np.float32)
            followup_hits = np.zeros(batch_size, dtype=np.int16)
            true_followup_hits = np.zeros(batch_size, dtype=np.int16)
            first_followup_step = np.full(batch_size, -1, dtype=np.int32)
            last_followup_step = np.full(batch_size, -1, dtype=np.int32)
            initial_combo_constraint, _ = _combo_constraint_masks(
                start_gamestate,
                defender_idx=defender_idx,
            )
            combo_started = initial_combo_constraint.copy()
            previous_defender = _slot_for_source(start_gamestate, defender_idx)
            previous_defender_percent = previous_defender["percent"].astype(np.float32, copy=True)
            previous_defender_stocks = previous_defender["stocks"].astype(np.int16, copy=True)
            previous_defender_in_hit = (
                (previous_defender["hitlag"] > 0) | (previous_defender["hitstun"] > 0)
            )
            previous_analyzed = _slot_for_source(start_gamestate, analyzed_idx)
            previous_analyzed_percent = previous_analyzed["percent"].astype(np.float32, copy=True)
            previous_analyzed_stocks = previous_analyzed["stocks"].astype(np.int16, copy=True)
            combo_reversed = np.zeros(batch_size, dtype=np.bool_)
            defender_ko_confirmed = np.zeros(batch_size, dtype=np.bool_)
            analyzed_ko_confirmed = np.zeros(batch_size, dtype=np.bool_)
            last_input = buffers.prev_input_t[start_records].copy()
            last_controllers = _controllers_from_input(last_input, num_players=num_players)
            prev_input = last_input.copy()
            needs_reset = np.zeros(batch_size, dtype=np.bool_)
            if args.history_mode == "dummy":
                needs_reset[:] = True
            chunk_timings["seed_s"] += time.perf_counter() - phase

            for step in range(max_rollout_steps):
                phase = time.perf_counter()
                game_buffers.fill(gamestate, needs_reset, controllers=last_controllers)
                chunk_timings["observation_fill_s"] += time.perf_counter() - phase

                force_mask = np.zeros(port_batch_size, dtype=np.bool_)
                if args.history_mode == "teacher-forced":
                    if step == 0:
                        if args.analyzed_port == 1:
                            force_mask[:batch_size] = True
                        else:
                            force_mask[batch_size:] = True
                    defender_replay_mask = step < defender_takeover_steps
                    if defender_idx == 0:
                        force_mask[:batch_size] = defender_replay_mask
                    else:
                        force_mask[batch_size:] = defender_replay_mask
                if bool(np.any(force_mask)):
                    phase = time.perf_counter()
                    replay_records = np.minimum(start_records + step, buffers.num_records - 1)
                    prev_controllers = _controllers_from_input(
                        buffers.prev_input_t[replay_records],
                        num_players=num_players,
                    )
                    _force_prev_controller(
                        agent,
                        _concat_port_controllers(prev_controllers, batch_size=batch_size),
                        force_mask,
                    )
                    chunk_timings["force_s"] += time.perf_counter() - phase

                phase = time.perf_counter()
                outputs = agent.step(game_buffers.game, game_buffers.needs_reset)
                chunk_timings["agent_step_s"] += time.perf_counter() - phase
                needs_reset[:] = False

                phase = time.perf_counter()
                current_input = np.zeros(batch_size, dtype=INPUT_DTYPE)
                replay_records = np.minimum(start_records + step, buffers.num_records - 1)
                if args.opponent_mode == "replay":
                    current_input[...] = buffers.input_t[replay_records]
                analyzed_source = slice(0, batch_size) if analyzed_idx == 0 else slice(batch_size, port_batch_size)
                _write_encoded_to_input(
                    current_input,
                    player_index=analyzed_idx,
                    encoded_controller=outputs.controller_state,
                    source_slice=analyzed_source,
                    axis_spacing=axis_spacing,
                    shoulder_spacing=shoulder_spacing,
                )
                defender_model_mask = step >= defender_takeover_steps
                if bool(np.any(defender_model_mask)):
                    replay_defender_input = current_input["p"][:, defender_idx].copy()
                    defender_source = slice(0, batch_size) if defender_idx == 0 else slice(batch_size, port_batch_size)
                    _write_encoded_to_input(
                        current_input,
                        player_index=defender_idx,
                        encoded_controller=outputs.controller_state,
                        source_slice=defender_source,
                        axis_spacing=axis_spacing,
                        shoulder_spacing=shoulder_spacing,
                    )
                    current_input["p"][~defender_model_mask, defender_idx] = replay_defender_input[~defender_model_mask]
                if step < option_steps:
                    input_history[step] = current_input
                if stream_history is not None:
                    stream_history[step] = current_input
                chunk_timings["input_write_s"] += time.perf_counter() - phase

                phase = time.perf_counter()
                if args.rng_mode == "replay":
                    msl_binding.step_input_replay_frame_rng(
                        handle,
                        _u8_rows(buffers.seed_t[replay_records]),
                        _u8_rows(prev_input),
                        _u8_rows(current_input),
                    )
                else:
                    msl_binding.step_input(handle, _u8_rows(prev_input), _u8_rows(current_input))
                msl_binding.write_gamestate(handle, viewpoint, gamestate_raw)
                msl_binding.debug_write_processed_input(handle, processed_input_u8)
                analyzed_action_now = _slot_for_source(gamestate, analyzed_idx)["action_id"].astype(np.int16)
                if step < option_steps:
                    action_history[step] = analyzed_action_now.astype(np.uint16)
                divergent_throw = _divergent_throw_takeover_mask(
                    expected_actions=expected_replay_throw_actions,
                    observed_actions=analyzed_action_now,
                    takeover_steps=defender_takeover_steps,
                    step=step,
                    resolved_mask=resolved_mask,
                )
                if bool(np.any(divergent_throw)):
                    defender_takeover_steps[divergent_throw] = int(step + 1)
                    defender_takeover_reason[divergent_throw] = np.uint8(1)
                    divergent_throw_action[divergent_throw] = analyzed_action_now[divergent_throw]
                    divergent_throw_detected_frame[divergent_throw] = gamestate["frame_id"][divergent_throw].astype(np.int32)
                chunk_timings["sim_step_s"] += time.perf_counter() - phase

                last_controllers = _controllers_from_input(processed_input, num_players=num_players)
                prev_input = current_input.copy()

                active_mask = ~resolved_mask
                defender_now = _slot_for_source(gamestate, defender_idx)
                defender_percent_now = defender_now["percent"].astype(np.float32)
                defender_stocks_now = defender_now["stocks"].astype(np.int16)
                same_stock = defender_stocks_now == previous_defender_stocks
                damage_delta = np.where(
                    same_stock,
                    np.maximum(0.0, defender_percent_now - previous_defender_percent),
                    0.0,
                )
                defender_stock_event = defender_stocks_now < previous_defender_stocks
                defender_death_action = _slot_action_mask(defender_now, _dead_action_ids())
                defender_ko_event = active_mask & (defender_stock_event | defender_death_action)
                defender_ko_confirmed |= defender_ko_event
                # A death animation/stock decrement confirms the result but is
                # not itself another hit in the model's route.
                hit_event = active_mask & (damage_delta > 0.01)
                analyzed_now = _slot_for_source(gamestate, analyzed_idx)
                analyzed_percent_now = analyzed_now["percent"].astype(np.float32)
                analyzed_stocks_now = analyzed_now["stocks"].astype(np.int16)
                analyzed_death_action = _slot_action_mask(analyzed_now, _dead_action_ids())
                analyzed_ko_event = active_mask & (
                    (analyzed_stocks_now < previous_analyzed_stocks) | analyzed_death_action
                )
                analyzed_ko_confirmed |= analyzed_ko_event
                attacker_damage_event = active_mask & (
                    (analyzed_percent_now > previous_analyzed_percent + 0.01)
                    | analyzed_ko_event
                )
                combo_reversed |= attacker_damage_event
                if bool(np.any(hit_event)):
                    followup_damage += np.where(hit_event, damage_delta, 0.0).astype(np.float32)
                    followup_hits += hit_event.astype(np.int16)
                    true_event = hit_event & previous_defender_in_hit
                    true_followup_hits += true_event.astype(np.int16)
                    first_followup_step = np.where(
                        hit_event & (first_followup_step < 0),
                        int(step + 1),
                        first_followup_step,
                    )
                    last_followup_step = np.where(hit_event, int(step + 1), last_followup_step)
                    combo_started |= hit_event

                combo_escape = _combo_escape_masks(
                    gamestate,
                    analyzed_idx=analyzed_idx,
                    defender_idx=defender_idx,
                    min_separation=float(args.combo_escape_min_separation),
                    pressure_range=float(args.combo_escape_pressure_range),
                    max_air_height=float(args.combo_escape_max_air_height),
                    edge_x_threshold=float(args.combo_edge_x_threshold),
                )
                defender_in_hit = combo_escape["in_hit"]
                combo_escaped_this_frame = combo_started & combo_escape["escape_eligible"] & (~hit_event)
                combo_escape_counts = np.where(
                    active_mask & combo_escaped_this_frame,
                    combo_escape_counts + 1,
                    0,
                )
                previous_defender_percent = defender_percent_now.copy()
                previous_defender_stocks = defender_stocks_now.copy()
                previous_defender_in_hit = defender_in_hit.copy()
                previous_analyzed_percent = analyzed_percent_now.copy()
                previous_analyzed_stocks = analyzed_stocks_now.copy()

                last_step = step + 1 >= max_rollout_steps
                if args.objective == "combo-extension":
                    defender_death_resolved = (~resolved_mask) & defender_ko_event
                    reversal_resolved = (~resolved_mask) & attacker_damage_event & (~defender_death_resolved)
                    combo_escape_resolved = (
                        (~resolved_mask)
                        & combo_started
                        & (~reversal_resolved)
                        & (combo_escape_counts >= int(args.combo_escape_stable_frames))
                    )
                    timeout_resolved = (~resolved_mask) & last_step
                    newly_resolved = defender_death_resolved | reversal_resolved | combo_escape_resolved | timeout_resolved
                else:
                    stock_lost, stable_neutral, unresolved_defense = _defense_resolution_masks(
                        gamestate,
                        start_frame=start_gamestate,
                        analyzed_idx=analyzed_idx,
                    )
                    if step + 1 >= int(args.defense_resolution_min_frames):
                        defense_resolution_active |= unresolved_defense | stock_lost
                    stable_neutral_counts = np.where(stable_neutral, stable_neutral_counts + 1, 0)
                    can_resolve = step + 1 >= int(args.defense_resolution_min_frames)
                    reached_nominal_end = step + 1 >= int(args.rollout_frames)
                    death_resolved = (~resolved_mask) & stock_lost
                    neutral_resolved = (
                        (~resolved_mask)
                        & can_resolve
                        & (stable_neutral_counts >= int(args.defense_neutral_stable_frames))
                    )
                    non_defense_resolved = (
                        (~resolved_mask)
                        & reached_nominal_end
                        & (~defense_resolution_active)
                        & (~unresolved_defense)
                    )
                    timeout_resolved = (~resolved_mask) & last_step
                    newly_resolved = death_resolved | neutral_resolved | non_defense_resolved | timeout_resolved
                if bool(np.any(newly_resolved)):
                    resolved_gamestate[newly_resolved] = gamestate[newly_resolved]
                    resolved_step[newly_resolved] = int(step + 1)
                    resolved_combo_escape_counts[newly_resolved] = combo_escape_counts[newly_resolved]
                    if args.objective == "combo-extension":
                        resolved_reason_code[defender_death_resolved] = 5
                        resolved_reason_code[combo_escape_resolved & ~defender_death_resolved] = 6
                        resolved_reason_code[reversal_resolved] = 8
                        combo_timeout = timeout_resolved & ~(
                            defender_death_resolved | reversal_resolved | combo_escape_resolved
                        )
                        resolved_reason_code[combo_timeout] = 7
                        unresolved_timeout_penalty[combo_timeout] = float(args.unresolved_defense_penalty)
                    else:
                        resolved_reason_code[death_resolved] = 1
                        resolved_reason_code[neutral_resolved & ~death_resolved] = 2
                        resolved_reason_code[non_defense_resolved & ~(death_resolved | neutral_resolved)] = 3
                        unresolved_timeout = timeout_resolved & ~(death_resolved | neutral_resolved | non_defense_resolved)
                        resolved_reason_code[unresolved_timeout] = 4
                        unresolved_timeout_penalty[unresolved_timeout & defense_resolution_active] = float(args.unresolved_defense_penalty)
                    resolved_mask[newly_resolved] = True
                if bool(np.all(resolved_mask)):
                    break

            phase = time.perf_counter()
            stream_path = None
            if stream_history is not None:
                stored_stream_steps = int(max(1, np.max(resolved_step)))
                stream_history_to_store = stream_history[:stored_stream_steps]
                stream_path = streams_dir / f"chunk_{chunk_index:04d}_inputs.npz"
                np.savez_compressed(
                    stream_path,
                    inputs=stream_history_to_store,
                    lane_ids=np.asarray([lane.lane_id for lane in chunk], dtype=np.int32),
                    base_frames=np.asarray([lane.base_frame for lane in chunk], dtype=np.int32),
                    offsets=np.asarray([lane.offset for lane in chunk], dtype=np.int32),
                    sample_indices=np.asarray([lane.sample_index for lane in chunk], dtype=np.int32),
                    takeover_frames=np.asarray([lane.takeover_frame for lane in chunk], dtype=np.int32),
                    defender_takeover_frames=np.asarray(
                        [lane.takeover_frame for lane in chunk],
                        dtype=np.int32,
                    ) + defender_takeover_steps.astype(np.int32),
                )
            dealt, taken, defender_stocks_lost, analyzed_stocks_lost, score = _score_batch(
                start_gamestate,
                resolved_gamestate,
                analyzed_idx=analyzed_idx,
                defender_idx=defender_idx,
                unresolved_penalty=unresolved_timeout_penalty,
            )
            defender_stocks_lost = np.maximum(
                defender_stocks_lost,
                defender_ko_confirmed.astype(np.int16),
            )
            analyzed_stocks_lost = np.maximum(
                analyzed_stocks_lost,
                analyzed_ko_confirmed.astype(np.int16),
            )
            if args.objective == "combo-extension":
                score = _combo_score_batch(
                    followup_damage=followup_damage,
                    followup_hits=followup_hits,
                    true_followup_hits=true_followup_hits,
                    damage_taken=taken,
                    defender_stocks_lost=defender_stocks_lost,
                    analyzed_stocks_lost=analyzed_stocks_lost,
                    reversed_combo=combo_reversed,
                )
                score = score - unresolved_timeout_penalty
            terminal_combo_escape = _combo_escape_masks(
                resolved_gamestate,
                analyzed_idx=analyzed_idx,
                defender_idx=defender_idx,
                min_separation=float(args.combo_escape_min_separation),
                pressure_range=float(args.combo_escape_pressure_range),
                max_air_height=float(args.combo_escape_max_air_height),
                edge_x_threshold=float(args.combo_edge_x_threshold),
            )
            terminal_defender = _slot_for_source(resolved_gamestate, defender_idx)
            terminal_attacker = _slot_for_source(resolved_gamestate, analyzed_idx)
            reason_names = {
                1: "stock_lost",
                2: "stable_neutral",
                3: "nominal_non_defense",
                4: "timeout_unresolved_defense",
                5: "combo_stock_lost",
                6: "combo_escape",
                7: "combo_timeout",
                8: "combo_reversal",
            }
            for env, lane in enumerate(chunk):
                actual_defender_takeover_frame = int(lane.takeover_frame + defender_takeover_steps[env])
                baseline = frame_metadata.get(int(lane.base_frame), {})
                anchor_frame = contact_anchor_frames.get(int(lane.base_frame))
                if anchor_frame is not None:
                    (
                        original_followup_damage,
                        original_followup_hits,
                        original_followup_kill,
                        original_followup_score,
                    ) = _replay_baseline_after_anchor(baseline, anchor_frame)
                else:
                    original_followup_damage = float(baseline.get("original_followup_damage") or 0.0)
                    original_followup_hits = int(baseline.get("original_followup_hits") or 0)
                    original_followup_kill = bool(baseline.get("original_followup_kill"))
                    original_followup_score = float(
                        baseline.get("original_followup_score")
                        or (
                            original_followup_damage
                            + 6.0 * original_followup_hits
                            + 100.0 * int(original_followup_kill)
                        )
                    )
                damage_delta_vs_replay = float(followup_damage[env]) - original_followup_damage
                score_delta_vs_replay = float(score[env]) - original_followup_score
                model_killed = int(defender_stocks_lost[env]) > 0
                outperformed_replay = _combo_outperformed_replay(
                    combo_reversed=bool(combo_reversed[env]),
                    resolution_reason_code=int(resolved_reason_code[env]),
                    model_killed=model_killed,
                    replay_killed=original_followup_kill,
                    damage_delta=damage_delta_vs_replay,
                )
                lane_option_steps = min(option_steps, max(1, int(resolved_step[env])))
                option = _option_summary(
                    action_ids=[int(value) for value in action_history[:lane_option_steps, env]],
                    input_tokens=[
                        _input_token(input_history[step]["p"][env, analyzed_idx])
                        for step in range(lane_option_steps)
                    ],
                    initial_action_id=int(initial_actions[env]),
                    max_action_segments=int(args.option_max_action_segments),
                    max_input_segments=int(args.option_max_input_segments),
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
                        "recordedContactAnchorFrame": anchor_frame,
                        "defenderTakeoverFrame": actual_defender_takeover_frame,
                        "defenderTakeover": {
                            "reason": "divergent_throw" if int(defender_takeover_reason[env]) == 1 else "fixed_delay",
                            "expectedReplayThrowActionId": (
                                int(expected_replay_throw_actions[env])
                                if expected_replay_throw_actions[env] >= 0
                                else None
                            ),
                            "expectedReplayThrowAction": (
                                _action_name(int(expected_replay_throw_actions[env]))
                                if expected_replay_throw_actions[env] >= 0
                                else None
                            ),
                            "modelThrowActionId": (
                                int(divergent_throw_action[env])
                                if divergent_throw_action[env] >= 0
                                else None
                            ),
                            "modelThrowAction": (
                                _action_name(int(divergent_throw_action[env]))
                                if divergent_throw_action[env] >= 0
                                else None
                            ),
                            "detectedFrame": (
                                int(divergent_throw_detected_frame[env])
                                if divergent_throw_detected_frame[env] >= 0
                                else None
                            ),
                        },
                        "startState": _state_summary(start_gamestate, env, num_players=num_players),
                        "endState": _state_summary(resolved_gamestate, env, num_players=num_players),
                        "resolution": {
                            "step": int(resolved_step[env]),
                            "frame": int(lane.takeover_frame + max(0, int(resolved_step[env]) - 1)),
                            "reason": reason_names.get(int(resolved_reason_code[env]), "unknown"),
                            "nominalRolloutFrames": int(args.rollout_frames),
                            "maxRolloutFrames": int(max_rollout_steps),
                            "unresolvedDefensePenalty": round(float(unresolved_timeout_penalty[env]), 3),
                            "stableNeutralRequiredFrames": int(args.defense_neutral_stable_frames),
                            "escapeStableRequiredFrames": int(args.combo_escape_stable_frames),
                            "escapeStableObservedFrames": int(resolved_combo_escape_counts[env]),
                            "defenderKoConfirmed": bool(defender_ko_confirmed[env]),
                            "analyzedKoConfirmed": bool(analyzed_ko_confirmed[env]),
                            "terminalEvidence": {
                                "defenderActionId": int(terminal_defender[env]["action_id"]),
                                "defenderAction": _action_name(int(terminal_defender[env]["action_id"])),
                                "defenderActionFrame": int(terminal_defender[env]["action_frame"]),
                                "defenderGrounded": bool(terminal_defender[env]["on_ground"]),
                                "defenderPosition": {
                                    "x": round(float(terminal_defender[env]["pos_x"]), 3),
                                    "y": round(float(terminal_defender[env]["pos_y"]), 3),
                                },
                                "attackerActionId": int(terminal_attacker[env]["action_id"]),
                                "attackerAction": _action_name(int(terminal_attacker[env]["action_id"])),
                                "attackerActionFrame": int(terminal_attacker[env]["action_frame"]),
                                "playerSeparation": round(float(terminal_combo_escape["separation"][env]), 3),
                                "defenderInHit": bool(terminal_combo_escape["in_hit"][env]),
                                "defenderInCommittedState": bool(terminal_combo_escape["lockout_action"][env]),
                                "defenderActionableLocomotion": bool(terminal_combo_escape["actionable_action"][env]),
                                "offstageOrJuggled": bool(terminal_combo_escape["offstage_like"][env]),
                                "attackerPressuring": bool(terminal_combo_escape["attacker_pressuring"][env]),
                                "attackerTooClose": bool(terminal_combo_escape["too_close"][env]),
                                "attackerActivePressure": bool(terminal_combo_escape["active_pressure"][env]),
                                "escapeEligible": bool(terminal_combo_escape["escape_eligible"][env]),
                                "escapeBlockReason": COMBO_ESCAPE_BLOCK_REASONS.get(
                                    int(terminal_combo_escape["block_reason"][env]),
                                    "unknown",
                                ),
                            },
                        },
                        "damageDealt": round(float(dealt[env]), 3),
                        "damageTaken": round(float(taken[env]), 3),
                        "followupDamage": round(float(followup_damage[env]), 3),
                        "followupHits": int(followup_hits[env]),
                        "trueFollowupHits": int(true_followup_hits[env]),
                        "extendedCombo": bool(followup_hits[env] > 0),
                        "trueExtension": bool(true_followup_hits[env] > 0),
                        "comboReversed": bool(combo_reversed[env]),
                        "replayBaseline": {
                            "followupDamage": round(original_followup_damage, 3),
                            "followupHits": original_followup_hits,
                            "killed": original_followup_kill,
                            "score": round(original_followup_score, 3),
                        },
                        "damageDeltaVsReplay": round(damage_delta_vs_replay, 3),
                        "scoreDeltaVsReplay": round(score_delta_vs_replay, 3),
                        "outperformedReplay": bool(outperformed_replay),
                        "firstFollowupStep": int(first_followup_step[env]) if first_followup_step[env] >= 0 else None,
                        "lastFollowupStep": int(last_followup_step[env]) if last_followup_step[env] >= 0 else None,
                        "comboDurationFrames": int(last_followup_step[env]) if last_followup_step[env] >= 0 else 0,
                        "initialDefenderConstrained": bool(initial_combo_constraint[env]),
                        "defenderStocksLost": int(defender_stocks_lost[env]),
                        "analyzedStocksLost": int(analyzed_stocks_lost[env]),
                        "score": round(float(score[env]), 3),
                        "optionSignature": option["optionSignature"],
                        "comboOptionSignature": option["comboOptionSignature"],
                        "optionDetailedSignature": option["optionDetailedSignature"],
                        "inputSignature": option["inputSignature"],
                        "option": option,
                        "controllerStream": _stream_ref(
                            stream_path=stream_path,
                            env=env,
                            lane=lane,
                            stream_frames=max(1, int(resolved_step[env])),
                            analyzed_port=int(args.analyzed_port),
                            defender_port=defender_idx + 1,
                            defender_takeover_frame=actual_defender_takeover_frame,
                        ),
                    }
                )
            chunk_timings["summarize_s"] += time.perf_counter() - phase
        finally:
            if agent is not None:
                agent.stop()
            msl_binding.destroy(handle)
        chunk_timings["total_s"] = time.perf_counter() - chunk_t0
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

    resolution_reason_counts: dict[str, int] = {}
    escape_block_reason_counts: dict[str, int] = {}
    defender_takeover_reason_counts: dict[str, int] = {}
    for row in all_rows:
        resolution = row.get("resolution") or {}
        reason = str(resolution.get("reason") or "unknown")
        resolution_reason_counts[reason] = resolution_reason_counts.get(reason, 0) + 1
        terminal = resolution.get("terminalEvidence") or {}
        block_reason = str(terminal.get("escapeBlockReason") or "unknown")
        escape_block_reason_counts[block_reason] = escape_block_reason_counts.get(block_reason, 0) + 1
        takeover_reason = str((row.get("defenderTakeover") or {}).get("reason") or "fixed_delay")
        defender_takeover_reason_counts[takeover_reason] = defender_takeover_reason_counts.get(takeover_reason, 0) + 1

    timings["total_s"] = time.perf_counter() - t0
    summary = {
        "runner": "slippi_ai_review.simulation",
        "replay": str(replay),
        "model": str(args.model.resolve()),
        "mslRoot": str(args.msl_root.resolve()),
        "slippiAiRoot": str(args.slippi_ai_root.resolve()),
        "analyzedPort": int(args.analyzed_port),
        "objective": args.objective,
        "defenderPort": defender_idx + 1,
        "baseFrames": base_frames,
        "anchorRecordedContact": bool(args.anchor_recorded_contact),
        "recordedContactAnchorFrames": contact_anchor_frames,
        "offsets": offsets,
        "samplesPerPoint": int(args.samples_per_point),
        "laneCount": len(lane_specs),
        "maxBatchLanes": int(args.max_batch_lanes),
        "rolloutFrames": int(args.rollout_frames),
        "maxRolloutFrames": _max_rollout_steps(args),
        "defenseResolutionExtraFrames": int(args.defense_resolution_extra_frames),
        "comboMaxRolloutFrames": int(args.combo_max_rollout_frames),
        "defenseResolutionMinFrames": int(args.defense_resolution_min_frames),
        "defenseNeutralStableFrames": int(args.defense_neutral_stable_frames),
        "comboEscapeStableFrames": int(args.combo_escape_stable_frames),
        "comboEscapeMinSeparation": float(args.combo_escape_min_separation),
        "comboEscapePressureRange": float(args.combo_escape_pressure_range),
        "comboEscapeMaxAirHeight": float(args.combo_escape_max_air_height),
        "comboEdgeXThreshold": float(args.combo_edge_x_threshold),
        "resolutionReasonCounts": resolution_reason_counts,
        "terminalEscapeBlockReasonCounts": escape_block_reason_counts,
        "unresolvedDefensePenalty": float(args.unresolved_defense_penalty),
        "warmupFrames": int(args.warmup_frames),
        "defenderDelayFrames": int(args.defender_delay_frames),
        "defenderTakeoverMode": "fixed-delay-or-divergent-throw",
        "defenderTakeoverReasonCounts": defender_takeover_reason_counts,
        "historyMode": args.history_mode,
        "opponentMode": args.opponent_mode,
        "rngMode": args.rng_mode,
        "sampleTemperature": float(args.sample_temperature),
        "optionHorizonFrames": int(args.option_horizon_frames),
        "optionMaxActionSegments": int(args.option_max_action_segments),
        "optionMaxInputSegments": int(args.option_max_input_segments),
        "dumpControllerStreams": bool(args.dump_controller_streams),
        "controllerStreamsDir": str(streams_dir) if args.dump_controller_streams else None,
        "enableGpu": bool(args.enable_gpu),
        "axisSpacing": axis_spacing,
        "shoulderSpacing": shoulder_spacing,
        "lanesJsonl": str(rows_path),
        "groups": _groups(all_rows, objective=args.objective),
        "timings": timings,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
