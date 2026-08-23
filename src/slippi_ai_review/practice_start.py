"""Fast source-SLP selection of low-risk Training Mode restore frames."""

from __future__ import annotations

import struct
import math
from dataclasses import asdict, dataclass
from pathlib import Path


CMD_MESSAGE_SIZES = 0x35
CMD_POST_FRAME = 0x38
FIRST_SLIPPI_FRAME = -123
MIN_FALLBACK_PREROLL_FRAMES = 20
STABLE_POSITION_EPSILON = 0.05
STABLE_VELOCITY_EPSILON = 0.05
COUNTERATTACK_RISK_GAP_FRAMES = 30
COUNTERATTACK_SETUP_FRAMES = 18


@dataclass(frozen=True)
class PlayerFrame:
    action: int
    x: float
    y: float
    velocities: tuple[float, float, float, float, float]
    hitlag: float


@dataclass(frozen=True)
class PracticeStartDecision:
    frame: int
    mode: str
    preferred_latest_frame: int
    fallback_latest_frame: int
    earliest_frame: int
    opening_action_frame: int
    inspected_frames: int

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _raw_region(data: bytes) -> tuple[int, int]:
    if data and data[0] == 0x36:
        return 0, len(data)
    if len(data) >= 15 and data[:3] == b"{U\x03" and data[3:11] == b"raw[$U#l":
        return 15, int.from_bytes(data[11:15], "big")
    raise ValueError("Unsupported SLP header/raw data layout")


def load_player_frames(path: Path, players: set[int]) -> dict[int, dict[int, PlayerFrame]]:
    data = path.read_bytes()
    raw_start, raw_length = _raw_region(data)
    end = raw_start + raw_length
    payload_sizes: dict[int, int] = {}
    frames: dict[int, dict[int, PlayerFrame]] = {}
    position = raw_start
    while position < end:
        command = data[position]
        payload_offset = position + 1
        if command == CMD_MESSAGE_SIZES:
            if payload_offset >= end:
                break
            payload_size = data[payload_offset]
            cursor = payload_offset + 1
            while cursor + 2 < payload_offset + payload_size:
                payload_sizes[data[cursor]] = int.from_bytes(data[cursor + 1 : cursor + 3], "big")
                cursor += 3
        else:
            payload_size = payload_sizes.get(command, 0)
        if payload_offset + payload_size > end:
            raise ValueError(f"SLP event overruns raw region at offset {position}")
        if command == CMD_POST_FRAME and payload_size >= 73:
            player = int(data[payload_offset + 4])
            follower = bool(data[payload_offset + 5])
            if player in players and not follower:
                frame = struct.unpack_from(">i", data, payload_offset)[0]
                frames.setdefault(frame, {})[player] = PlayerFrame(
                    action=struct.unpack_from(">H", data, payload_offset + 7)[0],
                    x=struct.unpack_from(">f", data, payload_offset + 9)[0],
                    y=struct.unpack_from(">f", data, payload_offset + 13)[0],
                    velocities=tuple(
                        struct.unpack_from(">f", data, payload_offset + offset)[0]
                        for offset in (52, 56, 60, 64, 68)
                    ),
                    hitlag=struct.unpack_from(">f", data, payload_offset + 72)[0],
                )
        position = payload_offset + payload_size
    return frames


def _risky_action(action: int) -> bool:
    return (
        44 <= action <= 69  # attacks
        or 75 <= action <= 91  # damage states
        or 212 <= action <= 232  # captures, throws, and downed transitions
        or 233 <= action <= 236  # rolls, spotdodge, and airdodge
    )


def _stable_candidate(
    frames: dict[int, dict[int, PlayerFrame]], frame: int, players: set[int]
) -> bool:
    current = frames.get(frame) or {}
    previous = frames.get(frame - 1) or {}
    if not players.issubset(current) or not players.issubset(previous):
        return False
    for player in players:
        state = current[player]
        prior = previous[player]
        if _risky_action(state.action) or state.hitlag > 0.0:
            return False
        if max(abs(value) for value in state.velocities) > STABLE_VELOCITY_EPSILON:
            return False
        if max(abs(state.x - prior.x), abs(state.y - prior.y)) > STABLE_POSITION_EPSILON:
            return False
    return True


def _restorable_candidate(
    frames: dict[int, dict[int, PlayerFrame]], frame: int, players: set[int]
) -> bool:
    states = frames.get(frame) or {}
    if not players.issubset(states):
        return False
    for player in players:
        state = states[player]
        values = (state.x, state.y, state.hitlag, *state.velocities)
        if _risky_action(state.action) or state.hitlag > 0.0 or not all(math.isfinite(value) for value in values):
            return False
    return True


def _counterattack_interaction_start(
    frames: dict[int, dict[int, PlayerFrame]],
    *,
    players: set[int],
    earliest_frame: int,
    opening_action_frame: int,
) -> int | None:
    risky_frames = [
        frame
        for frame in range(earliest_frame, opening_action_frame)
        if players.issubset(frames.get(frame) or {})
        and any(_risky_action(frames[frame][player].action) for player in players)
    ]
    if not risky_frames:
        return None
    interaction_start = risky_frames[-1]
    previous = risky_frames[-1]
    for frame in reversed(risky_frames[:-1]):
        if previous - frame > COUNTERATTACK_RISK_GAP_FRAMES:
            break
        interaction_start = frame
        previous = frame
    return interaction_start


def _clean_path(
    frames: dict[int, dict[int, PlayerFrame]], start: int, end: int, players: set[int]
) -> bool:
    for frame in range(start, end):
        states = frames.get(frame) or {}
        if not players.issubset(states):
            return False
        if any(_risky_action(states[player].action) or states[player].hitlag > 0.0 for player in players):
            return False
    return True


def choose_stable_start(
    frames: dict[int, dict[int, PlayerFrame]],
    *,
    players: set[int],
    earliest_frame: int,
    preferred_latest_frame: int,
    fallback_latest_frame: int,
    opening_action_frame: int,
    default_frame: int,
    opening_type: str | None = None,
) -> PracticeStartDecision:
    inspected = 0
    if opening_type == "counter-attack":
        interaction_start = _counterattack_interaction_start(
            frames,
            players=players,
            earliest_frame=earliest_frame,
            opening_action_frame=opening_action_frame,
        )
        if interaction_start is not None:
            setup_frame = max(earliest_frame, interaction_start - COUNTERATTACK_SETUP_FRAMES)
            for frame in range(setup_frame, earliest_frame - 1, -1):
                inspected += 1
                if _restorable_candidate(frames, frame, players):
                    return PracticeStartDecision(
                        frame=frame,
                        mode="counterattack_setup",
                        preferred_latest_frame=preferred_latest_frame,
                        fallback_latest_frame=fallback_latest_frame,
                        earliest_frame=earliest_frame,
                        opening_action_frame=opening_action_frame,
                        inspected_frames=inspected,
                    )
    ranges = (
        ("preferred", preferred_latest_frame, earliest_frame),
        ("shortened_preroll", fallback_latest_frame, preferred_latest_frame + 1),
    )
    stable_active_path: int | None = None
    for mode, latest, lower in ranges:
        for frame in range(latest, lower - 1, -1):
            inspected += 1
            if not _stable_candidate(frames, frame, players):
                continue
            if stable_active_path is None:
                stable_active_path = frame
            if _clean_path(frames, frame, opening_action_frame, players):
                return PracticeStartDecision(
                    frame=frame,
                    mode=mode,
                    preferred_latest_frame=preferred_latest_frame,
                    fallback_latest_frame=fallback_latest_frame,
                    earliest_frame=earliest_frame,
                    opening_action_frame=opening_action_frame,
                    inspected_frames=inspected,
                )
    if stable_active_path is not None:
        return PracticeStartDecision(
            frame=stable_active_path,
            mode="stable_active_path",
            preferred_latest_frame=preferred_latest_frame,
            fallback_latest_frame=fallback_latest_frame,
            earliest_frame=earliest_frame,
            opening_action_frame=opening_action_frame,
            inspected_frames=inspected,
        )
    return PracticeStartDecision(
        frame=default_frame,
        mode="default_no_stable_candidate",
        preferred_latest_frame=preferred_latest_frame,
        fallback_latest_frame=fallback_latest_frame,
        earliest_frame=earliest_frame,
        opening_action_frame=opening_action_frame,
        inspected_frames=inspected,
    )


def select_practice_start(
    slp_path: Path,
    *,
    players: set[int],
    takeover_frame: int,
    opening_frame: int,
    opening_action_frame: int,
    preroll_frames: int,
    maximum_leadin_frames: int,
    default_frame: int,
    opening_type: str | None = None,
) -> PracticeStartDecision:
    requested_preroll = max(30, preroll_frames)
    fallback_preroll = max(MIN_FALLBACK_PREROLL_FRAMES, requested_preroll - 10)
    earliest = max(FIRST_SLIPPI_FRAME, takeover_frame - maximum_leadin_frames)
    preferred_latest = max(earliest, opening_frame - requested_preroll)
    fallback_latest = max(preferred_latest, opening_frame - fallback_preroll)
    frames = load_player_frames(slp_path, players)
    return choose_stable_start(
        frames,
        players=players,
        earliest_frame=earliest,
        preferred_latest_frame=preferred_latest,
        fallback_latest_frame=fallback_latest,
        opening_action_frame=max(opening_action_frame, fallback_latest + 1),
        default_frame=default_frame,
        opening_type=opening_type,
    )
