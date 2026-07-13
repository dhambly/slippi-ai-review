"""Patch an existing SLP with an MSL lane trajectory for replay playback.

This is intentionally conservative: it preserves the original raw event stream
and only edits existing pre/post-frame player packets for frames covered by the
selected MSL lane. That keeps game-start metadata, item packets, frame bookends,
and the footer intact while giving Slippi playback an artificial MSL branch.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_settings
from .paths import PROJECT_DIR


WORK_DIR = PROJECT_DIR
DEFAULT_MSL_ROOT = load_settings().msl_root

CMD_MESSAGE_SIZES = 0x35
CMD_GAME_START = 0x36
CMD_PRE_FRAME = 0x37
CMD_POST_FRAME = 0x38
CMD_GAME_END = 0x39
CMD_FRAME_BOOKEND = 0x3C

LEGACY_UPGRADE_VERSION = (3, 16, 0)
LEGACY_UPGRADE_SIZES = {
    CMD_GAME_START: 760,
    CMD_PRE_FRAME: 66,
    CMD_POST_FRAME: 84,
    CMD_FRAME_BOOKEND: 8,
}


@dataclass
class RawEvent:
  command: int
  command_offset: int
  payload_offset: int
  payload_size: int


def load_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def host_path(raw_path: str) -> Path:
  if raw_path.startswith("/mnt/") and len(raw_path) >= 7 and raw_path[6] == "/":
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


def u8_rows(arr: np.ndarray) -> np.ndarray:
  return arr.view(np.uint8).reshape(arr.shape[0], arr.dtype.itemsize)


def target_from_queue(queue_path: Path, target_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
  queue = load_json(queue_path)
  targets = queue.get("targets") or []
  if target_index < 1 or target_index > len(targets):
    raise SystemExit(f"--target-index {target_index} outside target range 1..{len(targets)}")
  return queue, targets[target_index - 1]


def lane_for_route(target: dict[str, Any], alternative_index: int) -> dict[str, Any]:
  if alternative_index < 0:
    raise SystemExit("--alternative-index must be zero or greater")
  if alternative_index == 0:
    return target.get("representative_lane") or {}
  alternatives = target.get("alternative_routes") or []
  if alternative_index > len(alternatives):
    raise SystemExit(
        f"--alternative-index {alternative_index} outside route range 0..{len(alternatives)}"
    )
  return alternatives[alternative_index - 1].get("representative_lane") or {}


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


def raw_region(slp: bytes) -> tuple[int, int]:
  if slp and slp[0] == 0x36:
    return 0, len(slp)
  if len(slp) >= 15 and slp[:3] == b"{U\x03" and slp[3:11] == b"raw[$U#l":
    return 15, int.from_bytes(slp[11:15], "big")
  raise SystemExit("Unsupported SLP header/raw data layout.")


def parse_events(buf: bytearray, raw_start: int, raw_len: int) -> list[RawEvent]:
  events: list[RawEvent] = []
  payload_sizes: dict[int, int] = {}
  pos = raw_start
  end = raw_start + raw_len
  while pos < end:
    command = buf[pos]
    payload_start = pos + 1
    if command == CMD_MESSAGE_SIZES:
      if payload_start >= end:
        break
      payload_size = buf[payload_start]
      cursor = payload_start + 1
      while cursor + 2 < payload_start + payload_size:
        cmd = buf[cursor]
        size = int.from_bytes(buf[cursor + 1:cursor + 3], "big")
        payload_sizes[cmd] = size
        cursor += 3
    else:
      payload_size = payload_sizes.get(command, 0)
    if payload_start + payload_size > end:
      raise SystemExit(f"Raw SLP event overruns raw region at offset {pos}")
    events.append(RawEvent(command, pos, payload_start, payload_size))
    pos = payload_start + payload_size
  return events


def reconstruct_legacy_post_physics(buf: bytearray, events: list[RawEvent]) -> dict[str, int]:
  """Fill post-frame physics fields that did not exist in pre-3.0 SLPs.

  A post-frame position delta is the velocity that produced the current
  position. For ordinary aerial and grounded states this maps directly to the
  self or ground velocity restored by Training Mode. Damage states use the
  knockback velocity field instead. This is primarily needed at the selected
  export frame; subsequent gameplay is driven by the recorded controllers.
  """
  records: dict[tuple[int, bool], list[RawEvent]] = {}
  for event in events:
    if event.command != CMD_POST_FRAME or event.payload_size < 73:
      continue
    off = event.payload_offset
    key = (int(buf[off + 4]), bool(buf[off + 5]))
    records.setdefault(key, []).append(event)

  reconstructed = 0
  inferred_hitlag = 0
  damage_states = set(range(75, 92))
  for player_events in records.values():
    player_events.sort(key=lambda event: struct.unpack_from(">i", buf, event.payload_offset)[0])
    for index, event in enumerate(player_events):
      off = event.payload_offset
      frame = struct.unpack_from(">i", buf, off)[0]
      x = struct.unpack_from(">f", buf, off + 9)[0]
      y = struct.unpack_from(">f", buf, off + 13)[0]
      action = struct.unpack_from(">H", buf, off + 7)[0]
      airborne = bool(buf[off + 46])
      dx = 0.0
      dy = 0.0
      if index > 0:
        previous = player_events[index - 1]
        prev_off = previous.payload_offset
        prev_frame = struct.unpack_from(">i", buf, prev_off)[0]
        if prev_frame + 1 == frame:
          dx = x - struct.unpack_from(">f", buf, prev_off + 9)[0]
          dy = y - struct.unpack_from(">f", buf, prev_off + 13)[0]
          # Do not turn a respawn, transformation, or other teleport into an
          # enormous velocity when old packet data lacks an explicit marker.
          if abs(dx) > 20.0 or abs(dy) > 20.0:
            dx = dy = 0.0

      self_x = self_y = knockback_x = knockback_y = ground_x = 0.0
      if action in damage_states:
        knockback_x, knockback_y = dx, dy
      elif airborne:
        self_x, self_y = dx, dy
      else:
        ground_x = dx
        self_y = dy
      f32(buf, off + 52, self_x)
      f32(buf, off + 56, self_y)
      f32(buf, off + 60, knockback_x)
      f32(buf, off + 64, knockback_y)
      f32(buf, off + 68, ground_x)
      reconstructed += 1

    # Repeated post-frame poses are the legacy signal available for hitlag.
    # Mark all but the final frozen frame so tm_replay will not choose one as
    # a safe export point.
    run_end = len(player_events) - 1
    while run_end >= 0:
      end_event = player_events[run_end]
      end_off = end_event.payload_offset
      signature = bytes(buf[end_off + 7:end_off + 21]) + bytes(buf[end_off + 33:end_off + 37])
      run_start = run_end
      while run_start > 0:
        prior = player_events[run_start - 1]
        prior_off = prior.payload_offset
        prior_signature = bytes(buf[prior_off + 7:prior_off + 21]) + bytes(buf[prior_off + 33:prior_off + 37])
        if prior_signature != signature:
          break
        current_frame = struct.unpack_from(">i", buf, player_events[run_start].payload_offset)[0]
        prior_frame = struct.unpack_from(">i", buf, prior_off)[0]
        if prior_frame + 1 != current_frame:
          break
        run_start -= 1
      if run_end > run_start:
        for frozen_index in range(run_start, run_end):
          remaining = float(run_end - frozen_index)
          f32(buf, player_events[frozen_index].payload_offset + 72, remaining)
          inferred_hitlag += 1
      run_end = run_start - 1

  return {"post_physics_reconstructed": reconstructed, "hitlag_frames_inferred": inferred_hitlag}


def reconstruct_legacy_attack_instances(buf: bytearray, events: list[RawEvent]) -> dict[str, int]:
  """Synthesize attack-instance IDs used to rebuild Melee's stale queue.

  Legacy post-frame packets identify the attacker and move but predate the two
  instance-ID fields. A stable ID per contiguous action-state run is enough to
  distinguish repeated attacks while keeping every hit of a multihit move in
  one stale-queue entry.
  """
  by_player: dict[tuple[int, bool], list[RawEvent]] = {}
  by_frame: dict[tuple[int, int, bool], RawEvent] = {}
  for event in events:
    if event.command != CMD_POST_FRAME or event.payload_size < 84:
      continue
    off = event.payload_offset
    player = int(buf[off + 4])
    follower = bool(buf[off + 5])
    frame = struct.unpack_from(">i", buf, off)[0]
    by_player.setdefault((player, follower), []).append(event)
    by_frame[(frame, player, follower)] = event

  next_instance = 1
  frame_instances: dict[tuple[int, int, bool], int] = {}
  for key, player_events in sorted(by_player.items()):
    player_events.sort(key=lambda event: struct.unpack_from(">i", buf, event.payload_offset)[0])
    previous_action = None
    previous_anim = None
    current_instance = 0
    for event in player_events:
      off = event.payload_offset
      frame = struct.unpack_from(">i", buf, off)[0]
      action = struct.unpack_from(">H", buf, off + 7)[0]
      anim = struct.unpack_from(">f", buf, off + 33)[0]
      if previous_action != action or (previous_anim is not None and anim < previous_anim):
        current_instance = next_instance
        next_instance = (next_instance + 1) & 0xFFFF
        if next_instance == 0:
          next_instance = 1
      frame_instances[(frame, key[0], key[1])] = current_instance
      u16(buf, off + 82, current_instance)
      previous_action = action
      previous_anim = anim

  hit_events = 0
  stale_instances: set[int] = set()
  for (victim, follower), player_events in by_player.items():
    if follower:
      continue
    player_events.sort(key=lambda event: struct.unpack_from(">i", buf, event.payload_offset)[0])
    previous_percent = None
    previous_stocks = None
    current_hit_instance = 0
    for event in player_events:
      off = event.payload_offset
      frame = struct.unpack_from(">i", buf, off)[0]
      percent = struct.unpack_from(">f", buf, off + 21)[0]
      stocks = int(buf[off + 32])
      if previous_stocks is not None and stocks != previous_stocks:
        current_hit_instance = 0
      if previous_percent is not None and percent > previous_percent + 0.001:
        attacker = int(buf[off + 31])
        attacker_event = by_frame.get((frame, attacker, False))
        if attacker_event is not None and attacker != victim:
          attacker_off = attacker_event.payload_offset
          attack_kind = int(buf[attacker_off + 29])
          instance = frame_instances.get((frame, attacker, False), 0)
          if attack_kind != 0 and instance != 0:
            current_hit_instance = instance
            u16(buf, attacker_off + 82, instance)
            hit_events += 1
            stale_instances.add(instance)
      u16(buf, off + 80, current_hit_instance)
      previous_percent = percent
      previous_stocks = stocks

  return {
      "attack_instances_reconstructed": len(frame_instances),
      "damage_events_linked": hit_events,
      "stale_instances_reconstructed": len(stale_instances),
  }


def upgrade_legacy_slp(buf: bytearray) -> tuple[bytearray, dict[str, Any]]:
  raw_start, raw_len = raw_region(bytes(buf))
  events = parse_events(buf, raw_start, raw_len)
  game_start = next((event for event in events if event.command == CMD_GAME_START), None)
  if game_start is None or game_start.payload_size < 4:
    raise SystemExit("SLP has no readable game-start event")
  original_version = tuple(int(value) for value in buf[game_start.payload_offset:game_start.payload_offset + 3])
  if original_version[0] >= 3:
    return buf, {"upgraded": False, "original_version": ".".join(map(str, original_version))}

  raw = bytearray()
  pending_frame: int | None = None
  bookend_count = 0

  def append_bookend(frame: int) -> None:
    nonlocal bookend_count
    raw.append(CMD_FRAME_BOOKEND)
    raw.extend(struct.pack(">ii", int(frame), int(frame)))
    bookend_count += 1

  for event in events:
    payload = bytearray(buf[event.payload_offset:event.payload_offset + event.payload_size])
    event_frame = None
    if event.command in (CMD_PRE_FRAME, CMD_POST_FRAME) and len(payload) >= 4:
      event_frame = struct.unpack_from(">i", payload, 0)[0]
      if pending_frame is not None and event_frame != pending_frame:
        append_bookend(pending_frame)
      pending_frame = event_frame
    elif event.command == CMD_GAME_END and pending_frame is not None:
      append_bookend(pending_frame)
      pending_frame = None

    target_size = LEGACY_UPGRADE_SIZES.get(event.command, event.payload_size)
    if len(payload) < target_size:
      payload.extend(b"\0" * (target_size - len(payload)))

    if event.command == CMD_MESSAGE_SIZES:
      commands = set()
      cursor = 1
      while cursor + 2 < len(payload):
        command = payload[cursor]
        commands.add(command)
        if command in LEGACY_UPGRADE_SIZES:
          payload[cursor + 1:cursor + 3] = LEGACY_UPGRADE_SIZES[command].to_bytes(2, "big")
        cursor += 3
      if CMD_FRAME_BOOKEND not in commands:
        payload.extend((CMD_FRAME_BOOKEND, *LEGACY_UPGRADE_SIZES[CMD_FRAME_BOOKEND].to_bytes(2, "big")))
      payload[0] = len(payload)
    elif event.command == CMD_GAME_START:
      payload[0:3] = bytes(LEGACY_UPGRADE_VERSION)
    elif event.command == CMD_PRE_FRAME:
      # Raw ADC fields were added after these Wii-era recordings. Reconstruct
      # them from the normalized sticks already present in every old packet.
      for destination, source in ((63, 28), (64, 32), (65, 36)):
        normalized = struct.unpack_from(">f", payload, source)[0]
        payload[destination] = int(clamp(round(normalized * 80.0), -128, 127)) & 0xFF

    raw.append(event.command)
    raw.extend(payload)

  if pending_frame is not None:
    append_bookend(pending_frame)

  upgraded = bytearray(buf[:raw_start])
  upgraded.extend(raw)
  upgraded.extend(buf[raw_start + raw_len:])
  if raw_start == 15:
    upgraded[11:15] = len(raw).to_bytes(4, "big")
  upgraded_raw_start, upgraded_raw_len = raw_region(bytes(upgraded))
  physics_details = reconstruct_legacy_post_physics(
      upgraded,
      parse_events(upgraded, upgraded_raw_start, upgraded_raw_len),
  )
  instance_details = reconstruct_legacy_attack_instances(
      upgraded,
      parse_events(upgraded, upgraded_raw_start, upgraded_raw_len),
  )
  return upgraded, {
      "upgraded": True,
      "original_version": ".".join(map(str, original_version)),
      "upgraded_version": ".".join(map(str, LEGACY_UPGRADE_VERSION)),
      "frame_bookends_added": bookend_count,
      **physics_details,
      **instance_details,
  }


def animation_index_lookup_from_events(buf: bytearray, events: list[RawEvent]) -> dict[tuple[int, int], int]:
  """Return the most common Slippi animation index for each (char_id, action_id).

  MSL rollout state currently exposes action_id/action_frame, but not the selected
  submotion stored by Slippi as post-frame animation index 0x4D. Dolphin's visual
  pose uses this value, so synthetic SLPs need to reconstruct it from the source
  replay where possible.
  """
  from collections import Counter, defaultdict

  counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
  for event in events:
    if event.command != CMD_POST_FRAME or event.payload_size < 81:
      continue
    off = event.payload_offset
    if buf[off + 5] != 0:
      continue
    char_id = int(buf[off + 6])
    action_id = struct.unpack_from(">H", buf, off + 7)[0]
    animation_index = struct.unpack_from(">I", buf, off + 76)[0]
    counts[(char_id, action_id)][animation_index] += 1
  return {key: counter.most_common(1)[0][0] for key, counter in counts.items() if counter}


def f32(buf: bytearray, offset: int, value: float) -> None:
  struct.pack_into(">f", buf, offset, float(value))


def u16(buf: bytearray, offset: int, value: int) -> None:
  struct.pack_into(">H", buf, offset, int(value) & 0xFFFF)


def u32(buf: bytearray, offset: int, value: int) -> None:
  struct.pack_into(">I", buf, offset, int(value) & 0xFFFFFFFF)


def clamp(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, float(value)))


def raw_axis_to_float(value: Any) -> float:
  return clamp(float(value) / 80.0, -1.0, 1.0)


def raw_axis_to_u8(value: Any) -> int:
  return int(clamp(round(float(value) + 128.0), 0, 255))


def active_patch_players(
    frame: int,
    controlled_player: int,
    defender_player: int | None,
    defender_takeover_frame: int | None,
) -> set[int]:
  players = {controlled_player}
  if (
      defender_player is not None
      and defender_takeover_frame is not None
      and frame >= defender_takeover_frame
  ):
    players.add(defender_player)
  return players


def input_for_source(frame_input: np.void, source_player: int) -> np.void:
  return frame_input["p"][source_player]


def slot_for_source(frame: np.ndarray, env: int, source_player: int) -> np.void | None:
  slots = frame["slots"][env]
  for slot in slots:
    if bool(slot["present"]) and int(slot["source_player"]) == int(source_player):
      return slot
  return None


def build_msl_rows(
    *,
    queue: dict[str, Any],
    lane: dict[str, Any],
    stream: dict[str, Any],
    stream_path: Path,
    env: int,
    frame_limit: int,
    msl_root: Path,
    animation_lookup: dict[tuple[int, int], int] | None = None,
) -> dict[int, dict[str, Any]]:
  replay = Path(queue["replay"]).resolve()
  takeover_frame = int(stream["takeoverFrame"])
  start_record = int(lane.get("startRecord"))
  input_data = np.load(stream_path, allow_pickle=False)
  inputs = input_data[str(stream.get("array") or "inputs")]
  frame_limit = max(0, min(frame_limit, int(inputs.shape[0])))

  setup_msl(msl_root)
  import melee_sim.dtypes as msl_dtypes  # type: ignore
  import msl_binding  # type: ignore
  from tools.eval.validation_dtypes import INPUT_DTYPE  # type: ignore
  from tools.slippi.validation_buffer_builder import build_validation_buffers_from_slp  # type: ignore

  buffers = build_validation_buffers_from_slp(slp_path=str(replay))
  num_players = int(buffers.num_players)
  if start_record < 0 or start_record >= int(buffers.num_records):
    raise SystemExit(f"startRecord {start_record} outside validation buffer")

  gamestate_dtype = msl_dtypes.gamestate_dtype()
  handle = msl_binding.init(1, num_players)
  viewpoint = np.zeros(1, dtype=np.uint8)
  gamestate_raw = np.zeros((1, gamestate_dtype.itemsize), dtype=np.uint8)
  gamestate = gamestate_raw.view(gamestate_dtype).reshape(1)
  prev_input = buffers.prev_input_t[[start_record]].copy()
  rows: dict[int, dict[str, Any]] = {}

  def capture(frame_input: np.void | None) -> None:
    frame = int(gamestate["frame_id"][0])
    players: dict[int, dict[str, Any]] = {}
    for source in range(num_players):
      slot = slot_for_source(gamestate, 0, source)
      if slot is None:
        continue
      char_id = int(slot["char_id"])
      action_id = int(slot["action_id"])
      action_frame = float(slot["action_frame"])
      animation_index = None
      if animation_lookup is not None:
        animation_index = animation_lookup.get((char_id, action_id))
      if animation_index is None:
        animation_index = 0xFFFFFFFF if action_frame < 0 else action_id
      player: dict[str, Any] = {
          "action_id": action_id,
          "action_frame": action_frame,
          "percent": float(slot["percent"]),
          "shield_hp": float(slot["shield_hp"]),
          "stocks": int(slot["stocks"]),
          "x": float(slot["pos_x"]),
          "y": float(slot["pos_y"]),
          "facing": 1.0 if int(slot["facing"]) else -1.0,
          "is_airborne": 0 if int(slot["on_ground"]) else 1,
          "jumps_left": int(slot["jumps_left"]),
          "hurtbox_state": int(slot["hurtbox_state"]),
          "speed_air_x_self": float(slot["speed_air_x_self"]),
          "speed_y_self": float(slot["speed_y_self"]),
          "speed_x_attack": float(slot["speed_x_attack"]),
          "speed_y_attack": float(slot["speed_y_attack"]),
          "speed_ground_x_self": float(slot["speed_ground_x_self"]),
          "hitlag": float(slot["hitlag"]),
          "char_id": char_id,
          "animation_index": int(animation_index),
      }
      if frame_input is not None:
        inp = input_for_source(frame_input, source)
        buttons = int(inp["buttons"])
        l_trigger = int(inp["l"])
        r_trigger = int(inp["r"])
        player["input"] = {
            "main_x": int(inp["main_x"]),
            "main_y": int(inp["main_y"]),
            "c_x": int(inp["c_x"]),
            "c_y": int(inp["c_y"]),
            "l": l_trigger,
            "r": r_trigger,
            "trigger": max(l_trigger, r_trigger) / 255.0,
            "buttons": buttons,
      }
      players[source] = player
    rows[frame] = {"players": players}

  try:
    msl_binding.reseed_seed_rollout(handle, u8_rows(buffers.seed_t[[start_record]]))
    msl_binding.write_gamestate(handle, viewpoint, gamestate_raw)
    capture(inputs[0, env] if frame_limit > 0 else None)
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
      capture(next_input)
      prev_input = current_input.copy()
  finally:
    msl_binding.destroy(handle)
    os.chdir(WORK_DIR)

  # Clamp to the actual lane frame range. MSL capture includes the initial row
  # plus one post-step row per input, so the final frame is inclusive.
  return {frame: row for frame, row in rows.items() if takeover_frame <= frame <= takeover_frame + frame_limit}


def patch_pre_frame(
    buf: bytearray,
    event: RawEvent,
    state_players: dict[int, dict[str, Any]],
    input_players: dict[int, dict[str, Any]],
    allowed_players: set[int] | None = None,
) -> bool:
  if event.payload_size < 66:
    return False
  off = event.payload_offset
  player_index = buf[off + 4]
  is_follower = buf[off + 5] != 0
  if allowed_players is not None and player_index not in allowed_players:
    return False
  state_player = state_players.get(player_index)
  input_player = input_players.get(player_index)
  if state_player is None or input_player is None or is_follower:
    return False
  inp = input_player.get("input") or {}
  u16(buf, off + 10, state_player["action_id"])
  f32(buf, off + 12, state_player["x"])
  f32(buf, off + 16, state_player["y"])
  f32(buf, off + 20, state_player["facing"])
  f32(buf, off + 24, raw_axis_to_float(inp.get("main_x", 0)))
  f32(buf, off + 28, raw_axis_to_float(inp.get("main_y", 0)))
  f32(buf, off + 32, raw_axis_to_float(inp.get("c_x", 0)))
  f32(buf, off + 36, raw_axis_to_float(inp.get("c_y", 0)))
  f32(buf, off + 40, inp.get("trigger", 0.0))
  u32(buf, off + 44, inp.get("buttons", 0))
  u16(buf, off + 48, inp.get("buttons", 0))
  f32(buf, off + 50, float(inp.get("l", 0)) / 255.0)
  f32(buf, off + 54, float(inp.get("r", 0)) / 255.0)
  buf[off + 58] = raw_axis_to_u8(inp.get("main_x", 0))
  f32(buf, off + 59, state_player["percent"])
  if event.payload_size > 63:
    buf[off + 63] = raw_axis_to_u8(inp.get("main_y", 0))
  if event.payload_size > 64:
    buf[off + 64] = raw_axis_to_u8(inp.get("c_x", 0))
  if event.payload_size > 65:
    buf[off + 65] = raw_axis_to_u8(inp.get("c_y", 0))
  return True


def patch_post_frame(
    buf: bytearray,
    event: RawEvent,
    players: dict[int, dict[str, Any]],
    allowed_players: set[int] | None = None,
) -> bool:
  if event.payload_size < 77:
    return False
  off = event.payload_offset
  player_index = buf[off + 4]
  is_follower = buf[off + 5] != 0
  if allowed_players is not None and player_index not in allowed_players:
    return False
  player = players.get(player_index)
  if player is None or is_follower:
    return False
  buf[off + 6] = int(player["char_id"]) & 0xFF
  u16(buf, off + 7, player["action_id"])
  f32(buf, off + 9, player["x"])
  f32(buf, off + 13, player["y"])
  f32(buf, off + 17, player["facing"])
  f32(buf, off + 21, player["percent"])
  f32(buf, off + 25, player["shield_hp"])
  buf[off + 32] = int(player["stocks"]) & 0xFF
  f32(buf, off + 33, player["action_frame"])
  buf[off + 46] = 1 if int(player["is_airborne"]) else 0
  # Keep lastGroundId from the original packet for now; MSL does not expose it in this dtype.
  buf[off + 49] = int(player["jumps_left"]) & 0xFF
  buf[off + 51] = int(player["hurtbox_state"]) & 0xFF
  f32(buf, off + 52, player["speed_air_x_self"])
  f32(buf, off + 56, player["speed_y_self"])
  f32(buf, off + 60, player["speed_x_attack"])
  f32(buf, off + 64, player["speed_y_attack"])
  f32(buf, off + 68, player["speed_ground_x_self"])
  f32(buf, off + 72, player["hitlag"])
  if event.payload_size >= 81:
    u32(buf, off + 76, player.get("animation_index", 0))
  return True


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--queue-json", required=True, type=Path)
  parser.add_argument("--target-index", required=True, type=int)
  parser.add_argument(
      "--alternative-index",
      type=int,
      default=0,
      help="Route to export: 0 is the primary representative lane; 1+ selects an alternative route.",
  )
  parser.add_argument("--out", required=True, type=Path)
  parser.add_argument("--msl-root", type=Path, default=DEFAULT_MSL_ROOT)
  parser.add_argument("--max-frames", type=int, default=None)
  parser.add_argument(
      "--patch-mode",
      choices=("full", "inputs-only", "none"),
      default="full",
      help=(
          "full patches controller and post-frame state; inputs-only patches "
          "controller packets; none only upgrades legacy packet framing."
      ),
  )
  parser.add_argument(
      "--pre-state-offset",
      type=int,
      default=0,
      help=(
          "MSL row offset used for pre-frame state fields. The historical "
          "default was -1; 0 matches real SLP semantics where frame F pre-state "
          "is the authoritative frame-start state for frame F."
      ),
  )
  parser.add_argument(
      "--post-state-offset",
      type=int,
      default=0,
      help=(
          "MSL row offset used for post-frame state fields. 0 matches real SLP "
          "semantics: post packet F is the visible/start state for frame F."
      ),
  )
  parser.add_argument(
      "--input-offset",
      type=int,
      default=0,
      help=(
          "MSL row offset used for pre-frame controller fields. Use -1 when "
          "row F is the state produced by the preceding row's input."
      ),
  )
  args = parser.parse_args()

  queue, target = target_from_queue(args.queue_json.resolve(), args.target_index)
  lane = full_lane_from_queue(queue, lane_for_route(target, args.alternative_index))
  stream = lane.get("controllerStream") or {}
  if not stream:
    raise SystemExit("target has no controllerStream")
  stream_path = host_path(str(stream["path"]))
  if not stream_path.exists():
    raise SystemExit(f"controller stream missing: {stream_path}")
  env = int(stream["env"])
  stream_data = np.load(stream_path, allow_pickle=False)
  inputs = stream_data[str(stream.get("array") or "inputs")]
  frame_limit = int(args.max_frames if args.max_frames is not None else inputs.shape[0])

  replay = Path(queue["replay"]).resolve()
  buf = bytearray(replay.read_bytes())
  buf, legacy_upgrade = upgrade_legacy_slp(buf)
  raw_start, raw_len = raw_region(bytes(buf))
  events = parse_events(buf, raw_start, raw_len)
  animation_lookup = animation_index_lookup_from_events(buf, events)

  rows = {}
  if args.patch_mode != "none":
    rows = build_msl_rows(
        queue=queue,
        lane=lane,
        stream=stream,
        stream_path=stream_path,
        env=env,
        frame_limit=frame_limit,
        msl_root=args.msl_root,
        animation_lookup=animation_lookup,
    )

  patched_pre = 0
  patched_post = 0
  touched_frames: set[int] = set()
  controlled_player = int(queue["controlled_port"]) - 1
  defender_port = stream.get("defenderPort")
  defender_player = int(defender_port) - 1 if defender_port is not None else None
  defender_takeover_frame = stream.get("defenderTakeoverFrame")
  if defender_takeover_frame is not None:
    defender_takeover_frame = int(defender_takeover_frame)
  for event in events:
    if event.command not in (CMD_PRE_FRAME, CMD_POST_FRAME) or event.payload_size < 4:
      continue
    frame = struct.unpack_from(">i", buf, event.payload_offset)[0]
    allowed_players = active_patch_players(
        frame,
        controlled_player,
        defender_player,
        defender_takeover_frame,
    )
    if event.command == CMD_PRE_FRAME:
      # State and controller timing are independently configurable because an
      # MSL row may describe the state before its associated input is stepped.
      state_row = rows.get(frame + args.pre_state_offset) or rows.get(frame)
      input_row = rows.get(frame + args.input_offset) or rows.get(frame)
      if state_row is None or input_row is None:
        continue
      if patch_pre_frame(
          buf,
          event,
          state_row["players"],
          input_row["players"],
          allowed_players,
      ):
        patched_pre += 1
        touched_frames.add(frame)
    elif event.command == CMD_POST_FRAME and args.patch_mode == "full":
      row = rows.get(frame + args.post_state_offset)
      if row is None:
        continue
      if patch_post_frame(buf, event, row["players"], allowed_players):
        patched_post += 1
        touched_frames.add(frame)

  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_bytes(buf)
  summary = {
      "out": str(args.out.resolve()),
      "source_replay": str(replay),
      "queue_json": str(args.queue_json.resolve()),
      "target_index": int(args.target_index),
      "alternative_index": int(args.alternative_index),
      "lane_id": int(lane.get("laneId")),
      "takeover_frame": int(stream["takeoverFrame"]),
      "controlled_player": controlled_player,
      "defender_player": defender_player,
      "defender_takeover_frame": defender_takeover_frame,
      "rows": len(rows),
      "input_steps": frame_limit,
      "patched_frames": len(touched_frames),
      "patched_pre_packets": patched_pre,
      "patched_post_packets": patched_post,
      "patch_mode": args.patch_mode,
      "pre_state_offset": int(args.pre_state_offset),
      "post_state_offset": int(args.post_state_offset),
      "input_offset": int(args.input_offset),
      "legacy_upgrade": legacy_upgrade,
      "first_patched_frame": min(touched_frames) if touched_frames else None,
      "last_patched_frame": max(touched_frames) if touched_frames else None,
  }
  summary_path = args.out.with_suffix(args.out.suffix + ".json")
  summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  main()
