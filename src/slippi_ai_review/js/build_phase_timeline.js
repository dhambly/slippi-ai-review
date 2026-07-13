#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

function parseArgs(argv) {
  const args = {
    replay: null,
    out: null,
    meaningfulDamage: 8,
    meaningfulDuration: 45,
    mergeNeutralGap: 12,
    shortNeutralFrames: 45,
    tradeLeadInFrames: 120,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const next = () => {
      i += 1;
      if (i >= argv.length) throw new Error(`Missing value for ${arg}`);
      return argv[i];
    };
    if (arg === "--replay") args.replay = next();
    else if (arg === "--out") args.out = next();
    else if (arg === "--meaningful-damage") args.meaningfulDamage = Number(next());
    else if (arg === "--meaningful-duration") args.meaningfulDuration = Number(next());
    else if (arg === "--merge-neutral-gap") args.mergeNeutralGap = Number(next());
    else if (arg === "--short-neutral-frames") args.shortNeutralFrames = Number(next());
    else if (arg === "--trade-lead-in-frames") args.tradeLeadInFrames = Number(next());
    else if (arg === "--help" || arg === "-h") {
      console.log("usage: node build_phase_timeline.js --replay <file.slp> --out <timeline.json>");
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!args.replay) throw new Error("--replay is required");
  if (!args.out) throw new Error("--out is required");
  return args;
}

function slippiJsNodeModule() {
  return "@slippi/slippi-js/node";
}

let melee = null;
let meleeFrameData = null;

const INTERNAL_CHARACTER_NAMES = {
  0x00: "Mario",
  0x01: "Fox",
  0x02: "Captain Falcon",
  0x03: "Donkey Kong",
  0x04: "Kirby",
  0x05: "Bowser",
  0x06: "Link",
  0x07: "Sheik",
  0x08: "Ness",
  0x09: "Peach",
  0x0a: "Popo",
  0x0b: "Nana",
  0x0c: "Pikachu",
  0x0d: "Samus",
  0x0e: "Yoshi",
  0x0f: "Jigglypuff",
  0x10: "Mewtwo",
  0x11: "Luigi",
  0x12: "Marth",
  0x13: "Zelda",
  0x14: "Young Link",
  0x15: "Dr. Mario",
  0x16: "Falco",
  0x17: "Pichu",
  0x18: "Mr. Game & Watch",
  0x19: "Ganondorf",
  0x1a: "Roy",
};

const MOVE_KEY_BY_SHORT_NAME = {
  jab: "jab1",
  "rapid-jabs": "rapidjabs_loop",
  dash: "dashattack",
  ftilt: "ftilt_m",
  utilt: "utilt",
  dtilt: "dtilt",
  fsmash: "fsmash_m",
  usmash: "usmash",
  dsmash: "dsmash",
  nair: "nair",
  fair: "fair",
  bair: "bair",
  uair: "upair",
  dair: "dair",
  pummel: "pummel",
  fthrow: "fthrow",
  bthrow: "bthrow",
  uthrow: "uthrow",
  dthrow: "dthrow",
};

function framedataJsonPath() {
  return path.resolve(__dirname, "..", "data", "framedata.json");
}

function loadFrameData() {
  if (!meleeFrameData) {
    meleeFrameData = JSON.parse(fs.readFileSync(framedataJsonPath(), "utf8"));
  }
  return meleeFrameData;
}

function moveInfo(moveId) {
  if (!melee || moveId == null) {
    return { id: moveId, name: `move ${moveId}`, shortName: `move${moveId}` };
  }
  return {
    id: moveId,
    name: melee.moves.getMoveName(moveId),
    shortName: melee.moves.getMoveShortName(moveId),
  };
}

function round(value, places = 2) {
  if (value == null || Number.isNaN(value)) return null;
  const scale = 10 ** places;
  return Math.round(value * scale) / scale;
}

function seconds(frame) {
  return round(frame / 60, 3);
}

function conversionDamage(conversion) {
  const end = conversion.endPercent ?? conversion.currentPercent ?? conversion.startPercent ?? 0;
  return Math.max(0, end - (conversion.startPercent ?? 0));
}

const DEAD_ACTIONS = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const RESPAWN_ACTIONS = new Set([12, 13, 322, 323, 324, 327, 328, 329, 330, 331, 332, 333, 334, 336, 337, 338, 339]);
const STATE = {
  WAIT: 0x0e,
  WALK_SLOW: 0x0f,
  WALK_MIDDLE: 0x10,
  WALK_FAST: 0x11,
  TURN: 0x12,
  DASH: 0x14,
  RUN: 0x15,
  RUN_DIRECT: 0x16,
  RUN_BRAKE: 0x17,
  KNEE_BEND: 0x18,
  LANDING: 0x2a,
  LANDING_FALL_SPECIAL: 0x2b,
  ATTACK_START: 0x2c,
  ATTACK_END: 0x45,
  AERIAL_NAIR: 0x41,
  AERIAL_FAIR: 0x42,
  AERIAL_BAIR: 0x43,
  AERIAL_UAIR: 0x44,
  AERIAL_DAIR: 0x45,
  LANDING_AIR_START: 0x46,
  LANDING_AIR_END: 0x4a,
  DAMAGE_START: 0x4b,
  DAMAGE_END: 0x5b,
  GUARD_START: 0xb2,
  GUARD_END: 0xb6,
  GRAB: 0xd4,
  GRAB_PULL: 0xd5,
  DASH_GRAB: 0xd6,
  GRAB_WAIT: 0xd7,
  PUMMEL: 0xd8,
  F_THROW: 0xd9,
  B_THROW: 0xda,
  U_THROW: 0xdb,
  D_THROW: 0xdc,
  AIR_DODGE: 0xec,
};

const AERIAL_BY_LANDING = {
  [STATE.LANDING_AIR_START]: "nair",
  [STATE.LANDING_AIR_START + 1]: "fair",
  [STATE.LANDING_AIR_START + 2]: "bair",
  [STATE.LANDING_AIR_START + 3]: "uair",
  [STATE.LANDING_AIR_START + 4]: "dair",
};

const AERIAL_BY_ATTACK = {
  [STATE.AERIAL_NAIR]: "nair",
  [STATE.AERIAL_FAIR]: "fair",
  [STATE.AERIAL_BAIR]: "bair",
  [STATE.AERIAL_UAIR]: "uair",
  [STATE.AERIAL_DAIR]: "dair",
};

const GROUNDED_ATTACK_NAMES = {
  0x2c: "jab 1",
  0x2d: "jab 2",
  0x2e: "jab 3",
  0x2f: "rapid jab start",
  0x30: "rapid jab loop",
  0x31: "rapid jab end",
  0x32: "dash attack",
  0x33: "high ftilt",
  0x34: "high-mid ftilt",
  0x35: "ftilt",
  0x36: "low-mid ftilt",
  0x37: "low ftilt",
  0x38: "utilt",
  0x39: "dtilt",
  0x3a: "high fsmash",
  0x3b: "high-mid fsmash",
  0x3c: "fsmash",
  0x3d: "low-mid fsmash",
  0x3e: "low fsmash",
  0x3f: "usmash",
  0x40: "dsmash",
};

function groundedAttackName(action) {
  return GROUNDED_ATTACK_NAMES[action] || actionName(null, action);
}

function isSmashAttack(action) {
  return action >= 0x3a && action <= 0x40;
}

function framePost(frames, frame, playerIndex) {
  return frames?.[frame]?.players?.[playerIndex]?.post ?? null;
}

function actionAt(frames, frame, playerIndex) {
  return framePost(frames, frame, playerIndex)?.actionStateId ?? null;
}

function actionName(slippi, actionStateId) {
  if (actionStateId == null) return "unknown";
  const localNames = {
    0x00: "dead down",
    0x01: "dead left",
    0x02: "dead right",
    0x03: "dead up",
    0x0e: "wait",
    0x0f: "walk slow",
    0x10: "walk middle",
    0x11: "walk fast",
    0x12: "turn",
    0x14: "dash",
    0x15: "run",
    0x16: "run direct",
    0x17: "run brake",
    0x18: "jump squat",
    0x1d: "fall",
    0x1e: "fall forward",
    0x1f: "fall back",
    0x20: "fall aerial",
    0x21: "fall aerial forward",
    0x22: "fall aerial back",
    0x27: "crouch start",
    0x28: "crouch hold",
    0x29: "crouch end",
    0x2a: "landing",
    0x2b: "landing fall special",
    ...GROUNDED_ATTACK_NAMES,
    [STATE.LANDING_AIR_START]: "landing nair",
    [STATE.LANDING_AIR_START + 1]: "landing fair",
    [STATE.LANDING_AIR_START + 2]: "landing bair",
    [STATE.LANDING_AIR_START + 3]: "landing uair",
    [STATE.LANDING_AIR_START + 4]: "landing dair",
    0xb2: "shield startup",
    0xb3: "shield hold",
    0xb4: "shield release",
    0xb5: "shield stun",
    0xb6: "powershield reflect",
    0xb7: "knockdown bounce face up",
    0xb8: "knockdown wait face up",
    0xb9: "knockdown damage face up",
    0xba: "getup face up",
    0xbb: "getup attack face up",
    0xbc: "tech roll forward face up",
    0xbd: "tech roll back face up",
    0xbe: "missed tech face up",
    0xbf: "knockdown bounce face down",
    0xc0: "knockdown wait face down",
    0xc1: "knockdown damage face down",
    0xc2: "getup face down",
    0xc3: "getup attack face down",
    0xc4: "tech roll forward face down",
    0xc5: "tech roll back face down",
    0xc6: "missed tech face down",
    0xc7: "tech in place",
    0xc8: "wall tech",
    0xc9: "ceiling tech",
    0xca: "missed tech getup",
    0xcb: "wall jump tech",
    0xcc: "missed wall tech",
    0xd4: "grab",
    0xd5: "grab pull",
    0xd6: "dash grab",
    0xd7: "grab hold",
    0xd8: "pummel",
    0xd9: "forward throw",
    0xda: "back throw",
    0xdb: "up throw",
    0xdc: "down throw",
    0xec: "air dodge",
    0xf5: "teeter",
    0xfc: "ledge catch",
    0xfd: "ledge hang",
  };
  if (localNames[actionStateId]) return localNames[actionStateId];
  try {
    const value = melee?.actions?.getName ? melee.actions.getName(actionStateId) : null;
    if (value) return value;
  } catch (_) {}
  const known = Object.entries(STATE).find(([, value]) => value === actionStateId);
  return known ? known[0].toLowerCase() : `state ${actionStateId}`;
}

function isDamageState(action) {
  return action >= STATE.DAMAGE_START && action <= STATE.DAMAGE_END;
}

function isAerialLanding(action) {
  return action >= STATE.LANDING_AIR_START && action <= STATE.LANDING_AIR_END;
}

function isAerialAttack(action) {
  return action >= STATE.AERIAL_NAIR && action <= STATE.AERIAL_DAIR;
}

function commitmentSource(action) {
  if (isAerialLanding(action)) {
    const aerial = AERIAL_BY_LANDING[action] || "unknown aerial";
    return {
      key: `landing ${aerial}`,
      category: "aerial landing",
      actionName: `landing ${aerial}`,
      insertionQuality: "low",
    };
  }
  if (action === STATE.LANDING_FALL_SPECIAL) {
    return {
      key: "special landing",
      category: "special landing",
      actionName: "landing fall special",
      insertionQuality: "high",
    };
  }
  if (action === STATE.LANDING) {
    return {
      key: "normal landing",
      category: "normal landing",
      actionName: "normal landing",
      insertionQuality: "low",
    };
  }
  if (isAerialAttack(action)) {
    return {
      key: `aerial attack ${action}`,
      category: "aerial attack commitment",
      actionName: AERIAL_BY_ATTACK[action] || actionName(null, action),
      insertionQuality: "low",
    };
  }
  if (action >= STATE.ATTACK_START && action <= STATE.ATTACK_END) {
    const smash = isSmashAttack(action);
    return {
      key: `attack state ${action}`,
      category: smash ? "smash attack commitment" : "grounded attack commitment",
      actionName: groundedAttackName(action),
      insertionQuality: "high",
    };
  }
  if (action >= STATE.GRAB && action <= STATE.D_THROW) {
    return {
      key: `grab/throw state ${action}`,
      category: "grab/throw commitment",
      actionName: actionName(null, action),
      insertionQuality: "medium",
    };
  }
  if (action === STATE.AIR_DODGE) {
    return {
      key: "air dodge",
      category: "dodge endlag",
      actionName: "air dodge",
      insertionQuality: "high",
    };
  }
  return null;
}

function commitmentOpportunityScore(candidate, openingFrame) {
  const categoryBonus = {
    "smash attack commitment": 12,
    "special landing": 8,
    "grounded attack commitment": 8,
    "dodge endlag": 6,
    "grab/throw commitment": 4,
    "normal landing": 1,
    "aerial landing": -4,
    "aerial attack commitment": -6,
  }[candidate.category] || 0;
  const qualityBonus = { high: 10, medium: 4, low: -12 }[candidate.insertionQuality] || 0;
  const earlyBonus = Math.max(0, 30 - Math.max(0, candidate.startFrame - openingFrame));
  return candidate.durationFrames * 4 + categoryBonus + qualityBonus + earlyBonus;
}

function phillipInsertionCommitment(frames, conversion, labels) {
  const openingFrame = conversion.startFrame;
  const attacker = conversion.lastHitBy;
  const conversionEnd = conversion.endFrame ?? openingFrame;
  if (openingFrame == null || attacker == null) return null;
  const scanStart = openingFrame;
  const scanEnd = Math.min(conversionEnd, openingFrame + 120);
  const candidates = [];
  let active = null;

  const flush = (endFrame) => {
    if (!active) return;
    const duration = endFrame - active.startFrame + 1;
    if (duration >= 3) {
      candidates.push({
        playerIndex: attacker,
        playerLabel: labels[attacker] || `P${attacker + 1}`,
        startFrame: active.startFrame,
        endFrame,
        durationFrames: duration,
        actionStateId: active.actionStateId,
        actionName: active.source.actionName,
        key: active.source.key,
        category: active.source.category,
        insertionQuality: active.source.insertionQuality,
        actionStateCounterAtStart: active.counterAtStart,
        actionStateCounterAtEnd: active.counterAtEnd,
        framesAfterOpeningAtStart: active.startFrame - openingFrame,
        framesAfterOpeningAtEnd: endFrame - openingFrame,
      });
    }
    active = null;
  };

  for (let frame = scanStart; frame <= scanEnd; frame += 1) {
    const post = framePost(frames, frame, attacker);
    const action = post?.actionStateId ?? null;
    const source = post && !isDamageState(action) ? commitmentSource(action) : null;
    if (!source) {
      flush(frame - 1);
      continue;
    }
    const sameRun = active && active.actionStateId === action && active.source.key === source.key;
    if (!sameRun) {
      flush(frame - 1);
      active = {
        startFrame: frame,
        actionStateId: action,
        source,
        counterAtStart: post?.actionStateCounter ?? null,
        counterAtEnd: post?.actionStateCounter ?? null,
      };
    } else {
      active.counterAtEnd = post?.actionStateCounter ?? active.counterAtEnd;
    }
  }
  flush(scanEnd);

  if (!candidates.length) return null;
  const ranked = candidates.slice().sort((a, b) => (
    commitmentOpportunityScore(b, openingFrame) - commitmentOpportunityScore(a, openingFrame) ||
    a.startFrame - b.startFrame
  ));
  const recommended = ranked.find((candidate) => candidate.insertionQuality !== "low") || null;
  const best = recommended || ranked[0];
  return {
    recommendedInsertionFrame: best.startFrame,
    expectedControlFrame: best.endFrame + 1,
    openingFrame,
    playerIndex: best.playerIndex,
    playerLabel: best.playerLabel,
    category: best.category,
    actionName: best.actionName,
    actionStateId: best.actionStateId,
    startFrame: best.startFrame,
    endFrame: best.endFrame,
    durationFrames: best.durationFrames,
    framesAfterOpeningAtStart: best.framesAfterOpeningAtStart,
    framesAfterOpeningAtEnd: best.framesAfterOpeningAtEnd,
    actionStateCounterAtStart: best.actionStateCounterAtStart,
    actionStateCounterAtEnd: best.actionStateCounterAtEnd,
    confidence: best.durationFrames >= 8 ? "medium" : "low",
    insertionQuality: best.insertionQuality,
    recommended: Boolean(recommended),
    fallbackOnly: !recommended,
    candidateCount: candidates.length,
    lowQualityCandidateCount: candidates.filter((candidate) => candidate.insertionQuality === "low").length,
    note: recommended
      ? "Longest observed non-aerial attacker commitment early in advantage. Use recommendedInsertionFrame as a low-ambiguity Phillip takeover point, then expect meaningful model control after expectedControlFrame."
      : "Only low-quality aerial/landing commitments were found early in advantage. Avoid automatic Phillip insertion here unless fallback aerial insertion is explicitly enabled.",
  };
}

function playerFrameFacts(frames, frame, playerIndex, label) {
  const post = framePost(frames, frame, playerIndex);
  if (!post) {
    return { frame, playerIndex, playerLabel: label || `P${playerIndex + 1}` };
  }
  const x = post.positionX;
  let stageSide = "unknown";
  if (x != null) {
    if (x <= -80) stageSide = "left corner/edge area";
    else if (x < -25) stageSide = "left side";
    else if (x >= 80) stageSide = "right corner/edge area";
    else if (x > 25) stageSide = "right side";
    else stageSide = "center";
  }
  return {
    frame,
    playerIndex,
    playerLabel: label || `P${playerIndex + 1}`,
    percent: round(post.percent, 1),
    stocksRemaining: post.stocksRemaining ?? null,
    positionX: round(post.positionX, 2),
    positionY: round(post.positionY, 2),
    stageSide,
    actionStateId: post.actionStateId ?? null,
    actionStateName: actionName(null, post.actionStateId),
    actionStateCounter: post.actionStateCounter == null ? null : round(post.actionStateCounter, 2),
    isAirborne: post.isAirborne ?? null,
    facingDirection: post.facingDirection ?? null,
  };
}

function spatialSample(frames, frame, playerIndices, labels) {
  const players = {};
  for (const playerIndex of playerIndices) {
    players[playerIndex] = playerFrameFacts(frames, frame, playerIndex, labels[playerIndex]);
  }
  const pair = playerIndices.length >= 2 ? {
    players: [playerIndices[0], playerIndices[1]],
    horizontalSeparation: null,
    verticalSeparation: null,
    euclideanDistance: null,
    sameSide: null,
  } : null;
  if (pair) {
    const a = players[playerIndices[0]];
    const b = players[playerIndices[1]];
    if (
      a?.positionX != null &&
      a?.positionY != null &&
      b?.positionX != null &&
      b?.positionY != null
    ) {
      const dx = b.positionX - a.positionX;
      const dy = b.positionY - a.positionY;
      pair.horizontalSeparation = round(Math.abs(dx), 2);
      pair.verticalSeparation = round(Math.abs(dy), 2);
      pair.euclideanDistance = round(Math.sqrt(dx * dx + dy * dy), 2);
      pair.sameSide = (
        a.stageSide &&
        b.stageSide &&
        a.stageSide !== "unknown" &&
        b.stageSide !== "unknown" &&
        a.stageSide === b.stageSide
      );
    }
  }
  return {
    frame,
    time: seconds(frame),
    players,
    pair,
  };
}

function spatialTimelineForSegment(segment, frames, playerIndices, labels) {
  const start = segment.startFrame;
  const end = segment.endFrame;
  const candidates = [
    start,
    start + 15,
    start + 30,
    Math.floor((start + end) / 2),
    end - 30,
    end - 15,
    end,
  ];
  const moves = segment.conversion?.moveSequence || [];
  for (const move of moves.slice(0, 8)) {
    if (move.frame != null) candidates.push(move.frame);
  }
  const unique = [...new Set(
    candidates
      .map((frame) => Math.max(start, Math.min(end, Math.round(frame))))
      .filter((frame) => Number.isFinite(frame))
  )].sort((a, b) => a - b);
  return {
    purpose: "Sampled raw Slippi positions through this segment. Use this before blaming commitments: judge whether spacing, side, height, and relative distance made the action reasonable.",
    sample_frames: unique,
    samples: unique.map((frame) => spatialSample(frames, frame, playerIndices, labels)),
  };
}

function actionTimelineForSegment(segment, frames, playerIndices, labels) {
  const segmentStart = segment.startFrame;
  const segmentEnd = segment.endFrame;
  const start = Math.max(segmentStart, segmentEnd - 120);
  const end = segmentEnd;
  const runsByPlayer = {};
  for (const playerIndex of playerIndices) {
    const runs = [];
    let active = null;
    const flush = (endFrame) => {
      if (!active) return;
      runs.push({
        playerIndex,
        playerLabel: labels[playerIndex] || `P${playerIndex + 1}`,
        startFrame: active.startFrame,
        endFrame,
        durationFrames: endFrame - active.startFrame + 1,
        actionStateId: active.actionStateId,
        actionStateName: active.actionStateName,
        startCounter: active.startCounter,
        endCounter: active.endCounter,
        start: playerFrameFacts(frames, active.startFrame, playerIndex, labels[playerIndex]),
        end: playerFrameFacts(frames, endFrame, playerIndex, labels[playerIndex]),
      });
      active = null;
    };
    for (let frame = start; frame <= end; frame += 1) {
      const post = framePost(frames, frame, playerIndex);
      if (!post) {
        flush(frame - 1);
        continue;
      }
      const actionStateId = post.actionStateId ?? null;
      const stateName = actionName(null, actionStateId);
      if (!active || active.actionStateId !== actionStateId) {
        flush(frame - 1);
        active = {
          startFrame: frame,
          actionStateId,
          actionStateName: stateName,
          startCounter: post.actionStateCounter == null ? null : round(post.actionStateCounter, 2),
          endCounter: post.actionStateCounter == null ? null : round(post.actionStateCounter, 2),
        };
      } else {
        active.endCounter = post.actionStateCounter == null ? active.endCounter : round(post.actionStateCounter, 2);
      }
    }
    flush(end);
    runsByPlayer[playerIndex] = runs.slice(-16);
  }
  return {
    purpose: "Action-state runs over the final 120f of this segment. Use this to reconstruct temporal causality before blaming old tech flags.",
    windowStartFrame: start,
    windowEndFrame: end,
    runsByPlayer,
  };
}

function conversionFrameFacts(conversion, frames, labels) {
  const startFrame = conversion.startFrame;
  const endFrame = conversion.endFrame ?? conversion.startFrame;
  const attacker = conversion.lastHitBy;
  const victim = conversion.playerIndex;
  return {
    startFrame,
    endFrame,
    attackerStart: playerFrameFacts(frames, startFrame, attacker, labels[attacker]),
    attackerEnd: playerFrameFacts(frames, endFrame, attacker, labels[attacker]),
    defenderStart: playerFrameFacts(frames, startFrame, victim, labels[victim]),
    defenderEnd: playerFrameFacts(frames, endFrame, victim, labels[victim]),
  };
}

function stickDirection(x, y) {
  if (x == null || y == null) return "unknown";
  const dead = 0.25;
  const horiz = x > dead ? "right" : x < -dead ? "left" : "";
  const vert = y > dead ? "up" : y < -dead ? "down" : "";
  if (!horiz && !vert) return "neutral";
  return [vert, horiz].filter(Boolean).join("-");
}

function stickMagnitude(x, y) {
  if (x == null || y == null) return 0;
  return Math.sqrt(x * x + y * y);
}

function sdiRegion(pre) {
  if (!pre) return "unknown";
  if (stickMagnitude(pre.joystickX, pre.joystickY) < 0.3) return "neutral";
  return stickDirection(pre.joystickX, pre.joystickY);
}

function estimateSdiDuringHitlag(frames, hitFrame, victim) {
  const timeline = [];
  const events = [];
  const preHitPre = frames?.[hitFrame - 1]?.players?.[victim]?.pre || null;
  let previousRegion = preHitPre ? sdiRegion(preHitPre) : null;
  for (let frame = hitFrame; frame < hitFrame + 40; frame += 1) {
    const entry = frames?.[frame]?.players?.[victim];
    const pre = entry?.pre || null;
    const post = entry?.post || null;
    if (!post || !post.hitlagRemaining) {
      if (timeline.length) break;
      continue;
    }
    const region = sdiRegion(pre);
    const row = {
      frame,
      hitlagRemaining: post.hitlagRemaining,
      stickX: pre ? round(pre.joystickX, 2) : null,
      stickY: pre ? round(pre.joystickY, 2) : null,
      region,
    };
    timeline.push(row);
    if (
      region !== "unknown" &&
      region !== "neutral" &&
      previousRegion != null &&
      region !== previousRegion
    ) {
      events.push({
        frame,
        region,
        stickX: row.stickX,
        stickY: row.stickY,
        previousRegion,
        source: frame === hitFrame ? "changed from frame before hit" : "changed during hitlag",
      });
    }
    previousRegion = region;
  }
  return {
    hitlagFramesObserved: timeline.length,
    preHitStick: preHitPre ? {
      frame: hitFrame - 1,
      stickX: round(preHitPre.joystickX, 2),
      stickY: round(preHitPre.joystickY, 2),
      region: sdiRegion(preHitPre),
    } : null,
    estimatedSdiEvents: events,
    summary: summarizeSdiEvents(events),
    stickTimeline: timeline,
    note: "Estimated from Slippi per-frame stick region changes during hitlag; not Melee's internal timer_lstick_tilt_x/y counter.",
  };
}

function summarizeSdiEvents(events) {
  if (!events.length) {
    return {
      count: 0,
      intent: "none",
      directions: [],
      text: "no estimated SDI",
    };
  }
  const axisCounts = { left: 0, right: 0, up: 0, down: 0 };
  for (const event of events) {
    const x = event.stickX ?? 0;
    const y = event.stickY ?? 0;
    if (x > 0.25) axisCounts.right += 1;
    if (x < -0.25) axisCounts.left += 1;
    if (y > 0.25) axisCounts.up += 1;
    if (y < -0.25) axisCounts.down += 1;
  }
  const horizontal = axisCounts.right - axisCounts.left;
  const vertical = axisCounts.up - axisCounts.down;
  const horizontalDirections = [];
  const verticalDirections = [];
  if (Math.abs(horizontal) >= Math.max(1, Math.ceil(events.length * 0.35))) {
    horizontalDirections.push(horizontal > 0 ? "right" : "left");
  }
  if (Math.abs(vertical) >= Math.max(1, Math.ceil(events.length * 0.35))) {
    verticalDirections.push(vertical > 0 ? "up" : "down");
  }
  const directions = [...verticalDirections, ...horizontalDirections];
  const intent = directions.length ? `SDI ${directions.join("-")}` : "SDI unclear";
  return {
    count: events.length,
    intent,
    directions,
    axisCounts,
    text: `${events.length} estimated SDI ${events.length === 1 ? "event" : "events"}; ${intent}`,
  };
}

function trajectoryDelta(post, after) {
  if (!post || !after) return null;
  const dx = (after.positionX ?? 0) - (post.positionX ?? 0);
  const dy = (after.positionY ?? 0) - (post.positionY ?? 0);
  return {
    x: round(dx, 2),
    y: round(dy, 2),
    angleDegrees: round(Math.atan2(dy, dx) * 180 / Math.PI, 1),
  };
}

function vectorAngleDegrees(x, y) {
  if (x == null || y == null || (x === 0 && y === 0)) return null;
  return Math.atan2(y, x) * 180 / Math.PI;
}

function normalizeAngleDegrees(angle) {
  if (angle == null) return null;
  let value = angle % 360;
  if (value < 0) value += 360;
  return value;
}

function angularDistanceDegrees(a, b) {
  if (a == null || b == null) return null;
  const diff = Math.abs(normalizeAngleDegrees(a) - normalizeAngleDegrees(b));
  return diff > 180 ? 360 - diff : diff;
}

function unitVectorFromAngleRadians(angle) {
  return {
    x: Math.cos(angle),
    y: Math.sin(angle),
  };
}

function stickVectorFromSample(sample) {
  if (!sample || sample.x == null || sample.y == null) return null;
  const mag = stickMagnitude(sample.x, sample.y);
  if (mag < 0.25) return null;
  return {
    x: sample.x / mag,
    y: sample.y / mag,
    angleDegrees: vectorAngleDegrees(sample.x, sample.y),
  };
}

function diAlignment(actual, target) {
  if (!actual || !target) return null;
  const actualAngle = actual.angleDegrees;
  const targetAngle = vectorAngleDegrees(target.x, target.y);
  const distance = angularDistanceDegrees(actualAngle, targetAngle);
  return {
    angleDiffDegrees: round(distance, 1),
    dot: round((actual.x * target.x) + (actual.y * target.y), 2),
  };
}

function percentDeltaAt(frames, frame, playerIndex) {
  const post = framePost(frames, frame, playerIndex);
  const prev = framePost(frames, frame - 1, playerIndex);
  if (!post || !prev || post.percent == null || prev.percent == null) return null;
  return round(Math.max(0, post.percent - prev.percent), 2);
}

function moveKeyForMove(shortName, moveName) {
  if (shortName && MOVE_KEY_BY_SHORT_NAME[shortName]) return MOVE_KEY_BY_SHORT_NAME[shortName];
  const normalized = String(moveName || "").toLowerCase().replace(/\s+/g, "");
  const byName = {
    dashattack: "dashattack",
    neutralair: "nair",
    forwardair: "fair",
    backair: "bair",
    upair: "upair",
    downair: "dair",
    uptilt: "utilt",
    downtilt: "dtilt",
    forwardtilt: "ftilt_m",
    upsmash: "usmash",
    downsmash: "dsmash",
    forwardsmash: "fsmash_m",
  };
  return byName[normalized] || null;
}

function specialMoveKeyFor(characterName, shortName, attackerPost, immediateDamage) {
  if (characterName !== "Fox") return null;
  if (shortName === "down-b") return attackerPost?.isAirborne ? "0x13d" : "0x139";
  if (shortName === "up-b") {
    if (immediateDamage != null && immediateDamage <= 3) return attackerPost?.isAirborne ? "0x134" : "0x133";
    return "0x135";
  }
  return null;
}

function resolvedHitboxAngleDegrees(angle, victimPost) {
  if (angle == null) return null;
  if (angle === 361) {
    return victimPost?.isAirborne ? 45 : 0;
  }
  return angle;
}

function actualLaunchAngleFromHitboxAngle(angle, opponentDir) {
  if (angle == null || opponentDir == null) return angle;
  const actual = opponentDir < 0 ? angle : 180 - angle;
  return normalizeAngleDegrees(actual);
}

function actualLaunchAngleFromThrowAngle(angle, moveKey, attackerPost, victimPost) {
  if (angle == null || !attackerPost || !victimPost) return angle;
  let hitAngle = angle;
  let dir = null;
  if (moveKey === "bthrow") {
    dir = victimPost.facingDirection;
    if (hitAngle > 95) hitAngle = 180 - hitAngle;
  } else if (hitAngle < 95) {
    dir = -attackerPost.facingDirection;
  } else {
    dir = attackerPost.facingDirection;
    hitAngle = 180 - hitAngle;
  }
  if (dir == null) return angle;
  return normalizeAngleDegrees(dir < 0 ? hitAngle : 180 - hitAngle);
}

function isThrowMoveKey(moveKey) {
  return moveKey === "fthrow" || moveKey === "bthrow" || moveKey === "uthrow" || moveKey === "dthrow";
}

function candidateFromThrowData(throwData, victimPost, immediateDamage) {
  if (!throwData) return null;
  const damageDiff = immediateDamage == null ? null : Math.abs((throwData.damage ?? 0) - immediateDamage);
  return {
    hitboxId: "throw",
    type: "throw",
    damage: throwData.damage ?? null,
    angle: throwData.angle ?? null,
    resolvedAngleDegrees: resolvedHitboxAngleDegrees(throwData.angle, victimPost),
    kbGrowth: throwData.kbGrowth ?? null,
    weightDepKb: throwData.weightDepKb ?? null,
    baseKb: throwData.baseKb ?? null,
    element: throwData.element ?? null,
    damageDiff: damageDiff == null ? null : round(damageDiff, 2),
  };
}

function candidateFromProjectile(shortName, immediateDamage) {
  if (shortName !== "neutral-b") return null;
  return {
    hitboxId: "projectile",
    type: "projectile",
    damage: immediateDamage,
    angle: null,
    resolvedAngleDegrees: null,
    kbGrowth: null,
    weightDepKb: null,
    baseKb: null,
    element: "laser",
    damageDiff: 0,
    note: "Projectile/no-flinch style hit; DI comparison is not meaningful.",
  };
}

function candidateFromNoDamageContact(shortName, immediateDamage, victimPost) {
  const speeds = victimPost?.selfInducedSpeeds || {};
  const hasAttackVelocity = Math.abs(speeds.attackX || 0) > 0.001 || Math.abs(speeds.attackY || 0) > 0.001;
  if (immediateDamage !== 0 || hasAttackVelocity) return null;
  return {
    hitboxId: "no_damage_contact",
    type: "no_damage_contact",
    damage: 0,
    angle: null,
    resolvedAngleDegrees: null,
    kbGrowth: null,
    weightDepKb: null,
    baseKb: null,
    element: shortName || "unknown",
    damageDiff: 0,
    note: "Recorded contact caused no damage and no attack velocity; DI comparison is not meaningful.",
  };
}

function candidateFromGenericEdgeAttack(shortName, immediateDamage, victimPost) {
  if (shortName !== "edge" && shortName !== "edge-slow") return null;
  if (immediateDamage == null || immediateDamage <= 0) return null;
  const speeds = victimPost?.selfInducedSpeeds || {};
  return {
    hitboxId: "generic_edge_attack",
    type: "generic_edge_attack",
    damage: immediateDamage,
    angle: null,
    resolvedAngleDegrees: null,
    observedVelocityAngleDegrees: round(vectorAngleDegrees(speeds.attackX, speeds.attackY), 1),
    kbGrowth: null,
    weightDepKb: null,
    baseKb: null,
    element: "edge_attack",
    damageDiff: 0,
    note: "Generic ledge attack is not present in bundled framedata; using observed attack velocity for DI comparison.",
  };
}

function selectCandidateByEvidence(candidates, velocityAngle, opponentDir) {
  if (!candidates.length) return null;
  const scored = candidates.map((candidate) => {
    const actualAngle = actualLaunchAngleFromHitboxAngle(candidate.resolvedAngleDegrees, opponentDir);
    const angleDiff = velocityAngle == null || actualAngle == null
      ? null
      : angularDistanceDegrees(velocityAngle, actualAngle);
    const damageScore = candidate.damageDiff == null ? 0 : Math.max(0, 8 - candidate.damageDiff * 2);
    const angleScore = angleDiff == null ? 0 : Math.max(0, 8 - angleDiff / 5);
    return {
      candidate,
      actualAngle,
      angleDiff,
      score: damageScore + angleScore,
    };
  }).sort((a, b) => b.score - a.score);
  const best = scored[0];
  const second = scored[1] || null;
  if (!best) return null;
  const scoreGap = second ? best.score - second.score : Infinity;
  const damageGap = second && best.candidate.damageDiff != null && second.candidate.damageDiff != null
    ? second.candidate.damageDiff - best.candidate.damageDiff
    : null;
  const angleGap = second && best.angleDiff != null && second.angleDiff != null
    ? second.angleDiff - best.angleDiff
    : null;
  if (scoreGap >= 1.5 || (damageGap != null && damageGap >= 0.75) || (angleGap != null && angleGap >= 12)) {
    return {
      candidate: best.candidate,
      confidence: scoreGap >= 3 || (damageGap != null && damageGap >= 1.5) || (angleGap != null && angleGap >= 20)
        ? "high"
        : "medium",
      reason: `selected by evidence: damageDiff=${best.candidate.damageDiff}, velocityAngleDiff=${best.angleDiff == null ? "unknown" : round(best.angleDiff, 1)}`,
    };
  }
  return null;
}

function inferHitboxCandidates(frames, hitFrame, attacker, victim, move, info) {
  const attackerPost = framePost(frames, hitFrame, attacker);
  const victimPost = framePost(frames, hitFrame, victim);
  const characterName = INTERNAL_CHARACTER_NAMES[attackerPost?.internalCharacterId];
  const immediateDamage = percentDeltaAt(frames, hitFrame, victim);
  const moveKey = moveKeyForMove(info.shortName, info.name)
    || specialMoveKeyFor(characterName, info.shortName, attackerPost, immediateDamage);
  const inference = {
    available: false,
    confidence: "low",
    source: "slippi-js framedata.json",
    characterName,
    moveKey,
    actionStateCounter: attackerPost?.actionStateCounter != null ? round(attackerPost.actionStateCounter, 2) : null,
    immediateDamage,
    candidates: [],
    selectedAngleDegrees: null,
    selectedAngleSource: null,
    note: "",
  };
  if (!characterName || !moveKey) {
    const projectileCandidate = candidateFromProjectile(info.shortName, immediateDamage);
    const noDamageCandidate = candidateFromNoDamageContact(info.shortName, immediateDamage, victimPost);
    const edgeCandidate = candidateFromGenericEdgeAttack(info.shortName, immediateDamage, victimPost);
    const specialCandidate = projectileCandidate || noDamageCandidate || edgeCandidate;
    if (specialCandidate) {
      inference.available = true;
      inference.confidence = edgeCandidate ? "high" : "exact";
      inference.candidates = [specialCandidate];
      inference.selectedAngleDegrees = null;
      inference.selectedAngleSource = specialCandidate.type;
      inference.note = specialCandidate.note;
      return inference;
    }
    inference.note = "Could not map attacker character or Slippi move id to framedata key.";
    return inference;
  }
  const data = loadFrameData();
  const moveData = data?.[characterName]?.[moveKey];
  if (!moveData) {
    inference.note = `No framedata entry for ${characterName} ${moveKey}.`;
    return inference;
  }
  if (isThrowMoveKey(moveKey)) {
    const candidate = candidateFromThrowData(moveData.throw, victimPost, immediateDamage);
    if (!candidate) {
      inference.note = `No throw data in framedata entry for ${characterName} ${moveKey}.`;
      return inference;
    }
    inference.available = true;
    inference.confidence = "exact";
    inference.candidates = [candidate];
    inference.selectedAngleDegrees = candidate.resolvedAngleDegrees;
    inference.selectedAngleSource = "throw data";
    inference.note = "Throw lookup uses framedata throw object rather than active hitbox frames.";
    return inference;
  }
  const counter = attackerPost?.actionStateCounter;
  if (counter == null) {
    inference.note = "No attacker actionStateCounter on hit frame.";
    return inference;
  }
  const activeGroups = (moveData.hitFrames || []).filter((group) => (
    counter >= group.start - 0.01 && counter <= group.end + 0.01
  ));
  const activeHitboxIds = new Set(activeGroups.flatMap((group) => group.hitboxes || []));
  if (!activeHitboxIds.size && (moveData.hitboxes || []).length === 1 && info.shortName === "up-b") {
    activeHitboxIds.add(0);
  }
  const rawCandidates = [...activeHitboxIds]
    .map((hitboxId) => {
      const hitbox = moveData.hitboxes?.[hitboxId];
      if (!hitbox) return null;
      const canHitState = victimPost?.isAirborne ? hitbox.hitAirborne !== false : hitbox.hitGrounded !== false;
      const damageDiff = immediateDamage == null ? null : Math.abs((hitbox.damage ?? 0) - immediateDamage);
      return {
        hitboxId,
        damage: hitbox.damage ?? null,
        angle: hitbox.angle ?? null,
        resolvedAngleDegrees: resolvedHitboxAngleDegrees(hitbox.angle, victimPost),
        kbGrowth: hitbox.kbGrowth ?? null,
        weightDepKb: hitbox.weightDepKb ?? null,
        baseKb: hitbox.baseKb ?? null,
        hitboxInteraction: hitbox.hitboxInteraction ?? null,
        hitGrounded: hitbox.hitGrounded ?? null,
        hitAirborne: hitbox.hitAirborne ?? null,
        canHitVictimState: canHitState,
        damageDiff: damageDiff == null ? null : round(damageDiff, 2),
      };
    })
    .filter(Boolean)
    .filter((candidate) => candidate.canHitVictimState);
  inference.available = rawCandidates.length > 0;
  inference.candidates = rawCandidates;
  if (!inference.available) {
    inference.note = activeHitboxIds.size
      ? "Active hitboxes existed, but none matched victim grounded/aerial state."
      : "No active framedata hitbox matched the attacker's action counter.";
    return inference;
  }

  const damageMatched = immediateDamage == null
    ? rawCandidates
    : rawCandidates.filter((candidate) => candidate.damageDiff != null && candidate.damageDiff <= 0.75);
  const narrowed = damageMatched.length ? damageMatched : rawCandidates;
  const uniqueAngles = new Set(narrowed.map((candidate) => candidate.resolvedAngleDegrees).filter((value) => value != null));
  const uniqueIds = new Set(narrowed.map((candidate) => candidate.hitboxId));
  inference.candidates = narrowed;
  if (uniqueIds.size === 1) inference.confidence = "exact";
  else if (damageMatched.length && uniqueAngles.size === 1) inference.confidence = "high";
  else if (uniqueAngles.size === 1) inference.confidence = "medium";
  else inference.confidence = "low";
  if (uniqueAngles.size === 1) {
    inference.selectedAngleDegrees = [...uniqueAngles][0];
    inference.selectedAngleSource = inference.confidence === "exact"
      ? "unique active hitbox"
      : "shared angle across candidate hitboxes";
  } else {
    const speeds = victimPost?.selfInducedSpeeds || {};
    const velocityAngle = vectorAngleDegrees(speeds.attackX, speeds.attackY);
    const victimX = victimPost?.positionX ?? null;
    const attackerX = attackerPost?.positionX ?? null;
    const opponentDir = victimX == null || attackerX == null || attackerX === victimX
      ? null
      : (attackerX > victimX ? 1 : -1);
    const selected = selectCandidateByEvidence(narrowed, velocityAngle, opponentDir);
    if (selected) {
      inference.confidence = selected.confidence;
      inference.selectedAngleDegrees = selected.candidate.resolvedAngleDegrees;
      inference.selectedAngleSource = selected.reason;
    } else {
      inference.note = "Multiple candidate hitboxes have different angles; use observed velocity fallback for DI comparison.";
    }
  }
  return inference;
}

function estimateCeDiOptions(frames, hitFrame, attacker, victim, diStick, post, hitboxInference = null) {
  const victimPost = post || framePost(frames, hitFrame, victim);
  const attackerPost = framePost(frames, hitFrame, attacker);
  const speeds = victimPost?.selfInducedSpeeds || {};
  const kbX = speeds.attackX ?? null;
  const kbY = speeds.attackY ?? null;
  const rawInferredAngle = hitboxInference?.selectedAngleDegrees ?? null;
  const velocityAngle = kbX == null || kbY == null || (kbX === 0 && kbY === 0)
    ? null
    : vectorAngleDegrees(kbX, kbY);
  const victimX = victimPost?.positionX ?? null;
  const attackerX = attackerPost?.positionX ?? null;
  const opponentDir = victimX == null || attackerX == null || attackerX === victimX
    ? null
    : (attackerX > victimX ? 1 : -1);
  const isThrowAngle = hitboxInference?.candidates?.length === 1 && hitboxInference.candidates[0]?.type === "throw";
  const inferredAngle = isThrowAngle
    ? actualLaunchAngleFromThrowAngle(rawInferredAngle, hitboxInference.moveKey, attackerPost, victimPost)
    : actualLaunchAngleFromHitboxAngle(rawInferredAngle, opponentDir);
  const kbAngleDegrees = inferredAngle ?? velocityAngle;
  const kbAngle = kbAngleDegrees == null ? null : kbAngleDegrees * Math.PI / 180;
  if (kbAngle == null || opponentDir == null) {
    return {
      available: false,
      note: "Could not estimate CE-style DI in/out because knockback vector or attacker side was unavailable.",
    };
  }

  const twoPi = Math.PI * 2;
  let angle = kbAngle;
  while (angle < 0) angle += twoPi;
  while (angle >= twoPi) angle -= twoPi;

  const inAngle = angle <= Math.PI
    ? angle - Math.PI * 0.5 * opponentDir
    : angle + Math.PI * 0.5 * opponentDir;
  const outAngle = angle <= Math.PI
    ? angle + Math.PI * 0.5 * opponentDir
    : angle - Math.PI * 0.5 * opponentDir;
  const inVector = unitVectorFromAngleRadians(inAngle);
  const outVector = unitVectorFromAngleRadians(outAngle);
  const actual = stickVectorFromSample(diStick);
  const alignIn = diAlignment(actual, inVector);
  const alignOut = diAlignment(actual, outVector);
  let closest = "neutral/no meaningful DI";
  if (actual && alignIn && alignOut) {
    const diffGap = Math.abs(alignIn.angleDiffDegrees - alignOut.angleDiffDegrees);
    closest = diffGap < 7
      ? "ambiguous/orthogonal to DI in-out"
      : (alignIn.angleDiffDegrees < alignOut.angleDiffDegrees ? "DI in" : "DI out");
  }
  return {
    available: true,
    estimatedFrom: inferredAngle != null
      ? `framedata hitbox angle (${hitboxInference.selectedAngleSource}); approximate CE-style comparison, not a full no-DI knockback simulation.`
      : "Slippi selfInducedSpeeds.attackX/Y at contact plus attacker side; approximate CE-style comparison, not a full no-DI knockback simulation.",
    angleSource: inferredAngle != null ? "framedata" : "observed_velocity",
    baseAngleDegrees: round(kbAngleDegrees, 1),
    rawHitboxAngleDegrees: rawInferredAngle == null ? null : round(rawInferredAngle, 1),
    knockbackVector: {
      x: round(kbX, 3),
      y: round(kbY, 3),
      angleDegrees: round(vectorAngleDegrees(kbX, kbY), 1),
    },
    opponentDir,
    diIn: {
      x: round(inVector.x, 2),
      y: round(inVector.y, 2),
      direction: stickDirection(inVector.x, inVector.y),
      angleDegrees: round(vectorAngleDegrees(inVector.x, inVector.y), 1),
      alignment: alignIn,
    },
    diOut: {
      x: round(outVector.x, 2),
      y: round(outVector.y, 2),
      direction: stickDirection(outVector.x, outVector.y),
      angleDegrees: round(vectorAngleDegrees(outVector.x, outVector.y), 1),
      alignment: alignOut,
    },
    actualClosest: closest,
  };
}

function diVerdictFromComparison(ceDiComparison, diStick, hitboxInference) {
  const specialType = hitboxInference?.candidates?.[0]?.type || null;
  if (specialType === "projectile" || specialType === "no_damage_contact") {
    return {
      verdict: "DI not meaningful",
      confidence: "high",
      basis: hitboxInference?.note || "Hit does not produce meaningful DI.",
    };
  }
  if (!ceDiComparison?.available) {
    return {
      verdict: "unknown",
      confidence: "low",
      basis: ceDiComparison?.note || "DI comparison unavailable.",
    };
  }
  const actual = stickVectorFromSample(diStick);
  if (!actual) {
    return {
      verdict: "no meaningful DI",
      confidence: "high",
      basis: `Last-hitlag main stick was neutral; angle source ${ceDiComparison.angleSource}.`,
    };
  }
  const closest = ceDiComparison.actualClosest || "unknown";
  let verdict = closest;
  if (closest === "ambiguous/orthogonal to DI in-out") verdict = "ambiguous";
  const confidenceByHitbox = {
    exact: "high",
    high: "high",
    medium: "medium",
    low: "low",
  };
  let confidence = confidenceByHitbox[hitboxInference?.confidence] || "medium";
  if (ceDiComparison.angleSource === "observed_velocity" && confidence === "high") {
    confidence = "medium";
  }
  if (verdict === "ambiguous") confidence = confidence === "low" ? "low" : "medium";
  return {
    verdict,
    confidence,
    basis: `${hitboxInference?.confidence || "unknown"} hitbox inference; angle source ${ceDiComparison.angleSource}; base angle ${ceDiComparison.baseAngleDegrees}.`,
    actualStickFrame: diStick?.frame ?? null,
    actualStickDirection: diStick?.direction ?? null,
    hitboxConfidence: hitboxInference?.confidence || null,
    angleSource: ceDiComparison.angleSource || null,
  };
}

function hitDiSamples(conversion, frames, labels) {
  return (conversion.moves || []).map((move, index) => {
    const victim = conversion.playerIndex;
    const attacker = move.playerIndex;
    const pre = frames?.[move.frame]?.players?.[victim]?.pre || null;
    const post = framePost(frames, move.frame, victim);
    const after = framePost(frames, move.frame + 12, victim);
    const sdiEstimate = estimateSdiDuringHitlag(frames, move.frame, victim);
    const diFrame = sdiEstimate.stickTimeline[sdiEstimate.stickTimeline.length - 1] || null;
    const info = moveInfo(move.moveId);
    const hitboxInference = inferHitboxCandidates(frames, move.frame, attacker, victim, move, info);
    const diStick = diFrame ? {
      frame: diFrame.frame,
      hitlagRemaining: diFrame.hitlagRemaining,
      x: diFrame.stickX,
      y: diFrame.stickY,
      direction: diFrame.region === "unknown" ? "unknown" : stickDirection(diFrame.stickX, diFrame.stickY),
    } : null;
    const ceDiComparison = estimateCeDiOptions(frames, move.frame, attacker, victim, diStick ? {
      x: diStick.x,
      y: diStick.y,
    } : null, post, hitboxInference);
    return {
      index: index + 1,
      frame: move.frame,
      moveName: info.name,
      moveShortName: info.shortName,
      attacker: labels[attacker] || `P${attacker + 1}`,
      victim: labels[victim] || `P${victim + 1}`,
      contactStick: pre ? {
        x: round(pre.joystickX, 2),
        y: round(pre.joystickY, 2),
        direction: stickDirection(pre.joystickX, pre.joystickY),
      } : null,
      diStick,
      hitboxInference,
      ceDiComparison,
      diVerdict: diVerdictFromComparison(ceDiComparison, diStick, hitboxInference),
      contactCStick: pre ? {
        x: round(pre.cStickX, 2),
        y: round(pre.cStickY, 2),
        direction: stickDirection(pre.cStickX, pre.cStickY),
      } : null,
      hitlagRemaining: post?.hitlagRemaining ?? null,
      positionDelta12f: trajectoryDelta(post, after),
      sdiEstimate,
    };
  });
}

function isRespawnLike(frames, frame, playerIndices) {
  for (const playerIndex of playerIndices) {
    const action = actionAt(frames, frame, playerIndex);
    if (action == null) return true;
    if (DEAD_ACTIONS.has(action) || RESPAWN_ACTIONS.has(action)) return true;
  }
  return false;
}

function resolveRespawnEnd(frames, start, lastFrame, playerIndices) {
  const minEnd = Math.min(lastFrame, start + 90);
  for (let frame = start; frame <= lastFrame; frame += 1) {
    if (frame < minEnd) continue;
    if (!isRespawnLike(frames, frame, playerIndices)) return frame;
  }
  return Math.min(lastFrame, start + 240);
}

function playerLabel(settings, playerIndex) {
  const player = (settings.players || []).find((item) => item.playerIndex === playerIndex);
  if (!player) return `P${playerIndex + 1}`;
  const port = player.port ?? playerIndex + 1;
  const name = player.displayName || player.connectCode || player.nametag || "";
  return name ? `P${port} ${name}` : `P${port}`;
}

function isMeaningfulConversion(conversion, args) {
  const damage = conversionDamage(conversion);
  const duration = Math.max(0, (conversion.endFrame ?? conversion.startFrame) - conversion.startFrame);
  const moves = (conversion.moves || []).length;
  const openingType = conversion.openingType || "unknown";
  if (conversion.didKill) return true;
  if (damage >= args.meaningfulDamage) return true;
  if (moves >= 2) return true;
  if (duration >= args.meaningfulDuration && damage >= 5) return true;
  if (openingType === "neutral-win" && damage >= 6) return true;
  return false;
}

function isOneHitKill(conversion) {
  return Boolean(conversion.didKill) && (conversion.moves || []).length <= 1;
}

function isMutualTradeConversion(conversion, conversions) {
  if ((conversion.openingType || "unknown") !== "trade" || conversion.didKill) {
    return false;
  }
  const start = conversion.startFrame;
  const end = conversion.endFrame ?? conversion.startFrame;
  return conversions.some((other) => {
    if (other === conversion) return false;
    if ((other.openingType || "unknown") !== "trade" || other.didKill) return false;
    if (Math.abs((other.startFrame ?? 0) - start) > 3) return false;
    const otherEnd = other.endFrame ?? other.startFrame;
    const overlaps = Math.min(end, otherEnd) >= Math.max(start, other.startFrame);
    if (!overlaps) return false;
    return (
      other.playerIndex === conversion.lastHitBy &&
      other.lastHitBy === conversion.playerIndex
    );
  });
}

function isTradePairedWithOneHitKill(conversion, conversions) {
  if ((conversion.openingType || "unknown") !== "trade" || conversion.didKill) {
    return false;
  }
  const start = conversion.startFrame;
  return conversions.some((other) => {
    if (other === conversion) return false;
    if ((other.openingType || "unknown") !== "trade" || !isOneHitKill(other)) return false;
    if (Math.abs((other.startFrame ?? 0) - start) > 3) return false;
    return (
      other.playerIndex === conversion.lastHitBy &&
      other.lastHitBy === conversion.playerIndex
    );
  });
}

function summarizeConversion(conversion, labels) {
  const damage = conversionDamage(conversion);
  const attacker = conversion.lastHitBy;
  const victim = conversion.playerIndex;
  const moves = (conversion.moves || []).length;
  const opening = conversion.openingType || "unknown";
  const attackerLabel = labels[attacker] || `P${attacker + 1}`;
  const victimLabel = labels[victim] || `P${victim + 1}`;
  const kill = conversion.didKill ? ", stock" : "";
  return `${attackerLabel} advantage: ${opening}, +${round(damage, 1)}%, ${moves} hit${moves === 1 ? "" : "s"}${kill} vs ${victimLabel}`;
}

function conversionMoveDetails(conversion, labels) {
  return (conversion.moves || []).map((move, index) => {
    const info = moveInfo(move.moveId);
    const playerIndex = move.playerIndex;
    return {
      index: index + 1,
      frame: move.frame,
      playerIndex,
      playerLabel: labels[playerIndex] || `P${playerIndex + 1}`,
      moveId: move.moveId,
      moveName: info.name,
      moveShortName: info.shortName,
      hitCount: move.hitCount,
      damage: round(move.damage, 2),
    };
  });
}

function advantageOpportunityDetails(conversion, conversionIndex, conversions, frames, labels, args) {
  const moveSequence = conversionMoveDetails(conversion, labels);
  const openingMove = moveSequence[0] || null;
  const startFrame = conversion.startFrame;
  const endFrame = conversion.endFrame ?? startFrame;
  const oneHitKill = isOneHitKill(conversion);
  const mutualTrade = isMutualTradeConversion(conversion, conversions);
  const pairedOneHitKillTrade = isTradePairedWithOneHitKill(conversion, conversions);
  return {
    conversionIndex,
    owner: conversion.lastHitBy,
    defender: conversion.playerIndex,
    startFrame,
    endFrame,
    openingFrame: openingMove?.frame ?? startFrame,
    openingType: conversion.openingType || "unknown",
    damage: round(conversionDamage(conversion), 2),
    startPercent: round(conversion.startPercent, 2),
    endPercent: round(conversion.endPercent ?? conversion.currentPercent, 2),
    moves: moveSequence.length,
    openingMove,
    moveSequence,
    didKill: Boolean(conversion.didKill),
    oneHitKill,
    mutualTrade,
    pairedOneHitKillTrade,
    meaningfulForPhaseTimeline: isMeaningfulConversion(conversion, args),
    frameFacts: conversionFrameFacts(conversion, frames, labels),
    diSamples: hitDiSamples(conversion, frames, labels),
    phillipInsertionCommitment: phillipInsertionCommitment(frames, conversion, labels),
  };
}

function summarizeOneHitKill(conversion, labels) {
  const damage = conversionDamage(conversion);
  const attacker = conversion.lastHitBy;
  const victim = conversion.playerIndex;
  const attackerLabel = labels[attacker] || `P${attacker + 1}`;
  const victimLabel = labels[victim] || `P${victim + 1}`;
  const opening = conversion.openingType || "unknown";
  return `${attackerLabel} one-hit kill: ${opening}, +${round(damage, 1)}% vs ${victimLabel}`;
}

function pushSegment(segments, segment) {
  if (segment.endFrame < segment.startFrame) return;
  const previous = segments[segments.length - 1];
  if (
    previous &&
    previous.phase === segment.phase &&
    previous.owner === segment.owner &&
    segment.startFrame - previous.endFrame <= 1
  ) {
    previous.endFrame = Math.max(previous.endFrame, segment.endFrame);
    previous.endTime = seconds(previous.endFrame);
    previous.notes = [...(previous.notes || []), ...(segment.notes || [])].slice(-12);
    return;
  }
  segments.push({
    ...segment,
    startTime: seconds(segment.startFrame),
    endTime: seconds(segment.endFrame),
  });
}

function absorbShortNeutralSegments(segments, maxNeutralFrames) {
  if (maxNeutralFrames <= 0) return segments;
  let changed = true;
  while (changed) {
    changed = false;
    for (let i = 1; i < segments.length - 1; i += 1) {
      const segment = segments[i];
      const previous = segments[i - 1];
      const next = segments[i + 1];
      const length = segment.endFrame - segment.startFrame + 1;
      if (segment.phase !== "neutral" || length > maxNeutralFrames) continue;
      if (!previous || !next) continue;
      if (previous.phase === "stock_transition" || next.phase === "stock_transition") continue;
      if (previous.phase === "pregame" || next.phase === "pregame") continue;
      if (previous.phase !== "advantage" && next.phase !== "advantage") continue;

      const mergeIntoPrevious =
        previous.phase === "advantage" &&
        (next.phase !== "advantage" || previous.owner === next.owner);
      const target = mergeIntoPrevious ? previous : next;
      const other = mergeIntoPrevious ? next : previous;
      const note = `Absorbed short neutral gap f${segment.startFrame}-${segment.endFrame} (${length}f) into ${target.phase}`;
      target.notes = [...(target.notes || []), note, ...(segment.notes || [])].slice(-12);
      if (mergeIntoPrevious) {
        target.endFrame = segment.endFrame;
        target.endTime = seconds(target.endFrame);
        if (target.endFrame + 1 >= other.startFrame && target.phase === other.phase && target.owner === other.owner) {
          target.endFrame = other.endFrame;
          target.endTime = seconds(target.endFrame);
          target.notes = [
            ...(target.notes || []),
            `Continues into f${other.startFrame}: ${other.label}`,
            ...(other.notes || []),
          ].slice(-12);
          segments.splice(i, 2);
        } else {
          segments.splice(i, 1);
        }
      } else {
        target.startFrame = segment.startFrame;
        target.startTime = seconds(target.startFrame);
        segments.splice(i, 1);
      }
      changed = true;
      break;
    }
  }
  return segments;
}

function main() {
  const args = parseArgs(process.argv);
  const slippi = require(slippiJsNodeModule());
  const { SlippiGame } = slippi;
  melee = slippi;
  const game = new SlippiGame(args.replay);
  const settings = game.getSettings();
  const stats = game.getStats();
  const frames = game.getFrames();
  if (!settings || !stats || !frames) {
    throw new Error("Could not load Slippi settings/stats/frames.");
  }

  const frameNumbers = Object.keys(frames).map(Number).filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  const firstFrame = frameNumbers[0];
  const lastFrame = frameNumbers[frameNumbers.length - 1];
  const playerIndices = (settings.players || []).map((player) => player.playerIndex);
  const labels = {};
  for (const playerIndex of playerIndices) labels[playerIndex] = playerLabel(settings, playerIndex);

  const conversions = (stats.conversions || [])
    .filter((conversion) => conversion.startFrame != null)
    .slice()
    .sort((a, b) => a.startFrame - b.startFrame);
  const advantageOpportunities = conversions.map((conversion, conversionIndex) => (
    advantageOpportunityDetails(
      conversion,
      conversionIndex,
      conversions,
      frames,
      labels,
      args,
    )
  ));

  const segments = [];
  const minorInteractions = [];
  const pendingMutualTrades = [];
  let cursor = firstFrame;

  if (firstFrame < 0) {
    pushSegment(segments, {
      phase: "pregame",
      owner: null,
      label: "Pregame / countdown",
      startFrame: firstFrame,
      endFrame: Math.min(-1, lastFrame),
      notes: [],
    });
    cursor = 0;
  }

  for (const conversion of conversions) {
    const originalStart = Math.max(firstFrame, conversion.startFrame);
    const start = Math.max(originalStart, cursor);
    const end = Math.min(lastFrame, conversion.endFrame ?? conversion.startFrame);
    if (end < cursor) continue;
    const oneHitKill = isOneHitKill(conversion);
    const pairedOneHitKillTrade = isTradePairedWithOneHitKill(conversion, conversions);
    const mutualTrade = isMutualTradeConversion(conversion, conversions);
    if (pairedOneHitKillTrade || mutualTrade || !isMeaningfulConversion(conversion, args)) {
      const prefix = pairedOneHitKillTrade
        ? "Trade context for one-hit kill"
        : mutualTrade
        ? "Mutual trade/scramble kept in neutral"
        : summarizeConversion(conversion, labels);
      const minor = {
        frame: originalStart,
        endFrame: end,
        time: seconds(originalStart),
        summary: pairedOneHitKillTrade || mutualTrade
          ? `${prefix}: ${summarizeConversion(conversion, labels)}`
          : prefix,
        damage: round(conversionDamage(conversion), 2),
        openingType: conversion.openingType || "unknown",
      };
      minorInteractions.push(minor);
      if (mutualTrade || pairedOneHitKillTrade) {
        pendingMutualTrades.push(minor);
      }
      continue;
    }

    const leadInTrades = pendingMutualTrades.filter((trade) => (
      trade.frame >= cursor &&
      trade.frame < start &&
      start - trade.endFrame <= args.tradeLeadInFrames
    ));
    const segmentStart = leadInTrades.length
      ? Math.min(...leadInTrades.map((trade) => trade.frame))
      : start;

    if (segmentStart > cursor) {
      const notes = minorInteractions
        .filter((item) => item.frame >= cursor && item.frame < segmentStart)
        .map((item) => `Minor kept in neutral at f${item.frame}: ${item.summary}`);
      pushSegment(segments, {
        phase: "neutral",
        owner: null,
        label: notes.length ? "Neutral (minor hits kept)" : "Neutral",
        startFrame: cursor,
        endFrame: segmentStart - 1,
        notes,
      });
    }

    const attacker = conversion.lastHitBy;
    const victim = conversion.playerIndex;
    const baseLabel = oneHitKill ? summarizeOneHitKill(conversion, labels) : summarizeConversion(conversion, labels);
    const advantageLabel = leadInTrades.length
      ? `Trade lead-in -> ${baseLabel}`
      : baseLabel;
    pushSegment(segments, {
      phase: oneHitKill ? "one_hit_kill" : "advantage",
      owner: attacker,
      defender: victim,
      label: advantageLabel,
      startFrame: segmentStart,
      endFrame: end,
      conversion: {
        openingType: conversion.openingType || "unknown",
        damage: round(conversionDamage(conversion), 2),
        startPercent: round(conversion.startPercent, 1),
        endPercent: round(conversion.endPercent ?? conversion.currentPercent, 1),
        moves: (conversion.moves || []).length,
        openingMove: conversionMoveDetails(conversion, labels)[0] || null,
        moveSequence: conversionMoveDetails(conversion, labels),
        didKill: Boolean(conversion.didKill),
        oneHitKill,
        frameFacts: conversionFrameFacts(conversion, frames, labels),
        diSamples: hitDiSamples(conversion, frames, labels),
        phillipInsertionCommitment: phillipInsertionCommitment(frames, conversion, labels),
      },
      notes: leadInTrades.map((trade) => `Trade lead-in at f${trade.frame}: ${trade.summary}`),
    });

    cursor = end + 1;
    for (const trade of leadInTrades) {
      trade.absorbedBy = {
        frame: start,
        summary: summarizeConversion(conversion, labels),
      };
    }
    if (conversion.didKill && cursor <= lastFrame) {
      const respawnEnd = resolveRespawnEnd(frames, cursor, lastFrame, playerIndices);
      pushSegment(segments, {
        phase: "stock_transition",
        owner: null,
        label: `Stock transition / respawn after ${labels[attacker] || `P${attacker + 1}`} kill`,
        startFrame: cursor,
        endFrame: respawnEnd,
        notes: [],
      });
      cursor = respawnEnd + 1;
    }
  }

  if (cursor <= lastFrame) {
    const notes = minorInteractions
      .filter((item) => item.frame >= cursor)
      .map((item) => `Minor kept in neutral at f${item.frame}: ${item.summary}`);
    pushSegment(segments, {
      phase: "neutral",
      owner: null,
      label: notes.length ? "Neutral (minor hits kept)" : "Neutral",
      startFrame: cursor,
      endFrame: lastFrame,
      notes,
    });
  }

  absorbShortNeutralSegments(segments, args.shortNeutralFrames);
  for (const segment of segments) {
    segment.spatialTimeline = spatialTimelineForSegment(segment, frames, playerIndices, labels);
    segment.actionTimeline = actionTimelineForSegment(segment, frames, playerIndices, labels);
  }

  const timeline = {
    replay: path.resolve(args.replay),
    generatedAt: new Date().toISOString(),
    firstFrame,
    lastFrame,
    durationSeconds: seconds(lastFrame - firstFrame + 1),
    settings: {
      stageId: settings.stageId,
      stageName: melee.stages.getStageName(settings.stageId),
      players: (settings.players || []).map((player) => ({
        playerIndex: player.playerIndex,
        port: player.port,
        characterId: player.characterId,
        characterName: melee.characters.getCharacterInfo(player.characterId)?.name || `Character ${player.characterId}`,
        displayName: player.displayName || "",
        connectCode: player.connectCode || "",
      })),
    },
    thresholds: {
      meaningfulDamage: args.meaningfulDamage,
      meaningfulDuration: args.meaningfulDuration,
      shortNeutralFrames: args.shortNeutralFrames,
      tradeLeadInFrames: args.tradeLeadInFrames,
    },
    advantageOpportunities,
    minorInteractions,
    segments,
  };

  fs.mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
  fs.writeFileSync(args.out, JSON.stringify(timeline, null, 2) + "\n", "utf8");
  console.log(JSON.stringify({
    out: path.resolve(args.out),
    segments: segments.length,
    advantageOpportunities: advantageOpportunities.length,
    minorInteractions: minorInteractions.length,
    firstFrame,
    lastFrame,
  }, null, 2));
}

main();
