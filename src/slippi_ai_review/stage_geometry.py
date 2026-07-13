"""Coarse Melee stage geometry for coaching prompts.

These values are meant for LLM spacing language: center, side, corner, ledge,
platform, and blast-zone context. They are not collision-grade stage data.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


COMMON_NOTE = (
    "Approximate static coaching geometry. Use for spacing language and "
    "position labels, not exact collision, ECB, camera, or physics claims."
)


STAGES_BY_ID: dict[int, dict[str, Any]] = {
    2: {
        "stageId": 2,
        "stageName": "Fountain of Dreams",
        "precision": "coarse_coaching_context",
        "sourceNote": COMMON_NOTE + " Side platforms move; listed y values are rough ranges.",
        "coordinateSystem": "Melee world coordinates; x increases to the right, y increases upward.",
        "mainPlatform": {"leftX": -63.35, "rightX": 63.35, "y": 0.0},
        "ledges": {"left": {"x": -63.35, "y": 0.0}, "right": {"x": 63.35, "y": 0.0}},
        "blastZones": {"leftX": -198.75, "rightX": 198.75, "topY": 202.5, "bottomY": -146.25},
        "platforms": [
            {"name": "left side platform", "leftX": -49.0, "rightX": -21.0, "approxY": "variable 13-35"},
            {"name": "right side platform", "leftX": 21.0, "rightX": 49.0, "approxY": "variable 13-35"},
            {"name": "top platform", "leftX": -18.0, "rightX": 18.0, "approxY": 42.0},
        ],
    },
    3: {
        "stageId": 3,
        "stageName": "Pokemon Stadium",
        "precision": "coarse_coaching_context",
        "sourceNote": COMMON_NOTE + " Transformations are ignored.",
        "coordinateSystem": "Melee world coordinates; x increases to the right, y increases upward.",
        "mainPlatform": {"leftX": -87.75, "rightX": 87.75, "y": 0.0},
        "ledges": {"left": {"x": -87.75, "y": 0.0}, "right": {"x": 87.75, "y": 0.0}},
        "blastZones": {"leftX": -230.0, "rightX": 230.0, "topY": 180.0, "bottomY": -111.0},
        "platforms": [
            {"name": "left platform", "leftX": -59.0, "rightX": -33.0, "approxY": 25.0},
            {"name": "right platform", "leftX": 33.0, "rightX": 59.0, "approxY": 25.0},
        ],
    },
    8: {
        "stageId": 8,
        "stageName": "Yoshi's Story",
        "precision": "coarse_coaching_context",
        "sourceNote": COMMON_NOTE,
        "coordinateSystem": "Melee world coordinates; x increases to the right, y increases upward.",
        "mainPlatform": {"leftX": -56.0, "rightX": 56.0, "y": 0.0},
        "ledges": {"left": {"x": -56.0, "y": 0.0}, "right": {"x": 56.0, "y": 0.0}},
        "blastZones": {"leftX": -175.7, "rightX": 175.7, "topY": 168.0, "bottomY": -91.0},
        "platforms": [
            {"name": "left platform", "leftX": -43.0, "rightX": -22.0, "approxY": 23.5},
            {"name": "right platform", "leftX": 22.0, "rightX": 43.0, "approxY": 23.5},
            {"name": "top platform", "leftX": -16.0, "rightX": 16.0, "approxY": 42.0},
        ],
    },
    28: {
        "stageId": 28,
        "stageName": "Dream Land N64",
        "precision": "coarse_coaching_context",
        "sourceNote": COMMON_NOTE,
        "coordinateSystem": "Melee world coordinates; x increases to the right, y increases upward.",
        "mainPlatform": {"leftX": -77.3, "rightX": 77.3, "y": 0.0},
        "ledges": {"left": {"x": -77.3, "y": 0.0}, "right": {"x": 77.3, "y": 0.0}},
        "blastZones": {"leftX": -255.0, "rightX": 255.0, "topY": 250.0, "bottomY": -123.0},
        "platforms": [
            {"name": "left platform", "leftX": -59.0, "rightX": -31.0, "approxY": 30.0},
            {"name": "right platform", "leftX": 31.0, "rightX": 59.0, "approxY": 30.0},
            {"name": "top platform", "leftX": -19.0, "rightX": 19.0, "approxY": 51.0},
        ],
    },
    31: {
        "stageId": 31,
        "stageName": "Battlefield",
        "precision": "coarse_coaching_context",
        "sourceNote": COMMON_NOTE,
        "coordinateSystem": "Melee world coordinates; x increases to the right, y increases upward.",
        "mainPlatform": {"leftX": -68.4, "rightX": 68.4, "y": 0.0},
        "ledges": {"left": {"x": -68.4, "y": 0.0}, "right": {"x": 68.4, "y": 0.0}},
        "blastZones": {"leftX": -224.0, "rightX": 224.0, "topY": 200.0, "bottomY": -108.0},
        "platforms": [
            {"name": "left platform", "leftX": -61.0, "rightX": -35.0, "approxY": 27.2},
            {"name": "right platform", "leftX": 35.0, "rightX": 61.0, "approxY": 27.2},
            {"name": "top platform", "leftX": -18.8, "rightX": 18.8, "approxY": 54.4},
        ],
    },
    32: {
        "stageId": 32,
        "stageName": "Final Destination",
        "precision": "coarse_coaching_context",
        "sourceNote": COMMON_NOTE,
        "coordinateSystem": "Melee world coordinates; x increases to the right, y increases upward.",
        "mainPlatform": {"leftX": -85.6, "rightX": 85.6, "y": 0.0},
        "ledges": {"left": {"x": -85.6, "y": 0.0}, "right": {"x": 85.6, "y": 0.0}},
        "blastZones": {"leftX": -246.0, "rightX": 246.0, "topY": 188.0, "bottomY": -140.0},
        "platforms": [],
    },
}


STAGE_NAME_TO_ID = {
    "fountainofdreams": 2,
    "fountain": 2,
    "pokemonstadium": 3,
    "pokemon": 3,
    "stadium": 3,
    "yoshisstory": 8,
    "yoshis": 8,
    "dreamlandn64": 28,
    "dreamland": 28,
    "battlefield": 31,
    "finaldestination": 32,
    "fd": 32,
}


def _normalize_name(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def stage_geometry_for_settings(settings: dict[str, Any] | None) -> dict[str, Any] | None:
    if not settings:
        return None
    stage_id = settings.get("stageId")
    try:
        stage_id = int(stage_id)
    except (TypeError, ValueError):
        stage_id = None
    if stage_id in STAGES_BY_ID:
        return deepcopy(STAGES_BY_ID[stage_id])
    stage_name = _normalize_name(settings.get("stageName"))
    mapped_id = STAGE_NAME_TO_ID.get(stage_name)
    if mapped_id in STAGES_BY_ID:
        return deepcopy(STAGES_BY_ID[mapped_id])
    return None


def _ledge_x(stage_geometry: dict[str, Any] | None, side: str) -> float | None:
    try:
        return float((stage_geometry or {}).get("ledges", {}).get(side, {}).get("x"))
    except (TypeError, ValueError):
        return None


def _platform_label(x: float, y: float, stage_geometry: dict[str, Any]) -> str | None:
    if y < 8:
        return None
    for platform in stage_geometry.get("platforms") or []:
        left = platform.get("leftX")
        right = platform.get("rightX")
        try:
            left_f = float(left)
            right_f = float(right)
        except (TypeError, ValueError):
            continue
        if left_f - 3 <= x <= right_f + 3:
            return str(platform.get("name") or "platform")
    return None


def classify_position(
    x: Any,
    y: Any = None,
    *,
    stage_geometry: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(x, (int, float)):
        return {"region": "unknown"}
    x_f = float(x)
    y_f = float(y) if isinstance(y, (int, float)) else 0.0
    left_ledge = _ledge_x(stage_geometry, "left")
    right_ledge = _ledge_x(stage_geometry, "right")
    if left_ledge is None or right_ledge is None:
        abs_x = abs(x_f)
        if abs_x >= 80:
            return {"region": "corner_or_ledge", "side": "left" if x_f < 0 else "right"}
        if abs_x >= 55:
            return {"region": "side", "side": "left" if x_f < 0 else "right"}
        return {"region": "center", "side": None}

    ledge_abs = max(abs(left_ledge), abs(right_ledge))
    distance_left = x_f - left_ledge
    distance_right = right_ledge - x_f
    nearest_side = "left" if abs(distance_left) < abs(distance_right) else "right"
    distance_to_nearest_ledge = min(abs(distance_left), abs(distance_right))

    if x_f < left_ledge:
        region = "offstage_left"
    elif x_f > right_ledge:
        region = "offstage_right"
    else:
        abs_x = abs(x_f)
        if abs_x >= ledge_abs - 8:
            region = "corner_or_ledge"
        elif abs_x >= ledge_abs * 0.65:
            region = "side"
        elif abs_x <= ledge_abs * 0.35:
            region = "center"
        else:
            region = "midstage"

    out = {
        "region": region,
        "side": "left" if x_f < 0 else ("right" if x_f > 0 else "center"),
        "nearestLedge": nearest_side,
        "distanceToNearestLedge": round(distance_to_nearest_ledge, 2),
    }
    platform = _platform_label(x_f, y_f, stage_geometry or {})
    if platform:
        out["platform"] = platform
    return out


def stage_geometry_text(stage_geometry: dict[str, Any] | None) -> str:
    if not stage_geometry:
        return "No stage geometry table entry is available for this stage."
    main = stage_geometry.get("mainPlatform") or {}
    ledges = stage_geometry.get("ledges") or {}
    blast = stage_geometry.get("blastZones") or {}
    platforms = stage_geometry.get("platforms") or []
    platform_text = "; ".join(
        f"{p.get('name')}: x {p.get('leftX')} to {p.get('rightX')}, y {p.get('approxY')}"
        for p in platforms
    ) or "none"
    return "\n".join(
        [
            f"- Stage geometry precision: {stage_geometry.get('precision')}",
            f"- Note: {stage_geometry.get('sourceNote')}",
            f"- Main platform: x {main.get('leftX')} to {main.get('rightX')}, y {main.get('y')}",
            f"- Ledges: left x {((ledges.get('left') or {}).get('x'))}, right x {((ledges.get('right') or {}).get('x'))}",
            f"- Blast zones: left {blast.get('leftX')}, right {blast.get('rightX')}, top {blast.get('topY')}, bottom {blast.get('bottomY')}",
            f"- Platforms: {platform_text}",
        ]
    )
