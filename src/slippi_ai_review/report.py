"""Build a game review around interactive traces and optional video fallbacks."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def percent(value: Any) -> str:
    return f"{100.0 * float(value or 0.0):.0f}%"


def elapsed(frame: int) -> str:
    seconds = max(0.0, frame / 60.0)
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}:{remainder:05.2f}"


def relative_media(path: str | None, out: Path) -> str:
    if not path:
        return ""
    return Path(os.path.relpath(Path(path), out.parent)).as_posix()


def reason_text(reasons: dict[str, Any]) -> str:
    if not reasons:
        return "No qualifying option"
    labels = {
        "missing_simulation_rows": "Not simulated",
        "missing_candidate_metadata": "Missing candidate metadata",
        "too_few_option_samples": "Option sample count too low",
        "option_share_below_threshold": "Option was too rare",
        "improvement_rate_below_threshold": "Improvement rate below threshold",
    }
    return ", ".join(
        f"{labels.get(key, key.replace('_', ' ').capitalize())} ({int(value)})"
        for key, value in reasons.items()
    )


def category_for(target: dict[str, Any], result: dict[str, Any]) -> str:
    option = target.get("option") or {}
    baseline = target.get("replay_baseline") or {}
    signature = str(option.get("optionSignature") or "").upper()
    label = str(target.get("label") or "").lower()
    lane = target.get("representative_lane") or {}
    if (
        float(option.get("killRate") or 0.0) >= 0.5
        or bool(baseline.get("original_followup_kill"))
        or int(lane.get("defenderStocksLost") or 0) > 0
    ):
        return "stock"
    if "EDGE" in signature or "edgeguard" in label:
        return "edgeguard"
    resolution = result.get("resolution") or {}
    if "stock" in str(resolution.get("reason") or ""):
        return "stock"
    return "extension"


def interactive_url(route: dict[str, Any]) -> str:
    params = {
        "replay": route.get("replay_trace"),
        "agent": route.get("agent_trace"),
        "switch": int(route.get("switch_frame") or 0),
        "defenderSwitch": int(route.get("defender_switch_frame") or 0),
        "start": int(route.get("start_frame") or 0),
        "frames": int(route.get("frame_count") or 1),
    }
    if route.get("timeline_events"):
        params["events"] = route["timeline_events"]
    return "viewer/compare.html?" + urlencode(params)


def ordinal(value: int) -> str:
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def insertion_label(baseline: dict[str, Any], move: str) -> str:
    role = str(baseline.get("opportunity_role") or "")
    index = baseline.get("branch_move_index")
    move_label = move.replace("conversion-start", "throw choice")
    if role == "conversion_commitment":
        return "During grab (throw choice)"
    if role == "failed_extension_commitment":
        return f"At missed follow-up ({move_label})"
    if index is not None:
        prefix = "Before" if role == "continuation_commitment" else "After"
        if move_label in {"dash", "run"}:
            if role == "opening_hit":
                return f"Before opening hit ({move_label} approach)"
            return f"{prefix} {ordinal(int(index))} hit ({move_label} approach)"
        return f"{prefix} {ordinal(int(index))} hit ({move_label})"
    return f"At {move_label}"


def insertion_text(item: dict[str, Any], *, display_index: int) -> dict[str, str]:
    baseline = item["baseline"]
    opening = item["opening"]
    option = item["option"]
    lane = item["lane"]
    result = item["result"]
    frame = item["frame"]
    move = opening.get("moveShortName") or opening.get("moveName") or "advantage"
    signature = option.get("optionSignature") or lane.get("comboOptionSignature") or "model continuation"
    replay_hits = int(baseline.get("original_followup_hits") or 0)
    replay_damage = float(baseline.get("original_followup_damage") or 0.0)
    lane_hits = int(lane.get("followupHits") or 0)
    lane_damage = float(lane.get("followupDamage") or 0.0)
    resolution = result.get("resolution") or lane.get("resolution") or {}
    terminal = str(resolution.get("reason") or "unknown").replace("_", " ")
    terminal_frame = resolution.get("frame")
    terminal_text = terminal + (f" · f{int(terminal_frame)}" if terminal_frame is not None else "")
    rank = option.get("selectionRank")
    return {
        "move": str(move),
        "signature": str(signature),
        "insertion_label": insertion_label(baseline, str(move)),
        "heading": f"{move} → {signature}",
        "eyebrow": f"{display_index:02d} · {item['category']} · {elapsed(frame)} elapsed · f{frame}",
        "rank": f"Quality rank {int(rank)}" if rank is not None else "Qualified improvement",
        "replay_main": f"{replay_hits} follow-up hit{'s' if replay_hits != 1 else ''} · {replay_damage:.1f}%",
        "replay_sub": "Stock taken" if bool(baseline.get("original_followup_kill")) else "No stock conversion",
        "agent_main": f"{lane_hits} follow-up hit{'s' if lane_hits != 1 else ''} · {lane_damage:.1f}%",
        "agent_sub": "Stock taken" if int(lane.get("defenderStocksLost") or 0) > 0 else "No stock conversion",
        "reliability_main": f"{percent(option.get('improvementRate'))} improved · {int(option.get('samples') or 0)} samples",
        "reliability_sub": f"{percent(option.get('optionShare'))} policy share · {percent(option.get('reversalRate'))} reversal",
        "boundary_main": terminal_text,
        "boundary_sub": f"{int(resolution.get('step') or 0)} simulated frames",
    }


def build_page(payload: dict[str, Any], *, out: Path) -> str:
    queue_path = Path(payload["queue_json"])
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    targets = list(queue.get("targets") or [])
    result_by_index = {int(item["target_index"]): item for item in payload.get("results") or []}
    audit = payload.get("selection_audit") or queue.get("selection_audit") or {}
    generation = audit.get("candidate_generation") or {}
    replay = Path(queue.get("replay") or "game.slp")
    display_name = str(queue.get("display_name") or replay.stem)
    analyzed_port = int(queue.get("controlled_port") or 1)

    moments: list[dict[str, Any]] = []
    for index, target in enumerate(targets, start=1):
        result = result_by_index.get(index, {})
        baseline = target.get("replay_baseline") or {}
        opening = baseline.get("opening_move") or {}
        option = target.get("option") or {}
        lane = target.get("representative_lane") or {}
        moments.append({
            "index": index,
            "target": target,
            "result": result,
            "baseline": baseline,
            "opening": opening,
            "option": option,
            "lane": lane,
            "frame": int(target.get("base_frame") or target.get("takeover_frame") or 0),
            "category": category_for(target, result),
        })
    moments.sort(key=lambda item: (item["frame"], item["index"]))
    scenario_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in moments:
        presentation = item["baseline"].get("presentation_segment") or {}
        start = presentation.get("startFrame")
        end = presentation.get("endFrame")
        sequence_id = item["baseline"].get("sequence_id")
        key = (
            "presentation",
            int(start),
            int(end),
        ) if start is not None and end is not None else ("sequence", sequence_id or item["frame"])
        scenario_groups.setdefault(key, []).append(item)
    scenarios = sorted(
        scenario_groups.values(),
        key=lambda group: min(item["frame"] for item in group),
    )
    has_video_fallback = any(
        item["result"].get("consolidated_clip") or item["result"].get("side_by_side")
        for item in moments
    )

    nav_items = []
    sections = []
    for display_index, scenario_items in enumerate(scenarios, start=1):
        scenario_items.sort(key=lambda entry: entry["frame"])
        item = min(
            scenario_items,
            key=lambda entry: (
                int(entry["option"].get("selectionRank") or 10**9),
                entry["frame"],
            ),
        )
        target = item["target"]
        result = item["result"]
        baseline = item["baseline"]
        opening = item["opening"]
        option = item["option"]
        lane = item["lane"]
        frame = item["frame"]
        scenario_frame = min(entry["frame"] for entry in scenario_items)
        category = item["category"]
        anchor = f"moment-{display_index}"
        move = opening.get("moveShortName") or opening.get("moveName") or "advantage"
        signature = option.get("optionSignature") or lane.get("comboOptionSignature") or "model continuation"
        nav_items.append(
            f'<a class="moment-link" href="#{anchor}" data-category="{esc(category)}" '
            f'data-frame="{scenario_frame}" data-rank="{int(option.get("selectionRank") or display_index)}">'
            f'<span class="moment-number">{display_index:02d}</span>'
            f'<span><strong>{esc(move)} → {esc(signature)}</strong>'
            f'<small>{len(scenario_items)} insertion point{"s" if len(scenario_items) != 1 else ""} · '
            f'{elapsed(scenario_frame)}</small></span></a>'
        )

        replay_hits = int(baseline.get("original_followup_hits") or 0)
        replay_damage = float(baseline.get("original_followup_damage") or 0.0)
        replay_kill = bool(baseline.get("original_followup_kill"))
        lane_hits = int(lane.get("followupHits") or 0)
        lane_damage = float(lane.get("followupDamage") or 0.0)
        lane_kill = int(lane.get("defenderStocksLost") or 0) > 0
        resolution = result.get("resolution") or lane.get("resolution") or {}
        clip = result.get("consolidated_clip") or result.get("side_by_side")
        clip_src = relative_media(clip, out)
        poster_src = relative_media(result.get("poster"), out)
        poster_attr = f' poster="{esc(poster_src)}"' if poster_src else ""
        clip_html = (
            f'<video controls preload="metadata" playsinline src="{esc(clip_src)}"{poster_attr}></video>'
            if clip_src
            else '<div class="missing-video">Open this report with open_review.cmd to use the interactive viewer.</div>'
        )
        interactive = result.get("interactive") or {}
        interactive_src = interactive_url(interactive) if interactive else ""
        interactive_html = (
            f'<div class="interactive-shell" data-start-frame="{int(interactive.get("start_frame") or 0)}">'
            f'<iframe loading="lazy" allowfullscreen title="Interactive comparison for frame {frame}" '
            f'data-src="{esc(interactive_src)}"></iframe>'
            f'<div class="dormant-poster" aria-hidden="true">'
            f'<div><span>Replay</span></div><div><span>Counterfactual</span></div>'
            f'<strong>Comparison paused at f{int(interactive.get("start_frame") or 0)}</strong></div></div>'
            if interactive_src
            else ""
        )
        primary_insertion_id = f"scenario-{display_index}-insertion-{item['index']}"
        insertion_buttons = []
        route_groups = []
        primary_alternative_count = 0
        for insertion in scenario_items:
            insertion_id = f"scenario-{display_index}-insertion-{insertion['index']}"
            insertion_view = insertion_text(insertion, display_index=display_index)
            insertion_interactive = insertion["result"].get("interactive") or {}
            alternatives = list(insertion_interactive.get("alternative_routes") or [])
            active_insertion = insertion_id == primary_insertion_id
            if active_insertion:
                primary_alternative_count = len(alternatives)
            insertion_buttons.append(
                f'<button type="button" class="insertion-option{" active" if active_insertion else ""}" '
                f'data-insertion-id="{insertion_id}" data-target-index="{insertion["index"]}" '
                f'data-opening="{esc(insertion_view["move"])}" '
                f'data-signature="{esc(insertion_view["signature"])}" data-heading="{esc(insertion_view["heading"])}" '
                f'data-eyebrow="{esc(insertion_view["eyebrow"])}" data-rank="{esc(insertion_view["rank"])}" '
                f'data-replay-main="{esc(insertion_view["replay_main"])}" data-replay-sub="{esc(insertion_view["replay_sub"])}" '
                f'data-agent-main="{esc(insertion_view["agent_main"])}" data-agent-sub="{esc(insertion_view["agent_sub"])}" '
                f'data-reliability-main="{esc(insertion_view["reliability_main"])}" data-reliability-sub="{esc(insertion_view["reliability_sub"])}" '
                f'data-boundary-main="{esc(insertion_view["boundary_main"])}" data-boundary-sub="{esc(insertion_view["boundary_sub"])}" '
                f'aria-pressed="{"true" if active_insertion else "false"}">'
                f'<strong>{esc(insertion_view["insertion_label"])}</strong>'
                f'<span>f{insertion["frame"]} · {esc(insertion_view["signature"])} · {esc(insertion_view["rank"])}</span></button>'
            )
            available_routes = [{
                **insertion_interactive,
                "route_index": 0,
                "option": insertion["option"],
                "signature": insertion_view["signature"],
            }, *alternatives]
            route_items = []
            for route_position, route in enumerate(available_routes):
                route_option = route.get("option") or {}
                route_signature = route.get("signature") or route_option.get("optionSignature") or "Route"
                route_items.append(
                    f'<button type="button" class="route-option{" active" if route_position == 0 else ""}" '
                    f'data-target-index="{insertion["index"]}" data-alternative-index="{route_position}" '
                    f'data-route-src="{esc(interactive_url(route))}" data-start-frame="{int(route.get("start_frame") or 0)}" '
                    f'data-signature="{esc(route_signature)}" aria-pressed="{"true" if route_position == 0 else "false"}">'
                    f'<strong>{esc(route_signature)}</strong><span>{percent(route_option.get("improvementRate"))} improved · '
                    f'{int(route_option.get("samples") or 0)} samples</span></button>'
                )
            route_groups.append(
                f'<div class="route-options" data-insertion-id="{insertion_id}" data-alternative-count="{len(alternatives)}"'
                f'{"" if active_insertion else " hidden"}>{"".join(route_items)}</div>'
            )
        insertion_html = (
            f'<details class="insertion-picker"><summary>Insertion points <span>{len(scenario_items)} tested</span></summary>'
            f'<div class="insertion-options">{"".join(insertion_buttons)}</div></details>'
            if len(scenario_items) > 1 else ""
        )
        route_html = (
            f'<details class="route-picker"{" hidden" if primary_alternative_count == 0 else ""}>'
            f'<summary>More routes <span>{primary_alternative_count} alternatives</span></summary>'
            f'<div class="route-groups">{"".join(route_groups)}</div></details>'
        )
        rank = option.get("selectionRank")
        rank_text = f"Quality rank {int(rank)}" if rank is not None else "Qualified improvement"
        terminal = str(resolution.get("reason") or "unknown").replace("_", " ")
        terminal_frame = resolution.get("frame")
        terminal_text = terminal + (f" · f{int(terminal_frame)}" if terminal_frame is not None else "")
        sections.append(f"""
        <article id="{anchor}" class="moment" data-category="{esc(category)}" data-frame="{scenario_frame}" data-rank="{int(option.get('selectionRank') or display_index)}" data-current-opening="{esc(move)}" data-current-target-index="{item['index']}" data-current-alternative-index="0">
          <header class="moment-header">
            <div>
              <p class="eyebrow">{display_index:02d} · {esc(category)} · {elapsed(frame)} elapsed · f{frame}</p>
              <h2 data-opening="{esc(move)}">{esc(move)} → {esc(signature)}</h2>
            </div>
            <span class="rank">{esc(rank_text)}</span>
          </header>
          {interactive_html}
          {insertion_html}
          {route_html}
          <div class="practice-actions">
            <button type="button" class="practice-ce" data-scenario-mode="replay" title="Older SLP files may not replay correctly in Training Mode CE.">
              <span>Replay in CE</span>
              <small>Older SLP files may not replay correctly</small>
            </button>
            <button type="button" class="practice-ce" data-scenario-mode="phillip" title="Older SLP files may not replay correctly in Training Mode CE.">
              <span>Phillip rollout in CE</span>
              <small>Older SLP files may not replay correctly</small>
            </button>
            <span class="practice-status" role="status" aria-live="polite"></span>
          </div>
          <div class="video-shell">{clip_html}</div>
          <div class="comparison" aria-label="Replay and model outcome comparison">
            <section>
              <h3>Replay continuation</h3>
              <strong data-summary="replay-main">{replay_hits} follow-up hit{'s' if replay_hits != 1 else ''} · {replay_damage:.1f}%</strong>
              <span data-summary="replay-sub">{'Stock taken' if replay_kill else 'No stock conversion'}</span>
            </section>
            <section>
              <h3>Phillip continuation</h3>
              <strong data-summary="agent-main">{lane_hits} follow-up hit{'s' if lane_hits != 1 else ''} · {lane_damage:.1f}%</strong>
              <span data-summary="agent-sub">{'Stock taken' if lane_kill else 'No stock conversion'}</span>
            </section>
            <section>
              <h3>Option reliability</h3>
              <strong data-summary="reliability-main">{percent(option.get('improvementRate'))} improved · {int(option.get('samples') or 0)} samples</strong>
              <span data-summary="reliability-sub">{percent(option.get('optionShare'))} policy share · {percent(option.get('reversalRate'))} reversal</span>
            </section>
            <section>
              <h3>Simulation boundary</h3>
              <strong data-summary="boundary-main">{esc(terminal_text)}</strong>
              <span data-summary="boundary-sub">{int((resolution.get('step') or 0))} simulated frames</span>
            </section>
          </div>
        </article>
        """)

    missing_count = len(audit.get("missing_simulation_frames") or [])
    failed = int(payload.get("failed_count") or 0)
    complete = failed == 0 and missing_count == 0
    status_class = "good" if complete else "bad"
    opportunity_count = int(
        generation.get("total_advantage_sequences")
        or generation.get("analyzed_player_conversion_count")
        or generation.get("timeline_segment_count")
        or audit.get("candidate_count")
        or 0
    )
    omitted = list(audit.get("not_selected") or [])
    omitted_rows = "".join(
        f"<tr><td>f{int(item.get('frame') or 0)}</td>"
        f"<td>{elapsed(int(item.get('frame') or 0))}</td>"
        f"<td>{esc(item.get('stage') or 'selection')}</td>"
        f"<td>{esc(reason_text(item.get('reasons') or {}))}</td></tr>"
        for item in omitted
    )
    disposition_html = f"""
      <details class="disposition" {'open' if missing_count else ''}>
        <summary>Candidate disposition <span>{len(omitted)} not selected</span></summary>
        <div class="table-scroll"><table>
          <thead><tr><th>Frame</th><th>Elapsed</th><th>Stage</th><th>Reason</th></tr></thead>
          <tbody>{omitted_rows}</tbody>
        </table></div>
      </details>
    """ if omitted else ""
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{esc(display_name)} advantage review</title>
  <style>
    :root {{ color-scheme:dark; --ink:#e8ece9; --muted:#9ba59f; --line:#343b37; --paper:#111311; --white:#191c19; --raised:#20241f; --green:#65d69a; --red:#ff7b75; --amber:#f0b35a; --blue:#71b7df; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font:15px/1.45 Inter,Segoe UI,Arial,sans-serif; letter-spacing:0; }}
    a {{ color:inherit; }}
    .topbar {{ position:sticky; top:0; z-index:10; display:flex; align-items:center; gap:18px; min-height:58px; padding:9px 22px; color:var(--ink); background:#0b0d0c; border-bottom:3px solid #4cbd82; }}
    .topbar h1 {{ margin:0; font-size:17px; font-weight:700; letter-spacing:0; }}
    .topbar p {{ margin:0; color:var(--muted); font-size:12px; }}
    .dashboard-link {{ margin-left:auto; min-height:34px; padding:7px 10px; border:1px solid #566059; border-radius:4px; color:var(--ink); background:var(--raised); font-size:12px; font-weight:700; text-decoration:none; }}
    .dashboard-link:hover,.dashboard-link:focus {{ border-color:var(--green); outline:none; }}
    .upload-trigger {{ min-height:34px; padding:6px 10px; border:1px solid #566059; border-radius:4px; color:var(--ink); background:var(--raised); font-weight:700; cursor:pointer; }}
    .upload-trigger:hover,.upload-trigger:focus {{ border-color:var(--green); outline:none; }}
    .topbar .status {{ margin-left:auto; padding:4px 8px; border:1px solid currentColor; border-radius:4px; font-size:12px; font-weight:700; }}
    .upload-trigger + .status {{ margin-left:0; }}
    .status.good {{ color:#78deb2; }} .status.bad {{ color:#ff9c9c; }}
    .upload-dialog {{ width:min(470px,calc(100vw - 24px)); padding:0; border:1px solid #566059; border-radius:6px; color:var(--ink); background:var(--white); box-shadow:0 18px 60px rgba(0,0,0,.5); }}
    .upload-dialog::backdrop {{ background:rgba(0,0,0,.72); }}
    .upload-dialog header {{ display:flex; align-items:center; justify-content:space-between; padding:13px 15px; border-bottom:1px solid var(--line); }}
    .upload-dialog h2 {{ font-size:16px; }}
    .dialog-close {{ width:32px; height:32px; padding:0; border:1px solid #566059; border-radius:4px; color:var(--ink); background:var(--raised); font-size:22px; line-height:1; cursor:pointer; }}
    #slp-upload-mount {{ padding:15px; }}
    .slp-upload-panel {{ display:grid; gap:10px; }}
    .slp-upload-panel__label {{ color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; }}
    .slp-upload-panel__input {{ width:100%; padding:9px; border:1px solid #566059; border-radius:4px; color:var(--ink); background:var(--raised); }}
    .slp-upload-panel__submit {{ min-height:38px; border:1px solid #477b60; border-radius:4px; color:#07130d; background:var(--green); font-weight:800; cursor:pointer; }}
    .slp-upload-panel__submit:disabled {{ opacity:.6; cursor:wait; }}
    .slp-upload-panel__target {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .slp-upload-panel__target button {{ min-height:36px; padding:7px 10px; border:1px solid #477b60; border-radius:4px; color:var(--ink); background:var(--raised); font-weight:700; cursor:pointer; }}
    .slp-upload-panel__status {{ min-height:21px; color:var(--muted); font-size:12px; }}
    .slp-upload-panel__status[data-state="validated"] {{ color:var(--amber); }}
    .slp-upload-panel__status[data-state="accepted"] {{ color:var(--green); }}
    .slp-upload-panel__status[data-state="error"] {{ color:var(--red); }}
    .slp-upload-panel__review-link {{ display:inline-block; margin-top:5px; color:var(--ink); font-weight:700; }}
    .summary {{ display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); border-bottom:1px solid var(--line); background:var(--white); }}
    .metric {{ min-height:82px; padding:15px 20px; border-right:1px solid var(--line); }}
    .metric:last-child {{ border-right:0; }}
    .metric span {{ display:block; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; }}
    .metric strong {{ display:block; margin-top:4px; font-size:24px; line-height:1.1; }}
    .metric.warn strong {{ color:var(--amber); }} .metric.fail strong {{ color:var(--red); }}
    .workspace {{ display:grid; grid-template-columns:260px minmax(0,1fr); max-width:1500px; margin:0 auto; }}
    .sidebar {{ position:sticky; top:58px; align-self:start; height:calc(100vh - 58px); overflow:auto; padding:20px 14px; border-right:1px solid var(--line); background:#151815; }}
    .sidebar h2 {{ margin:0 8px 10px; font-size:12px; text-transform:uppercase; color:var(--muted); }}
    .filters {{ display:grid; gap:6px; margin:0 8px 16px; }}
    select {{ width:100%; min-height:36px; padding:6px 28px 6px 8px; border:1px solid #566059; border-radius:4px; background:var(--raised); color:var(--ink); }}
    .moment-link {{ display:grid; grid-template-columns:34px 1fr; gap:8px; align-items:center; padding:9px 8px; border-top:1px solid var(--line); text-decoration:none; }}
    .moment-link:hover,.moment-link:focus {{ background:var(--raised); outline:none; }}
    .moment-number {{ color:var(--green); font:700 12px/1 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .moment-link strong,.moment-link small {{ display:block; overflow-wrap:anywhere; }}
    .moment-link strong {{ font-size:12px; }} .moment-link small {{ margin-top:2px; color:var(--muted); font-size:11px; }}
    main {{ min-width:0; padding:24px clamp(16px,3vw,44px) 80px; }}
    .combined {{ margin-bottom:30px; padding-bottom:28px; border-bottom:1px solid var(--line); }}
    .combined h2 {{ margin:0 0 12px; font-size:18px; }}
    .combined-video {{ display:block; width:min(100%,1100px); aspect-ratio:1280/526; background:#000; }}
    .disposition {{ margin:0 0 30px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); background:var(--white); }}
    .disposition summary {{ cursor:pointer; padding:12px 14px; font-weight:700; }}
    .disposition summary span {{ margin-left:8px; color:var(--muted); font-size:12px; font-weight:400; }}
    .table-scroll {{ overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; }} th,td {{ padding:9px 14px; border-top:1px solid var(--line); text-align:left; font-size:12px; }} th {{ color:var(--muted); font-size:10px; text-transform:uppercase; }}
    .moment {{ margin:0 0 42px; padding:0 0 38px; border-bottom:1px solid var(--line); scroll-margin-top:78px; }}
    .moment-header {{ display:flex; justify-content:space-between; gap:20px; align-items:end; margin-bottom:13px; }}
    .eyebrow {{ margin:0 0 3px; color:var(--blue); font-size:11px; font-weight:700; text-transform:uppercase; }}
    h2 {{ margin:0; font-size:21px; }}
    .rank {{ flex:0 0 auto; padding:4px 7px; color:var(--green); border:1px solid #477b60; border-radius:4px; font-size:11px; font-weight:700; }}
    .video-shell {{ width:100%; background:#000; }}
    .video-shell video {{ display:block; width:100%; aspect-ratio:1280/526; }}
    .interactive-shell {{ position:relative; display:none; width:100%; aspect-ratio:1460/650; overflow:hidden; background:#101417; }}
    .interactive-shell iframe {{ display:block; width:100%; height:100%; border:0; }}
    .dormant-poster {{ position:absolute; inset:0; display:none; grid-template-columns:1fr 1fr; gap:1px; color:#dce5e8; background:#596166; pointer-events:none; }}
    .dormant-poster div {{ display:grid; place-items:center; background:#20272b; }}
    .dormant-poster span {{ padding:5px 8px; border:1px solid #536067; border-radius:4px; color:#afbbc0; font-size:11px; font-weight:700; text-transform:uppercase; }}
    .dormant-poster strong {{ position:absolute; left:50%; bottom:16px; translate:-50% 0; padding:6px 9px; border-radius:4px; color:#fff; background:rgba(16,20,23,.88); font:700 12px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace; }}
    .interactive-shell.dormant .dormant-poster {{ display:grid; }}
    .interactive-shell.dormant iframe {{ visibility:hidden; }}
    .insertion-picker,.route-picker {{ border:1px solid var(--line); border-top:0; background:var(--white); }}
    .insertion-picker summary,.route-picker summary {{ cursor:pointer; padding:9px 12px; color:var(--ink); font-size:12px; font-weight:700; }}
    .insertion-picker summary span,.route-picker summary span {{ margin-left:6px; color:var(--muted); font-weight:400; }}
    .insertion-options,.route-options {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:6px; padding:0 10px 10px; }}
    .insertion-option,.route-option {{ min-width:0; padding:8px 10px; border:1px solid #4b554f; border-radius:4px; color:var(--ink); background:var(--raised); text-align:left; cursor:pointer; }}
    .insertion-option:hover,.insertion-option:focus,.route-option:hover,.route-option:focus {{ border-color:var(--green); outline:none; }}
    .insertion-option.active,.route-option.active {{ border-color:var(--green); box-shadow:inset 3px 0 0 var(--green); }}
    .insertion-option strong,.insertion-option span,.route-option strong,.route-option span {{ display:block; overflow-wrap:anywhere; }}
    .insertion-option strong,.route-option strong {{ font-size:12px; }}
    .insertion-option span,.route-option span {{ margin-top:2px; color:var(--muted); font-size:10px; }}
    .practice-actions {{ display:flex; align-items:center; gap:10px; min-height:48px; padding:8px 10px; border:1px solid var(--line); border-top:0; background:var(--white); }}
    .practice-ce {{ display:flex; flex-direction:column; align-items:flex-start; justify-content:center; min-height:44px; max-width:100%; padding:6px 10px; border:1px solid #477b60; border-radius:4px; color:#07130d; background:var(--green); font-weight:800; cursor:pointer; text-align:left; }}
    .practice-ce span,.practice-ce small {{ display:block; max-width:100%; overflow-wrap:anywhere; }}
    .practice-ce small {{ margin-top:1px; font-size:10px; font-weight:650; line-height:1.2; }}
    .practice-ce:hover,.practice-ce:focus {{ border-color:#a8f1c9; outline:none; }}
    .practice-ce:disabled {{ opacity:.6; cursor:wait; }}
    .practice-status {{ min-width:0; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }}
    .practice-status[data-state="ready"] {{ color:var(--green); }}
    .practice-status[data-state="error"] {{ color:var(--red); }}
    body.interactive-mode .moment:has(.interactive-shell) .interactive-shell {{ display:block; }}
    body.interactive-mode .moment:has(.interactive-shell) .video-shell {{ display:none; }}
    .missing-video {{ display:grid; place-items:center; aspect-ratio:1280/526; color:#fff; background:#2c3337; }}
    .comparison {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-top:0; background:var(--white); }}
    .comparison section {{ min-width:0; padding:13px 14px; border-right:1px solid var(--line); }}
    .comparison section:last-child {{ border-right:0; }}
    .comparison h3 {{ margin:0 0 6px; color:var(--muted); font-size:10px; text-transform:uppercase; }}
    .comparison strong,.comparison span {{ display:block; overflow-wrap:anywhere; }}
    .comparison strong {{ font-size:13px; }} .comparison span {{ margin-top:3px; color:var(--muted); font-size:11px; }}
    [hidden] {{ display:none !important; }}
    @media (max-width:900px) {{
      .summary {{ grid-template-columns:repeat(3,1fr); }} .metric {{ border-bottom:1px solid var(--line); }}
      .workspace {{ grid-template-columns:1fr; }} .sidebar {{ position:static; height:auto; border-right:0; border-bottom:1px solid var(--line); }}
      .moment-links {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .comparison {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .comparison section:nth-child(2) {{ border-right:0; }} .comparison section:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }}
    }}
    @media (max-width:560px) {{
      .topbar {{ flex-wrap:wrap; align-items:center; gap:8px; padding:9px 12px; }} .topbar > div {{ flex:1 1 180px; min-width:0; }} .topbar h1 {{ overflow-wrap:anywhere; }} .topbar p {{ display:none; }}
      .upload-trigger {{ margin-left:0; }}
      .summary {{ grid-template-columns:repeat(2,1fr); }} .metric {{ min-height:70px; padding:12px; }} .metric strong {{ font-size:20px; }}
      .moment-links {{ grid-template-columns:1fr; }} main {{ padding:18px 10px 60px; }}
      .interactive-shell {{ aspect-ratio:4/3; }}
      .moment-header {{ align-items:start; }} .comparison {{ grid-template-columns:1fr; }} .comparison section {{ border-right:0; border-bottom:1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div><h1>{esc(display_name)}</h1><p>P{analyzed_port} advantage review · chronological counterfactuals</p></div>
    <a class="dashboard-link" href="/">Dashboard</a>
    <button class="upload-trigger" id="upload-trigger" type="button">Upload replay</button>
    <span class="status {status_class}">{'COMPLETE' if complete else f'{failed} RENDER FAILED' if failed else 'INCOMPLETE'}</span>
  </header>
  <dialog class="upload-dialog" id="upload-dialog">
    <header><h2>Upload Slippi replay</h2><button class="dialog-close" type="button" aria-label="Close upload dialog">&times;</button></header>
    <div id="slp-upload-mount"></div>
  </dialog>
  <section class="summary" aria-label="Game review summary">
    <div class="metric"><span>Advantage opportunities</span><strong>{opportunity_count}</strong></div>
    <div class="metric"><span>Injection points</span><strong>{int(audit.get('candidate_count') or 0)}</strong></div>
    <div class="metric {'warn' if missing_count else ''}"><span>Preflighted</span><strong>{int(audit.get('simulated_candidate_count') or 0)}</strong></div>
    <div class="metric"><span>Refined</span><strong>{int(audit.get('refined_candidate_count') or audit.get('simulated_candidate_count') or 0)}</strong></div>
    <div class="metric"><span>Review situations</span><strong>{len(scenarios)}</strong></div>
    <div class="metric {'fail' if failed else ''}"><span>Artifact failures</span><strong>{failed}</strong></div>
  </section>
  <div class="workspace">
    <aside class="sidebar">
      <h2>Moments</h2>
      <div class="filters"><select id="category-filter" aria-label="Filter review moments">
        <option value="all">All improvements</option><option value="extension">Extensions</option><option value="edgeguard">Edgeguards</option><option value="stock">Stock conversions</option>
      </select><select id="order-mode" aria-label="Order review moments"><option value="timeline" selected>Timeline order</option><option value="quality">Quality order</option></select><select id="review-mode" aria-label="Review display mode" data-video-fallback="{str(has_video_fallback).lower()}" hidden><option value="interactive">Interactive comparison</option>{'<option value="video">Rendered video</option>' if has_video_fallback else ''}</select></div>
      <nav class="moment-links">{''.join(nav_items)}</nav>
    </aside>
    <main>
      {disposition_html}
      {''.join(sections)}
    </main>
  </div>
  <script src="/advantage-review-static/slp_upload_panel.js"></script>
  <script>
    const uploadDialog = document.querySelector('#upload-dialog');
    document.querySelector('#upload-trigger').addEventListener('click', () => uploadDialog.showModal());
    uploadDialog.querySelector('.dialog-close').addEventListener('click', () => uploadDialog.close());
    uploadDialog.addEventListener('click', event => {{
      if (event.target === uploadDialog) uploadDialog.close();
    }});
    if (window.SlpUploadPanel) SlpUploadPanel.mount(document.querySelector('#slp-upload-mount'));
    const filter = document.querySelector('#category-filter');
    const orderMode = document.querySelector('#order-mode');
    const reviewMode = document.querySelector('#review-mode');
    const moments = [...document.querySelectorAll('.moment')];
    const links = [...document.querySelectorAll('.moment-link')];
    const momentParent = moments[0]?.parentElement;
    const linkParent = links[0]?.parentElement;
    filter.addEventListener('change', () => {{
      const value = filter.value;
      moments.forEach(el => el.hidden = value !== 'all' && el.dataset.category !== value);
      links.forEach(el => el.hidden = value !== 'all' && el.dataset.category !== value);
    }});
    const applyOrder = () => {{
      const quality = orderMode.value === 'quality';
      const compare = (left, right) => quality
        ? Number(left.dataset.rank) - Number(right.dataset.rank) || Number(left.dataset.frame) - Number(right.dataset.frame)
        : Number(left.dataset.frame) - Number(right.dataset.frame) || Number(left.dataset.rank) - Number(right.dataset.rank);
      [...moments].sort(compare).forEach(el => momentParent?.append(el));
      [...links].sort(compare).forEach(el => linkParent?.append(el));
    }};
    orderMode.addEventListener('change', applyOrder);
    applyOrder();
    const interactiveFrames = [...document.querySelectorAll('.interactive-shell iframe[data-src]')];
    if (/^https?:$/.test(location.protocol) && interactiveFrames.length) {{
      reviewMode.hidden = reviewMode.dataset.videoFallback !== 'true';
      const loadedFrames = new Set();
      const visibleFrames = new Set();
      const maxLoadedFrames = 4;
      const savedFrames = new Map();
      const shellFor = frame => frame.closest('.interactive-shell');
      const setPosterFrame = (frame, relativeFrame) => {{
        const shell = shellFor(frame);
        const label = shell?.querySelector('.dormant-poster strong');
        const absoluteFrame = Number(shell?.dataset.startFrame || 0) + relativeFrame;
        if (label) label.textContent = `Comparison paused at f${{absoluteFrame}}`;
      }};
      addEventListener('message', event => {{
        if (event.data?.type !== 'comparison-frame') return;
        const frame = interactiveFrames.find(candidate => candidate.contentWindow === event.source);
        if (!frame || frame.dataset.loading === 'true') return;
        const relativeFrame = Number(event.data.frame) || 0;
        savedFrames.set(frame, relativeFrame);
        setPosterFrame(frame, relativeFrame);
      }});
      const captureFrameState = frame => {{
        let relativeFrame = savedFrames.get(frame) || 0;
        try {{
          const seek = frame.contentDocument?.querySelector('#seek');
          if (seek) relativeFrame = Number(seek.value) || 0;
        }} catch (_error) {{
          // The bundled viewer is same-origin, but retain the last known frame if it is not readable.
        }}
        savedFrames.set(frame, relativeFrame);
        setPosterFrame(frame, relativeFrame);
        return relativeFrame;
      }};
      const unloadFrame = frame => {{
        captureFrameState(frame);
        frame.dataset.loading = 'false';
        shellFor(frame)?.classList.add('dormant');
        frame.src = 'about:blank';
        frame.removeAttribute('src');
        loadedFrames.delete(frame);
      }};
      const trimLoadedFrames = keep => {{
        for (const frame of [...loadedFrames]) {{
          if (loadedFrames.size <= maxLoadedFrames) break;
          if (frame !== keep && !visibleFrames.has(frame)) unloadFrame(frame);
        }}
      }};
      const restoreFrameState = frame => {{
        if (frame.dataset.loading !== 'true') return;
        let attempts = 0;
        const restore = () => {{
          if (frame.dataset.loading !== 'true') return;
          let seek = null;
          try {{ seek = frame.contentDocument?.querySelector('#seek'); }} catch (_error) {{}}
          if (!seek) {{
            if (attempts++ < 120) setTimeout(restore, 50);
            return;
          }}
          const relativeFrame = savedFrames.get(frame) || 0;
          seek.value = String(Math.max(0, Math.min(Number(seek.max) || 0, relativeFrame)));
          seek.dispatchEvent(new Event('input', {{ bubbles:true }}));
          frame.dataset.loading = 'false';
          shellFor(frame)?.classList.remove('dormant');
        }};
        restore();
      }};
      const loadFrame = frame => {{
        if (!frame.hasAttribute('src')) {{
          frame.dataset.loading = 'true';
          shellFor(frame)?.classList.add('dormant');
          frame.src = frame.dataset.src;
        }}
        loadedFrames.delete(frame);
        loadedFrames.add(frame);
        trimLoadedFrames(frame);
      }};
      const selectRoute = (button, force = false) => {{
        const moment = button.closest('.moment');
        const shell = moment?.querySelector('.interactive-shell');
        const frame = shell?.querySelector('iframe[data-src]');
        if (!shell || !frame || (!force && button.classList.contains('active'))) return;
        const shouldLoad = frame.hasAttribute('src') || visibleFrames.has(frame);
        if (frame.hasAttribute('src')) unloadFrame(frame);
        frame.dataset.src = button.dataset.routeSrc;
        moment.dataset.currentTargetIndex = button.dataset.targetIndex;
        moment.dataset.currentAlternativeIndex = button.dataset.alternativeIndex || '0';
        shell.dataset.startFrame = button.dataset.startFrame || '0';
        savedFrames.set(frame, 0);
        setPosterFrame(frame, 0);
        moment.querySelectorAll('.route-option').forEach(option => {{
          const active = option === button;
          option.classList.toggle('active', active);
          option.setAttribute('aria-pressed', String(active));
        }});
        const heading = moment.querySelector('.moment-header h2[data-opening]');
        if (heading) heading.textContent = `${{moment.dataset.currentOpening}} → ${{button.dataset.signature}}`;
        moment.querySelector('.route-picker')?.removeAttribute('open');
        if (shouldLoad) loadFrame(frame);
      }};
      document.addEventListener('click', event => {{
        const routeButton = event.target.closest('.route-option');
        if (routeButton) {{
          selectRoute(routeButton);
          return;
        }}
        const insertionButton = event.target.closest('.insertion-option');
        if (!insertionButton || insertionButton.classList.contains('active')) return;
        const moment = insertionButton.closest('.moment');
        moment.querySelectorAll('.insertion-option').forEach(option => {{
          const active = option === insertionButton;
          option.classList.toggle('active', active);
          option.setAttribute('aria-pressed', String(active));
        }});
        moment.dataset.currentOpening = insertionButton.dataset.opening;
        moment.dataset.currentTargetIndex = insertionButton.dataset.targetIndex;
        moment.dataset.currentAlternativeIndex = '0';
        moment.querySelector('.eyebrow').textContent = insertionButton.dataset.eyebrow;
        moment.querySelector('.moment-header h2').textContent = insertionButton.dataset.heading;
        moment.querySelector('.rank').textContent = insertionButton.dataset.rank;
        const summaryKeys = ['replayMain','replaySub','agentMain','agentSub','reliabilityMain','reliabilitySub','boundaryMain','boundarySub'];
        summaryKeys.forEach(key => {{
          const attribute = key.replace(/[A-Z]/g, letter => `-${{letter.toLowerCase()}}`);
          const target = moment.querySelector(`[data-summary="${{attribute}}"]`);
          if (target) target.textContent = insertionButton.dataset[key];
        }});
        const insertionId = insertionButton.dataset.insertionId;
        const routeGroups = [...moment.querySelectorAll('.route-options[data-insertion-id]')];
        routeGroups.forEach(group => group.hidden = group.dataset.insertionId !== insertionId);
        const activeGroup = routeGroups.find(group => group.dataset.insertionId === insertionId);
        const alternativeCount = Number(activeGroup?.dataset.alternativeCount || 0);
        const routePicker = moment.querySelector('.route-picker');
        routePicker.hidden = alternativeCount === 0;
        routePicker.querySelector('summary span').textContent = `${{alternativeCount}} alternatives`;
        const primaryRoute = activeGroup?.querySelector('.route-option');
        if (primaryRoute) selectRoute(primaryRoute, true);
        insertionButton.closest('.insertion-picker')?.removeAttribute('open');
      }});
      interactiveFrames.forEach(frame => frame.addEventListener('load', () => {{
        if (frame.getAttribute('src') && frame.getAttribute('src') !== 'about:blank') restoreFrameState(frame);
      }}));
      const observer = new IntersectionObserver(entries => {{
        entries.forEach(entry => {{
          if (entry.isIntersecting) {{
            visibleFrames.add(entry.target);
            loadFrame(entry.target);
          }} else {{
            visibleFrames.delete(entry.target);
          }}
        }});
        trimLoadedFrames(null);
      }}, {{ rootMargin: '500px 0px' }});
      interactiveFrames.forEach(frame => observer.observe(frame));
      const activateInteractive = () => {{
        const active = reviewMode.value === 'interactive';
        document.body.classList.toggle('interactive-mode', active);
        if (!active) {{
          interactiveFrames.forEach(unloadFrame);
          return;
        }}
        requestAnimationFrame(() => {{
          const margin = 500;
          interactiveFrames.forEach(frame => {{
            const bounds = frame.getBoundingClientRect();
            if (bounds.bottom >= -margin && bounds.top <= innerHeight + margin) {{
              visibleFrames.add(frame);
              loadFrame(frame);
            }}
          }});
          trimLoadedFrames(null);
        }});
      }};
      reviewMode.addEventListener('change', activateInteractive);
      activateInteractive();
    }}
    const reviewIdMatch = location.pathname.match(new RegExp('^/review-artifacts/([0-9a-f-]+)/'));
    document.addEventListener('click', async event => {{
      const button = event.target.closest('.practice-ce');
      if (!button) return;
      const moment = button.closest('.moment');
      const status = moment.querySelector('.practice-status');
      if (!reviewIdMatch) {{
        status.dataset.state = 'error';
        status.textContent = 'Open this report from the dashboard first.';
        return;
      }}
      button.disabled = true;
      status.dataset.state = 'working';
      status.textContent = 'Preparing scenario...';
      try {{
        const response = await fetch(`/api/reviews/${{reviewIdMatch[1]}}/training-mode`, {{
          method:'POST',
          headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{
            targetIndex:Number(moment.dataset.currentTargetIndex),
            alternativeIndex:Number(moment.dataset.currentAlternativeIndex || 0),
            scenarioMode:button.dataset.scenarioMode,
          }}),
        }});
        const body = await response.json();
        if (!response.ok || !body.ok) throw new Error(body.error?.message || 'Scenario export failed.');
        status.dataset.state = 'ready';
        status.textContent = body.scenario.scenarioMode === 'replay'
          ? `Replay ready in CE from safe f${{body.scenario.practiceStartFrame}}; opening f${{body.scenario.openingHitFrame}}.`
          : `Phillip rollout ready in CE from safe f${{body.scenario.practiceStartFrame}}; opening f${{body.scenario.openingHitFrame}}; takeover f${{body.scenario.takeoverFrame}} (P${{body.scenario.humanPort}}).`;
      }} catch (error) {{
        status.dataset.state = 'error';
        status.textContent = error.message || 'Scenario export failed.';
      }} finally {{
        button.disabled = false;
      }}
    }});
    document.querySelectorAll('video').forEach(video => video.addEventListener('play', () => {{
      document.querySelectorAll('video').forEach(other => {{ if (other !== video) other.pause(); }});
    }}));
  </script>
</body>
</html>
"""
    return document


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    out = args.out.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(payload, out=out), encoding="utf-8")
    print(json.dumps({"out": str(out), "bytes": out.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
