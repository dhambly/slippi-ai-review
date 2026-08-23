"""Build a dedicated interactive report for neutral-loss avoidance routes."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .disadvantage_report import write_placeholder


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def pct(value: Any) -> str:
    return f"{100.0 * float(value or 0.0):.0f}%"


def elapsed(frame: int) -> str:
    seconds = max(0.0, frame / 60.0)
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def option_label(value: Any) -> str:
    return str(value or "Option").replace("_", " ").title()


def local_path(value: str | Path) -> Path:
    """Read manifests produced from either Windows or the WSL simulation process."""
    path = Path(value)
    if path.is_file() or os.name != "nt":
        return path
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", str(value))
    if match:
        return Path(f"{match.group(1).upper()}:/{match.group(2)}")
    return path


def interactive_url(route: dict[str, Any]) -> str:
    params = {
        "replay": route.get("replay_trace"),
        "agent": route.get("agent_trace"),
        "switch": int(route.get("switch_frame") or 0),
        "takeover": int(route.get("model_control_frame") or route.get("switch_frame") or 0),
        "defenderSwitch": int(route.get("defender_switch_frame") or 0),
        "start": int(route.get("start_frame") or 0),
        "frames": int(route.get("frame_count") or 1),
    }
    if route.get("timeline_events"):
        params["events"] = route["timeline_events"]
    return "viewer/compare.html?" + urlencode(params)


def route_button(route: dict[str, Any], *, index: int, active: bool) -> str:
    option = route.get("option") or {}
    signature = option_label(option.get("optionSignature"))
    lead = int(route.get("lookback_frames") or 0)
    route_kind = str(route.get("route_kind") or "avoid")
    goal = "Win neutral" if route_kind == "win" else "Avoid hit"
    emergency = bool(option.get("emergency"))
    character = "emergency escape" if emergency else "neutral option"
    return (
        f'<button type="button" class="route-option{" active" if active else ""}" '
        f'data-route-index="{index}" data-route-src="{esc(interactive_url(route))}" '
        f'data-option="{esc(signature)}" data-avoid="{esc(pct(option.get("avoidRate")))}" '
        f'data-share="{esc(pct(option.get("successfulPolicyShare")))}" data-lead="{lead}" '
        f'data-injection="{esc(route.get("injection_frame") or "")}" '
        f'data-clean="{esc(pct(option.get("cleanEscapeRate")))}" data-stable="{esc(pct(option.get("stableNeutralRate")))}" '
        f'data-win="{esc(pct(option.get("neutralWinRate")))}" data-goal="{esc(goal)}" '
        f'data-character="{esc(character)}" '
        f'aria-pressed="{"true" if active else "false"}">'
        f'<strong>{lead}f &middot; {esc(goal)}: {esc(signature)}</strong><span>{pct(option.get("avoidRate"))} avoids &middot; {pct(option.get("neutralWinRate"))} wins &middot; '
        f'{character}</span></button>'
    )


def build_page(payload: dict[str, Any]) -> str:
    queue_path = local_path(payload["queue_json"])
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    targets = list(queue.get("targets") or [])
    results = {int(item["target_index"]): item for item in payload.get("results") or []}
    replay_name = str(queue.get("display_name") or Path(queue.get("replay") or "game.slp").name)
    controlled_port = int(queue.get("controlled_port") or 1)
    sections = []
    nav = []
    for display_index, target in enumerate(targets, start=1):
        neutral = target.get("neutral_loss") or {}
        option = target.get("option") or {}
        lane = target.get("representative_lane") or {}
        result = results.get(display_index, {})
        primary = dict(result.get("interactive") or {})
        primary.setdefault(
            "lookback_frames",
            neutral.get("selectedRouteLookbackFrames") or neutral.get("lookbackFrames"),
        )
        primary.setdefault("injection_frame", neutral.get("injectionFrame") or target.get("base_frame"))
        primary["option"] = option
        routes = [primary, *(primary.get("alternative_routes") or [])]
        primary["alternative_routes"] = []
        route_buttons = "".join(
            route_button(route, index=index, active=index == 0)
            for index, route in enumerate(routes)
        )
        opening_frame = int(neutral.get("openingFrame") or 0)
        injection_frame = int(neutral.get("injectionFrame") or target.get("base_frame") or 0)
        move = str(primary.get("opening_display_name") or option_label(neutral.get("openingMove")))
        chosen = option_label(option.get("optionSignature"))
        selected_lead = int(primary.get("lookback_frames") or neutral.get("selectedRouteLookbackFrames") or neutral.get("lookbackFrames") or 0)
        contact = lane.get("neutralContact") or {}
        dealt_frame = contact.get("firstDamageDealtFrame")
        counter_hit = dealt_frame is not None and int(dealt_frame) <= int(neutral.get("deadlineFrame") or opening_frame)
        nav.append(
            f'<a href="#neutral-{display_index}"><span>{display_index:02d}</span>'
            f'<strong>{esc(move)}</strong><small>{selected_lead}-{int(max(neutral.get("availableLookbackFrames") or [selected_lead]))}f lead</small></a>'
        )
        sections.append(f"""
        <article class="moment" id="neutral-{display_index}">
          <header class="moment-head">
            <div>
              <p class="eyebrow">{display_index:02d} &middot; neutral loss &middot; {elapsed(opening_frame)} elapsed &middot; f{opening_frame}</p>
              <h2>Options before {esc(move)}</h2>
              <p>Compare Phillip's best observed route at each 10f insertion. Late options can be emergency escapes rather than recommendations.</p>
            </div>
            <div class="boundary"><span>Nearest reliable boundary</span><strong>{int(neutral.get('lookbackFrames') or 0)}f</strong></div>
          </header>
          <div class="viewer-shell">
            <iframe title="Replay versus Phillip neutral avoidance at frame {opening_frame}" loading="lazy" allowfullscreen data-src="{esc(interactive_url(primary))}"></iframe>
            <div class="viewer-placeholder"><span>Replay</span><span>Phillip avoidance</span><strong>Comparison ready</strong></div>
          </div>
          <div class="routes" role="group" aria-label="Phillip options by insertion lead time">{route_buttons}</div>
          <div class="facts">
            <section><span>Recorded outcome</span><strong>Hit by {esc(move)}</strong><small>Opening at f{opening_frame}</small></section>
            <section><span data-option-goal>Avoid hit</span><strong data-option-name>{esc(chosen)}</strong><small data-option-character>{'emergency escape' if option.get('emergency') else 'neutral option'}</small></section>
            <section><span>Lead time</span><strong data-option-lead>{selected_lead}f</strong><small data-option-injection>Control starts at f{injection_frame}</small></section>
            <section><span>Avoidance / follow-through</span><strong data-option-avoid>{pct(option.get('avoidRate'))}</strong><small data-option-quality>{pct(option.get('cleanEscapeRate'))} clean &middot; {pct(option.get('stableNeutralRate'))} stable neutral</small></section>
            <section><span>Counter-hit</span><strong>{'Yes' if counter_hit else 'No'}</strong><small>{'Phillip lands first by the deadline' if counter_hit else 'Avoidance without a required punish'}</small></section>
          </div>
        </article>
        """)

    empty = "" if sections else '<section class="empty"><h2>No reliable neutral-loss boundaries</h2><p>No tested lookback cleared the configured confidence threshold.</p></section>'
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Neutral Loss Review &middot; {esc(replay_name)}</title>
  <style>
    :root {{ color-scheme:dark; --bg:#0c0f13; --panel:#151a20; --line:#303943; --text:#f4f6f8; --muted:#9ba7b2; --cyan:#64c8e8; --green:#62d39a; --amber:#edbd61; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; background:var(--bg); }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }}
    a {{ color:inherit; text-decoration:none; }}
    button {{ font:inherit; }}
    .topbar {{ position:sticky; top:0; z-index:10; display:flex; align-items:center; justify-content:space-between; gap:18px; min-height:58px; padding:8px 24px; border-bottom:1px solid var(--line); background:rgba(12,15,19,.96); backdrop-filter:blur(12px); }}
    .brand strong {{ display:block; }} .brand small {{ color:var(--muted); }}
    .tabs {{ display:flex; gap:4px; }} .tabs a {{ padding:8px 12px; color:var(--muted); border-bottom:2px solid transparent; }} .tabs a.active {{ color:var(--text); border-color:var(--cyan); }}
    .phase-tabs {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:1px; width:min(1420px,100%); margin:0 auto; border:1px solid var(--line); border-top:0; background:var(--line); }}
    .phase-tabs a {{ display:flex; align-items:center; justify-content:center; min-height:48px; padding:9px 12px; color:var(--muted); background:var(--panel); font-size:13px; font-weight:800; }} .phase-tabs a.active {{ color:var(--text); background:#17313a; box-shadow:inset 0 -3px var(--cyan); }} .phase-tabs a:hover,.phase-tabs a:focus {{ color:var(--text); background:#1e252d; outline:none; }}
    .layout {{ display:grid; grid-template-columns:230px minmax(0,1fr); width:min(1420px,100%); margin:0 auto; }}
    aside {{ position:sticky; top:58px; align-self:start; height:calc(100vh - 58px); padding:24px 16px; border-right:1px solid var(--line); overflow:auto; }}
    aside h1 {{ margin:0 8px 6px; font-size:20px; }} aside>p {{ margin:0 8px 18px; color:var(--muted); font-size:13px; }}
    .moment-nav {{ display:grid; gap:2px; }} .moment-nav a {{ display:grid; grid-template-columns:28px 1fr; column-gap:8px; padding:9px 8px; }} .moment-nav a:hover {{ background:var(--panel); }}
    .moment-nav span {{ grid-row:1/3; color:var(--cyan); font:12px ui-monospace,monospace; }} .moment-nav strong {{ font-size:13px; }} .moment-nav small {{ color:var(--muted); }}
    main {{ min-width:0; padding:0 36px 64px; }}
    .moment {{ max-width:1120px; margin:0 auto; padding:38px 0 48px; border-bottom:1px solid var(--line); }}
    main .moment {{ display:none; }} main .moment.slide-active {{ display:block; }}
    .slide-controls {{ display:flex; align-items:center; justify-content:space-between; gap:12px; max-width:1120px; margin:0 auto; padding:16px 0; border-bottom:1px solid var(--line); }}
    .slide-controls button {{ min-height:36px; padding:7px 11px; border:1px solid var(--line); border-radius:4px; color:var(--text); background:var(--panel); font-weight:700; cursor:pointer; }} .slide-controls button:disabled {{ opacity:.45; cursor:default; }} .slide-position {{ color:var(--muted); font-size:12px; font-weight:700; }}
    .moment-nav a.active {{ background:var(--panel); box-shadow:inset 3px 0 var(--cyan); }}
    .moment-head {{ display:flex; justify-content:space-between; gap:28px; margin-bottom:18px; }}
    .eyebrow {{ margin:0 0 6px; color:var(--cyan); font-size:12px; font-weight:700; text-transform:uppercase; }}
    h2 {{ margin:0; font-size:28px; }} .moment-head p:last-child {{ margin:6px 0 0; color:var(--muted); }}
    .boundary {{ min-width:190px; padding-left:14px; border-left:3px solid var(--green); }} .boundary span {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; }} .boundary strong {{ font-size:26px; }}
    .viewer-shell {{ position:relative; width:100%; aspect-ratio:16/7; min-height:330px; border:1px solid var(--line); background:#050709; overflow:hidden; }}
    iframe {{ position:absolute; inset:0; z-index:2; width:100%; height:100%; border:0; background:#050709; }}
    .viewer-placeholder {{ position:absolute; z-index:2; inset:0; display:grid; grid-template-columns:1fr 1fr; color:var(--text); pointer-events:none; }} .viewer-placeholder span {{ align-self:start; justify-self:start; margin:8px; padding:3px 6px; border:1px solid var(--line); border-radius:3px; background:rgba(12,15,19,.82); font-size:11px; font-weight:800; line-height:1; text-transform:uppercase; }} .viewer-placeholder strong {{ display:none; }}
    .routes {{ display:flex; flex-wrap:wrap; gap:8px; padding:14px 0; border-bottom:1px solid var(--line); }}
    .route-option {{ min-width:190px; padding:10px 12px; text-align:left; color:var(--muted); border:1px solid var(--line); border-radius:4px; background:var(--panel); cursor:pointer; }}
    .route-option strong,.route-option span {{ display:block; }} .route-option strong {{ color:var(--text); }} .route-option span {{ margin-top:2px; font-size:12px; }} .route-option.active {{ border-color:var(--cyan); box-shadow:inset 3px 0 var(--cyan); }}
    .facts {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); border:1px solid var(--line); border-top:0; }} .facts section {{ min-width:0; padding:15px; border-right:1px solid var(--line); }} .facts section:last-child {{ border:0; }}
    .facts span,.facts strong,.facts small {{ display:block; }} .facts span {{ color:var(--muted); font-size:11px; text-transform:uppercase; }} .facts strong {{ margin-top:4px; font-size:18px; }} .facts small {{ margin-top:3px; color:var(--muted); }}
    .empty {{ padding:80px 20px; text-align:center; color:var(--muted); }}
    @media(max-width:850px) {{ .topbar {{ padding-inline:12px; }} .phase-tabs a {{ min-height:44px; padding:8px 4px; font-size:11px; }} .layout {{ display:block; }} aside {{ position:static; height:auto; border:0; border-bottom:1px solid var(--line); }} .moment-nav {{ display:flex; overflow:auto; }} .moment-nav a {{ min-width:150px; }} main {{ padding-inline:12px; }} .moment-head {{ display:block; }} .boundary {{ margin-top:14px; }} .viewer-shell {{ min-height:260px; aspect-ratio:4/3; }} .facts {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .facts section:nth-child(2) {{ border-right:0; }} .facts section:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }} }}
  </style>
</head>
<body>
  <header class="topbar"><div class="brand"><strong>Slippi AI Review</strong><small>{esc(replay_name)} &middot; analyzing P{controlled_port}</small></div><nav class="tabs"><a href="advantage_review.html">Advantage</a><a class="active" href="neutral_review.html">Neutral</a><a href="disadvantage_review.html">Disadvantage</a><a href="/">Dashboard</a></nav></header>
  <nav class="phase-tabs" aria-label="Game phase"><a href="advantage_review.html">Advantage</a><a class="active" href="neutral_review.html">Neutral</a><a href="disadvantage_review.html">Disadvantage</a></nav>
  <div class="layout">
    <aside><h1>Neutral losses</h1><p>{len(targets)} reliable avoidance moment{'s' if len(targets) != 1 else ''}</p><nav class="moment-nav">{''.join(nav)}</nav></aside>
    <main><div class="slide-controls" aria-label="Neutral review slide navigation"><button type="button" data-slide-nav="previous">Previous</button><span class="slide-position" aria-live="polite"></span><button type="button" data-slide-nav="next">Next</button></div>{''.join(sections)}{empty}</main>
  </div>
  <script>
    const loadFrame = frame => {{ if (!frame.src && frame.dataset.src) frame.src = frame.dataset.src; }};
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {{ if (entry.isIntersecting) {{ loadFrame(entry.target); observer.unobserve(entry.target); }} }}), {{ rootMargin:'500px 0px' }});
    document.querySelectorAll('iframe[data-src]').forEach(frame => observer.observe(frame));
    document.querySelectorAll('.routes').forEach(routes => routes.addEventListener('click', event => {{
      const button = event.target.closest('.route-option'); if (!button) return;
      const moment = button.closest('.moment'); const frame = moment.querySelector('iframe');
      routes.querySelectorAll('.route-option').forEach(item => {{ item.classList.toggle('active', item === button); item.setAttribute('aria-pressed', item === button ? 'true' : 'false'); }});
      frame.src = button.dataset.routeSrc;
      moment.querySelector('[data-option-name]').textContent = button.dataset.option;
      moment.querySelector('[data-option-goal]').textContent = button.dataset.goal;
      moment.querySelector('[data-option-character]').textContent = button.dataset.character;
      moment.querySelector('[data-option-lead]').textContent = button.dataset.lead + 'f';
      if (button.dataset.injection) moment.querySelector('[data-option-injection]').textContent = 'Control starts at f' + button.dataset.injection;
      moment.querySelector('[data-option-avoid]').textContent = button.dataset.avoid;
      moment.querySelector('[data-option-quality]').textContent = button.dataset.clean + ' clean · ' + button.dataset.stable + ' stable neutral';
    }}));
    const moments = [...document.querySelectorAll('.moment')];
    const navLinks = [...document.querySelectorAll('.moment-nav a')];
    const position = document.querySelector('.slide-position');
    const slideButtons = [...document.querySelectorAll('[data-slide-nav]')];
    let active = null;
    const selectSlide = (moment, scroll = false) => {{
      if (!moment) return;
      active = moment;
      moments.forEach(item => item.classList.toggle('slide-active', item === moment));
      navLinks.forEach(link => link.classList.toggle('active', link.getAttribute('href') === `#${{moment.id}}`));
      const index = moments.indexOf(moment);
      position.textContent = `Moment ${{index + 1}} of ${{moments.length}}`;
      slideButtons.forEach(button => button.disabled = button.dataset.slideNav === 'previous' ? index === 0 : index === moments.length - 1);
      if (scroll) window.scrollTo({{top:0, behavior:'smooth'}});
    }};
    if (moments.length) selectSlide(moments[0]);
    slideButtons.forEach(button => button.addEventListener('click', () => {{
      const index = moments.indexOf(active); selectSlide(moments[index + (button.dataset.slideNav === 'previous' ? -1 : 1)], true);
    }}));
    navLinks.forEach(link => link.addEventListener('click', event => {{ event.preventDefault(); selectSlide(document.querySelector(link.getAttribute('href')), true); }}));
    document.addEventListener('keydown', event => {{
      if (event.target.matches('input,select,textarea,button')) return;
      if (event.key === 'ArrowLeft') slideButtons.find(button => button.dataset.slideNav === 'previous')?.click();
      if (event.key === 'ArrowRight') slideButtons.find(button => button.dataset.slideNav === 'next')?.click();
    }});
    document.addEventListener('keydown', event => {{
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      const key = event.key.toLowerCase();
      if (!([" ", ",", ".", "e"].includes(key) || /^[0-9]$/.test(key))) return;
      event.preventDefault();
      if (event.repeat && key === " ") return;
      const frame = active?.querySelector('iframe[data-src]');
      if (!frame) return;
      loadFrame(frame);
      frame.contentWindow?.postMessage({{ type:'comparison-shortcut', key }}, '*');
    }}, {{ capture:true }});
  </script>
</body>
</html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.out.resolve().write_text(build_page(payload), encoding="utf-8")
    queue = json.loads(local_path(payload["queue_json"]).read_text(encoding="utf-8"))
    write_placeholder(
        args.out.resolve().parent / "disadvantage_review.html",
        display_name=str(queue.get("display_name") or Path(queue.get("replay") or "game.slp").stem),
        controlled_port=int(queue.get("controlled_port") or 1),
    )
    print(json.dumps({"out": str(args.out.resolve()), "targets": int(payload.get("target_count") or 0)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
