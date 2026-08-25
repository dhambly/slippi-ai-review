"""Build phase-specific slide decks from low-sample whole-game sweep traces."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


PHASES = ("advantage", "neutral", "disadvantage")


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def pct(value: Any) -> str:
    return f"{100 * float(value or 0):.0f}%"


def elapsed(frame: int) -> str:
    seconds = max(0, frame) / 60
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def option_label(value: Any) -> str:
    return str(value or "No commitment").replace("_", " ").title()


def interactive_url(route: dict[str, Any], queue: dict[str, Any]) -> str:
    players = sorted((queue.get("game") or {}).get("players") or [], key=lambda player: int(player.get("port") or 99))
    names = {
        f"p{int(player.get('port') or 0)}": str(player.get("displayName") or player.get("connectCode") or "").strip()
        for player in players
        if int(player.get("port") or 0) in (1, 2)
    }
    params = {
        "replay": route.get("replay_trace"), "agent": route.get("agent_trace"),
        "switch": int(route.get("switch_frame") or 0),
        "takeover": int(route.get("model_control_frame") or route.get("switch_frame") or 0),
        "defenderSwitch": int(route.get("defender_switch_frame") or 0),
        "start": int(route.get("start_frame") or 0), "frames": int(route.get("frame_count") or 1),
        **names,
    }
    if route.get("timeline_events"):
        params["events"] = route["timeline_events"]
    return "viewer/compare.html?" + urlencode(params)


def tabs(active: str) -> str:
    return "".join(
        f'<a class="{"active" if phase == active else ""}" href="{phase}_review.html">{phase.title()}</a>'
        for phase in PHASES
    )


def game_heading(queue: dict[str, Any]) -> tuple[str, str]:
    game = queue.get("game") or {}
    players = game.get("players") or []
    player_names = []
    characters = []
    for player in sorted(players, key=lambda item: int(item.get("port") or 99)):
        player_names.append(str(player.get("displayName") or player.get("connectCode") or f"P{player.get('port')}").strip())
        if player.get("characterName"):
            characters.append(str(player["characterName"]).strip())
    title = " vs ".join(player_names) or Path(str(queue.get("replay") or "game.slp")).stem
    detail = " vs ".join(characters)
    stage = str(game.get("stageName") or "").strip()
    return title, f"{detail} · {stage}" if detail and stage else detail or stage


def build_page(
    queue: dict[str, Any],
    manifest: dict[str, Any],
    *,
    phase: str,
    context_label: str = "whole-game sweep",
    verdict_heading: str = "Best observed low-sample branch",
    option_heading: str = "Phillip option",
) -> str:
    results = {int(item["target_index"]): item.get("interactive") or {} for item in manifest.get("results") or []}
    slides = []
    for index, target in enumerate(queue.get("targets") or [], start=1):
        if target.get("phase") != phase:
            continue
        segment = target.get("phase_segment") or {}
        option = target.get("option") or {}
        interactive = results.get(index) or {}
        frame = int(target.get("base_frame") or 0)
        control_frame = int(target.get("takeover_frame") or frame)
        slides.append((frame, f"""
          <article class="slide" id="slide-{index}" data-target-index="{index}" data-clip-start-frame="{int(interactive.get('start_frame') or frame)}">
            <header><p>{esc(segment.get('subtitle') or phase)} · {elapsed(frame)} elapsed · f{frame}</p><h2>{esc(segment.get('title') or phase.title())}</h2></header>
            <div class="viewer"><iframe loading="lazy" allowfullscreen title="Replay versus Phillip at f{frame}" data-src="{esc(interactive_url(interactive, queue))}"></iframe><div class="placeholder"><span>Replay</span><span>Phillip</span></div></div>
            <div class="verdict"><strong>{esc(verdict_heading)}</strong><span>{int(option.get('samples') or 0)} of {int(option.get('sweepSamples') or option.get('samples') or 0)} samples chose this route ({pct(option.get('optionShare'))}); inspect it rather than treating it as a prescription.</span></div>
            <div class="facts"><section><span>{esc(option_heading)}</span><strong>{esc(option_label(option.get('optionSignature')))}</strong><small>Median score {float(option.get('medianScore') or 0):.1f}</small></section><section><span>Damage trade</span><strong>{float(option.get('damageDealt') or 0):.1f}% dealt · {float(option.get('damageTaken') or 0):.1f}% taken</strong><small>Best score {float(option.get('bestScore') or 0):.1f}</small></section><section><span>Risk</span><strong>{pct(option.get('reversalRate'))} reversal</strong><small>{pct(option.get('killRate'))} kill rate</small></section><section><span>Segment</span><strong>f{int(segment.get('startFrame') or frame)}–f{int(segment.get('endFrame') or frame)}</strong><small>Phillip control at f{control_frame}</small></section></div>
            <div class="practice"><button class="practice-ce" type="button" data-scenario-mode="replay">Replay in CE</button><button class="practice-ce" type="button" data-scenario-mode="phillip">Phillip rollout in CE</button><span class="variation-source" role="group" aria-label="Playback before random defense"><button type="button" class="variation-source-option active" data-variation-source="replay" aria-pressed="true">Replay</button><button type="button" class="variation-source-option" data-variation-source="rollout" aria-pressed="false">Rollout</button></span><span class="variation-action"><button class="practice-ce" type="button" data-scenario-mode="variations">Random defense in CE</button><button class="info-mark" type="button" title="The selected source replays normally until the frame currently selected in the viewer. At that frame, Training Mode CE begins choosing random defensive options.">?</button></span><span class="practice-status" role="status" aria-live="polite"></span></div>
          </article>"""))
    slides.sort(key=lambda item: item[0])
    body = "".join(markup for _, markup in slides) or '<section class="empty">No sweep slides for this phase.</section>'
    port = int(queue.get("controlled_port") or 1)
    game_title, game_detail = game_heading(queue)
    refined_link = '<a class="refined-link" href="advantage_refined.html">Refined advantage routes</a>' if phase == "advantage" else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(game_title)} · {phase.title()}</title><style>
    :root{{color-scheme:dark;--bg:#0c0f13;--panel:#151a20;--line:#303943;--text:#f4f6f8;--muted:#9ba7b2;--cyan:#64c8e8;--green:#62d39a}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 Inter,system-ui,sans-serif}}a{{color:inherit;text-decoration:none}}.top{{display:flex;align-items:center;gap:14px;min-height:58px;padding:8px 24px;border-bottom:1px solid var(--line)}}.top strong{{margin-right:auto}}.top small{{color:var(--muted)}}.refined-link{{color:var(--cyan);font-weight:700}}.phase-tabs{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;max-width:1420px;margin:0 auto;border:1px solid var(--line);border-top:0;background:var(--line)}}.phase-tabs a{{display:flex;align-items:center;justify-content:center;min-height:50px;background:var(--panel);color:var(--muted);font-weight:800}}.phase-tabs a.active{{background:#17313a;color:var(--text);box-shadow:inset 0 -3px var(--cyan)}}main{{width:min(1120px,100%);margin:0 auto;padding:0 24px 60px}}.controls{{display:flex;justify-content:space-between;align-items:center;padding:16px 0;border-bottom:1px solid var(--line)}}button{{min-height:36px;padding:7px 11px;border:1px solid var(--line);border-radius:4px;background:var(--panel);color:var(--text);font:inherit;font-weight:700;cursor:pointer}}button:disabled{{opacity:.45;cursor:default}}.position{{color:var(--muted);font-size:12px;font-weight:700}}.slide{{display:none;padding:28px 0}}.slide.active{{display:block}}header p{{margin:0;color:var(--cyan);font-size:12px;font-weight:800;text-transform:uppercase}}h2{{margin:4px 0 16px;font-size:24px}}.viewer{{position:relative;aspect-ratio:16/7;min-height:320px;border:1px solid var(--line);background:#050709}}iframe{{position:absolute;inset:0;width:100%;height:100%;border:0}}.placeholder{{position:absolute;z-index:2;inset:0;display:grid;grid-template-columns:1fr 1fr;color:var(--text);pointer-events:none}}.placeholder span{{align-self:start;justify-self:start;margin:8px;padding:3px 6px;border:1px solid #46505b;border-radius:3px;background:rgba(12,15,19,.82);font-size:11px;font-weight:800;line-height:1;text-transform:uppercase}}.verdict{{display:flex;gap:9px;padding:10px 12px;border:1px solid #3b5e49;border-top:0;background:#14241b;color:#b9e7cd}}.verdict span{{color:#b9c7bf;font-size:12px}}.facts{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);border-top:0}}.facts section{{min-width:0;padding:13px;border-right:1px solid var(--line)}}.facts section:last-child{{border:0}}.facts span,.facts strong,.facts small{{display:block;overflow-wrap:anywhere}}.facts span,.facts small{{color:var(--muted);font-size:11px}}.facts strong{{margin:4px 0;font-size:15px}}.practice{{display:flex;align-items:center;gap:8px;padding-top:12px;flex-wrap:wrap}}.practice-ce{{border-color:#477b60;background:var(--green);color:#07130d}}.variation-source,.variation-action{{display:flex;align-items:center}}.variation-source-option{{min-width:0;border-radius:0;color:var(--muted)}}.variation-source-option:first-child{{border-radius:4px 0 0 4px}}.variation-source-option:last-child{{border-radius:0 4px 4px 0}}.variation-source-option.active{{border-color:#65d69a;color:var(--text);background:#244333}}.info-mark{{width:24px;min-width:24px;padding:0;border-radius:0 4px 4px 0;border-left:0;color:var(--muted)}}.variation-action .practice-ce{{border-radius:4px 0 0 4px}}.practice-status{{color:var(--muted);font-size:12px}}.empty{{padding:70px 0;text-align:center;color:var(--muted)}}@media(max-width:700px){{.top{{padding-inline:12px}}.phase-tabs a{{min-height:44px;font-size:12px}}main{{padding-inline:12px}}.viewer{{min-height:250px;aspect-ratio:4/3}}.facts{{grid-template-columns:repeat(2,minmax(0,1fr))}}.facts section:nth-child(2){{border-right:0}}.facts section:nth-child(-n+2){{border-bottom:1px solid var(--line)}}.practice{{align-items:stretch}}}}
    </style></head><body><header class="top"><strong>{esc(game_title)}</strong><small>{esc(game_detail)} · P{port} review · {esc(context_label)}</small>{refined_link}<a href="/">Dashboard</a></header><nav class="phase-tabs">{tabs(phase)}</nav><main><div class="controls"><button data-nav="previous">Previous</button><span class="position"></span><button data-nav="next">Next</button></div>{body}</main><script>
      const slides=[...document.querySelectorAll('.slide')],position=document.querySelector('.position'),buttons=[...document.querySelectorAll('[data-nav]')],reviewIdMatch=location.pathname.match(/review-artifacts\\/([0-9a-f-]+)/);let active=0;function select(index,updateHash=true){{if(!slides.length)return;active=Math.max(0,Math.min(slides.length-1,index));slides.forEach((slide,i)=>slide.classList.toggle('active',i===active));position.textContent=`Slide ${{active+1}} of ${{slides.length}}`;buttons.forEach(button=>button.disabled=button.dataset.nav==='previous'?active===0:active===slides.length-1);const frame=slides[active].querySelector('iframe');if(frame&&!frame.src)frame.src=frame.dataset.src;if(updateHash&&slides[active].id)history.replaceState(null,'',`#${{slides[active].id}}`)}}function currentFrame(slide){{const clipStart=Number(slide.dataset.clipStartFrame||0);try{{const seek=slide.querySelector('iframe')?.contentDocument?.querySelector('#seek');if(seek)return clipStart+Number(seek.value||0)}}catch(_error){{}}return clipStart}}function forwardVisualizerShortcut(event){{if(event.ctrlKey||event.metaKey||event.altKey)return;const key=event.key.toLowerCase();if(!([" ",",",".","e"].includes(key)||/^[0-9]$/.test(key)))return;event.preventDefault();if(event.repeat&&key===" ")return;const frame=slides[active]?.querySelector('iframe');if(!frame)return;if(!frame.src&&frame.dataset.src)frame.src=frame.dataset.src;frame.contentWindow?.postMessage({{type:'comparison-shortcut',key}},'*')}}const initial=Math.max(0,slides.findIndex(slide=>`#${{slide.id}}`===location.hash));select(initial,false);buttons.forEach(button=>button.addEventListener('click',()=>select(active+(button.dataset.nav==='previous'?-1:1))));addEventListener('hashchange',()=>{{const index=slides.findIndex(slide=>`#${{slide.id}}`===location.hash);if(index>=0)select(index,false)}});addEventListener('keydown',event=>{{if(event.target.matches('button,input,select,textarea'))return;if(event.key==='ArrowLeft')select(active-1);if(event.key==='ArrowRight')select(active+1)}});document.addEventListener('keydown',forwardVisualizerShortcut,{{capture:true}});document.addEventListener('click',async event=>{{const source=event.target.closest('.variation-source-option');if(source){{const group=source.closest('.variation-source');group.querySelectorAll('button').forEach(item=>{{const on=item===source;item.classList.toggle('active',on);item.setAttribute('aria-pressed',String(on))}});return}}const button=event.target.closest('.practice-ce');if(!button)return;const slide=button.closest('.slide'),status=slide.querySelector('.practice-status');if(!reviewIdMatch){{status.textContent='Open this report from the dashboard first.';return}}button.disabled=true;status.textContent='Preparing scenario...';try{{const mode=button.dataset.scenarioMode,variationSource=slide.querySelector('.variation-source-option.active')?.dataset.variationSource||'replay';const response=await fetch(`/api/reviews/${{reviewIdMatch[1]}}/training-mode`,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{targetIndex:Number(slide.dataset.targetIndex),alternativeIndex:0,scenarioMode:mode,variationStartFrame:mode==='variations'?currentFrame(slide):undefined,variationSource,queueMode:'phase-sweep'}})}});const body=await response.json();if(!response.ok||!body.ok)throw new Error(body.error?.message||'Scenario export failed.');status.textContent=mode==='replay'?`Replay ready in CE from f${{body.scenario.practiceStartFrame}}.`:mode==='variations'?`Random defense starts at f${{body.scenario.variationStartFrame}}.`:`Phillip rollout ready in CE from f${{body.scenario.practiceStartFrame}}.`}}catch(error){{status.textContent=error.message||'Scenario export failed.'}}finally{{button.disabled=false}}}});
    </script></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue_json.resolve().read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    args.out_dir.resolve().mkdir(parents=True, exist_ok=True)
    for phase in PHASES:
        (args.out_dir.resolve() / f"{phase}_review.html").write_text(build_page(queue, manifest, phase=phase), encoding="utf-8")
    print(json.dumps({"out": str(args.out_dir.resolve()), "targets": len(queue.get("targets") or [])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
