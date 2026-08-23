"""Build the target player's refined defensive review."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from .phase_sweep_report import build_page


def write_placeholder(out: Path, *, display_name: str, controlled_port: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Disadvantage review</title><style>
        :root{{color-scheme:dark;--bg:#0c0f13;--line:#303943;--text:#f4f6f8;--muted:#9ba7b2;--cyan:#64c8e8}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 Inter,system-ui,sans-serif}}a{{color:inherit;text-decoration:none}}.top{{display:flex;align-items:center;gap:8px;min-height:58px;padding:8px 24px;border-bottom:1px solid var(--line)}}.top strong{{margin-right:auto}}.top a{{padding:8px 12px;color:var(--muted);border-bottom:2px solid transparent}}.top a.active{{color:var(--text);border-color:var(--cyan)}}main{{width:min(760px,100%);margin:0 auto;padding:72px 24px}}h1{{margin:0 0 8px;font-size:28px}}p{{color:var(--muted)}}</style></head><body>
        <nav class="top"><strong>{html.escape(display_name)} &middot; P{controlled_port}</strong><a href="advantage_review.html">Advantage</a><a href="neutral_review.html">Neutral</a><a class="active" href="disadvantage_review.html">Disadvantage</a><a href="/">Dashboard</a></nav>
        <main><h1>Disadvantage review</h1><p>No defensive segments qualified in this game.</p></main></body></html>""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    queue = json.loads(Path(manifest["queue_json"]).resolve().read_text(encoding="utf-8"))
    args.out.resolve().parent.mkdir(parents=True, exist_ok=True)
    page = build_page(
        queue,
        manifest,
        phase="disadvantage",
        context_label="defensive refinement",
        verdict_heading="Best observed defensive branch",
        option_heading="Defensive option",
    )
    page = page.replace("queueMode:'phase-sweep'", "queueMode:'disadvantage'")
    args.out.resolve().write_text(page, encoding="utf-8")
    print(json.dumps({"out": str(args.out.resolve()), "targets": len(queue.get("targets") or [])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
