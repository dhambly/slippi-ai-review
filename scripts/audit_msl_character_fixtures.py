"""Run downloaded replay fixtures through MSL's validation buffer builder."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from slippi_ai_review.config import load_settings


_BUILD_VALIDATION_BUFFERS = None


def worker_init(msl_root: str) -> None:
    global _BUILD_VALIDATION_BUFFERS
    root = Path(msl_root)
    sys.path.insert(0, str(root))
    os.chdir(root)
    from tools.slippi.validation_buffer_builder import build_validation_buffers_from_slp

    _BUILD_VALIDATION_BUFFERS = build_validation_buffers_from_slp


def audit_one(fixture: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    result = {**fixture}
    try:
        if _BUILD_VALIDATION_BUFFERS is None:
            raise RuntimeError("validator worker was not initialized")
        buffers = _BUILD_VALIDATION_BUFFERS(slp_path=str(Path(fixture["localPath"])))
        result.update(
            {
                "status": "pass",
                "records": int(buffers.num_records),
                "players": int(buffers.num_players),
            }
        )
    except BaseException as exc:  # PyO3 parser panics inherit BaseException, not Exception.
        result.update(
            {
                "status": "fail",
                "errorType": type(exc).__name__,
                "error": str(exc),
            }
        )
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpora/slippi_public_v3_7"))
    parser.add_argument("--msl-root", type=Path, default=settings.msl_root)
    parser.add_argument("--out", type=Path, default=Path("data/audits/msl_character_compatibility.json"))
    parser.add_argument("--character", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    corpus = args.corpus.resolve()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    requested = set(args.character)
    fixtures = [
        row for row in manifest["files"]
        if not requested or requested.intersection(row.get("characterBuckets") or [row.get("characterBucket")])
    ]
    msl_root = args.msl_root.resolve()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=worker_init,
        initargs=(str(msl_root),),
    ) as executor:
        for index, result in enumerate(executor.map(audit_one, fixtures, chunksize=8), start=1):
            results.append(result)
            print(json.dumps({"index": index, "total": len(fixtures), **result}), flush=True)

    by_character: dict[str, Counter[str]] = defaultdict(Counter)
    errors: Counter[str] = Counter()
    for result in results:
        for character in result.get("characterBuckets") or [result.get("characterBucket")]:
            by_character[str(character)][str(result["status"])] += 1
        if result["status"] == "fail":
            errors[str(result.get("error") or result.get("errorType"))] += 1
    payload = {
        "version": 1,
        "corpus": str(corpus),
        "mslRoot": str(msl_root),
        "files": len(results),
        "passed": sum(result["status"] == "pass" for result in results),
        "failed": sum(result["status"] == "fail" for result in results),
        "byCharacter": {key: dict(value) for key, value in sorted(by_character.items())},
        "errorCounts": dict(errors.most_common()),
        "results": results,
    }
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "passed": payload["passed"], "failed": payload["failed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
