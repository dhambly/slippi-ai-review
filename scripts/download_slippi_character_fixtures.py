"""Download a small per-character fixture set from the public Slippi v3.7 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, RepoFile, RepoFolder


DATASET = "erickfm/slippi-public-dataset-v3.7"
DOWNLOAD_ROOT = f"https://huggingface.co/datasets/{DATASET}/resolve/main"
HF = HfApi()


def dataset_characters() -> list[str]:
    rows = HF.list_repo_tree(DATASET, repo_type="dataset", recursive=False, expand=False)
    return sorted(row.path for row in rows if isinstance(row, RepoFolder))


def character_files(character: str, count: int) -> list[dict[str, Any]]:
    rows = HF.list_repo_tree(
        DATASET,
        path_in_repo=character,
        repo_type="dataset",
        recursive=True,
        expand=False,
    )
    files = [row for row in rows if isinstance(row, RepoFile) and row.path.endswith(".slp")]

    def fixture_rank(row: RepoFile) -> str:
        return hashlib.sha256(row.path.encode("utf-8")).hexdigest()

    selected = sorted(files, key=fixture_rank)[:count]
    return [
        {
            "path": row.path,
            "size": int(row.size or 0),
            "sha256": str(row.lfs.sha256 if row.lfs else row.blob_id),
            "characterBucket": character,
        }
        for row in selected
    ]


def download_one(row: dict[str, Any], out: Path) -> dict[str, Any]:
    source_path = str(row["path"])
    sha256 = str(row["sha256"])
    target = out / "files" / f"{sha256}.slp"
    target.parent.mkdir(parents=True, exist_ok=True)
    expected_size = int(row.get("size") or 0)
    if target.is_file() and (expected_size <= 0 or target.stat().st_size == expected_size):
        status = "existing"
    else:
        url = f"{DOWNLOAD_ROOT}/{urllib.parse.quote(source_path, safe='/')}?download=true"
        temporary = target.with_suffix(".slp.part")
        for attempt in range(1, 6):
            try:
                urllib.request.urlretrieve(url, temporary)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 5:
                    raise
                time.sleep(2 ** (attempt - 1))
        if expected_size > 0 and temporary.stat().st_size != expected_size:
            temporary.unlink(missing_ok=True)
            raise IOError(f"size mismatch for {source_path}: expected {expected_size}")
        temporary.replace(target)
        status = "downloaded"
    return {
        "characterBuckets": sorted(row["characterBuckets"]),
        "sourcePath": source_path,
        "sourceSize": expected_size,
        "sha256": sha256,
        "localPath": str(target.resolve()),
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/corpora/slippi_public_v3_7"))
    parser.add_argument("--samples-per-character", type=int, default=500)
    parser.add_argument("--character", action="append", default=[])
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_character <= 0:
        raise ValueError("--samples-per-character must be positive")
    out = args.out.resolve()
    characters = sorted(set(args.character or dataset_characters()))
    available = set(dataset_characters())
    unknown = sorted(set(characters) - available)
    if unknown:
        raise ValueError(f"unknown character buckets: {unknown}")

    selected_by_hash: dict[str, dict[str, Any]] = {}
    bucket_selection_count = 0
    for character in characters:
        rows = character_files(character, int(args.samples_per_character))
        bucket_selection_count += len(rows)
        for row in rows:
            digest = str(row["sha256"])
            existing = selected_by_hash.get(digest)
            if existing is None:
                selected_by_hash[digest] = {**row, "characterBuckets": {character}}
            else:
                existing["characterBuckets"].add(character)
    selected = list(selected_by_hash.values())

    results = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {executor.submit(download_one, row, out): row for row in selected}
        for future in as_completed(futures):
            source = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "characterBuckets": sorted(source["characterBuckets"]),
                    "sourcePath": str(source["path"]),
                    "sourceSize": int(source.get("size") or 0),
                    "sha256": str(source["sha256"]),
                    "status": "unavailable",
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                }
            results.append(result)
            print(json.dumps(result), flush=True)

    available_results = [row for row in results if row["status"] != "unavailable"]
    unavailable_results = [row for row in results if row["status"] == "unavailable"]
    manifest = {
        "version": 1,
        "dataset": DATASET,
        "samplesPerCharacter": int(args.samples_per_character),
        "characters": characters,
        "bucketSelections": bucket_selection_count,
        "files": sorted(available_results, key=lambda row: row["sha256"]),
        "unavailable": sorted(unavailable_results, key=lambda row: row["sha256"]),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "files": len(available_results),
        "unavailable": len(unavailable_results),
        "characters": len(characters),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
