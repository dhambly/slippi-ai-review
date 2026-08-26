from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from slippi_ai_review.paths import WEB_DIR
from slippi_ai_review.server import SUPPORTED_CHARACTER_NAMES


VIEWER_ZIP_BY_CHARACTER = {
    "Fox": "fox.zip",
    "Falco": "falco.zip",
    "Marth": "marth.zip",
    "Sheik": "sheik.zip",
    "Captain Falcon": "captainFalcon.zip",
    "Ganondorf": "ganondorf.zip",
    "Jigglypuff": "jigglypuff.zip",
    "Pikachu": "pikachu.zip",
    "Samus": "samus.zip",
    "Peach": "peach.zip",
    "Yoshi": "yoshi.zip",
}


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_internal_character_ids_map_to_slippi_external_ids(tmp_path: Path) -> None:
    source = WEB_DIR / "msl_static" / "tools" / "viewer" / "msltrace1.js"
    module_path = tmp_path / "msltrace1.mjs"
    shutil.copy2(source, module_path)
    module = module_path.as_uri()
    script = f"""
      import {{ externalCharId }} from {json.dumps(module)};
      const expected = [8,2,0,1,4,5,6,19,11,12,14,14,13,16,17,15,10,7,9,18,21,22,20,24,3,25,23];
      const actual = expected.map((_, internalId) => externalCharId(internalId));
      if (JSON.stringify(actual) !== JSON.stringify(expected)) {{
        throw new Error(`character mapping mismatch: ${{JSON.stringify(actual)}}`);
      }}
    """

    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        check=True,
    )


def test_every_supported_character_has_a_valid_viewer_bundle() -> None:
    zips = WEB_DIR / "msl_static" / "tools" / "viewer" / "slippi-viewer" / "public" / "zips"

    assert set(VIEWER_ZIP_BY_CHARACTER) == set(SUPPORTED_CHARACTER_NAMES)
    for filename in VIEWER_ZIP_BY_CHARACTER.values():
        assert zipfile.is_zipfile(zips / filename), filename
