from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("review_bootstrap", ROOT / "scripts" / "bootstrap.py")
assert SPEC and SPEC.loader
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def test_locked_runtime_artifacts_match_checksums(self) -> None:
        lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
        for platform_key, specs in lock["runtimeBundles"].items():
            for runtime in ("legacy", "decomp"):
                artifact = ROOT / specs[runtime]["path"]
                with self.subTest(platform=platform_key, runtime=runtime):
                    actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
                    self.assertEqual(actual, specs[runtime]["sha256"])

    def test_locked_model_matches_checksum_and_size(self) -> None:
        lock = json.loads((ROOT / "dependencies.lock.json").read_text(encoding="utf-8"))
        spec = lock["modelBundle"]
        model = ROOT / spec["path"]
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), spec["sha256"])
        self.assertEqual(model.stat().st_size, spec["bytes"])

    def test_runtime_platform_key_normalizes_x86_64(self) -> None:
        with mock.patch.object(bootstrap.sys, "platform", "linux"), mock.patch.object(
            bootstrap.platform, "machine", return_value="AMD64"
        ):
            self.assertEqual(bootstrap.runtime_platform_key(), "linux-x86_64-cp312")

    def test_runtime_platform_key_selects_macos_architecture(self) -> None:
        for machine, expected in (
            ("arm64", "macos-arm64-cp312"),
            ("x86_64", "macos-x86_64-cp312"),
        ):
            with self.subTest(machine=machine), mock.patch.object(
                bootstrap.sys, "platform", "darwin"
            ), mock.patch.object(bootstrap.platform, "machine", return_value=machine):
                self.assertEqual(bootstrap.runtime_platform_key(), expected)

    def test_runtime_artifact_is_verified_extracted_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "runtime.whl"
            with zipfile.ZipFile(artifact, "w") as archive:
                archive.writestr("melee_sim/__init__.py", "VALUE = 1\n")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            destination = root / "runtime"
            kwargs = {
                "name": "test-runtime",
                "artifact": artifact,
                "expected_hash": digest,
                "destination": destination,
                "identity": {"sourceCommit": "abc", "platform": "test"},
            }
            first = bootstrap.install_runtime_artifact(**kwargs)
            second = bootstrap.install_runtime_artifact(**kwargs)
            self.assertEqual(first, destination)
            self.assertEqual(second, destination)
            self.assertEqual((destination / "melee_sim" / "__init__.py").read_text(), "VALUE = 1\n")

    def test_runtime_artifact_rejects_wrong_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "runtime.whl"
            artifact.write_bytes(b"not a wheel")
            with self.assertRaisesRegex(SystemExit, "checksum mismatch"):
                bootstrap.install_runtime_artifact(
                    name="bad-runtime",
                    artifact=artifact,
                    expected_hash="0" * 64,
                    destination=root / "runtime",
                    identity={},
                )


if __name__ == "__main__":
    unittest.main()
