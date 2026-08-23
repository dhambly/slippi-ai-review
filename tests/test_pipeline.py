from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from slippi_ai_review.pipeline import (
    disadvantage_command,
    msl_command,
    neutral_command,
    phase_sweep_command,
    publish_phase_sweep_artifacts,
    windows_to_wsl,
)


class PipelineTests(unittest.TestCase):
    def test_wsl_unc_root_maps_back_to_linux_path(self) -> None:
        root = Path(r"\\wsl.localhost\Ubuntu\home\user\melee-sim-light")
        self.assertEqual(windows_to_wsl(root), "/home/user/melee-sim-light")

    def test_decomp_command_uses_replay_exact_runner_and_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decomp = root / "melee-sim-light-decomp"
            args = SimpleNamespace(
                replay=root / "game.slp", controlled_port=2, model=root / "gm-v2",
                msl_root=root / "legacy", msl_decomp_root=decomp,
                slippi_ai_root=root / "slippi-ai", simulation_backend="decomp",
                runtime_mode="native", runtime_python="/opt/review/.venv/bin/python",
                cuda_library_path="", max_batch_lanes=128, gpu_duty_cycle=0.2,
            )
            command = msl_command(
                args,
                metadata=root / "candidates.json",
                out_dir=root / "run",
                offsets="0",
                samples=8,
                dump_streams=True,
            )
        self.assertTrue(command[command.index("-u") + 1].endswith("decomp_simulation.py"))
        self.assertEqual(command[command.index("--msl-root") + 1], str(decomp.resolve()))
        self.assertIn(str(decomp.resolve()), next(value for value in command if value.startswith("PYTHONPATH=")))
        self.assertIn("--dump-controller-streams", command)

    def test_native_command_runs_directly_without_wsl_or_micromamba(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                replay=root / "game.slp", controlled_port=1, model=root / "gm-v2",
                msl_root=root / "melee-sim-light", slippi_ai_root=root / "slippi-ai",
                runtime_mode="native", runtime_python="/opt/review/.venv/bin/python",
                cuda_library_path="", max_batch_lanes=1024, gpu_duty_cycle=0.2,
            )
            command = msl_command(
                args,
                metadata=root / "candidates.json",
                out_dir=root / "run",
                offsets="0",
                samples=16,
                dump_streams=False,
            )
        self.assertEqual(command[0], "env")
        self.assertIn("/opt/review/.venv/bin/python", command)
        self.assertNotIn("wsl", command)
        self.assertNotIn("micromamba", command)
        self.assertTrue(any(item.startswith("MSL_DATA_DIR=") for item in command))
        self.assertTrue(any(item.startswith("PYTHONPATH=") for item in command))

    def test_phase_sweep_keeps_refined_advantage_as_primary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sweep"
            destination = root / "review"
            (source / "traces").mkdir(parents=True)
            (source / "viewer").mkdir()
            destination.mkdir()
            for phase in ("advantage", "neutral", "disadvantage"):
                (source / f"{phase}_review.html").write_text(f"sweep {phase}", encoding="utf-8")
            (destination / "advantage_review.html").write_text("refined advantage", encoding="utf-8")
            publish_phase_sweep_artifacts(source, destination)
            self.assertEqual((destination / "advantage_review.html").read_text(encoding="utf-8"), "refined advantage")
            self.assertEqual((destination / "neutral_review.html").read_text(encoding="utf-8"), "sweep neutral")

    def test_disadvantage_command_uses_defensive_budget_and_wsl_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                replay=root / "game.slp", controlled_port=1, display_name="Defensive Game",
                disadvantage_samples=64, disadvantage_rollout_frames=180,
                disadvantage_opponent_delay_frames=60,
                disadvantage_enable_gpu=True, max_batch_lanes=2048, render_workers=4,
                model=root / "gm-v2", msl_root=root / "melee-sim-light",
                slippi_ai_root=root / "slippi-ai", cuda_library_path="/usr/lib/wsl/lib",
                micromamba="/home/user/micromamba", msl_env="/home/user/env",
                runtime_mode="wsl",
                gpu_duty_cycle=0.2, force=False,
            )
            command = disadvantage_command(args, inventory=root / "inventory.json", out_dir=root / "disadvantage")
        self.assertEqual(command[0:2], ["wsl", "env"])
        self.assertIn("slippi_ai_review.disadvantage_pipeline", command)
        self.assertEqual(command[command.index("--samples") + 1], "64")
        self.assertEqual(command[command.index("--rollout-frames") + 1], "180")
        self.assertEqual(command[command.index("--opponent-takeover-max-delay-frames") + 1], "60")
        self.assertIn("--enable-gpu", command)

    def test_neutral_command_uses_wsl_environment_and_neutral_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                replay=root / "game.slp",
                controlled_port=2,
                display_name="Friendly Game",
                slp_version="3.18.0",
                neutral_preflight_samples=48,
                neutral_refinement_samples=192,
                max_batch_lanes=2048,
                render_workers=4,
                model=root / "gm-v2",
                msl_root=root / "melee-sim-light",
                slippi_ai_root=root / "slippi-ai",
                cuda_library_path="/usr/lib/wsl/lib",
                micromamba="/home/user/micromamba",
                msl_env="/home/user/env",
                runtime_mode="wsl",
            )
            command = neutral_command(args, out_dir=root / "neutral", raw_events=root / "raw_events.json")
        self.assertEqual(command[0:2], ["wsl", "env"])
        self.assertIn("slippi_ai_review.neutral_pipeline", command)
        self.assertEqual(command[command.index("--analyzed-port") + 1], "2")
        self.assertEqual(command[command.index("--preflight-samples") + 1], "48")
        self.assertEqual(command[command.index("--refinement-samples") + 1], "192")
        self.assertEqual(command[command.index("--slp-version") + 1], "3.18.0")
        self.assertIn("--raw-events-json", command)
        self.assertIn("--enable-gpu", command)
        args.neutral_enable_gpu = False
        cpu_only = neutral_command(args, out_dir=root / "neutral", raw_events=root / "raw_events.json")
        self.assertNotIn("--enable-gpu", cpu_only)

    def test_phase_sweep_is_small_and_keeps_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                replay=root / "game.slp", controlled_port=1, phase_sweep_samples=12,
                phase_sweep_enable_gpu=True, max_batch_lanes=2048, model=root / "gm-v2",
                msl_root=root / "melee-sim-light", slippi_ai_root=root / "slippi-ai",
                cuda_library_path="/usr/lib/wsl/lib", micromamba="/home/user/micromamba", msl_env="/home/user/env",
                runtime_mode="wsl",
            )
            command = phase_sweep_command(args, inventory=root / "inventory.json", out_dir=root / "sweep")
        self.assertEqual(command[command.index("--samples-per-point") + 1], "12")
        self.assertEqual(command[command.index("--rollout-frames") + 1], "120")
        self.assertEqual(command[command.index("--defender-takeover-mode") + 1], "observed-phase-followup")
        self.assertIn("--dump-controller-streams", command)
        self.assertIn("--enable-gpu", command)
        self.assertEqual(command[command.index("--gpu-duty-cycle") + 1], "0.2")


if __name__ == "__main__":
    unittest.main()
