from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from slippi_ai_review.training import (
    CARD_FILENAME,
    SCENARIO_MODES,
    build_tm_replay_command,
    derive_practice_window,
    install_gci,
    resolved_safe_frame,
    select_route,
)


class ScenarioSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = {
            "targets": [
                {
                    "representative_lane": {
                        "laneId": 1,
                        "controllerStream": {"takeoverFrame": 100, "endFrame": 200},
                    },
                    "alternative_routes": [
                        {
                            "representative_lane": {
                                "laneId": 2,
                                "controllerStream": {"takeoverFrame": 101, "endFrame": 180},
                            }
                        }
                    ],
                }
            ]
        }

    def test_selects_primary_and_alternative_streams(self) -> None:
        _, lane, stream = select_route(self.queue, 1, 0)
        self.assertEqual((lane["laneId"], stream["takeoverFrame"]), (1, 100))
        _, lane, stream = select_route(self.queue, 1, 1)
        self.assertEqual((lane["laneId"], stream["takeoverFrame"]), (2, 101))

    def test_rejects_missing_target_or_route(self) -> None:
        with self.assertRaises(ValueError):
            select_route(self.queue, 2, 0)
        with self.assertRaises(ValueError):
            select_route(self.queue, 1, 2)

    def test_exposes_replay_and_phillip_modes(self) -> None:
        self.assertEqual(SCENARIO_MODES, ("replay", "phillip", "variations"))

    def test_practice_window_starts_before_first_sequence_hit(self) -> None:
        target = {
            "takeover_frame": 100,
            "replay_baseline": {
                "segment_start_frame": 70,
                "presentation_segment": {"openingFrame": 70},
                "sequence_opening_move": {"frame": 70},
            },
        }
        stream = {"takeoverFrame": 100, "endFrame": 160}
        self.assertEqual(derive_practice_window(target, stream, 30), (40, 70, 160))
        self.assertEqual(derive_practice_window(target, stream, 0), (40, 70, 160))

    def test_practice_window_clamps_to_first_slippi_frame(self) -> None:
        target = {"replay_baseline": {"segment_start_frame": -100}}
        stream = {"takeoverFrame": -80, "endFrame": 20}
        self.assertEqual(derive_practice_window(target, stream, 60), (-123, -100, 20))

    def test_practice_window_caps_leadin_at_200_frames(self) -> None:
        target = {
            "replay_baseline": {
                "segment_start_frame": 1190,
                "sequence_opening_move": {"frame": 1190},
                "presentation_segment": {"openingFrame": 924},
            },
        }
        stream = {"takeoverFrame": 1345, "endFrame": 1499}
        self.assertEqual(derive_practice_window(target, stream, 30), (1160, 1190, 1499))


class AtomicInstallTests(unittest.TestCase):
    def test_installs_to_reserved_card_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.gci"
            source.write_bytes(b"scenario")
            destination = install_gci(source, root / "Card A")
            self.assertEqual(destination.name, CARD_FILENAME)
            self.assertEqual(destination.read_bytes(), b"scenario")
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])


class TmReplayCommandTests(unittest.TestCase):
    def test_both_players_start_in_playback(self) -> None:
        command = build_tm_replay_command(
            Path("tm_replay.exe"),
            Path("scenario.slp"),
            Path("scenario.gci"),
            start_frame=40,
            duration=120,
            display_name="CODEX test",
            human_port=1,
        )
        self.assertEqual(command[command.index("--hmn-mode") + 1], "playback")
        self.assertEqual(command[command.index("--cpu-mode") + 1], "playback")
        self.assertIn("--auto-restore", command)

    def test_variation_handoff_uses_absolute_slippi_frame(self) -> None:
        command = build_tm_replay_command(
            Path("tm_replay.exe"),
            Path("scenario.slp"),
            Path("scenario.gci"),
            start_frame=1634,
            duration=300,
            display_name="CODEX RANDOM",
            human_port=1,
            cpu_handoff_frame=1704,
        )
        self.assertEqual(command[command.index("--cpu-handoff-frame") + 1], "1704")

    def test_reads_resolved_safe_frame_from_export_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "tm_replay.log"
            log_path.write_text(
                "Resolved safe Slippi frame: 1043\nSavestate file created\n",
                encoding="utf-8",
            )
            self.assertEqual(resolved_safe_frame(log_path), 1043)


if __name__ == "__main__":
    unittest.main()
