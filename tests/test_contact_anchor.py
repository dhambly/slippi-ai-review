from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from slippi_ai_review.simulation import (
    _load_frame_metadata,
    _post_contact_anchor_record,
    _recorded_contact_frame,
    _replay_baseline_after_anchor,
)
from slippi_ai_review.simulation_support import _load_base_frames


class ContactAnchorTest(unittest.TestCase):
    def test_refinement_preserves_candidate_base_frame(self) -> None:
        payload = {
            "targets": [{"base_frame": 1330, "takeover_frame": 1345, "label": "anchored"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            args = SimpleNamespace(takeover_frame=[], takeover_frames_json=path)
            self.assertEqual(_load_base_frames(args), [1330])
            self.assertEqual(_load_frame_metadata(path)[1330]["label"], "anchored")

    def test_finds_first_progressed_frame_after_hitlag(self) -> None:
        dtype = np.dtype([
            ("hitlag", np.uint16, (2,)),
            ("action_frame", np.int16, (2,)),
            ("pos_x", np.float32, (2,)),
            ("pos_y", np.float32, (2,)),
        ])
        seed = np.zeros(8, dtype=dtype)
        seed[2:5]["hitlag"] = [[4, 4], [2, 2], [1, 1]]
        seed[2:6]["action_frame"] = [[8, 1], [8, 1], [8, 1], [8, 1]]
        seed[6]["action_frame"] = [9, 2]
        buffers = SimpleNamespace(seed_t=seed, num_records=len(seed))
        frame_ids = np.arange(100, 108, dtype=np.int32)

        record = _post_contact_anchor_record(
            buffers,
            frame_ids,
            {"point_evidence": {"hit_frame": 102}},
            analyzed_idx=1,
            defender_idx=0,
        )

        self.assertEqual(record, 6)

    def test_reads_contact_from_move_and_recalculates_later_baseline(self) -> None:
        metadata = {
            "branch_move": {"frame": 1099},
            "remaining_move_sequence": [
                {"frame": 1099, "damage": 10, "hitCount": 1},
                {"frame": 1120, "damage": 7, "hitCount": 2},
            ],
            "original_followup_kill": True,
        }

        self.assertEqual(_recorded_contact_frame(metadata), 1099)
        self.assertEqual(_replay_baseline_after_anchor(metadata, 1104), (7.0, 2, True, 119.0))


if __name__ == "__main__":
    unittest.main()
