from __future__ import annotations

import unittest

from slippi_ai_review.practice_start import PlayerFrame, choose_stable_start


def state(action: int = 14, *, x: float = 0.0, velocity: float = 0.0) -> PlayerFrame:
    return PlayerFrame(
        action=action,
        x=x,
        y=0.0,
        velocities=(velocity, 0.0, 0.0, 0.0, 0.0),
        hitlag=0.0,
    )


class PracticeStartTests(unittest.TestCase):
    def test_counterattack_starts_before_connected_damage_exchange(self) -> None:
        frames = {frame: {0: state(), 1: state()} for frame in range(1508, 1691)}
        for frame in range(1652, 1664):
            frames[frame][0] = state(75)
        for frame in range(1673, 1686):
            frames[frame][0] = state(75)
        for frame in range(1678, 1690):
            frames[frame][1] = state(44)
        decision = choose_stable_start(
            frames,
            players={0, 1},
            earliest_frame=1509,
            preferred_latest_frame=1674,
            fallback_latest_frame=1684,
            opening_action_frame=1690,
            default_frame=1674,
            opening_type="counter-attack",
        )
        self.assertEqual((decision.frame, decision.mode), (1634, "counterattack_setup"))

    def test_uses_shortened_preroll_to_avoid_spotdodge_path(self) -> None:
        frames = {
            frame: {0: state(), 1: state(235 if 1150 <= frame <= 1169 else 14)}
            for frame in range(1144, 1188)
        }
        decision = choose_stable_start(
            frames,
            players={0, 1},
            earliest_frame=1145,
            preferred_latest_frame=1160,
            fallback_latest_frame=1170,
            opening_action_frame=1187,
            default_frame=1160,
        )
        self.assertEqual((decision.frame, decision.mode), (1170, "shortened_preroll"))

    def test_prefers_full_preroll_when_path_is_clean(self) -> None:
        frames = {frame: {0: state(), 1: state()} for frame in range(39, 91)}
        decision = choose_stable_start(
            frames,
            players={0, 1},
            earliest_frame=40,
            preferred_latest_frame=70,
            fallback_latest_frame=80,
            opening_action_frame=90,
            default_frame=70,
        )
        self.assertEqual((decision.frame, decision.mode), (70, "preferred"))

    def test_falls_back_when_no_candidate_is_stable(self) -> None:
        frames = {
            frame: {0: state(velocity=1.0), 1: state()}
            for frame in range(39, 91)
        }
        decision = choose_stable_start(
            frames,
            players={0, 1},
            earliest_frame=40,
            preferred_latest_frame=70,
            fallback_latest_frame=80,
            opening_action_frame=90,
            default_frame=55,
        )
        self.assertEqual((decision.frame, decision.mode), (55, "default_no_stable_candidate"))

    def test_uses_stable_start_when_replay_path_contains_actions(self) -> None:
        frames = {frame: {0: state(), 1: state()} for frame in range(39, 91)}
        for frame in range(61, 90):
            frames[frame][0] = state(44)
        decision = choose_stable_start(
            frames,
            players={0, 1},
            earliest_frame=40,
            preferred_latest_frame=60,
            fallback_latest_frame=70,
            opening_action_frame=90,
            default_frame=60,
        )
        self.assertEqual((decision.frame, decision.mode), (60, "stable_active_path"))


if __name__ == "__main__":
    unittest.main()
