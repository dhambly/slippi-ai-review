import unittest

from slippi_ai_review.timeline import timeline_payload, trace_events


FIELDS = ["charId", "actionId"]


def trace(rows):
    return {"frames": {"playerFields": FIELDS, "rows": rows}}


class TimelineEventsTest(unittest.TestCase):
    def test_sparse_deltas_form_compact_events(self):
        payload = trace([
            [0, 0, None, [[1, 20], [7, 178]]],
            [1, 1, None, [[], []]],
            [1, 3, None, [[[1, 65]], [[1, 181]]]],
            [1, 8, None, [[[1, 14]], [[1, 212]]]],
        ])
        events = trace_events(payload)
        self.assertEqual(
            [(event["name"], event["player"], event["frame"], event["endFrame"]) for event in events],
            [("Nair", 0, 3, 8), ("Grab", 1, 8, 9)],
        )

    def test_specials_use_input_names(self):
        payload = trace([[0, 0, None, [[7, 358], [1, 381]]]])
        self.assertEqual([event["name"] for event in trace_events(payload)], ["Up-B", "Up-B"])

    def test_takeover_inside_move_does_not_create_second_start(self):
        replay = trace([
            [0, 0, None, [[1, 14], [7, 14]]],
            [1, 52, None, [[[1, 65]], []]],
            [1, 92, None, [[[1, 14]], []]],
        ])
        agent = trace([
            [0, 0, None, [[1, 65], [7, 14]]],
            [1, 28, None, [[[1, 63]], []]],
        ])
        payload = timeline_payload(replay, agent, switch_frame=60)
        self.assertEqual(
            [(event["name"], event["frame"]) for event in payload["lanes"]["msl"]],
            [("Up smash", 88)],
        )

    def test_agent_lane_is_shifted_to_takeover(self):
        replay = trace([[0, 0, None, [[1, 65], [7, 66]]]])
        agent = trace([[0, 0, None, [[1, 63], [7, 235]]]])
        payload = timeline_payload(replay, agent, switch_frame=60)
        self.assertEqual(payload["lanes"]["replay"][0]["frame"], 0)
        self.assertEqual(payload["lanes"]["msl"][0]["frame"], 60)
        self.assertEqual(payload["lanes"]["msl"][0]["name"], "Up smash")


if __name__ == "__main__":
    unittest.main()
