from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from slippi_ai_review import server as server_module


def player(port: int, character: str) -> dict[str, object]:
    return {
        "playerIndex": port - 1,
        "port": port,
        "name": None,
        "connectCode": None,
        "character": {"id": port, "name": character},
    }


class SlpValidationTests(unittest.TestCase):
    def test_accepts_supported_characters(self) -> None:
        details = {"players": [player(1, "Fox"), player(2, "Marth")]}
        self.assertEqual(server_module.validate_slp(Path("unused"), lambda _: details), details)

    def test_rejects_unsupported_character(self) -> None:
        details = {"players": [player(1, "Fox"), player(2, "Peach")]}
        with self.assertRaisesRegex(server_module.UploadError, "Unsupported: Peach") as raised:
            server_module.validate_slp(Path("unused"), lambda _: details)
        self.assertEqual(raised.exception.code, "unsupported_character")

    def test_rejects_invalid_player_shape(self) -> None:
        with self.assertRaises(server_module.UploadError) as raised:
            server_module.validate_slp(Path("unused"), lambda _: {"players": [player(1, "Fox")]})
        self.assertEqual(raised.exception.code, "unsupported_player_count")

    def test_real_parser_rejects_malformed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "broken.slp"
            replay.write_bytes(b"this is not a Slippi replay")
            with self.assertRaises(server_module.UploadError) as raised:
                server_module.inspect_slp(replay)
        self.assertEqual(raised.exception.code, "malformed_slp")

    def test_public_clients_require_matching_token(self) -> None:
        handler = SimpleNamespace(
            client_address=("8.8.8.8", 1234),
            path="/?token=secret",
            headers={"Host": "reviews.example.test"},
        )
        self.assertTrue(server_module._request_is_allowed(handler, allow_public=True, access_token="secret"))
        handler.path = "/?token=wrong"
        self.assertFalse(server_module._request_is_allowed(handler, allow_public=True, access_token="secret"))


class UploadEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.uploads = root / "uploads"
        def inspector(_: Path) -> dict[str, object]:
            return {
                "players": [player(1, "Fox"), player(2, "Sheik")],
                "match": {"stageId": 31, "slpVersion": "3.19.0", "lastFrame": 600},
            }
        self.training_calls: list[tuple[str, int, int, str]] = []

        def training_launcher(_uploads: Path, review_id: str, target_index: int, alternative_index: int, scenario_mode: str) -> dict[str, object]:
            self.training_calls.append((review_id, target_index, alternative_index, scenario_mode))
            return {"ok": True, "scenarioMode": scenario_mode, "takeoverFrame": 420, "humanPort": 2}

        handler = server_module.make_handler(
            root,
            root,
            self.uploads,
            max_upload_bytes=16,
            inspector=inspector,
            training_launcher=training_launcher,
        )
        self.server = server_module.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, body: bytes, filename: str = "game.slp", **headers: str) -> tuple[int, dict[str, object]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        request_headers = {
            "Content-Type": "application/octet-stream",
            "X-Slp-Filename": filename,
            **headers,
        }
        connection.request("POST", server_module.UPLOAD_ENDPOINT, body=body, headers=request_headers)
        response = connection.getresponse()
        payload = json.loads(response.read())
        connection.close()
        return response.status, payload

    def get(self, path: str) -> tuple[int, str, str]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        content_type = response.getheader("Content-Type") or ""
        connection.close()
        return response.status, content_type, body

    def post_json(self, path: str, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        body = json.loads(response.read())
        connection.close()
        return response.status, body

    def test_accepts_and_stores_with_structured_players(self) -> None:
        status, payload = self.request(b"valid replay")
        self.assertEqual(status, 202)
        self.assertTrue(payload["ok"])
        self.assertEqual([item["character"]["name"] for item in payload["players"]], ["Fox", "Sheik"])
        review_id = payload["review"]["reviewId"]
        self.assertEqual(server_module._review_id(review_id), review_id)
        review_dir = self.uploads / review_id
        self.assertEqual((review_dir / "replay.slp").read_bytes(), b"valid replay")
        self.assertTrue((review_dir / "review.json").is_file())
        self.assertEqual(payload["review"]["status"], "awaiting_target")

        api_status, content_type, body = self.get(payload["review"]["urls"]["status"])
        self.assertEqual(api_status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(json.loads(body)["review"]["reviewId"], review_id)

        page_status, content_type, body = self.get(payload["review"]["urls"]["review"])
        self.assertEqual(page_status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn(review_id[:8], body)

        selection_status, selection = self.post_json(
            f"/api/reviews/{review_id}/target",
            {"playerIndex": 1},
        )
        self.assertEqual(selection_status, 200)
        self.assertEqual(selection["review"]["status"], "queued")
        self.assertEqual(selection["review"]["targetPlayer"]["character"]["name"], "Sheik")
        duplicate_status, duplicate = self.post_json(
            f"/api/reviews/{review_id}/target",
            {"playerIndex": 0},
        )
        self.assertEqual(duplicate_status, 409)
        self.assertEqual(duplicate["error"]["code"], "invalid_player")

    def test_multiple_uploads_get_isolated_review_ids(self) -> None:
        first_status, first = self.request(b"first replay", "first.slp")
        second_status, second = self.request(b"second replay", "second.slp")
        self.assertEqual((first_status, second_status), (202, 202))
        first_id = first["review"]["reviewId"]
        second_id = second["review"]["reviewId"]
        self.assertNotEqual(first_id, second_id)
        self.assertEqual((self.uploads / first_id / "replay.slp").read_bytes(), b"first replay")
        self.assertEqual((self.uploads / second_id / "replay.slp").read_bytes(), b"second replay")

    def test_legacy_queue_without_target_returns_to_selection(self) -> None:
        _, payload = self.request(b"valid replay")
        review_id = payload["review"]["reviewId"]
        review_path = self.uploads / review_id / "review.json"
        legacy = json.loads(review_path.read_text(encoding="utf-8"))
        legacy["schemaVersion"] = 1
        legacy["status"] = "queued"
        legacy.pop("targetPlayer", None)
        review_path.write_text(json.dumps(legacy), encoding="utf-8")
        migrated = server_module._load_review(self.uploads, review_id)
        self.assertEqual(migrated["schemaVersion"], 3)
        self.assertEqual(migrated["status"], "awaiting_target")
        self.assertEqual(migrated["urls"]["report"], f"/reviews/{review_id}/report")

    def test_duplicate_hash_and_quality_preset_are_recorded(self) -> None:
        _, first = self.request(b"same replay", "first.slp")
        _, second = self.request(b"same replay", "second.slp")
        self.assertEqual(second["duplicates"][0]["reviewId"], first["review"]["reviewId"])
        review_id = second["review"]["reviewId"]
        status, selection = self.post_json(f"/api/reviews/{review_id}/target", {"playerIndex": 0, "qualityPreset": "deep"})
        self.assertEqual(status, 200)
        self.assertEqual(selection["review"]["settings"]["qualityPreset"], "deep")

    def test_cancel_retry_archive_and_delete_lifecycle(self) -> None:
        _, payload = self.request(b"lifecycle")
        review_id = payload["review"]["reviewId"]
        self.post_json(f"/api/reviews/{review_id}/target", {"playerIndex": 0})
        status, cancelled = self.post_json(f"/api/reviews/{review_id}/actions/cancel", {})
        self.assertEqual(status, 200)
        self.assertEqual(cancelled["review"]["status"], "cancelled")
        status, retried = self.post_json(f"/api/reviews/{review_id}/actions/retry", {})
        self.assertEqual(status, 200)
        self.assertEqual(retried["review"]["status"], "queued")

    def test_review_status_lifecycle_and_report_url(self) -> None:
        _, payload = self.request(b"valid replay")
        review_id = payload["review"]["reviewId"]
        server_module.select_review_target(self.uploads, review_id, 0)
        updated = server_module.update_review(
            self.uploads,
            review_id,
            "processing",
            message="Simulating candidates.",
        )
        self.assertEqual(updated["status"], "processing")
        artifacts = self.uploads / review_id / "artifacts"
        artifacts.mkdir()
        (artifacts / "advantage_review.html").write_text("review", encoding="utf-8")
        completed = server_module.update_review(
            self.uploads,
            review_id,
            "complete",
            message="Analysis complete.",
            report="advantage_review.html",
        )
        self.assertEqual(completed["urls"]["report"], f"/reviews/{review_id}/report")
        self.assertEqual(completed["urls"]["artifactReport"], f"/review-artifacts/{review_id}/advantage_review.html")
        permalink_status, _, _ = self.get(completed["urls"]["report"])
        self.assertEqual(permalink_status, 302)
        report_status, content_type, body = self.get(completed["urls"]["artifactReport"])
        self.assertEqual(report_status, 200)
        self.assertIn("text/html", content_type)
        self.assertEqual(body, "review")

    def test_dashboard_lists_uuid_reviews(self) -> None:
        _, payload = self.request(b"valid replay")
        dashboard_status, content_type, body = self.get("/")
        self.assertEqual(dashboard_status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn("Replay review queue", body)
        api_status, _, body = self.get("/api/reviews")
        self.assertEqual(api_status, 200)
        reviews = json.loads(body)["reviews"]
        self.assertEqual(reviews[0]["reviewId"], payload["review"]["reviewId"])

    def test_training_mode_endpoint_launches_selected_route_only_after_completion(self) -> None:
        _, payload = self.request(b"valid replay")
        review_id = payload["review"]["reviewId"]
        pending_status, pending = self.post_json(
            f"/api/reviews/{review_id}/training-mode",
            {"targetIndex": 3, "alternativeIndex": 1, "scenarioMode": "replay"},
        )
        self.assertEqual(pending_status, 409)
        self.assertEqual(pending["error"]["code"], "review_not_ready")

        artifacts = self.uploads / review_id / "artifacts"
        artifacts.mkdir()
        (artifacts / "advantage_review.html").write_text("review", encoding="utf-8")
        server_module.update_review(
            self.uploads,
            review_id,
            "complete",
            message="Ready",
            report="advantage_review.html",
        )
        status, response = self.post_json(
            f"/api/reviews/{review_id}/training-mode",
            {"targetIndex": 3, "alternativeIndex": 1, "scenarioMode": "replay"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(response["scenario"]["takeoverFrame"], 420)
        self.assertEqual(response["scenario"]["scenarioMode"], "replay")
        self.assertEqual(self.training_calls, [(review_id, 3, 1, "replay")])

    def test_training_mode_endpoint_rejects_boolean_indices(self) -> None:
        _, payload = self.request(b"valid replay")
        review_id = payload["review"]["reviewId"]
        artifacts = self.uploads / review_id / "artifacts"
        artifacts.mkdir()
        (artifacts / "advantage_review.html").write_text("review", encoding="utf-8")
        server_module.update_review(
            self.uploads,
            review_id,
            "complete",
            message="Ready",
            report="advantage_review.html",
        )
        status, response = self.post_json(
            f"/api/reviews/{review_id}/training-mode",
            {"targetIndex": True, "alternativeIndex": 0},
        )
        self.assertEqual(status, 400)
        self.assertEqual(response["error"]["code"], "invalid_training_route")
        self.assertEqual(self.training_calls, [])

    def test_rejects_wrong_extension_without_storage(self) -> None:
        status, payload = self.request(b"not slp", "game.txt")
        self.assertEqual(status, 415)
        self.assertEqual(payload["error"]["code"], "invalid_extension")
        self.assertFalse(self.uploads.exists())

    def test_rejects_oversized_without_storage(self) -> None:
        status, payload = self.request(b"x" * 17)
        self.assertEqual(status, 413)
        self.assertEqual(payload["error"]["code"], "file_too_large")
        self.assertFalse(self.uploads.exists())

    def test_rejected_validation_removes_temporary_file(self) -> None:
        def reject(_: Path) -> dict[str, object]:
            raise server_module.UploadError(422, "malformed_slp", "Unreadable replay.")

        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        root = Path(self.temp.name)
        handler = server_module.make_handler(root, root, self.uploads, inspector=reject)
        self.server = server_module.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        status, payload = self.request(b"bad replay")
        self.assertEqual(status, 422)
        self.assertEqual(payload["error"]["code"], "malformed_slp")
        self.assertEqual(list(self.uploads.iterdir()), [])

    def test_rejects_non_local_host(self) -> None:
        status, payload = self.request(b"valid replay", Host="example.test")
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"]["code"], "local_only")
        self.assertFalse(self.uploads.exists())

    def test_remote_mode_accepts_same_origin_tailscale_hostname(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        root = Path(self.temp.name)
        def inspector(_: Path) -> dict[str, object]:
            return {
                "players": [player(1, "Fox"), player(2, "Sheik")],
                "match": {"stageId": 31},
            }
        handler = server_module.make_handler(root, root, self.uploads, inspector=inspector, allow_remote=True)
        self.server = server_module.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        status, payload = self.request(
            b"valid replay",
            Host="melee-box.tailnet.ts.net:8877",
            Origin="http://melee-box.tailnet.ts.net:8877",
        )
        self.assertEqual(status, 202)
        self.assertEqual(payload["review"]["status"], "awaiting_target")


if __name__ == "__main__":
    unittest.main()
