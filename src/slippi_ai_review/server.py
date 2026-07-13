"""Serve an advantage HTML report with its interactive MSL comparison viewer."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import webbrowser
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from .config import load_settings
from .paths import JS_DIR, WEB_DIR, module_command


WORK_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MSL_STATIC = WEB_DIR / "msl_static"
DEFAULT_UPLOAD_DIR = load_settings().data_dir / "reviews"
DEFAULT_MAX_UPLOAD_BYTES = 64 * 1024 * 1024
SLP_INSPECTOR = JS_DIR / "inspect_slp_upload.js"
UPLOAD_ENDPOINT = "/api/slp-upload"
UPLOAD_PANEL_PATH = "/advantage-review-static/slp_upload_panel.js"
SUPPORTED_CHARACTERS = frozenset({"Fox", "Falco", "Marth", "Sheik"})
REVIEW_SCHEMA_VERSION = 3
REVIEW_STATES = frozenset({"awaiting_target", "queued", "processing", "complete", "failed", "cancelled", "archived"})
QUALITY_PRESETS = {
    "quick": {"label": "Quick", "preflightSamples": 16, "refinementSamples": 32, "relativeCost": 0.25},
    "standard": {"label": "Standard", "preflightSamples": 64, "refinementSamples": 128, "relativeCost": 1.0},
    "deep": {"label": "Deep", "preflightSamples": 128, "refinementSamples": 256, "relativeCost": 2.0},
}
TRAINING_MODE_LOCK = threading.Lock()


class UploadError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional pre-generated example report")
    parser.add_argument("--msl-static", type=Path, default=DEFAULT_MSL_STATIC)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--max-upload-mib", type=int, default=DEFAULT_MAX_UPLOAD_BYTES // (1024 * 1024))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-remote", action="store_true", help="Allow private-LAN and Tailscale clients")
    parser.add_argument("--allow-public", action="store_true", help="Allow public clients; requires a token")
    parser.add_argument("--access-token", default=os.environ.get("ADVANTAGE_REVIEW_TOKEN"), help=argparse.SUPPRESS)
    parser.add_argument("--start-worker", action=argparse.BooleanOptionalAction, default=True, help="Start the persistent analysis queue worker")
    parser.add_argument("--worker-poll-seconds", type=float, default=5.0)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def safe_file(root: Path, raw_path: str) -> Path | None:
    relative = unquote(raw_path).lstrip("/")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def inspect_slp(path: Path) -> dict[str, object]:
    if not SLP_INSPECTOR.is_file():
        raise UploadError(500, "parser_unavailable", "The local Slippi parser is not available.")
    try:
        result = subprocess.run(
            ["node", str(SLP_INSPECTOR), str(path)],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise UploadError(500, "parser_unavailable", "The local Slippi parser could not be run.") from exc
    if result.returncode != 0:
        raise UploadError(422, "malformed_slp", "The file is not a readable Slippi replay.")
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise UploadError(500, "parser_error", "The local Slippi parser returned an invalid result.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("players"), list):
        raise UploadError(422, "malformed_slp", "The replay does not contain valid player settings.")
    return payload


def validate_slp(path: Path, inspector: Callable[[Path], dict[str, object]] = inspect_slp) -> dict[str, object]:
    payload = inspector(path)
    players = payload.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise UploadError(422, "unsupported_player_count", "The replay must contain exactly two players.")
    characters = []
    for player in players:
        if not isinstance(player, dict):
            raise UploadError(422, "malformed_slp", "The replay contains invalid player settings.")
        character = player.get("character")
        name = character.get("name") if isinstance(character, dict) else None
        if not isinstance(name, str):
            raise UploadError(422, "malformed_slp", "A player character could not be identified.")
        characters.append(name)
    unsupported = sorted(set(characters) - SUPPORTED_CHARACTERS)
    if unsupported:
        names = ", ".join(unsupported)
        raise UploadError(
            422,
            "unsupported_character",
            f"Both players must use Fox, Falco, Marth, or Sheik. Unsupported: {names}.",
        )
    return payload


def _trusted_client_address(raw_address: str) -> bool:
    try:
        address = ipaddress.ip_address(raw_address)
    except ValueError:
        return False
    tailscale = ipaddress.ip_network("100.64.0.0/10")
    return address.is_loopback or address.is_private or address.is_link_local or address in tailscale


def _request_is_allowed(
    handler: SimpleHTTPRequestHandler,
    allow_remote: bool = False,
    allow_public: bool = False,
    access_token: str | None = None,
) -> bool:
    client = handler.client_address[0]
    try:
        is_loopback = ipaddress.ip_address(client).is_loopback
    except ValueError:
        return False
    trusted_remote = _trusted_client_address(client)
    if not is_loopback and not (allow_remote and trusted_remote) and not allow_public:
        return False
    raw_host = handler.headers.get("Host", "")
    host = (urlsplit(f"//{raw_host}").hostname or "").lower()
    if not allow_remote and not allow_public and host not in {"localhost", "127.0.0.1", "::1"}:
        return False
    origin = handler.headers.get("Origin")
    if origin:
        origin_host = (urlsplit(origin).hostname or "").lower()
        if (not allow_remote and origin_host not in {"localhost", "127.0.0.1", "::1"}) or (
            (allow_remote or allow_public) and origin_host != host
        ):
            return False
    if allow_public and not (is_loopback or trusted_remote):
        if not access_token:
            return False
        query = dict(part.split("=", 1) for part in urlsplit(handler.path).query.split("&") if "=" in part)
        authorization = handler.headers.get("Authorization", "")
        bearer = authorization[7:] if authorization.startswith("Bearer ") else None
        cookies = dict(part.strip().split("=", 1) for part in handler.headers.get("Cookie", "").split(";") if "=" in part)
        supplied = query.get("token") or bearer or cookies.get("advantage_review_token")
        if not supplied or not hmac.compare_digest(supplied, access_token):
            return False
    return True


def _upload_filename(headers: object) -> str:
    encoded = headers.get("X-Slp-Filename", "")  # type: ignore[attr-defined]
    try:
        filename = unquote(encoded, errors="strict")
    except UnicodeError as exc:
        raise UploadError(400, "invalid_filename", "The upload filename is invalid.") from exc
    if not filename or filename in {".", ".."} or any(value in filename for value in ("/", "\\", "\0")):
        raise UploadError(400, "invalid_filename", "Provide a plain .slp filename without a path.")
    if Path(filename).suffix.lower() != ".slp":
        raise UploadError(415, "invalid_extension", "Only .slp replay files are accepted.")
    return filename


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _review_id(raw_value: str) -> str | None:
    try:
        value = str(uuid.UUID(raw_value))
    except (ValueError, AttributeError):
        return None
    return value if value == raw_value.lower() else None


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_review(upload_dir: Path, review_id: str) -> dict[str, object] | None:
    validated = _review_id(review_id)
    if validated is None:
        return None
    path = upload_dir / validated / "review.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    schema_version = int(payload.get("schemaVersion") or 1)
    if schema_version < 2:
        review_id = str(payload.get("reviewId") or review_id)
        if payload.get("status") == "queued" and not payload.get("targetPlayer"):
            payload["status"] = "awaiting_target"
            payload["message"] = "Select the player to analyze."
        urls = payload.get("urls") if isinstance(payload.get("urls"), dict) else {}
        old_report = urls.get("report")
        urls.update({
            "review": f"/reviews/{review_id}/",
            "status": f"/api/reviews/{review_id}",
            "artifacts": f"/review-artifacts/{review_id}/",
            "report": f"/reviews/{review_id}/report",
            "artifactReport": old_report if isinstance(old_report, str) and old_report.startswith("/review-artifacts/") else None,
        })
        payload["urls"] = urls
    if schema_version < 3:
        payload.setdefault("settings", {"qualityPreset": "standard"})
        payload.setdefault("artifacts", {
            "review": {"status": "complete" if payload.get("status") == "complete" else "pending"},
            "cePractice": {"status": "available_after_review", "mode": "per_clip"},
        })
        payload["schemaVersion"] = REVIEW_SCHEMA_VERSION
        payload["updatedAt"] = _utc_now()
        _write_json_atomic(path, payload)
    stable_report = f"/reviews/{review_id}/report"
    urls = payload.get("urls") if isinstance(payload.get("urls"), dict) else {}
    if urls.get("report") != stable_report:
        old_report = urls.get("report")
        if "artifactReport" not in urls:
            urls["artifactReport"] = old_report if isinstance(old_report, str) and old_report.startswith("/review-artifacts/") else None
        urls["report"] = stable_report
        payload["urls"] = urls
        payload["updatedAt"] = _utc_now()
        _write_json_atomic(path, payload)
    return payload


def launch_training_mode_scenario(
    upload_dir: Path,
    review_id: str,
    target_index: int,
    alternative_index: int,
    scenario_mode: str = "phillip",
) -> dict[str, Any]:
    validated = _review_id(review_id)
    if validated is None:
        raise UploadError(404, "review_not_found", "Review not found.")
    if scenario_mode not in {"replay", "phillip"}:
        raise UploadError(400, "invalid_training_mode", "Select replay or Phillip practice.")
    review_dir = upload_dir / validated
    queue_path = review_dir / "pipeline" / "render_queue.json"
    if not queue_path.is_file():
        raise UploadError(409, "training_unavailable", "This review has no Training Mode export queue.")
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8"))
        targets = queue.get("targets") or []
        if target_index < 1 or target_index > len(targets):
            raise ValueError("The selected insertion is not in this review.")
        alternatives = targets[target_index - 1].get("alternative_routes") or []
        if alternative_index < 0 or alternative_index > len(alternatives):
            raise ValueError("The selected route is not in this review.")
    except (OSError, json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise UploadError(409, "training_unavailable", "The Training Mode queue is unreadable.") from exc
    except ValueError as exc:
        raise UploadError(409, "invalid_training_route", str(exc)) from exc

    command = module_command(
        "training",
        "--queue-json",
        str(queue_path),
        "--target-index",
        str(target_index),
        "--alternative-index",
        str(alternative_index),
        "--scenario-mode",
        scenario_mode,
        "--launch",
    )
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with TRAINING_MODE_LOCK:
        try:
            result = subprocess.run(
                command,
                cwd=WORK_DIR,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
                creationflags=creationflags,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise UploadError(500, "training_export_failed", "Training Mode export could not be started.") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        message = detail[-1] if detail else "Training Mode export failed."
        raise UploadError(500, "training_export_failed", message[:500])
    try:
        scenario = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise UploadError(500, "training_export_failed", "Training Mode export returned an invalid result.") from exc
    if not isinstance(scenario, dict) or not scenario.get("ok"):
        raise UploadError(500, "training_export_failed", "Training Mode export did not complete.")
    return scenario


def _list_reviews(upload_dir: Path) -> list[dict[str, object]]:
    if not upload_dir.is_dir():
        return []
    reviews = []
    for child in upload_dir.iterdir():
        if not child.is_dir() or _review_id(child.name) is None:
            continue
        payload = _load_review(upload_dir, child.name)
        if payload is not None:
            reviews.append(payload)
    reviews.sort(key=lambda item: str(item.get("createdAt") or ""), reverse=True)
    return reviews


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _review_metrics(upload_dir: Path, reviews: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    completed = []
    for review in reviews:
        timing = review.get("timing") if isinstance(review.get("timing"), dict) else {}
        seconds = timing.get("processingSeconds")
        if isinstance(seconds, (int, float)) and seconds > 0:
            preset = ((review.get("settings") or {}).get("qualityPreset") if isinstance(review.get("settings"), dict) else None) or "standard"
            completed.append(float(seconds) / float(QUALITY_PRESETS.get(str(preset), QUALITY_PRESETS["standard"])["relativeCost"]))
    standard_seconds = sum(completed[-10:]) / len(completed[-10:]) if completed else 20 * 60.0
    queued = sorted((r for r in reviews if r.get("status") == "queued"), key=lambda r: str(r.get("createdAt") or ""))
    elapsed = 0.0
    queue_meta: dict[str, tuple[int, int]] = {}
    for position, review in enumerate(queued, 1):
        preset = str(((review.get("settings") or {}).get("qualityPreset") if isinstance(review.get("settings"), dict) else None) or "standard")
        elapsed += standard_seconds * float(QUALITY_PRESETS.get(preset, QUALITY_PRESETS["standard"])["relativeCost"])
        queue_meta[str(review.get("reviewId"))] = (position, round(elapsed))
    decorated = []
    total_bytes = 0
    for review in reviews:
        item = dict(review)
        size = _directory_size(upload_dir / str(review.get("reviewId")))
        total_bytes += size
        item["storageBytes"] = size
        if str(review.get("reviewId")) in queue_meta:
            position, eta = queue_meta[str(review.get("reviewId"))]
            item["queue"] = {"position": position, "estimatedSeconds": eta}
        decorated.append(item)
    return decorated, {"totalBytes": total_bytes, "reviewCount": len(reviews), "estimatedStandardSeconds": round(standard_seconds)}


def _find_duplicates(upload_dir: Path, digest: str) -> list[dict[str, object]]:
    matches = []
    for review in _list_reviews(upload_dir):
        replay = review.get("replay") if isinstance(review.get("replay"), dict) else {}
        if replay.get("sha256") == digest:
            matches.append({
                "reviewId": review.get("reviewId"),
                "status": review.get("status"),
                "targetPlayer": review.get("targetPlayer"),
                "qualityPreset": ((review.get("settings") or {}).get("qualityPreset") if isinstance(review.get("settings"), dict) else "standard"),
                "url": (review.get("urls") or {}).get("review") if isinstance(review.get("urls"), dict) else None,
            })
    return matches


def _read_json_body(handler: SimpleHTTPRequestHandler, *, limit: int = 4096) -> dict[str, object]:
    if handler.headers.get_content_type() != "application/json":
        raise UploadError(415, "invalid_content_type", "Send this request as JSON.")
    try:
        length = int(handler.headers.get("Content-Length") or "")
    except ValueError as exc:
        raise UploadError(411, "length_required", "A valid Content-Length header is required.") from exc
    if length <= 0 or length > limit:
        raise UploadError(400, "invalid_request", "The request body is invalid.")
    try:
        body = json.loads(handler.rfile.read(length))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise UploadError(400, "invalid_json", "The request is not valid JSON.") from exc
    if not isinstance(body, dict):
        raise UploadError(400, "invalid_json", "The request must be a JSON object.")
    return body


def manage_review(upload_dir: Path, review_id: str, action: str) -> dict[str, object] | None:
    review = _load_review(upload_dir, review_id)
    if review is None:
        raise UploadError(404, "review_not_found", "Review not found.")
    status = str(review.get("status"))
    if action == "cancel":
        if status not in {"queued", "processing"}:
            raise UploadError(409, "invalid_state", "Only queued or processing reviews can be cancelled.")
        pipeline_pid = int(((review.get("worker") or {}).get("pipelinePid") if isinstance(review.get("worker"), dict) else 0) or 0)
        if pipeline_pid and _process_is_alive(pipeline_pid):
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pipeline_pid), "/T", "/F"], capture_output=True, check=False)
            else:
                os.kill(pipeline_pid, 15)
        return update_review(upload_dir, review_id, "cancelled", message="Analysis cancelled. Completed stages are retained for retry.")
    if action == "retry":
        if status not in {"failed", "cancelled", "complete", "archived"}:
            raise UploadError(409, "invalid_state", "This review cannot be retried yet.")
        review.pop("error", None)
        review["status"] = "queued"
        review["message"] = "Waiting to resume analysis; completed stages will be reused."
        review["updatedAt"] = _utc_now()
        _write_json_atomic(upload_dir / review_id / "review.json", review)
        return review
    if action == "archive":
        if status in {"queued", "processing", "awaiting_target"}:
            raise UploadError(409, "invalid_state", "Finish or cancel this review before archiving it.")
        return update_review(upload_dir, review_id, "archived", message="Review archived.")
    if action == "delete":
        if status == "processing":
            raise UploadError(409, "invalid_state", "Cancel this review before deleting it.")
        review_dir = (upload_dir / review_id).resolve()
        review_dir.relative_to(upload_dir.resolve())
        shutil.rmtree(review_dir)
        return None
    raise UploadError(400, "invalid_action", "Unknown review action.")


def cleanup_review_intermediates(upload_dir: Path, older_than_days: int = 30) -> dict[str, object]:
    if older_than_days < 0 or older_than_days > 3650:
        raise UploadError(400, "invalid_retention", "Retention must be between 0 and 3650 days.")
    cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
    reclaimed = 0
    cleaned = []
    for review in _list_reviews(upload_dir):
        if review.get("status") not in {"archived", "failed", "cancelled"}:
            continue
        review_dir = upload_dir / str(review.get("reviewId"))
        pipeline = review_dir / "pipeline"
        try:
            updated = datetime.fromisoformat(str(review.get("updatedAt")).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if updated > cutoff or not pipeline.is_dir():
            continue
        size = _directory_size(pipeline)
        shutil.rmtree(pipeline)
        reclaimed += size
        cleaned.append(str(review.get("reviewId")))
    return {"cleanedReviewIds": cleaned, "reclaimedBytes": reclaimed, "olderThanDays": older_than_days}


def _worker_state(upload_dir: Path) -> dict[str, object]:
    path = upload_dir / "worker.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"online": False, "status": "offline", "message": "Worker is not running"}
    pid = int(payload.get("pid") or 0) if isinstance(payload, dict) else 0
    online = _process_is_alive(pid)
    return {**payload, "online": online, "status": payload.get("status") if online else "offline"}


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _dashboard_page(example_report_url: str | None = None) -> bytes:
    example_link = (
        f'<a class="example-link" href="{html.escape(example_report_url)}">Open example report</a>'
        if example_report_url
        else ""
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Slippi analysis</title><style>
:root{{color-scheme:dark;--ink:#e8ece9;--muted:#9ba59f;--line:#343b37;--paper:#111311;--white:#191c19;--raised:#20241f;--green:#65d69a;--red:#ff7b75;--amber:#f0b35a}}
*{{box-sizing:border-box}}[hidden]{{display:none!important}}body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.45 Inter,Segoe UI,Arial,sans-serif}}
a{{color:inherit}}header{{display:flex;align-items:center;justify-content:space-between;gap:16px;min-height:62px;padding:13px 22px;border-bottom:3px solid #4cbd82;background:#0b0d0c}}h1{{margin:0;font-size:19px}}header p{{margin:2px 0 0;color:var(--muted);font-size:12px}}
.example-link{{padding:7px 9px;border:1px solid var(--line);border-radius:4px;text-decoration:none;font-size:12px;font-weight:700}}
main{{width:min(1240px,calc(100% - 28px));margin:30px auto 70px}}.upload-band{{display:grid;grid-template-columns:220px minmax(0,1fr);gap:24px;align-items:start;padding:18px 0 24px;border-bottom:1px solid var(--line)}}
h2{{margin:0;font-size:16px}}.section-copy{{margin:4px 0 0;color:var(--muted);font-size:12px}}#slp-upload-mount{{max-width:620px}}
.slp-upload-panel{{display:grid;grid-template-columns:minmax(0,1fr) 150px;gap:10px}}.slp-upload-panel__drop{{grid-column:1/-1;display:grid;place-items:center;min-height:108px;padding:18px;border:1px dashed #566059;border-radius:6px;background:var(--raised);cursor:pointer}}.slp-upload-panel__drop[data-dragging=true]{{border-color:var(--green);background:#17261e}}.slp-upload-panel__drop span{{color:var(--muted);font-size:12px}}.slp-upload-panel__input{{position:absolute;width:1px;height:1px;overflow:hidden;opacity:0}}
.slp-upload-panel__quality{{display:flex;margin:0;padding:0;border:0}}.slp-upload-panel__quality legend{{position:absolute;width:1px;height:1px;overflow:hidden}}.slp-upload-panel__quality label{{cursor:pointer}}.slp-upload-panel__quality input{{position:absolute;opacity:0}}.slp-upload-panel__quality span{{display:block;padding:9px 12px;border:1px solid #566059}}.slp-upload-panel__quality label:first-of-type span{{border-radius:4px 0 0 4px}}.slp-upload-panel__quality label:last-of-type span{{border-radius:0 4px 4px 0}}.slp-upload-panel__quality input:checked+span{{color:#07130d;background:var(--green);border-color:#477b60;font-weight:800}}.slp-upload-panel__submit{{min-height:40px;border:1px solid #477b60;border-radius:4px;color:#07130d;background:var(--green);font-weight:800;cursor:pointer}}.slp-upload-panel__progress{{grid-column:1/-1;width:100%}}
.slp-upload-panel__status,.slp-upload-panel__target{{grid-column:1/-1}}.slp-upload-panel__status{{min-height:20px;color:var(--muted);font-size:12px}}.slp-upload-panel__status[data-state="validated"]{{color:var(--amber)}}.slp-upload-panel__status[data-state="accepted"]{{color:var(--green)}}.slp-upload-panel__status[data-state="error"]{{color:var(--red)}}
.slp-upload-panel__target{{display:flex;flex-wrap:wrap;gap:8px}}.slp-upload-panel__target button{{min-height:36px;padding:7px 10px;border:1px solid #477b60;border-radius:4px;color:var(--ink);background:var(--raised);font-weight:700;cursor:pointer}}.slp-upload-panel__review-link{{display:inline-block;margin-top:4px;font-weight:700}}
.reviews-head{{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-top:28px;padding-bottom:10px}}.reviews-meta,.review-tools{{display:flex;align-items:center;gap:9px;flex-wrap:wrap}}.review-tools{{margin-bottom:10px}}.review-tools input,.review-tools select,.review-tools button{{min-height:36px;padding:7px 9px;border:1px solid #566059;border-radius:4px;color:var(--ink);background:var(--raised)}}.review-tools button{{cursor:pointer;font-weight:700}}.review-tools input{{min-width:260px}}.count,#storage-state{{color:var(--muted);font-size:12px}}.worker-state{{padding:4px 7px;border:1px solid #566059;border-radius:4px;color:var(--muted);font-size:10px;font-weight:800;text-transform:uppercase}}.worker-state[data-online="true"]{{color:var(--green)}}
.review-list{{border-top:1px solid var(--line)}}.review-row{{display:grid;grid-template-columns:minmax(180px,1.4fr) 150px minmax(170px,1fr) auto;gap:14px;align-items:center;padding:13px 8px;border-bottom:1px solid var(--line)}}
.matchup strong,.matchup span,.matchup small{{display:block}}.matchup span,.matchup small,.created{{color:var(--muted);font-size:11px}}.state{{justify-self:start;padding:4px 7px;border:1px solid #566059;border-radius:4px;font-size:10px;font-weight:800;text-transform:uppercase}}.state[data-state="queued"],.state[data-state="processing"]{{color:var(--amber)}}.state[data-state="complete"]{{color:var(--green)}}.state[data-state="failed"],.state[data-state="cancelled"]{{color:var(--red)}}
.actions{{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}}.actions a,.actions button{{min-height:32px;padding:6px 9px;border:1px solid #566059;border-radius:4px;color:var(--ink);background:var(--raised);font-size:11px;font-weight:700;text-decoration:none;cursor:pointer}}.empty{{padding:24px 8px;color:var(--muted)}}
@media(max-width:700px){{header{{padding:11px 14px}}.upload-band{{grid-template-columns:1fr;gap:12px}}.slp-upload-panel{{grid-template-columns:1fr}}.slp-upload-panel__submit{{grid-column:1}}.review-row{{grid-template-columns:1fr auto}}.created{{display:none}}.actions{{grid-column:1/-1}}}}
</style></head><body><header><div><h1>Slippi analysis</h1><p>Replay review queue</p></div>{example_link}</header><main>
<section class="upload-band"><div><h2>New analysis</h2><p class="section-copy">Upload, choose a player, then track processing by review ID.</p></div><div id="slp-upload-mount"></div></section>
<section><div class="reviews-head"><div><h2>Reviews</h2><p class="section-copy">Each review has a permanent UUID link.</p></div><div class="reviews-meta"><span class="worker-state" id="worker-state">Worker...</span><span id="storage-state"></span><span class="count" id="review-count"></span></div></div><div class="review-tools"><input id="review-search" type="search" placeholder="Search player, character, or replay"><select id="status-filter"><option value="all">All statuses</option><option value="awaiting_target">Needs player</option><option value="queued">Queued</option><option value="processing">Processing</option><option value="complete">Complete</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option><option value="archived">Archived</option></select><select id="age-filter"><option value="all">Any date</option><option value="1">Today</option><option value="7">Last 7 days</option><option value="30">Last 30 days</option></select><button id="cleanup-storage" type="button">Cleanup storage</button></div><div class="review-list" id="review-list"><div class="empty">Loading reviews...</div></div></section>
</main><script src="/advantage-review-static/slp_upload_panel.js"></script><script type="text/plain">
const list=document.querySelector('#review-list');const count=document.querySelector('#review-count');const workerState=document.querySelector('#worker-state');
function playerLabel(player){{const color=player.character?.colorName?` (${{player.character.colorName}})`:'';const identities=[...new Set([player.displayName,player.nametag,player.connectCode].filter(Boolean))];return `P${{player.port}} ${{player.character?.name||'Unknown'}}${{color}}${{identities.length?` · ${{identities.join(' · ')}}`:''}}`}}
function matchup(review){{return (review.players||[]).map(playerLabel).join(' vs ')}}
async function copyText(value){{if(navigator.clipboard?.writeText){{await navigator.clipboard.writeText(value);return}}const input=document.createElement('textarea');input.value=value;input.style.position='fixed';input.style.opacity='0';document.body.append(input);input.select();document.execCommand('copy');input.remove()}}
function render(reviews){{list.replaceChildren();count.textContent=`${{reviews.length}} total`;if(!reviews.length){{const empty=document.createElement('div');empty.className='empty';empty.textContent='No reviews yet.';list.append(empty);return}}
  for(const review of reviews){{const row=document.createElement('article');row.className='review-row';
    const target=review.targetPlayer?`Analyze ${{playerLabel(review.targetPlayer)}}`:'Choose analyzed player';
    row.innerHTML=`<div class="matchup"><strong></strong><span></span></div><span class="state"></span><span class="created"></span><div class="actions"><a>Open</a><button type="button">Copy link</button></div>`;
    row.querySelector('.matchup strong').textContent=matchup(review);row.querySelector('.matchup span').textContent=target;
    const state=row.querySelector('.state');state.textContent=review.status.replace('_',' ');state.dataset.state=review.status;
    row.querySelector('.created').textContent=new Date(review.createdAt).toLocaleString();const url=new URL(review.urls.review,location.href).href;
    row.querySelector('a').href=url;row.querySelector('button').addEventListener('click',async event=>{{await copyText(url);event.currentTarget.textContent='Copied';setTimeout(()=>event.currentTarget.textContent='Copy link',1200)}});list.append(row)}}}}
async function refresh(){{try{{const response=await fetch('/api/reviews',{{cache:'no-store'}});const body=await response.json();render(body.reviews||[]);workerState.textContent=`Worker ${{body.worker?.status||'offline'}}`;workerState.dataset.online=String(Boolean(body.worker?.online));workerState.title=body.worker?.message||''}}catch(_error){{list.textContent='Could not load reviews.'}}}}
SlpUploadPanel.mount(document.querySelector('#slp-upload-mount'),{{onAccepted:refresh,onValidated:refresh}});refresh();setInterval(refresh,5000);
</script><script src="/advantage-review-static/advantage_review_dashboard.js"></script></body></html>"""
    return document.encode("utf-8")


def _player_description(player: dict[str, object]) -> str:
    character = player.get("character") if isinstance(player.get("character"), dict) else {}
    color = f" ({character.get('colorName')})" if character.get("colorName") else ""
    identities = []
    for value in (player.get("displayName"), player.get("nametag"), player.get("connectCode")):
        if value and value not in identities:
            identities.append(str(value))
    identity = f" · {' · '.join(identities)}" if identities else ""
    return f"P{player.get('port') or '?'} {character.get('name') or 'Unknown'}{color}{identity}"


def update_review(
    upload_dir: Path,
    review_id: str,
    status: str,
    *,
    message: str | None = None,
    report: str | None = None,
) -> dict[str, object]:
    validated = _review_id(review_id)
    if validated is None or status not in REVIEW_STATES:
        raise ValueError("Invalid review ID or status")
    payload = _load_review(upload_dir.resolve(), validated)
    if payload is None:
        raise FileNotFoundError(f"Review not found: {validated}")
    payload["status"] = status
    payload["updatedAt"] = _utc_now()
    if message is not None:
        payload["message"] = message
    urls = payload.get("urls")
    if not isinstance(urls, dict):
        urls = {}
        payload["urls"] = urls
    if report is not None:
        relative = Path(report)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Report path must be relative to the review artifacts directory")
        report_path = upload_dir / validated / "artifacts" / relative
        if not report_path.is_file():
            raise FileNotFoundError(f"Review report not found: {report_path}")
        urls["artifactReport"] = f"/review-artifacts/{validated}/{relative.as_posix()}"
    if status == "complete" and not urls.get("artifactReport"):
        raise ValueError("A completed review must have a report")
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    review_artifact = artifacts.get("review") if isinstance(artifacts.get("review"), dict) else {}
    if status == "complete":
        review_artifact.update({"status": "complete", "url": urls.get("report")})
    elif status in {"failed", "cancelled"}:
        review_artifact["status"] = status
    elif status == "processing":
        review_artifact["status"] = "processing"
    artifacts["review"] = review_artifact
    artifacts.setdefault("cePractice", {"status": "available_after_review", "mode": "per_clip"})
    payload["artifacts"] = artifacts
    _write_json_atomic(upload_dir / validated / "review.json", payload)
    return payload


def select_review_target(upload_dir: Path, review_id: str, player_index: int, quality_preset: str = "standard") -> dict[str, object]:
    validated = _review_id(review_id)
    if validated is None:
        raise ValueError("Invalid review ID")
    payload = _load_review(upload_dir.resolve(), validated)
    if payload is None:
        raise FileNotFoundError(f"Review not found: {validated}")
    if payload.get("status") != "awaiting_target":
        raise ValueError("The analyzed player has already been selected")
    players = payload.get("players")
    if not isinstance(players, list):
        raise ValueError("Review players are unavailable")
    target = next(
        (player for player in players if isinstance(player, dict) and player.get("playerIndex") == player_index),
        None,
    )
    if target is None:
        raise ValueError("Selected player is not part of this replay")
    if quality_preset not in QUALITY_PRESETS:
        raise ValueError("Select Quick, Standard, or Deep quality")
    payload["targetPlayer"] = target
    settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    settings["qualityPreset"] = quality_preset
    payload["settings"] = settings
    payload["status"] = "queued"
    payload["message"] = f"Waiting to analyze {_player_description(target)}."
    payload["updatedAt"] = _utc_now()
    _write_json_atomic(upload_dir / validated / "review.json", payload)
    return payload


def _review_page(payload: dict[str, object]) -> bytes:
    review_id = html.escape(str(payload.get("reviewId") or ""))
    players = payload.get("players") if isinstance(payload.get("players"), list) else []
    matchup = " vs ".join(html.escape(_player_description(player)) for player in players if isinstance(player, dict))
    state = html.escape(str(payload.get("status") or "queued"))
    created = html.escape(str(payload.get("createdAt") or ""))
    target_buttons = "".join(
        f'<button type="button" data-player-index="{int(player.get("playerIndex") or 0)}">Analyze {html.escape(_player_description(player))}</button>'
        for player in players
        if isinstance(player, dict)
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replay review {review_id[:8]}</title><style>
:root{{color-scheme:dark;--ink:#e8ece9;--muted:#9ba59f;--line:#343b37;--paper:#111311;--raised:#20241f;--green:#65d69a;--red:#ff7b75}}
*{{box-sizing:border-box}}[hidden]{{display:none!important}}body{{margin:0;color:var(--ink);background:var(--paper);font:15px/1.45 Inter,Segoe UI,Arial,sans-serif}}
header{{display:flex;align-items:center;justify-content:space-between;gap:16px;min-height:58px;padding:13px 22px;border-bottom:3px solid #4cbd82;background:#0b0d0c}}h1{{margin:0;font-size:18px}}header a{{padding:7px 9px;border:1px solid var(--line);border-radius:4px;text-decoration:none;font-size:12px;font-weight:700}}
main{{width:min(720px,calc(100% - 28px));margin:54px auto}}.job{{border:1px solid var(--line);border-radius:6px;background:#191c19}}
.job-head{{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:16px;border-bottom:1px solid var(--line)}}
.job-body{{padding:16px}}.badge{{padding:5px 8px;border:1px solid #477b60;border-radius:4px;color:var(--green);font-size:11px;font-weight:800;text-transform:uppercase}}
.badge[data-state="failed"]{{color:var(--red);border-color:#80504d}}p{{margin:0 0 8px}}.muted{{color:var(--muted);font-size:12px}}
.target-picker{{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}}.target-picker button{{min-height:38px;padding:8px 11px;border:1px solid #477b60;border-radius:4px;color:#07130d;background:var(--green);font-weight:800;cursor:pointer}}
.review-link{{display:none;margin-top:16px;padding:9px 12px;border-radius:4px;color:#07130d;background:var(--green);font-weight:800;text-decoration:none}}
</style></head><body><header><h1>Replay analysis</h1><a href="/">Dashboard</a></header><main><section class="job">
<div class="job-head"><div><strong>{matchup}</strong><div class="muted">Review {review_id[:8]}</div></div><span class="badge" id="state" data-state="{state}">{state}</span></div>
<div class="job-body"><p id="message">Waiting for an analysis worker.</p><p class="muted">Created {created}</p><div class="target-picker" id="target-picker">{target_buttons}</div><a class="review-link" id="review-link">Open completed review</a></div>
</section></main><script>
const reviewId={json.dumps(str(payload.get('reviewId') or ''))};
async function refresh(){{
  try{{const response=await fetch(`/api/reviews/${{reviewId}}`,{{cache:'no-store'}});if(!response.ok)return;
    const body=await response.json();const review=body.review;const state=document.querySelector('#state');
    state.textContent=review.status;state.dataset.state=review.status;
    document.querySelector('#message').textContent=review.message||({{awaiting_target:'Select the player to analyze.',queued:'Waiting for an analysis worker.',processing:'Analyzing replay...',complete:'Analysis complete.',failed:'Analysis failed.'}}[review.status]||'Waiting.');
    document.querySelector('#target-picker').hidden=review.status!=='awaiting_target';
    if(review.status==='complete'&&review.urls?.report){{const link=document.querySelector('#review-link');link.href=review.urls.report;link.style.display='inline-block';}}
    if(!['complete','failed'].includes(review.status))setTimeout(refresh,2000);
  }}catch(_error){{setTimeout(refresh,3000)}}
}}refresh();
document.querySelector('#target-picker').addEventListener('click',async event=>{{
  const button=event.target.closest('button[data-player-index]');if(!button)return;
  document.querySelectorAll('#target-picker button').forEach(item=>item.disabled=true);
  await fetch(`/api/reviews/${{reviewId}}/target`,{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{playerIndex:Number(button.dataset.playerIndex)}})}});
  refresh();
}});
</script></body></html>"""
    return document.encode("utf-8")


def make_handler(
    root: Path,
    msl_static: Path,
    upload_dir: Path,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    inspector: Callable[[Path], dict[str, object]] = inspect_slp,
    example_report_url: str | None = None,
    allow_remote: bool = False,
    allow_public: bool = False,
    access_token: str | None = None,
    training_launcher: Callable[[Path, str, int, int, str], dict[str, Any]] = launch_training_mode_scenario,
) -> type[SimpleHTTPRequestHandler]:
    root = root.resolve()
    msl_static = msl_static.resolve()
    upload_dir = upload_dir.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            query = dict(part.split("=", 1) for part in parsed.query.split("&") if "=" in part)
            query_token = query.get("token")
            if allow_public and access_token and query_token and hmac.compare_digest(query_token, access_token):
                self.send_response(302)
                self.send_header("Set-Cookie", f"advantage_review_token={access_token}; HttpOnly; SameSite=Strict; Path=/")
                self.send_header("Location", parsed.path or "/")
                self.end_headers()
                return
            if not _request_is_allowed(self, allow_remote, allow_public, access_token):
                self.send_error(403, "Client is not allowed")
                return
            pathname = urlsplit(self.path).path
            if pathname == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            if pathname == "/":
                body = _dashboard_page(example_report_url)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if pathname == "/api/reviews":
                reviews, storage = _review_metrics(upload_dir, _list_reviews(upload_dir))
                self._send_json(200, {
                    "ok": True,
                    "reviews": reviews,
                    "worker": _worker_state(upload_dir),
                    "storage": storage,
                    "qualityPresets": QUALITY_PRESETS,
                })
                return
            log_match = re.fullmatch(r"/api/reviews/([0-9a-f-]+)/log", pathname)
            if log_match:
                review_id = _review_id(log_match.group(1))
                log_path = safe_file(upload_dir / str(review_id) / "logs", "worker_pipeline.log") if review_id else None
                if log_path is None or not log_path.is_file():
                    self._json_error(404, "log_not_found", "This review has no worker log yet.")
                else:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-250:]
                    self._send_json(200, {"ok": True, "lines": lines, "url": f"/api/reviews/{review_id}/log"})
                return
            api_match = re.fullmatch(r"/api/reviews/([0-9a-f-]+)", pathname)
            if api_match:
                review = _load_review(upload_dir, api_match.group(1))
                if review is None:
                    self._json_error(404, "review_not_found", "Review not found.")
                else:
                    self._send_json(200, {"ok": True, "review": review})
                return
            report_match = re.fullmatch(r"/reviews/([0-9a-f-]+)/report", pathname)
            if report_match:
                review = _load_review(upload_dir, report_match.group(1))
                if review is None:
                    self.send_error(404, "Review not found")
                    return
                urls = review.get("urls") if isinstance(review.get("urls"), dict) else {}
                destination = urls.get("artifactReport") if review.get("status") == "complete" else urls.get("review")
                if not isinstance(destination, str):
                    self.send_error(409, "Review report is not ready")
                    return
                self.send_response(302)
                self.send_header("Location", destination)
                self.end_headers()
                return
            page_match = re.fullmatch(r"/reviews/([0-9a-f-]+)/?", pathname)
            if page_match:
                review = _load_review(upload_dir, page_match.group(1))
                if review is None:
                    self.send_error(404, "Review not found")
                else:
                    body = _review_page(review)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self) -> None:
            pathname = urlsplit(self.path).path
            training_match = re.fullmatch(r"/api/reviews/([0-9a-f-]+)/training-mode", pathname)
            if training_match:
                try:
                    self._handle_training_mode(training_match.group(1))
                except UploadError as exc:
                    self._json_error(exc.status, exc.code, exc.message)
                except Exception:
                    self._json_error(500, "internal_error", "The Training Mode scenario could not be prepared.")
                return
            target_match = re.fullmatch(r"/api/reviews/([0-9a-f-]+)/target", pathname)
            if target_match:
                try:
                    self._handle_target_selection(target_match.group(1))
                except UploadError as exc:
                    self._json_error(exc.status, exc.code, exc.message)
                except Exception:
                    self._json_error(500, "internal_error", "The analyzed player could not be selected.")
                return
            action_match = re.fullmatch(r"/api/reviews/([0-9a-f-]+)/actions/(cancel|retry|archive|delete)", pathname)
            if action_match:
                try:
                    if not _request_is_allowed(self, allow_remote, allow_public, access_token):
                        raise UploadError(403, "local_only", "Review controls are unavailable from this client.")
                    result = manage_review(upload_dir, action_match.group(1), action_match.group(2))
                    self._send_json(200, {"ok": True, "review": result})
                except UploadError as exc:
                    self._json_error(exc.status, exc.code, exc.message)
                return
            if pathname == "/api/storage/cleanup":
                try:
                    if not _request_is_allowed(self, allow_remote, allow_public, access_token):
                        raise UploadError(403, "local_only", "Storage controls are unavailable from this client.")
                    body = _read_json_body(self)
                    days = body.get("olderThanDays", 30)
                    if not isinstance(days, int) or isinstance(days, bool):
                        raise UploadError(400, "invalid_retention", "Retention must be a number of days.")
                    self._send_json(200, {"ok": True, "cleanup": cleanup_review_intermediates(upload_dir, days)})
                except UploadError as exc:
                    self._json_error(exc.status, exc.code, exc.message)
                return
            if pathname != UPLOAD_ENDPOINT:
                self._json_error(404, "not_found", "Unknown API endpoint.")
                return
            try:
                self._handle_slp_upload()
            except UploadError as exc:
                self._json_error(exc.status, exc.code, exc.message)
            except Exception:
                self._json_error(500, "internal_error", "The upload could not be validated.")

        def _handle_training_mode(self, review_id: str) -> None:
            if not _request_is_allowed(self, allow_remote, allow_public, access_token):
                raise UploadError(403, "local_only", "Training Mode launch is not available from this client.")
            review = _load_review(upload_dir, review_id)
            if review is None:
                raise UploadError(404, "review_not_found", "Review not found.")
            if review.get("status") != "complete":
                raise UploadError(409, "review_not_ready", "Finish this review before exporting a scenario.")
            if self.headers.get_content_type() != "application/json":
                raise UploadError(415, "invalid_content_type", "Select the Training Mode route with JSON.")
            try:
                length = int(self.headers.get("Content-Length") or "")
            except ValueError as exc:
                raise UploadError(411, "length_required", "A valid Content-Length header is required.") from exc
            if length <= 0 or length > 4096:
                raise UploadError(400, "invalid_request", "The Training Mode request is invalid.")
            try:
                body = json.loads(self.rfile.read(length))
                target_index = body.get("targetIndex") if isinstance(body, dict) else None
                alternative_index = body.get("alternativeIndex", 0) if isinstance(body, dict) else None
                scenario_mode = body.get("scenarioMode", "phillip") if isinstance(body, dict) else None
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise UploadError(400, "invalid_json", "The Training Mode request is not valid JSON.") from exc
            if (
                not isinstance(target_index, int)
                or isinstance(target_index, bool)
                or not isinstance(alternative_index, int)
                or isinstance(alternative_index, bool)
                or scenario_mode not in {"replay", "phillip"}
            ):
                raise UploadError(400, "invalid_training_route", "Select a valid insertion and route.")
            scenario = training_launcher(upload_dir, review_id, target_index, alternative_index, str(scenario_mode))
            self._send_json(200, {"ok": True, "scenario": scenario})

        def _handle_target_selection(self, review_id: str) -> None:
            if not _request_is_allowed(self, allow_remote, allow_public, access_token):
                raise UploadError(403, "local_only", "Review selection is available only from this local site.")
            if self.headers.get_content_type() != "application/json":
                raise UploadError(415, "invalid_content_type", "Select the analyzed player with JSON.")
            try:
                length = int(self.headers.get("Content-Length") or "")
            except ValueError as exc:
                raise UploadError(411, "length_required", "A valid Content-Length header is required.") from exc
            if length <= 0 or length > 4096:
                raise UploadError(400, "invalid_request", "The player selection request is invalid.")
            try:
                body = json.loads(self.rfile.read(length))
                player_index = body.get("playerIndex") if isinstance(body, dict) else None
                quality_preset = body.get("qualityPreset", "standard") if isinstance(body, dict) else "standard"
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise UploadError(400, "invalid_json", "The player selection is not valid JSON.") from exc
            if not isinstance(player_index, int) or isinstance(player_index, bool):
                raise UploadError(400, "invalid_player", "Select one of the replay players.")
            if not isinstance(quality_preset, str) or quality_preset not in QUALITY_PRESETS:
                raise UploadError(400, "invalid_quality", "Select Quick, Standard, or Deep quality.")
            try:
                review = select_review_target(upload_dir, review_id, player_index, quality_preset)
            except FileNotFoundError as exc:
                raise UploadError(404, "review_not_found", "Review not found.") from exc
            except ValueError as exc:
                raise UploadError(409, "invalid_player", str(exc)) from exc
            self._send_json(200, {"ok": True, "review": review})

        def _handle_slp_upload(self) -> None:
            if not _request_is_allowed(self, allow_remote, allow_public, access_token):
                raise UploadError(403, "local_only", "Replay uploads are available only from this local site.")
            content_type = self.headers.get_content_type()
            if content_type != "application/octet-stream":
                raise UploadError(415, "invalid_content_type", "Upload the replay as application/octet-stream.")
            filename = _upload_filename(self.headers)
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length or "")
            except ValueError as exc:
                raise UploadError(411, "length_required", "A valid Content-Length header is required.") from exc
            if length <= 0:
                raise UploadError(400, "empty_file", "The replay file is empty.")
            if length > max_upload_bytes:
                raise UploadError(413, "file_too_large", f"Replay files may not exceed {max_upload_bytes} bytes.")

            upload_dir.mkdir(parents=True, exist_ok=True)
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(prefix=".slp-upload-", suffix=".tmp", dir=upload_dir, delete=False) as stream:
                    temporary = Path(stream.name)
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise UploadError(400, "incomplete_upload", "The replay upload ended before Content-Length bytes arrived.")
                        stream.write(chunk)
                        remaining -= len(chunk)

                details = validate_slp(temporary, inspector)
                replay_sha256 = hashlib.sha256(temporary.read_bytes()).hexdigest()
                duplicates = _find_duplicates(upload_dir, replay_sha256)
                review_id = str(uuid.uuid4())
                review_dir = upload_dir / review_id
                review_dir.mkdir()
                stored_name = "replay.slp"
                stored_path = review_dir / stored_name
                os.replace(temporary, stored_path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

            relative_path = stored_path.relative_to(WORK_DIR).as_posix() if stored_path.is_relative_to(WORK_DIR) else stored_name
            created_at = _utc_now()
            urls = {
                "review": f"/reviews/{review_id}/",
                "status": f"/api/reviews/{review_id}",
                "artifacts": f"/review-artifacts/{review_id}/",
                "report": f"/reviews/{review_id}/report",
                "artifactReport": None,
            }
            review = {
                "schemaVersion": REVIEW_SCHEMA_VERSION,
                "reviewId": review_id,
                "status": "awaiting_target",
                "message": "Select the player to analyze.",
                "createdAt": created_at,
                "updatedAt": created_at,
                "originalFilename": filename,
                "replay": {
                    "filename": stored_name,
                    "relativePath": relative_path,
                    "sizeBytes": length,
                    "sha256": replay_sha256,
                },
                "players": details["players"],
                "match": details.get("match", {}),
                "settings": {"qualityPreset": "standard"},
                "artifacts": {
                    "review": {"status": "pending"},
                    "cePractice": {"status": "available_after_review", "mode": "per_clip"},
                },
                "duplicates": duplicates,
                "urls": urls,
            }
            _write_json_atomic(review_dir / "review.json", review)
            response = {
                "ok": True,
                "upload": {
                    "id": review_id,
                    "originalFilename": filename,
                    "filename": stored_name,
                    "relativePath": relative_path,
                    "sizeBytes": length,
                },
                "review": review,
                "players": details["players"],
                "match": details.get("match", {}),
                "duplicates": duplicates,
                "qualityPresets": QUALITY_PRESETS,
            }
            self._send_json(202, response)

        def _json_error(self, status: int, code: str, message: str) -> None:
            self._send_json(status, {"ok": False, "error": {"code": code, "message": message}})

        def _send_json(self, status: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def translate_path(self, path: str) -> str:
            pathname = urlsplit(path).path
            if pathname == UPLOAD_PANEL_PATH:
                target = WEB_DIR / "slp_upload_panel.js"
            elif pathname == "/advantage-review-static/advantage_review_dashboard.js":
                target = WEB_DIR / "advantage_review_dashboard.js"
            elif pathname.startswith("/msl-static/"):
                target = safe_file(msl_static, pathname[len("/msl-static/"):])
            elif pathname.startswith("/review-artifacts/"):
                artifact_match = re.fullmatch(r"/review-artifacts/([0-9a-f-]+)/(.*)", pathname)
                review_id = _review_id(artifact_match.group(1)) if artifact_match else None
                target = safe_file(upload_dir / review_id / "artifacts", artifact_match.group(2)) if review_id and artifact_match else None
            else:
                target = safe_file(root, pathname)
            return str(target or root / "__forbidden__")

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
            super().end_headers()

        def log_message(self, format: str, *values: object) -> None:
            print(format % values)

        def copyfile(self, source: object, outputfile: object) -> None:
            try:
                super().copyfile(source, outputfile)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main() -> int:
    args = parse_args()
    report = args.report.resolve() if args.report else None
    msl_static = args.msl_static.resolve()
    upload_dir = args.upload_dir.resolve()
    if report is not None and not report.exists():
        raise SystemExit(f"Report not found: {report}")
    root = report.parent if report is not None else upload_dir
    if not (msl_static / "tools" / "viewer" / "slippi-viewer" / "dist" / "index.js").exists():
        raise SystemExit(f"MSL viewer assets not found: {msl_static}")
    if args.max_upload_mib <= 0:
        raise SystemExit("--max-upload-mib must be positive")
    if args.allow_public and not args.access_token:
        raise SystemExit("--allow-public requires --access-token or ADVANTAGE_REVIEW_TOKEN")
    mimetypes.add_type("application/javascript", ".js")
    upload_dir.mkdir(parents=True, exist_ok=True)
    example_report_url = f"/{report.name}" if report is not None else None
    handler = make_handler(
        root,
        msl_static,
        upload_dir,
        args.max_upload_mib * 1024 * 1024,
        example_report_url=example_report_url,
        allow_remote=args.allow_remote,
        allow_public=args.allow_public,
        access_token=args.access_token,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    worker_process: subprocess.Popen[bytes] | None = None
    if args.start_worker and not bool(_worker_state(upload_dir).get("online")):
        worker_command = [
            sys.executable,
            "-u",
            "-m",
            "slippi_ai_review.worker",
            "--upload-dir",
            str(upload_dir),
            "--poll-seconds",
            str(args.worker_poll_seconds),
        ]
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        worker_process = subprocess.Popen(
            worker_command,
            cwd=WORK_DIR,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    url = f"http://{args.host}:{server.server_port}/"
    print(url, flush=True)
    if not args.no_open:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if worker_process is not None and worker_process.poll() is None:
            worker_process.terminate()
            try:
                worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker_process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
