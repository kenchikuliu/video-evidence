from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_video import (  # noqa: E402
    Scene,
    evenly_spaced_indexes,
    evenly_spaced_times,
    find_new_lines,
    merge_sample_times,
    transcribe_audio,
)
from assemble_remake import build_ffmpeg_args as build_remake_ffmpeg_args  # noqa: E402
from edit_plan import build_edit_plan  # noqa: E402
from generate_remake import select_shots, validate_submission, write_run  # noqa: E402
from object_detection import DEFAULT_MODEL_PATH, postprocess, sample_times  # noqa: E402
from remake_spec import build_remake_spec  # noqa: E402
from render_edit import build_ffmpeg_args  # noqa: E402
from score_manifest import score_manifest  # noqa: E402
from semantic_review import build_evidence_index, validate_review  # noqa: E402
from video_providers import (  # noqa: E402
    build_payload,
    download_file,
    normalize_status,
    query_job,
    submit_job,
)


class SamplingTests(unittest.TestCase):
    def test_even_sampling(self) -> None:
        self.assertEqual(evenly_spaced_times(11.77, 5), [0.5, 3.192, 5.885, 8.578, 11.27])

    def test_short_video_midpoint(self) -> None:
        self.assertEqual(evenly_spaced_times(0.8, 5), [0.4])

    def test_scene_midpoints(self) -> None:
        result = merge_sample_times(
            10.0, [Scene(0.0, 4.0), Scene(4.0, 10.0)], 2, 10
        )
        self.assertEqual(result, [0.5, 2.0, 7.0, 9.5])

    def test_frame_cap_keeps_ends(self) -> None:
        self.assertEqual(evenly_spaced_indexes(10, 4), [0, 3, 6, 9])

    def test_object_sampling_covers_tail(self) -> None:
        self.assertEqual(sample_times(2.8, 1.0), [0.0, 1.0, 2.0, 2.75])


class EvidenceTests(unittest.TestCase):
    def test_bundled_model_path(self) -> None:
        self.assertEqual(DEFAULT_MODEL_PATH, ROOT / "models" / "yolo11n.onnx")
        self.assertTrue(DEFAULT_MODEL_PATH.is_file())

    def test_yolo_postprocess(self) -> None:
        output = np.zeros((1, 84, 1), dtype=np.float32)
        output[0, 0:4, 0] = [320.0, 320.0, 100.0, 200.0]
        output[0, 4, 0] = 0.9
        detections = postprocess(
            output, (640, 640), 1.0, (0, 0), 0.3, 0.45, ["person"]
        )
        self.assertEqual(detections[0]["label"], "person")
        self.assertEqual(detections[0]["box"], [270.0, 220.0, 370.0, 420.0])

    def test_new_lines_ignore_case_and_punctuation(self) -> None:
        result = find_new_lines(
            ["Run source is valid", "Exported final.video"],
            ["run-source IS valid!"],
        )
        self.assertEqual(result, ["Exported final.video"])

    def test_silent_audio_skips_model(self) -> None:
        result = transcribe_audio(
            Path("silent.mp4"),
            {"has_audio_stream": True, "silent": True},
            True,
            "base",
            "auto",
            "default",
        )
        self.assertEqual(result["status"], "skipped_silent_audio")

    def test_evidence_ids_are_stable(self) -> None:
        report = {
            "timeline": [
                {
                    "timestamp": 1.0,
                    "ocr_lines": [{"text": "Ready", "score": 0.99}],
                    "new_text_since_previous_sample": ["Ready"],
                }
            ],
            "transcript": {
                "segments": [{"start": 1.0, "end": 2.0, "text": "hello"}]
            },
        }
        evidence = build_evidence_index(report)
        self.assertEqual([item["id"] for item in evidence], ["ocr-000-000", "asr-000"])

    def test_validation_rejects_bad_citations(self) -> None:
        evidence = [
            {"id": "ocr-000-000", "source": "ocr", "timestamp": 1.0, "text": "Ready"}
        ]
        content = {
            "events": [
                {
                    "timestamp": 2.0,
                    "event": "Ready",
                    "evidence_ids": ["ocr-000-000", "unknown"],
                }
            ],
            "overall_evidence_ids": ["ocr-000-000"],
        }
        result = validate_review(content, evidence)
        self.assertFalse(result["passed"])
        self.assertEqual(result["invalid_evidence_ids"], ["unknown"])
        self.assertEqual(result["time_misaligned_evidence_ids"], ["ocr-000-000"])


class EditPlanTests(unittest.TestCase):
    def report(self) -> dict:
        return {
            "input": {"path": "input.mp4"},
            "media": {"duration_seconds": 10.0},
            "audio": {"has_audio_stream": False},
            "timeline": [{"timestamp": 1.0}, {"timestamp": 5.0}, {"timestamp": 9.0}],
            "evidence_index": [
                {
                    "id": "ocr-000-000",
                    "source": "ocr",
                    "timestamp": 1.0,
                    "text": "start",
                    "new_since_previous_sample": True,
                },
                {
                    "id": "ocr-002-000",
                    "source": "ocr",
                    "timestamp": 9.0,
                    "text": "export completed",
                    "new_since_previous_sample": True,
                },
            ],
        }

    def test_plan_respects_target_and_order(self) -> None:
        plan = build_edit_plan(self.report(), 5.0)
        self.assertLessEqual(plan["planned_duration"], 5.0)
        starts = [item["source_start"] for item in plan["segments"]]
        self.assertEqual(starts, sorted(starts))

    def test_renderer_maps_video(self) -> None:
        args = build_ffmpeg_args(build_edit_plan(self.report(), 5.0), Path("out.mp4"))
        self.assertIn("[vout]", args)
        self.assertNotIn("[aout]", args)


class ManifestTests(unittest.TestCase):
    def test_scores_predeclared_signals(self) -> None:
        report = {
            "timeline": [{"ocr_lines": [{"text": "Export complete"}]}],
            "transcript": {"segments": [{"text": "The result is ready"}]},
            "objects": {"unique_classes": ["person"]},
            "audio": {"has_audio_stream": True, "silent": False},
            "scenes": [{}, {}],
        }
        manifest = {
            "schema_version": 1,
            "cases": [
                {
                    "id": "case-01",
                    "analysis": "analysis.json",
                    "signals": [
                        {"id": "ocr", "kind": "text", "source": "ocr", "any": ["export complete"]},
                        {"id": "asr", "kind": "text", "source": "asr", "any": ["result is ready"]},
                        {"id": "object", "kind": "object", "any": ["person"]},
                        {"id": "audio", "kind": "audio", "has_audio_stream": True, "silent": False},
                        {"id": "scenes", "kind": "scenes", "min": 2, "weight": 0.5},
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis.json").write_text(json.dumps(report), encoding="utf-8")
            result = score_manifest(manifest, root / "manifest.json", None)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["matched_weight"], 4.5)

    def test_manifest_reports_miss_without_false_precision_claim(self) -> None:
        report = {
            "timeline": [],
            "transcript": {"segments": []},
            "objects": {"unique_classes": []},
            "audio": {"has_audio_stream": False, "silent": True},
            "scenes": [],
        }
        manifest = {
            "schema_version": 1,
            "cases": [
                {
                    "id": "case-01",
                    "analysis": "analysis.json",
                    "signals": [{"id": "car", "kind": "object", "any": ["car"]}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis.json").write_text(json.dumps(report), encoding="utf-8")
            result = score_manifest(manifest, root / "manifest.json", None)
        self.assertEqual(result["recall"], 0.0)
        self.assertIn("does not penalize false positives", result["limitations"][1])


class RemakeSpecTests(unittest.TestCase):
    def report(self) -> dict:
        return {
            "schema_version": 2,
            "input": {"path": "source.mp4"},
            "media": {"duration_seconds": 8.0, "width": 1080, "height": 1920},
            "scenes": [{"start": 0.0, "end": 8.0}],
            "timeline": [
                {"timestamp": 1.0, "frame": "frame-1.jpg"},
                {"timestamp": 7.0, "frame": "frame-2.jpg"},
            ],
            "evidence_index": [
                {"id": "ocr-000-000", "source": "ocr", "timestamp": 1.0, "text": "Try it"},
                {
                    "id": "asr-000",
                    "source": "asr",
                    "start": 5.0,
                    "end": 7.0,
                    "timestamp": 6.0,
                    "text": "The result is ready",
                },
            ],
            "objects": {
                "detections": [
                    {"timestamp": 1.0, "objects": [{"label": "person"}], "count": 1}
                ]
            },
        }

    def test_builds_editable_grounded_shots(self) -> None:
        spec = build_remake_spec(
            self.report(), "Create an original product demonstration", max_shot_duration=4
        )
        self.assertEqual(spec["output"]["aspect_ratio"], "9:16")
        self.assertEqual(len(spec["shots"]), 2)
        self.assertEqual(spec["shots"][0]["narrative_role"], "hook")
        self.assertEqual(spec["shots"][1]["narrative_role"], "payoff_or_call_to_action")
        self.assertEqual(spec["shots"][0]["evidence_ids"], ["ocr-000-000"])
        self.assertIn("person", spec["shots"][0]["visual_prompt"])
        self.assertFalse(spec["shots"][0]["approved"])

    def test_minimax_and_seedance_payloads(self) -> None:
        shot = {"id": "shot-001", "visual_prompt": "A product reveal", "requested_duration": 3.2}
        minimax = build_payload(
            "minimax", shot, "MiniMax-Hailuo-2.3", "768P", "9:16"
        )
        self.assertEqual(minimax["duration"], 6)
        self.assertFalse(minimax["prompt_optimizer"])
        seedance = build_payload(
            "seedance",
            shot,
            "doubao-seedance-2-0-260128",
            "720p",
            "9:16",
            "data:image/jpeg;base64,abc",
        )
        self.assertEqual(seedance["duration"], 4)
        self.assertEqual(seedance["content"][1]["role"], "first_frame")

    def test_paid_submission_guards(self) -> None:
        shots = [{"id": "shot-001", "approved": False}]
        with self.assertRaisesRegex(ValueError, "confirm-paid-api"):
            validate_submission(shots, True, False, True, "none", False)
        with self.assertRaisesRegex(ValueError, "unapproved"):
            validate_submission(shots, True, True, True, "none", False)
        shots[0]["approved"] = True
        with self.assertRaisesRegex(ValueError, "reference-guided"):
            validate_submission(shots, True, True, True, "first-frame", False)

    def test_shot_selection(self) -> None:
        spec = {"shots": [{"id": "shot-001"}, {"id": "shot-002"}]}
        self.assertEqual(select_shots(spec, ["shot-002"], False)[0]["id"], "shot-002")
        with self.assertRaisesRegex(ValueError, "unknown shot"):
            select_shots(spec, ["shot-999"], False)

    def test_remake_assembly_trims_to_requested_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "one.mp4"
            second = root / "two.mp4"
            first.write_bytes(b"one")
            second.write_bytes(b"two")
            run = {
                "output": {"aspect_ratio": "9:16"},
                "jobs": [
                    {"shot_id": "shot-001", "order": 1, "requested_duration": 1.5, "local_path": str(first)},
                    {"shot_id": "shot-002", "order": 2, "requested_duration": 2.0, "local_path": str(second)},
                ],
            }
            with patch("assemble_remake.has_audio", return_value=False):
                args = build_remake_ffmpeg_args(run, root / "out.mp4")
        filters = args[args.index("-filter_complex") + 1]
        self.assertIn("scale=720:1280", filters)
        self.assertIn("trim=duration=1.500", filters)
        self.assertIn("concat=n=2:v=1:a=1", filters)

    def test_generation_run_is_written_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generation.json"
            write_run(output, {"jobs": [{"shot_id": "shot-001"}]})
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["jobs"][0]["shot_id"],
                "shot-001",
            )
            self.assertFalse(output.with_suffix(".json.tmp").exists())


class MockProviderHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(size) or b"{}")
        self.requests.append({"method": "POST", "path": self.path, "body": body})
        if self.path == "/v1/video_generation":
            self.send_json({"task_id": "mini-1", "base_resp": {"status_code": 0}})
        else:
            self.send_json({"id": "seed-1"})

    def do_GET(self) -> None:  # noqa: N802
        self.requests.append({"method": "GET", "path": self.path})
        base = f"http://127.0.0.1:{self.server.server_address[1]}"
        if self.path.startswith("/v1/query/video_generation"):
            self.send_json(
                {
                    "status": "Success",
                    "file_id": "file-1",
                    "base_resp": {"status_code": 0},
                }
            )
        elif self.path.startswith("/v1/files/retrieve"):
            self.send_json(
                {
                    "file": {"download_url": f"{base}/clip.mp4"},
                    "base_resp": {"status_code": 0},
                }
            )
        elif self.path == "/contents/generations/tasks/seed-1":
            self.send_json(
                {"status": "succeeded", "content": {"video_url": f"{base}/clip.mp4"}}
            )
        elif self.path == "/clip.mp4":
            body = b"mock-video"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)


class ProviderContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        MockProviderHandler.requests = []
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockProviderHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def test_minimax_submit_query_and_download(self) -> None:
        submitted = submit_job("minimax", self.endpoint, "test-key", {"prompt": "x"})
        self.assertEqual(submitted["task_id"], "mini-1")
        result = query_job("minimax", self.endpoint, "test-key", "mini-1")
        self.assertEqual(result["status"], "succeeded")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip.mp4"
            download_file(result["download_url"], output)
            self.assertEqual(output.read_bytes(), b"mock-video")

    def test_failed_download_removes_partial_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip.mp4"
            partial = output.with_suffix(".mp4.part")
            with patch("video_providers.urllib.request.urlopen") as urlopen:
                response = urlopen.return_value.__enter__.return_value
                response.read.side_effect = [b"partial", OSError("connection lost")]
                with self.assertRaisesRegex(OSError, "connection lost"):
                    download_file("https://example.invalid/clip.mp4", output)
            self.assertFalse(output.exists())
            self.assertFalse(partial.exists())

    def test_seedance_submit_and_query(self) -> None:
        submitted = submit_job("seedance", self.endpoint, "test-key", {"content": []})
        self.assertEqual(submitted["task_id"], "seed-1")
        result = query_job("seedance", self.endpoint, "test-key", "seed-1")
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(result["download_url"].endswith("/clip.mp4"))

    def test_status_normalization(self) -> None:
        self.assertEqual(normalize_status("Queueing"), "queued")
        self.assertEqual(normalize_status("Processing"), "running")
        self.assertEqual(normalize_status("failed"), "failed")


if __name__ == "__main__":
    unittest.main()
