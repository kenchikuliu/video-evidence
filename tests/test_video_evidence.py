from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

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
from edit_plan import build_edit_plan  # noqa: E402
from object_detection import DEFAULT_MODEL_PATH, postprocess, sample_times  # noqa: E402
from render_edit import build_ffmpeg_args  # noqa: E402
from score_manifest import score_manifest  # noqa: E402
from semantic_review import build_evidence_index, validate_review  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
