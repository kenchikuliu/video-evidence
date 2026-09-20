from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


CANONICAL_RATIOS = {
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:4": 3 / 4,
}


def evidence_timestamp(item: dict) -> float:
    return float(item.get("timestamp", item.get("start", 0.0)))


def infer_aspect_ratio(report: dict) -> str:
    media = report.get("media", {})
    width = float(media.get("width") or 0)
    height = float(media.get("height") or 0)
    if width <= 0 or height <= 0:
        return "16:9"
    ratio = width / height
    return min(CANONICAL_RATIOS, key=lambda name: abs(CANONICAL_RATIOS[name] - ratio))


def split_interval(start: float, end: float, maximum: float) -> list[tuple[float, float]]:
    duration = end - start
    if duration <= 0:
        return []
    count = max(1, math.ceil(duration / maximum))
    step = duration / count
    return [
        (round(start + index * step, 3), round(start + (index + 1) * step, 3))
        for index in range(count)
    ]


def source_intervals(report: dict, maximum: float) -> list[tuple[float, float]]:
    duration = float(report.get("media", {}).get("duration_seconds") or 0)
    if duration <= 0:
        raise ValueError("analysis report has no positive media duration")
    scenes = []
    for scene in report.get("scenes", []):
        start = max(0.0, float(scene.get("start", 0.0)))
        end = min(duration, float(scene.get("end", duration)))
        if end > start:
            scenes.append((start, end))
    if not scenes:
        scenes = [(0.0, duration)]

    intervals = []
    for start, end in scenes:
        intervals.extend(split_interval(start, end, maximum))
    return intervals


def items_in_interval(items: Iterable[dict], start: float, end: float) -> list[dict]:
    return [item for item in items if start <= evidence_timestamp(item) < end + 0.001]


def objects_in_interval(report: dict, start: float, end: float) -> list[str]:
    labels = set()
    for sample in report.get("objects", {}).get("detections", []):
        timestamp = float(sample.get("timestamp", -1.0))
        if start <= timestamp < end + 0.001:
            labels.update(
                str(item.get("label", "")).strip()
                for item in sample.get("objects", [])
                if item.get("label")
            )
    return sorted(labels)


def nearest_reference_frame(report: dict, start: float, end: float) -> str | None:
    timeline = report.get("timeline", [])
    if not timeline:
        return None
    midpoint = (start + end) / 2.0
    inside = [
        item
        for item in timeline
        if start <= float(item.get("timestamp", -1.0)) < end + 0.001
    ]
    candidates = inside or timeline
    selected = min(candidates, key=lambda item: abs(float(item["timestamp"]) - midpoint))
    return selected.get("frame")


def grounded_events(review: dict | None, start: float, end: float) -> list[dict]:
    if not review:
        return []
    return [
        item
        for item in review.get("validation", {}).get("grounded_events", [])
        if item.get("grounded")
        and start <= float(item.get("timestamp", -1.0)) < end + 0.001
    ]


def narrative_role(index: int, count: int) -> str:
    if index == 0:
        return "hook"
    if index == count - 1:
        return "payoff_or_call_to_action"
    return "progression"


def build_visual_prompt(
    brief: str,
    role: str,
    duration: float,
    aspect_ratio: str,
    objects: list[str],
    events: list[dict],
    rights_mode: str,
) -> str:
    parts = [
        f"Create an original {aspect_ratio} advertising shot lasting about {duration:.1f} seconds.",
        f"Narrative role: {role}.",
        f"Creative brief: {brief}.",
    ]
    if events:
        parts.append(
            "Grounded source event to reinterpret: "
            + " ".join(str(item.get("event", "")) for item in events).strip()
            + "."
        )
    if objects:
        parts.append("Useful visible categories: " + ", ".join(objects) + ".")
    parts.append(
        "Keep the main subject readable, preserve temporal continuity, and leave text rendering to post-production."
    )
    if rights_mode == "structure-only":
        parts.append(
            "Reuse only timing and narrative structure; replace source identities, logos, music, characters, and brand assets with user-owned originals."
        )
    else:
        parts.append("The user states that the supplied source material may be reused.")
    return " ".join(parts)


def build_remake_spec(
    report: dict,
    brief: str,
    aspect_ratio: str = "auto",
    max_shot_duration: float = 6.0,
    review: dict | None = None,
    reference_policy: str = "review-only",
    rights_mode: str = "structure-only",
) -> dict:
    if not brief.strip():
        raise ValueError("brief must not be empty")
    if max_shot_duration < 2 or max_shot_duration > 12:
        raise ValueError("max_shot_duration must be between 2 and 12 seconds")
    if reference_policy not in {"review-only", "first-frame"}:
        raise ValueError("unsupported reference policy")
    if rights_mode not in {"structure-only", "owned-source"}:
        raise ValueError("unsupported rights mode")

    selected_ratio = infer_aspect_ratio(report) if aspect_ratio == "auto" else aspect_ratio
    if selected_ratio not in CANONICAL_RATIOS:
        raise ValueError(f"unsupported aspect ratio: {selected_ratio}")
    intervals = source_intervals(report, max_shot_duration)
    evidence_index = report.get("evidence_index", [])
    shots = []
    for index, (start, end) in enumerate(intervals):
        evidence = items_in_interval(evidence_index, start, end)
        events = grounded_events(review, start, end)
        objects = objects_in_interval(report, start, end)
        role = narrative_role(index, len(intervals))
        duration = round(end - start, 3)
        ocr = [item for item in evidence if item.get("source") == "ocr"]
        asr = [item for item in evidence if item.get("source") == "asr"]
        shots.append(
            {
                "id": f"shot-{index + 1:03d}",
                "order": index + 1,
                "approved": False,
                "source_start": start,
                "source_end": end,
                "requested_duration": duration,
                "narrative_role": role,
                "visual_prompt": build_visual_prompt(
                    brief,
                    role,
                    duration,
                    selected_ratio,
                    objects,
                    events,
                    rights_mode,
                ),
                "on_screen_text": [str(item.get("text", "")) for item in ocr],
                "dialogue_reference": [str(item.get("text", "")) for item in asr],
                "objects": objects,
                "evidence_ids": [str(item.get("id")) for item in evidence if item.get("id")],
                "grounded_events": [str(item.get("event", "")) for item in events],
                "reference_frame": nearest_reference_frame(report, start, end),
                "reference_policy": reference_policy,
                "review_notes": "Review the prompt, replace source-specific assets, then set approved to true.",
            }
        )

    return {
        "schema_version": 1,
        "kind": "video_remake_spec",
        "status": "draft",
        "creative_brief": brief,
        "rights_mode": rights_mode,
        "source": {
            "analysis_schema_version": report.get("schema_version"),
            "path": report.get("input", {}).get("path"),
            "duration_seconds": report.get("media", {}).get("duration_seconds"),
            "width": report.get("media", {}).get("width"),
            "height": report.get("media", {}).get("height"),
        },
        "output": {
            "aspect_ratio": selected_ratio,
            "planned_duration": round(sum(item["requested_duration"] for item in shots), 3),
        },
        "provider_defaults": {
            "minimax": {"model": "MiniMax-Hailuo-2.3", "resolution": "768P"},
            "seedance": {
                "model": "doubao-seedance-2-0-260128",
                "resolution": "720p",
                "generate_audio": False,
            },
        },
        "shots": shots,
        "warnings": [
            "This is an automatically derived draft; every shot requires creative review before paid submission.",
            "OCR, ASR, sampled objects, and grounded events are incomplete evidence, not a full visual description.",
            "Reference frames remain local unless a later command explicitly allows upload.",
            "Viral performance cannot be inferred or guaranteed from structural similarity.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert video evidence into an editable, provider-neutral remake specification."
    )
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--brief", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--semantic-review", type=Path)
    parser.add_argument("--aspect-ratio", default="auto", choices=("auto", *CANONICAL_RATIOS))
    parser.add_argument("--max-shot-duration", type=float, default=6.0)
    parser.add_argument(
        "--reference-policy", choices=("review-only", "first-frame"), default="review-only"
    )
    parser.add_argument(
        "--rights-mode", choices=("structure-only", "owned-source"), default="structure-only"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    analysis_path = args.analysis.resolve()
    report = json.loads(analysis_path.read_text(encoding="utf-8"))
    review = None
    if args.semantic_review:
        review = json.loads(args.semantic_review.resolve().read_text(encoding="utf-8"))
    spec = build_remake_spec(
        report,
        args.brief,
        args.aspect_ratio,
        args.max_shot_duration,
        review,
        args.reference_policy,
        args.rights_mode,
    )
    output = (args.output or analysis_path.with_name("remake_spec.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "shots": len(spec["shots"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
