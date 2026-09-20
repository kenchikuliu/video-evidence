from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rapidocr import RapidOCR
from scenedetect import ContentDetector, SceneManager, open_video

from edit_plan import build_edit_plan
from object_detection import DEFAULT_MODEL_PATH, detect_video_objects
from semantic_review import build_evidence_index, run_semantic_review


@dataclass(frozen=True)
class Scene:
    start: float
    end: float


def run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def probe_media(video_path: Path) -> dict:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(video_path),
        ]
    )
    probe = json.loads(result.stdout)
    duration_text = probe.get("format", {}).get("duration")
    if duration_text is None:
        durations = [
            float(stream["duration"])
            for stream in probe.get("streams", [])
            if stream.get("duration") is not None
        ]
        if not durations:
            raise ValueError("ffprobe did not report a media duration")
        duration = max(durations)
    else:
        duration = float(duration_text)

    video_stream = next(
        (stream for stream in probe.get("streams", []) if stream.get("codec_type") == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError("input has no video stream")

    audio_streams = [
        stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"
    ]
    return {
        "duration_seconds": round(duration, 6),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "video_codec": video_stream.get("codec_name"),
        "audio_stream_count": len(audio_streams),
        "audio_codecs": [stream.get("codec_name") for stream in audio_streams],
    }


def measure_audio(video_path: Path, has_audio: bool) -> dict:
    if not has_audio:
        return {"has_audio_stream": False, "max_volume_db": None, "silent": True}

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    match = re.search(r"max_volume:\s+(-?inf|-?[0-9.]+)\s+dB", result.stderr, re.IGNORECASE)
    if not match:
        return {"has_audio_stream": True, "max_volume_db": None, "silent": None}

    value = match.group(1).lower()
    max_volume_db = float("-inf") if value == "-inf" else float(value)
    return {
        "has_audio_stream": True,
        "max_volume_db": value if math.isinf(max_volume_db) else max_volume_db,
        "silent": math.isinf(max_volume_db) or max_volume_db <= -60.0,
    }


def transcribe_audio(
    video_path: Path,
    audio: dict,
    requested: bool,
    model_name: str,
    device: str,
    compute_type: str,
) -> dict:
    if not requested:
        return {"status": "not_requested", "segments": []}
    if not audio["has_audio_stream"]:
        return {"status": "skipped_no_audio_stream", "segments": []}
    if audio["silent"] is True:
        return {"status": "skipped_silent_audio", "segments": []}

    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "faster-whisper is required for --transcribe; install requirements-audio.txt"
        ) from error

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments_iterator, info = model.transcribe(
        str(video_path),
        vad_filter=True,
        word_timestamps=True,
    )
    segments = []
    for segment in segments_iterator:
        words = [
            {
                "start": round(float(word.start), 3),
                "end": round(float(word.end), 3),
                "text": word.word,
                "probability": round(float(word.probability), 5),
            }
            for word in (segment.words or [])
        ]
        segments.append(
            {
                "start": round(float(segment.start), 3),
                "end": round(float(segment.end), 3),
                "text": segment.text.strip(),
                "words": words,
            }
        )
    return {
        "status": "completed",
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 5),
        "segments": segments,
    }


def detect_scenes(video_path: Path) -> list[Scene]:
    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector())
    manager.detect_scenes(video, show_progress=False)
    return [
        Scene(start.seconds, end.seconds)
        for start, end in manager.get_scene_list(start_in_scene=True)
    ]


def evenly_spaced_times(duration: float, count: int) -> list[float]:
    if duration <= 0 or count <= 0:
        return []
    if duration <= 1.0:
        return [round(duration / 2.0, 3)]

    first = min(0.5, duration / 2.0)
    last = max(first, duration - 0.5)
    if count == 1 or math.isclose(first, last):
        return [round((first + last) / 2.0, 3)]
    step = (last - first) / (count - 1)
    return [round(first + step * index, 3) for index in range(count)]


def merge_sample_times(
    duration: float,
    scenes: Sequence[Scene],
    uniform_count: int,
    max_frames: int,
) -> list[float]:
    candidates = evenly_spaced_times(duration, uniform_count)
    if len(scenes) > 1:
        for scene in scenes:
            candidates.append(round((scene.start + scene.end) / 2.0, 3))

    deduplicated: list[float] = []
    for timestamp in sorted(candidates):
        clamped = min(max(timestamp, 0.0), max(duration - 0.001, 0.0))
        if not deduplicated or clamped - deduplicated[-1] >= 0.2:
            deduplicated.append(round(clamped, 3))

    if len(deduplicated) <= max_frames:
        return deduplicated
    indexes = evenly_spaced_indexes(len(deduplicated), max_frames)
    return [deduplicated[index] for index in indexes]


def evenly_spaced_indexes(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted({round(index * (length - 1) / (count - 1)) for index in range(count)})


def parse_timestamps(value: str, duration: float) -> list[float]:
    timestamps = sorted({round(float(item.strip()), 3) for item in value.split(",") if item.strip()})
    invalid = [value for value in timestamps if value < 0 or value >= duration]
    if invalid:
        raise ValueError(f"timestamps outside video duration: {invalid}")
    return timestamps


def extract_frame(video_path: Path, timestamp: float, output_path: Path) -> None:
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(output_path),
        ]
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\W+", "", text, flags=re.UNICODE).casefold()


def find_new_lines(lines: Iterable[str], previous_lines: Iterable[str]) -> list[str]:
    previous = {normalize_text(line) for line in previous_lines}
    return [line for line in lines if normalize_text(line) not in previous]


def analyze_frames(
    video_path: Path,
    timestamps: Sequence[float],
    frames_dir: Path,
    ocr_threshold: float,
) -> list[dict]:
    engine = RapidOCR()
    timeline = []
    previous_lines: list[str] = []
    for index, timestamp in enumerate(timestamps):
        frame_path = frames_dir / f"frame_{index:03d}_{timestamp:.3f}s.jpg"
        extract_frame(video_path, timestamp, frame_path)
        result = engine(frame_path)
        lines = []
        if result.txts and result.scores and result.boxes is not None:
            for text, score, box in zip(result.txts, result.scores, result.boxes):
                if float(score) < ocr_threshold:
                    continue
                lines.append(
                    {
                        "text": text,
                        "score": round(float(score), 5),
                        "box": [[round(float(x), 1), round(float(y), 1)] for x, y in box],
                    }
                )
        texts = [line["text"] for line in lines]
        timeline.append(
            {
                "timestamp": timestamp,
                "frame": str(frame_path),
                "ocr_line_count": len(lines),
                "ocr_lines": lines,
                "new_text_since_previous_sample": find_new_lines(texts, previous_lines),
            }
        )
        previous_lines = texts
    return timeline


def render_markdown(report: dict) -> str:
    detected_objects = report.get("objects", {}).get("unique_classes", [])
    lines = [
        "# Video Evidence Timeline",
        "",
        f"Source: `{report['input']['path']}`",
        f"Duration: {report['media']['duration_seconds']:.3f}s",
        f"Scenes: {len(report['scenes'])}",
        f"Silent: {report['audio']['silent']}",
        f"Objects: {', '.join(detected_objects) if detected_objects else 'not detected or not requested'}",
        "",
    ]
    for item in report["timeline"]:
        lines.extend([f"## {item['timestamp']:.3f}s", "", f"Frame: `{item['frame']}`", ""])
        evidence = item["new_text_since_previous_sample"] or [
            line["text"] for line in item["ocr_lines"]
        ]
        if evidence:
            lines.extend(f"- {text}" for text in evidence)
        else:
            lines.append("- No OCR evidence above the confidence threshold.")
        lines.append("")
    return "\n".join(lines)


def render_brief(source: Path, brief: str, target_duration: float | None) -> str:
    duration_line = (
        f"- 目标时长：约 {target_duration:.3f} 秒。"
        if target_duration is not None
        else "- 目标时长：未指定。"
    )
    return "\n".join(
        [
            "# Brief",
            "",
            brief,
            "",
            "## 输入与约束",
            "",
            f"- 源视频：`{source}`",
            duration_line,
            "- 全程本地处理，不登录，不调用付费 API。",
            "- OCR、ASR、时间戳和画面坐标是事实源；模型结论必须引用证据 ID。",
            "",
        ]
    )


def render_treatment(plan: dict) -> str:
    lines = [
        "# Treatment",
        "",
        "制作一条按源时间顺序推进的证据精华片。优先保留新信息、语音内容以及可见的校验或完成状态，避免把计划误写成结果。",
        "",
        f"计划成片时长：{plan['planned_duration']:.3f} 秒。",
        "",
        "## 选段",
        "",
    ]
    for index, segment in enumerate(plan["segments"], start=1):
        reasons = "；".join(segment["reasons"])
        lines.append(
            f"{index}. `{segment['source_start']:.3f}s - {segment['source_end']:.3f}s`：{reasons}。"
        )
    lines.extend(
        [
            "",
            "最终成片必须重新观看并检查切点、声音连续性、字幕可读性和事实准确性。",
            "",
        ]
    )
    return "\n".join(lines)


def render_semantic_markdown(review: dict) -> str:
    validation = review.get("validation", {})
    content = review.get("content", {})
    lines = [
        "# Analysis",
        "",
        "> 这是本地视觉模型对确定性证据的解释层；OCR、ASR、时间戳和坐标仍是事实源。",
        "",
        f"模型：`{review.get('model', 'unknown')}`",
        f"JSON 有效：{review.get('valid_json', False)}",
        f"引用校验通过：{validation.get('passed', False)}",
        f"事件证据覆盖：{validation.get('grounded_event_count', 0)}/{validation.get('event_count', 0)}",
        "",
        "## 整体理解",
        "",
        str(content.get("overall_purpose", "模型未产生整体理解。")),
        "",
    ]
    overall_citations = validation.get("overall_citations", [])
    if overall_citations:
        lines.append("整体判断引用：")
        lines.append("")
        lines.extend(
            f"- `{item['id']}` ({item['timestamp']:.3f}s): {item['text']}"
            for item in overall_citations
        )
        lines.append("")

    lines.extend(["## 证据化事件", ""])
    grounded_events = validation.get("grounded_events", [])
    if not grounded_events:
        lines.extend(["没有通过引用校验的事件。", ""])
    for event in grounded_events:
        timestamp = event.get("timestamp")
        lines.extend(
            [
                f"### {float(timestamp):.3f}s" if timestamp is not None else "### 未知时间",
                "",
                str(event.get("event", "")),
                "",
            ]
        )
        citations = event.get("citations", [])
        if citations:
            lines.extend(
                f"- `{item['id']}`: {item['text']}" for item in citations
            )
        else:
            lines.append("- 没有合法且时间对齐的证据引用。")
        uncertainty = event.get("uncertainty")
        if uncertainty:
            lines.extend(["", f"不确定项：{uncertainty}"])
        lines.append("")

    limitations = content.get("limitations", [])
    lines.extend(["## 限制", ""])
    if limitations:
        lines.extend(f"- {item}" for item in limitations)
    else:
        lines.append("- 模型未列出限制；仍需检查采样覆盖与原始证据。")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a local, evidence-first video timeline with scene detection and OCR."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument("--timestamps", help="Comma-separated timestamps in seconds.")
    parser.add_argument("--ocr-threshold", type=float, default=0.85)
    parser.add_argument(
        "--detect-objects",
        action="store_true",
        help="Run local YOLO11n ONNX object detection on uniformly sampled frames.",
    )
    parser.add_argument("--object-model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--object-confidence", type=float, default=0.3)
    parser.add_argument("--object-iou", type=float, default=0.45)
    parser.add_argument("--object-sample-interval", type=float, default=1.0)
    parser.add_argument(
        "--object-classes",
        help="Optional comma-separated COCO classes to retain. Detection remains image-only.",
    )
    parser.add_argument("--transcribe", action="store_true")
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--whisper-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--whisper-compute-type", default="default")
    parser.add_argument(
        "--semantic-review",
        action="store_true",
        help="Run an optional local Ollama vision review after deterministic analysis.",
    )
    parser.add_argument("--ollama-model", default="qwen3-vl:8b")
    parser.add_argument("--ollama-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--brief", help="Target purpose used by review and production notes.")
    parser.add_argument(
        "--target-duration",
        type=float,
        help="Create an evidence-first edit plan for this approximate duration in seconds.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    video_path = args.video.resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if args.samples <= 0 or args.max_frames <= 0:
        raise ValueError("--samples and --max-frames must be positive")
    if not 0.0 <= args.ocr_threshold <= 1.0:
        raise ValueError("--ocr-threshold must be between 0 and 1")
    if not 0.0 <= args.object_confidence <= 1.0:
        raise ValueError("--object-confidence must be between 0 and 1")
    if not 0.0 <= args.object_iou <= 1.0:
        raise ValueError("--object-iou must be between 0 and 1")
    if args.object_sample_interval <= 0:
        raise ValueError("--object-sample-interval must be positive")
    if args.target_duration is not None and args.target_duration <= 0:
        raise ValueError("--target-duration must be positive")

    output_dir = (args.output_dir or video_path.with_name(f"{video_path.stem}-analysis")).resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    media = probe_media(video_path)
    audio = measure_audio(video_path, media["audio_stream_count"] > 0)
    transcript = transcribe_audio(
        video_path,
        audio,
        args.transcribe,
        args.whisper_model,
        args.whisper_device,
        args.whisper_compute_type,
    )
    scenes = detect_scenes(video_path)
    if args.timestamps:
        timestamps = parse_timestamps(args.timestamps, media["duration_seconds"])
    else:
        timestamps = merge_sample_times(
            media["duration_seconds"], scenes, args.samples, args.max_frames
        )

    timeline = analyze_frames(
        video_path,
        timestamps,
        frames_dir,
        args.ocr_threshold,
    )
    objects = {"status": "not_requested", "detections": [], "unique_classes": []}
    object_seconds = 0.0
    if args.detect_objects:
        object_classes = None
        if args.object_classes:
            object_classes = [
                item.strip() for item in args.object_classes.split(",") if item.strip()
            ]
        started = time.perf_counter()
        objects = detect_video_objects(
            video_path,
            media["duration_seconds"],
            args.object_model,
            args.object_sample_interval,
            args.object_confidence,
            args.object_iou,
            object_classes,
        )
        object_seconds = round(time.perf_counter() - started, 3)
    report = {
        "schema_version": 2,
        "input": {"path": str(video_path)},
        "execution": {
            "local_only": True,
            "login_required": False,
            "paid_api_used": False,
            "ocr_threshold": args.ocr_threshold,
            "object_detection": args.detect_objects,
        },
        "media": media,
        "audio": audio,
        "transcript": transcript,
        "scenes": [scene.__dict__ for scene in scenes],
        "sample_timestamps": timestamps,
        "timeline": timeline,
        "objects": objects,
        "timings": {"objects": object_seconds},
    }
    report["evidence_index"] = build_evidence_index(report)
    json_path = output_dir / "analysis.json"
    markdown_path = output_dir / "timeline.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    artifacts = {"analysis": str(json_path), "timeline": str(markdown_path)}
    if args.brief:
        brief_path = output_dir / "BRIEF.md"
        brief_path.write_text(
            render_brief(video_path, args.brief, args.target_duration), encoding="utf-8"
        )
        artifacts["brief"] = str(brief_path)

    if args.target_duration is not None:
        plan = build_edit_plan(report, args.target_duration, args.brief)
        plan_path = output_dir / "edit_plan.json"
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        treatment_path = output_dir / "TREATMENT.md"
        treatment_path.write_text(render_treatment(plan), encoding="utf-8")
        artifacts["edit_plan"] = str(plan_path)
        artifacts["treatment"] = str(treatment_path)
        artifacts["render_command"] = (
            f'{sys.executable} "{Path(__file__).with_name("render_edit.py")}" '
            f'"{plan_path}" "{output_dir / "edited.mp4"}"'
        )

    if args.semantic_review:
        review = run_semantic_review(
            report,
            json_path,
            args.ollama_model,
            args.ollama_endpoint,
            args.brief,
        )
        review_path = output_dir / "semantic_review.json"
        review_path.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
        semantic_markdown_path = output_dir / "ANALYSIS.md"
        semantic_markdown_path.write_text(
            render_semantic_markdown(review), encoding="utf-8"
        )
        artifacts["semantic_review"] = str(review_path)
        artifacts["semantic_analysis"] = str(semantic_markdown_path)

    report["artifacts"] = artifacts
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(artifacts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
