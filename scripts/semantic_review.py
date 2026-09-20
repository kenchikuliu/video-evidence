from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path
from typing import Any


PROMPT_VERSION = 1
MAX_REVIEW_IMAGES = 12


def parse_json_content(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def build_evidence_index(report: dict) -> list[dict]:
    evidence: list[dict] = []
    for frame_index, item in enumerate(report.get("timeline", [])):
        new_text = set(item.get("new_text_since_previous_sample", []))
        for line_index, line in enumerate(item.get("ocr_lines", [])):
            evidence.append(
                {
                    "id": f"ocr-{frame_index:03d}-{line_index:03d}",
                    "source": "ocr",
                    "timestamp": item["timestamp"],
                    "text": line["text"],
                    "confidence": line.get("score"),
                    "new_since_previous_sample": line["text"] in new_text,
                }
            )

    for segment_index, segment in enumerate(
        report.get("transcript", {}).get("segments", [])
    ):
        evidence.append(
            {
                "id": f"asr-{segment_index:03d}",
                "source": "asr",
                "start": segment["start"],
                "end": segment["end"],
                "timestamp": round((segment["start"] + segment["end"]) / 2.0, 3),
                "text": segment["text"],
            }
        )
    return evidence


def _prompt_evidence(evidence_index: list[dict]) -> list[dict]:
    fields = ("id", "source", "timestamp", "start", "end", "text")
    return [{key: item[key] for key in fields if key in item} for item in evidence_index]


def select_review_timeline(timeline: list[dict], limit: int = MAX_REVIEW_IMAGES) -> list[dict]:
    if len(timeline) <= limit:
        return timeline
    indexes = sorted(
        {round(index * (len(timeline) - 1) / (limit - 1)) for index in range(limit)}
    )
    return [timeline[index] for index in indexes]


def build_prompt(report: dict, evidence_index: list[dict], brief: str | None) -> str:
    frame_order = [
        {"timestamp": item["timestamp"], "image_order": index + 1}
        for index, item in enumerate(report.get("timeline", []))
    ]
    evidence = json.dumps(_prompt_evidence(evidence_index), ensure_ascii=False)
    frames = json.dumps(frame_order, ensure_ascii=False)
    audio_status = json.dumps(report.get("audio", {}), ensure_ascii=False)
    target = brief or "理解视频实际展示的任务、过程与结果"
    return f"""/no_think
不要输出思考过程，立即根据证据返回最终 JSON。
你是视频证据复核员。图片按下面的 image_order 顺序提供：
{frames}

用户目标：{target}
音频状态：{audio_status}

以下是可引用的完整证据目录。OCR 和 ASR 才是事实来源，图片只帮助理解布局：
{evidence}

只返回以下 JSON，不要使用 Markdown：
{{
  "events": [
    {{
      "timestamp": 0.5,
      "event": "只描述该时间点被证据明确支持的事件",
      "evidence_ids": ["ocr-000-000"],
      "uncertainty": "无法确认的内容，没有则为空字符串"
    }}
  ],
  "overall_purpose": "整段视频的具体目标或主题",
  "overall_evidence_ids": ["ocr-000-000"],
  "limitations": ["采样、声音或可见证据的限制"]
}}

约束：
1. 覆盖每个提供的图片时间点，并按时间排序。
2. 每个事件只能引用目录中真实存在、且时间与事件一致的 evidence_ids。
3. overall_purpose 也必须引用证据，不得把计划、命令或界面文案写成已完成结果。
4. 不得补写点击、声音、人物身份或画面外信息；不确定就写入 uncertainty。
5. 优先保留目标、输入、关键决策、校验状态和产物路径；使用简体中文。
"""


def _is_time_aligned(event_timestamp: float, evidence: dict) -> bool:
    if evidence["source"] == "asr":
        return evidence["start"] - 0.25 <= event_timestamp <= evidence["end"] + 0.25
    return abs(float(evidence["timestamp"]) - event_timestamp) <= 0.25


def validate_review(content: dict, evidence_index: list[dict]) -> dict:
    by_id = {item["id"]: item for item in evidence_index}
    grounded_events = []
    invalid_ids: set[str] = set()
    misaligned_ids: set[str] = set()

    events = content.get("events", []) if isinstance(content, dict) else []
    if not isinstance(events, list):
        events = []
    for event in events:
        if not isinstance(event, dict):
            continue
        try:
            timestamp = float(event.get("timestamp"))
        except (TypeError, ValueError):
            timestamp = -1.0
        citations = []
        event_invalid = []
        event_misaligned = []
        evidence_ids = event.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        for evidence_id in evidence_ids:
            if not isinstance(evidence_id, str) or evidence_id not in by_id:
                event_invalid.append(str(evidence_id))
                invalid_ids.add(str(evidence_id))
                continue
            item = by_id[evidence_id]
            if not _is_time_aligned(timestamp, item):
                event_misaligned.append(evidence_id)
                misaligned_ids.add(evidence_id)
                continue
            citations.append(item)
        grounded_events.append(
            {
                "timestamp": event.get("timestamp"),
                "event": event.get("event", ""),
                "uncertainty": event.get("uncertainty", ""),
                "grounded": bool(citations),
                "citations": citations,
                "invalid_evidence_ids": event_invalid,
                "time_misaligned_evidence_ids": event_misaligned,
            }
        )

    overall_ids = content.get("overall_evidence_ids", []) if isinstance(content, dict) else []
    if not isinstance(overall_ids, list):
        overall_ids = []
    overall_citations = []
    for evidence_id in overall_ids:
        if isinstance(evidence_id, str) and evidence_id in by_id:
            overall_citations.append(by_id[evidence_id])
        else:
            invalid_ids.add(str(evidence_id))

    grounded_count = sum(1 for event in grounded_events if event["grounded"])
    return {
        "grounded_events": grounded_events,
        "overall_citations": overall_citations,
        "event_count": len(grounded_events),
        "grounded_event_count": grounded_count,
        "grounded_event_ratio": round(grounded_count / len(grounded_events), 4)
        if grounded_events
        else 0.0,
        "invalid_evidence_ids": sorted(invalid_ids),
        "time_misaligned_evidence_ids": sorted(misaligned_ids),
        "passed": bool(grounded_events)
        and grounded_count == len(grounded_events)
        and bool(overall_citations)
        and not invalid_ids
        and not misaligned_ids,
    }


def run_semantic_review(
    report: dict,
    analysis_path: Path,
    model: str,
    endpoint: str,
    brief: str | None = None,
) -> dict:
    evidence_index = report.get("evidence_index") or build_evidence_index(report)
    full_timeline = report.get("timeline", [])
    review_timeline = select_review_timeline(full_timeline)
    review_report = {**report, "timeline": review_timeline}
    images = [
        base64.b64encode(Path(item["frame"]).read_bytes()).decode("ascii")
        for item in review_timeline
    ]
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "think": False,
        "messages": [
            {
                "role": "user",
                "content": build_prompt(review_report, evidence_index, brief),
                "images": images,
            }
        ],
        "options": {"temperature": 0, "num_predict": 1800, "num_ctx": 32768},
    }

    def submit(request_payload: dict) -> dict:
        request = urllib.request.Request(
            endpoint.rstrip("/") + "/api/chat",
            data=json.dumps(request_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=1200) as response:
            return json.load(response)

    result = submit(payload)
    attempts = [
        {
            "content_length": len(result.get("message", {}).get("content", "")),
            "thinking_length": len(result.get("message", {}).get("thinking", "")),
            "output_tokens": result.get("eval_count"),
            "total_duration_seconds": round(
                result.get("total_duration", 0) / 1_000_000_000, 3
            ),
        }
    ]
    if not result.get("message", {}).get("content", "").strip():
        retry_payload = json.loads(json.dumps(payload))
        retry_payload["options"]["num_predict"] = 6000
        retry_payload["messages"][0]["content"] += (
            "\n上一轮没有返回最终内容。本轮禁止内部推理，只输出指定 JSON。"
        )
        result = submit(retry_payload)
        attempts.append(
            {
                "stage": "vision_retry",
                "content_length": len(result.get("message", {}).get("content", "")),
                "thinking_length": len(result.get("message", {}).get("thinking", "")),
                "output_tokens": result.get("eval_count"),
                "total_duration_seconds": round(
                    result.get("total_duration", 0) / 1_000_000_000, 3
                ),
            }
        )

    if not result.get("message", {}).get("content", "").strip():
        observation_draft = result.get("message", {}).get("thinking", "")
        synthesis_payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "think": False,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        build_prompt(review_report, evidence_index, brief)
                        + "\n上轮视觉检查留下了以下内部观察草稿。它不是事实源，"
                        "只能帮助你整理；最终陈述仍必须由给定 evidence_ids 支持：\n"
                        + observation_draft
                        + "\n现在停止继续分析，只输出指定 JSON。"
                    ),
                }
            ],
            "options": {"temperature": 0, "num_predict": 6000, "num_ctx": 32768},
        }
        result = submit(synthesis_payload)
        attempts.append(
            {
                "stage": "text_synthesis",
                "content_length": len(result.get("message", {}).get("content", "")),
                "thinking_length": len(result.get("message", {}).get("thinking", "")),
                "output_tokens": result.get("eval_count"),
                "total_duration_seconds": round(
                    result.get("total_duration", 0) / 1_000_000_000, 3
                ),
            }
        )

    content_text = result["message"].get("content", "")
    try:
        content = parse_json_content(content_text)
        valid_json = True
        parse_error = None
    except json.JSONDecodeError as error:
        content = {}
        valid_json = False
        parse_error = str(error)

    validation = validate_review(content, evidence_index)
    return {
        "schema_version": 1,
        "execution": {
            "local_only": True,
            "login_required": False,
            "paid_api_used": False,
        },
        "model": result.get("model", model),
        "prompt_version": PROMPT_VERSION,
        "source_frame_count": len(full_timeline),
        "reviewed_frame_count": len(review_timeline),
        "reviewed_timestamps": [item["timestamp"] for item in review_timeline],
        "analysis_source": str(analysis_path.resolve()),
        "valid_json": valid_json,
        "parse_error": parse_error,
        "attempts": attempts,
        "raw_content": content_text,
        "content": content,
        "validation": validation,
        "performance": {
            "total_duration_seconds": round(
                result.get("total_duration", 0) / 1_000_000_000, 3
            ),
            "load_duration_seconds": round(
                result.get("load_duration", 0) / 1_000_000_000, 3
            ),
            "prompt_tokens": result.get("prompt_eval_count"),
            "output_tokens": result.get("eval_count"),
        },
    }
