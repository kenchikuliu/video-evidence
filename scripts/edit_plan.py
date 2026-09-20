from __future__ import annotations

import math
from typing import Sequence


COMPLETION_KEYWORDS = (
    "complete",
    "completed",
    "export",
    "exported",
    "success",
    "successful",
    "valid",
    "完成",
    "成功",
    "导出",
    "有效",
    "通过",
)


def _block_boundaries(timestamps: Sequence[float], duration: float) -> list[tuple[float, float]]:
    if not timestamps:
        return [(0.0, duration)]
    boundaries = [0.0]
    boundaries.extend(
        (timestamps[index - 1] + timestamps[index]) / 2.0
        for index in range(1, len(timestamps))
    )
    boundaries.append(duration)
    return [
        (
            round(boundaries[index], 6),
            min(round(boundaries[index + 1], 6), duration),
        )
        for index in range(len(boundaries) - 1)
    ]


def _evidence_timestamp(item: dict) -> float:
    return float(item.get("timestamp", item.get("start", 0.0)))


def _score_block(start: float, end: float, evidence: list[dict], is_first: bool) -> tuple[float, list[str]]:
    ocr = [item for item in evidence if item.get("source") == "ocr"]
    asr = [item for item in evidence if item.get("source") == "asr"]
    new_ocr = [item for item in ocr if item.get("new_since_previous_sample")]
    text = " ".join(str(item.get("text", "")).casefold() for item in evidence)
    completion_hits = sorted({word for word in COMPLETION_KEYWORDS if word in text})

    score = 1.0
    reasons = []
    if is_first:
        score += 1.5
        reasons.append("保留开场上下文")
    if new_ocr:
        score += min(4.0, math.log2(len(new_ocr) + 1))
        reasons.append(f"包含 {len(new_ocr)} 条新增 OCR 证据")
    if asr:
        word_count = sum(len(str(item.get("text", "")).split()) for item in asr)
        score += min(5.0, 1.5 + word_count / 12.0)
        reasons.append(f"包含 {len(asr)} 条语音证据")
    if completion_hits:
        score += 3.0
        reasons.append("包含结果或校验状态词")
    if not reasons:
        reasons.append("维持时间线连续性")
    return round(score, 4), reasons


def _merge_selected(blocks: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for block in sorted(blocks, key=lambda item: item["source_start"]):
        if merged and abs(merged[-1]["source_end"] - block["source_start"]) <= 0.01:
            previous = merged[-1]
            previous["source_end"] = block["source_end"]
            previous["duration"] = round(
                previous["source_end"] - previous["source_start"], 3
            )
            previous["score"] = round(previous["score"] + block["score"], 4)
            previous["reasons"] = list(dict.fromkeys(previous["reasons"] + block["reasons"]))
            previous["evidence_ids"] = list(
                dict.fromkeys(previous["evidence_ids"] + block["evidence_ids"])
            )
        else:
            merged.append(dict(block))
    return merged


def _speech_blocks(evidence_index: list[dict]) -> list[dict]:
    asr_items = [item for item in evidence_index if item.get("source") == "asr"]
    ocr_items = [item for item in evidence_index if item.get("source") == "ocr"]
    blocks = []
    for index, speech in enumerate(asr_items):
        start = float(speech["start"])
        end = float(speech["end"])
        evidence = [speech] + [
            item
            for item in ocr_items
            if start <= _evidence_timestamp(item) <= end
        ]
        score, reasons = _score_block(start, end, evidence, index == 0)
        blocks.append(
            {
                "source_start": start,
                "source_end": end,
                "duration": round(end - start, 3),
                "score": score,
                "reasons": reasons + ["保留完整语音片段"],
                "evidence_ids": [item["id"] for item in evidence],
                "atomic_speech": True,
            }
        )
    return blocks


def build_edit_plan(report: dict, target_duration: float, brief: str | None = None) -> dict:
    duration = float(report["media"]["duration_seconds"])
    if target_duration <= 0:
        raise ValueError("target_duration must be positive")
    target = min(float(target_duration), duration)
    timestamps = sorted(float(item["timestamp"]) for item in report.get("timeline", []))
    evidence_index = report.get("evidence_index", [])

    blocks = _speech_blocks(evidence_index)
    speech_mode = bool(blocks)
    if not blocks:
        for index, (start, end) in enumerate(_block_boundaries(timestamps, duration)):
            evidence = [
                item
                for item in evidence_index
                if start <= _evidence_timestamp(item) <= end
            ]
            score, reasons = _score_block(start, end, evidence, index == 0)
            blocks.append(
                {
                    "source_start": start,
                    "source_end": end,
                    "duration": round(end - start, 3),
                    "score": score,
                    "reasons": reasons,
                    "evidence_ids": [item["id"] for item in evidence],
                    "atomic_speech": False,
                }
            )

    selected = []
    selected_duration = 0.0
    soft_limit = min(duration, target + max(0.5, target * 0.2)) if speech_mode else target
    for block in sorted(
        blocks,
        key=lambda item: (item["score"] / max(item["duration"], 0.001), item["score"]),
        reverse=True,
    ):
        remaining = target - selected_duration
        if remaining < 0.5 and not block["atomic_speech"]:
            break
        if block["atomic_speech"]:
            if selected and selected_duration + block["duration"] > soft_limit:
                continue
            chosen = dict(block)
            selected.append(chosen)
            selected_duration += chosen["duration"]
            continue

        take = min(block["duration"], max(remaining, 0.0))
        if take < 0.5:
            continue
        chosen = dict(block)
        if take < block["duration"]:
            midpoint = (block["source_start"] + block["source_end"]) / 2.0
            chosen["source_start"] = max(block["source_start"], midpoint - take / 2.0)
            chosen["source_end"] = min(block["source_end"], chosen["source_start"] + take)
            chosen["source_start"] = max(
                block["source_start"], chosen["source_end"] - take
            )
            chosen["source_start"] = round(chosen["source_start"], 3)
            chosen["source_end"] = round(chosen["source_end"], 3)
            chosen["duration"] = round(chosen["source_end"] - chosen["source_start"], 3)
            chosen["reasons"] = chosen["reasons"] + ["为满足目标时长进行了边界裁切"]
        selected.append(chosen)
        selected_duration += chosen["duration"]

    if not selected and blocks:
        selected.append(dict(max(blocks, key=lambda item: item["score"])))

    segments = _merge_selected(selected)
    for segment in segments:
        segment.pop("atomic_speech", None)
    planned_duration = round(sum(item["duration"] for item in segments), 3)
    warnings = [
        "该计划只根据可核验证据自动选段，仍应人工观看最终成片。",
        "模型语义复核不会覆盖或修改这个确定性计划。",
    ]
    if planned_duration > target:
        warnings.append("为避免截断语句，计划时长小幅超过目标时长。")
    return {
        "schema_version": 1,
        "mode": "evidence_reel",
        "brief": brief or "保留最有证据价值的片段，并维持源视频时间顺序",
        "source": report["input"]["path"],
        "source_has_audio": bool(report["audio"]["has_audio_stream"]),
        "source_duration": duration,
        "requested_target_duration": float(target_duration),
        "planned_duration": planned_duration,
        "strategy": (
            "将 ASR 片段视为不可从中间切断的原子段，按证据价值选段后恢复源时间顺序"
            if speech_mode
            else "按 OCR 新增内容和完成状态词确定性打分；选段后恢复源时间顺序"
        ),
        "segments": segments,
        "warnings": warnings,
    }
