from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def text_corpus(report: dict, source: str) -> str:
    values: list[str] = []
    if source in {"any", "ocr"}:
        for frame in report.get("timeline", []):
            values.extend(str(line.get("text", "")) for line in frame.get("ocr_lines", []))
    if source in {"any", "asr"}:
        values.extend(
            str(segment.get("text", ""))
            for segment in report.get("transcript", {}).get("segments", [])
        )
    return normalize_text("\n".join(values))


def require_list(signal: dict, key: str) -> list[Any]:
    value = signal.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"signal {signal.get('id', '<unknown>')} requires a non-empty '{key}' list")
    return value


def evaluate_signal(report: dict, signal: dict) -> tuple[bool, dict]:
    kind = signal.get("kind")
    if kind == "text":
        alternatives = [normalize_text(str(item)) for item in require_list(signal, "any")]
        source = signal.get("source", "any")
        if source not in {"any", "ocr", "asr"}:
            raise ValueError(f"unsupported text source: {source}")
        corpus = text_corpus(report, source)
        matches = [item for item in alternatives if item in corpus]
        return bool(matches), {"source": source, "matched_alternatives": matches}

    if kind == "object":
        expected = {normalize_text(str(item)) for item in require_list(signal, "any")}
        observed = {
            normalize_text(str(item))
            for item in report.get("objects", {}).get("unique_classes", [])
        }
        matches = sorted(expected & observed)
        return bool(matches), {"matched_classes": matches, "observed_classes": sorted(observed)}

    if kind == "audio":
        audio = report.get("audio", {})
        checks = []
        for key in ("has_audio_stream", "silent"):
            if key in signal:
                checks.append(audio.get(key) is signal[key])
        if not checks:
            raise ValueError("audio signals require 'has_audio_stream' and/or 'silent'")
        return all(checks), {
            "observed": {
                "has_audio_stream": audio.get("has_audio_stream"),
                "silent": audio.get("silent"),
            }
        }

    if kind == "scenes":
        count = len(report.get("scenes", []))
        minimum = int(signal.get("min", 0))
        maximum = int(signal.get("max", count))
        return minimum <= count <= maximum, {
            "observed_count": count,
            "expected_min": minimum,
            "expected_max": maximum,
        }

    raise ValueError(f"unsupported signal kind: {kind}")


def resolve_analysis(case: dict, manifest_path: Path, analysis_root: Path | None) -> Path:
    relative = Path(case.get("analysis", Path(str(case["id"])) / "analysis.json"))
    base = analysis_root if analysis_root is not None else manifest_path.parent
    return relative if relative.is_absolute() else base / relative


def score_manifest(manifest: dict, manifest_path: Path, analysis_root: Path | None) -> dict:
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest requires a non-empty cases list")

    case_results = []
    matched_weight = 0.0
    total_weight = 0.0
    for case in cases:
        if "id" not in case:
            raise ValueError("every case requires an id")
        signals = case.get("signals")
        if not isinstance(signals, list) or not signals:
            raise ValueError(f"case {case['id']} requires a non-empty signals list")
        analysis_path = resolve_analysis(case, manifest_path, analysis_root).resolve()
        report = json.loads(analysis_path.read_text(encoding="utf-8"))
        signal_results = []
        case_matched = 0.0
        case_total = 0.0
        seen_ids: set[str] = set()
        for signal in signals:
            signal_id = str(signal.get("id", ""))
            if not signal_id or signal_id in seen_ids:
                raise ValueError(f"case {case['id']} has a missing or duplicate signal id")
            seen_ids.add(signal_id)
            weight = float(signal.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"signal {signal_id} weight must be positive")
            matched, details = evaluate_signal(report, signal)
            earned = weight if matched else 0.0
            case_matched += earned
            case_total += weight
            signal_results.append(
                {
                    "id": signal_id,
                    "kind": signal.get("kind"),
                    "matched": matched,
                    "weight": weight,
                    "earned": earned,
                    "details": details,
                }
            )
        matched_weight += case_matched
        total_weight += case_total
        case_results.append(
            {
                "id": case["id"],
                "analysis": str(analysis_path),
                "matched_weight": case_matched,
                "total_weight": case_total,
                "recall": round(case_matched / case_total, 6),
                "signals": signal_results,
            }
        )

    return {
        "schema_version": 1,
        "metric": "predeclared_signal_recall",
        "matched_weight": matched_weight,
        "total_weight": total_weight,
        "recall": round(matched_weight / total_weight, 6),
        "cases": case_results,
        "limitations": [
            "This metric measures only predeclared signal recall.",
            "It does not penalize false positives or measure narrative understanding.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score analysis.json files against a predeclared evidence manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--analysis-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = score_manifest(
        manifest,
        manifest_path,
        args.analysis_root.resolve() if args.analysis_root else None,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
