---
name: video-evidence
description: "Analyze videos locally into auditable evidence timelines using FFprobe, scene detection, OCR, optional Whisper transcription, YOLO object detection, grounded Ollama review, and reproducible manifest scoring. Use when Codex needs to inspect, compare, summarize, benchmark, or create an evidence-based edit plan for a local video without uploading it or using a paid API."
---

# Video Evidence

Create a local evidence record before making semantic claims about a video. Treat timestamps, OCR boxes, ASR segments, object detections, and media metadata as observations. Treat summaries and edit recommendations as interpretations.

## Run The Core Workflow

1. Locate this skill's directory and use the scripts from its `scripts/` subdirectory.
2. Confirm `ffmpeg`, `ffprobe`, Python, and the packages in `requirements.txt` are available. Install dependencies only when needed and only with the user's authorization if that changes their environment.
3. Choose an output directory outside the skill directory. Never write generated frames, reports, or edited videos into the installed skill.
4. Run the deterministic analysis first:

```powershell
python <skill-dir>/scripts/analyze_video.py <video> --output-dir <output-dir>
```

5. Inspect `analysis.json` and `timeline.md`. Cite timestamps and evidence IDs in conclusions. State sampling gaps and failed stages rather than filling them with assumptions.

Read [references/workflow.md](references/workflow.md) for commands, outputs, optional stages, and failure handling.

## Enable Optional Evidence Only When Relevant

- Add `--detect-objects` for coarse COCO object recall. The bundled YOLO11n ONNX model runs locally on CPU.
- Add `--transcribe` for spoken content. Install `requirements-audio.txt` first. Whisper model weights may download on first use.
- Add `--target-duration <seconds>` to produce an auditable edit plan. Render it only after checking selected source intervals.
- Add `--semantic-review` only when a local Ollama vision model is already available or the user approves downloading one. This step uses the configured local endpoint and must not replace deterministic evidence.
- Use `scripts/render_edit.py` to render a checked `edit_plan.json` with FFmpeg.

Do not require an account, upload media, or call a paid API. Do not claim that local processing is cost-free when the machine, bandwidth, or model downloads still have resource costs.

## Keep Claims Grounded

- Prefer direct wording such as "OCR detected X at 03.4s" over intent claims.
- Report `semantic_review.validation.passed` and grounded-event counts whenever using semantic output.
- Do not present ungrounded model events as facts. A valid JSON response is not the same as a grounded response.
- Describe YOLO results as sampled detections, not exhaustive object tracking. COCO recall does not measure false positives, action recognition, brand recognition, or narrative understanding.
- Describe scene boundaries as algorithmic cut candidates, not editorial truth.
- Preserve the original media path and analysis parameters in reproducibility notes, but redact private paths before publishing artifacts.

## Benchmark Reproducibly

Write expectations before looking at a system's output, then score each system against the same manifest:

```powershell
python <skill-dir>/scripts/score_manifest.py <manifest.json> --analysis-root <results-dir> --output <score.json>
```

Read [references/benchmarking.md](references/benchmarking.md) before comparing systems. Report the manifest, source provenance, versions, runtime, failures, denominators, and metric limitations. Never describe a local-media-tool baseline as an official hosted model result unless it actually is one.

## Protect User Data

- Keep source videos and generated frames local unless the user explicitly requests publication and owns the rights.
- Exclude videos, analysis outputs, model caches, tokens, and machine-specific absolute paths from repositories.
- Do not infer sensitive traits from faces, voices, or surroundings.
- Stop before rendering over an existing output unless the overwrite is explicitly intended.
