# Workflow Reference

## Prerequisites

- Python 3.10 or newer.
- `ffmpeg` and `ffprobe` available on `PATH`.
- Packages from `requirements.txt` installed in the selected Python environment.
- Enough disk space for extracted JPEG frames and JSON reports.

Install the core Python dependencies from the skill directory:

```powershell
python -m pip install -r requirements.txt
```

Use a virtual environment when changing the user's Python environment would be undesirable.

## Deterministic Analysis

Run the core evidence pass:

```powershell
python scripts/analyze_video.py C:\path\input.mp4 `
  --output-dir C:\path\output
```

The default sampler combines evenly spaced frames with detected scene midpoints and caps the result at 24 frames. Use explicit timestamps for a controlled comparison:

```powershell
python scripts/analyze_video.py C:\path\input.mp4 `
  --timestamps 0.5,3.4,6.4,9.3 `
  --output-dir C:\path\output
```

Core artifacts:

| Artifact | Meaning |
| --- | --- |
| `analysis.json` | Machine-readable media, audio, scenes, OCR, optional ASR/objects, evidence IDs, parameters, and paths. |
| `timeline.md` | Human-readable timestamped OCR timeline. |
| `frames/*.jpg` | Sampled source frames used for OCR and optional semantic review. |

## Object Detection

The bundled YOLO11n ONNX model recognizes 80 COCO classes from sampled frames:

```powershell
python scripts/analyze_video.py C:\path\input.mp4 `
  --detect-objects `
  --object-sample-interval 1 `
  --object-confidence 0.3 `
  --output-dir C:\path\output
```

Use `--object-classes "person,car,cell phone"` to retain only selected COCO classes. A missing class means it was not detected in sampled frames; it does not prove the object is absent from the full video.

## Local Speech Transcription

Install the optional dependency:

```powershell
python -m pip install -r requirements-audio.txt
```

Then request transcription:

```powershell
python scripts/analyze_video.py C:\path\input.mp4 `
  --transcribe --whisper-model base `
  --output-dir C:\path\output
```

Faster Whisper may download model weights on first use. For a configured CUDA machine, add `--whisper-device cuda --whisper-compute-type float16`. The analyzer skips model loading when the input has no audio stream or is effectively silent.

## Grounded Semantic Review

Only use this stage with a local Ollama endpoint and a vision-capable model:

```powershell
python scripts/analyze_video.py C:\path\input.mp4 `
  --semantic-review --ollama-model qwen3-vl:8b `
  --brief "Identify the demonstrated workflow and its verified result" `
  --output-dir C:\path\output
```

Additional artifacts:

- `semantic_review.json` contains the raw model response, parsed content, citations, and validation details.
- `ANALYSIS.md` shows only the human-readable interpretation and grounded event details.
- `validation.passed` is strict: unknown evidence IDs, time-misaligned citations, or ungrounded events cause failure.

A failed validation is still useful diagnostic output, but its events must not be reported as verified facts.

## Edit Planning And Rendering

Generate a plan without rendering:

```powershell
python scripts/analyze_video.py C:\path\input.mp4 `
  --brief "Preserve the key demonstrated steps and final verification" `
  --target-duration 30 `
  --output-dir C:\path\output
```

Review `edit_plan.json` and `TREATMENT.md`, then render:

```powershell
python scripts/render_edit.py C:\path\output\edit_plan.json C:\path\output\edited.mp4
```

The plan keeps source order. ASR segments are atomic, so a rendered result may exceed the target slightly rather than cut a sentence in half. Always watch the rendered file and check cuts, audio continuity, legibility, and factual accuracy.

## Failure Handling

- If `ffprobe` or `ffmpeg` is missing, stop and report the missing executable.
- If OCR finds no text, report that no text passed the configured confidence threshold; do not label the video blank.
- If object detection finds nothing, report sampling interval, class filter, and confidence threshold.
- If Ollama is unavailable, retain deterministic artifacts and omit semantic review.
- If a stage fails, do not silently reuse output from an older run.
- Before sharing reports, remove source paths, frames, transcripts, or other private content that the user did not approve for publication.
