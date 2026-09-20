# Video Evidence

A Codex skill and local CLI for turning video into an auditable evidence timeline and an editable structural-remake plan. It extracts media metadata, audio state, scene boundaries, sampled frames, and OCR. Optional stages add local Whisper transcription, YOLO11n object detection, grounded Ollama review, MiniMax/Seedance request planning, and deterministic assembly.

Analysis and generation dry-runs do not require an account, upload the source video, or call a paid API. Real cloud generation is opt-in and guarded separately.

## What It Produces

- `analysis.json`: structured metadata, timestamps, OCR boxes, optional ASR and object detections, and stable evidence IDs.
- `timeline.md`: readable timestamped evidence.
- `semantic_review.json` and `ANALYSIS.md`: optional local-model interpretation with citation validation.
- `edit_plan.json`, `BRIEF.md`, and `TREATMENT.md`: optional deterministic edit planning.
- `remake_spec.json`: provider-neutral, evidence-linked shot prompts with explicit approval state.
- `*_generation_run.json`: dry-run or submitted MiniMax/Seedance jobs without stored API keys.
- A rendered evidence reel after explicit FFmpeg rendering.

## Install As A Codex Skill

Ask Codex to install the public repository:

```text
Use $skill-installer to install https://github.com/kenchikuliu/video-evidence as video-evidence.
```

Alternatively, place this repository at:

```text
~/.codex/skills/video-evidence
```

Restart Codex after installation. Invoke it with `$video-evidence`, for example:

```text
Use $video-evidence to analyze C:\path\video.mp4, build an evidence-linked structural remake, and dry-run both providers without paid submission.
```

## Runtime Setup

Requirements:

- Python 3.10+
- FFmpeg and FFprobe on `PATH`

Install the core Python packages:

```powershell
python -m pip install -r requirements.txt
```

Optional speech transcription:

```powershell
python -m pip install -r requirements-audio.txt
```

Whisper and Ollama model weights may require a one-time download. They do not require a paid API, but local compute, storage, and bandwidth still have costs.

## CLI Usage

Basic local analysis:

```powershell
python scripts/analyze_video.py C:\path\video.mp4 `
  --output-dir C:\path\video-analysis
```

OCR plus sampled COCO object detection:

```powershell
python scripts/analyze_video.py C:\path\video.mp4 `
  --detect-objects --object-sample-interval 1 `
  --output-dir C:\path\video-analysis
```

Optional local transcription, grounded Ollama review, and edit planning:

```powershell
python scripts/analyze_video.py C:\path\video.mp4 `
  --transcribe --whisper-model base `
  --semantic-review --ollama-model qwen3-vl:8b `
  --brief "Preserve the key demonstrated steps and verified result" `
  --target-duration 30 `
  --output-dir C:\path\video-analysis
```

Render a reviewed plan:

```powershell
python scripts/render_edit.py C:\path\video-analysis\edit_plan.json `
  C:\path\video-analysis\edited.mp4
```

See [`references/workflow.md`](references/workflow.md) for the full workflow.

## Structural Remake With MiniMax Or Seedance

Create a provider-neutral shot specification:

```powershell
python scripts/remake_spec.py C:\path\video-analysis\analysis.json `
  --brief "Create an original vertical product ad with the same pacing roles" `
  --aspect-ratio 9:16 `
  --output C:\path\video-analysis\remake_spec.json
```

Dry-run either provider without an API key or network submission:

```powershell
python scripts/generate_remake.py C:\path\video-analysis\remake_spec.json `
  --provider minimax `
  --output C:\path\video-analysis\minimax_generation_run.json

python scripts/generate_remake.py C:\path\video-analysis\remake_spec.json `
  --provider seedance `
  --output C:\path\video-analysis\seedance_generation_run.json
```

Real submission requires reviewed shots with `approved: true`, explicit shot selection, `--submit`, and `--confirm-paid-api`. Uploading a reference frame additionally requires `--allow-reference-upload`. API keys are read only from `MINIMAX_API_KEY` or `ARK_API_KEY`, never from command arguments.

After submission, poll and download clips, then assemble them:

```powershell
python scripts/poll_generation.py C:\path\video-analysis\minimax_generation_run.json `
  --wait --download-dir C:\path\video-analysis\clips

python scripts/assemble_remake.py C:\path\video-analysis\minimax_generation_run.json `
  C:\path\video-analysis\remake.mp4
```

See [`references/remake-generation.md`](references/remake-generation.md) for provider contracts, cost guards, model limits, and the complete workflow.

## Reproducible Scoring

Score one or more `analysis.json` files against expectations written before viewing system outputs:

```powershell
python scripts/score_manifest.py benchmark.json `
  --analysis-root C:\path\results `
  --output C:\path\score.json
```

This is a signal-recall metric. It does not penalize false positives or measure narrative understanding. See [`references/benchmarking.md`](references/benchmarking.md) before publishing comparisons.

## Development Evidence

During development, the pipeline was evaluated on 10 real brand advertisements with the same 33 predeclared signals for both systems:

| Pipeline | Signal recall | Runtime |
| --- | ---: | ---: |
| Hypit local media tools + VideoHighlighter 0.11.3 | 21.5 / 33 (65.15%) | 302.314 s |
| Video Evidence with YOLO11n ONNX | 29.1 / 33 (88.18%) | 1894.987 s |

Important limits: the baseline was not Hypit's official hosted model; object items measured recall only and did not penalize false positives; all 8 declared object items were found; YOLO added 63.827 seconds; strict Qwen citation validation passed 0 of 10 cases. These results show broader signal capture in this test, not general semantic superiority. The copyrighted advertisements and generated caches are intentionally excluded from this repository.

## Privacy And Scope

- Analysis stays local unless you explicitly move or publish its artifacts.
- Generated frames and transcripts may contain sensitive information.
- Cloud generation uploads prompts and any explicitly enabled reference media to the selected provider.
- OCR, ASR, and sampled detections are incomplete observations, not ground truth.
- The semantic layer must cite evidence and must not overwrite deterministic output.
- Structural similarity is not an exact copy and cannot guarantee viral performance.

## License

AGPL-3.0. The bundled YOLO11n ONNX model is also distributed under AGPL-3.0. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
