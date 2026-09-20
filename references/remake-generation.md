# Structural Remake And Generation

Use this workflow to reinterpret the timing and narrative structure of an analyzed video with MiniMax or Seedance. It does not guarantee an exact copy or viral performance.

## 1. Build The Draft Specification

Start from a completed `analysis.json`:

```powershell
python scripts/remake_spec.py C:\path\analysis.json `
  --brief "Create an original product demonstration with a fast hook and clear payoff" `
  --aspect-ratio 9:16 `
  --max-shot-duration 6 `
  --output C:\path\remake_spec.json
```

Add `--semantic-review C:\path\semantic_review.json` only when that review contains validated grounded events. The builder ignores ungrounded events.

The default `--rights-mode structure-only` preserves timing and narrative roles while instructing the generator to replace source-specific identities, logos, music, characters, and brand assets. Use `--rights-mode owned-source` only when the user states that the source material may be reused.

Each shot contains:

- Source interval and requested output duration.
- Narrative role: hook, progression, or payoff/call to action.
- Editable visual prompt.
- OCR text and ASR dialogue kept outside the visual prompt for later post-production.
- Evidence IDs and grounded events.
- A local reference frame for human review.
- `approved: false` until a person or agent checks the shot.

Review the JSON and set `approved` to `true` only for shots that are ready to incur generation cost.

## 2. Dry-Run Provider Requests

MiniMax:

```powershell
python scripts/generate_remake.py C:\path\remake_spec.json `
  --provider minimax `
  --output C:\path\minimax_generation_run.json
```

Seedance through Volcengine Ark:

```powershell
python scripts/generate_remake.py C:\path\remake_spec.json `
  --provider seedance `
  --output C:\path\seedance_generation_run.json
```

Dry-run is the default. It does not read an API key or contact either generation endpoint. Compare the generated request payloads, generated durations, model, resolution, and shot count before choosing a provider.

The output file is protected from accidental replacement. If a local process was interrupted, rerun the same command with `--resume`; existing planned dry-run jobs and jobs that already have a provider task ID are retained.

Current defaults, verified against official documentation and official SDK examples on 2026-09-20:

| Provider | Default model | Base endpoint | Credential environment variable |
| --- | --- | --- | --- |
| MiniMax | `MiniMax-Hailuo-2.3` | `https://api.minimax.io` | `MINIMAX_API_KEY` |
| Seedance | `doubao-seedance-2-0-260128` | `https://ark.cn-beijing.volces.com/api/v3` | `ARK_API_KEY` |

Override `--model`, `--endpoint`, `--resolution`, or `--ratio` when the account exposes a different model or region. Provider model catalogs change; do not assume a default remains available.

MiniMax v1 accepts model-dependent 6- or 10-second generations. The adapter generates at the supported duration and the assembler trims to `requested_duration`. Seedance duration support is model-dependent; the adapter accepts 2-12 seconds and rounds fractional durations up.

## 3. Submit An Explicit Paid Run

Never put API keys in command arguments or JSON files. Set the provider's environment variable through the user's normal secret-management method.

Submit one reviewed MiniMax shot:

```powershell
python scripts/generate_remake.py C:\path\remake_spec.json `
  --provider minimax `
  --shot shot-001 `
  --submit --confirm-paid-api `
  --output C:\path\minimax_generation_run.json
```

Submit all reviewed Seedance shots:

```powershell
python scripts/generate_remake.py C:\path\remake_spec.json `
  --provider seedance `
  --all-shots `
  --submit --confirm-paid-api `
  --output C:\path\seedance_generation_run.json
```

Submission is rejected when:

- `--confirm-paid-api` is missing.
- Neither `--shot` nor `--all-shots` was specified.
- Any selected shot is not approved.
- Reference-guided generation was requested without `--allow-reference-upload`.
- The expected API-key environment variable is absent.

If submission is interrupted after some tasks were created, rerun the same command with `--resume`. Jobs with an existing task ID are skipped, while a job recorded as `submission_failed` can be retried. Keep the provider, model, endpoint, and API-key environment unchanged when resuming.

Text-to-video is the default. For a user-authorized first-frame upload, add:

```powershell
--reference-mode first-frame --allow-reference-upload
```

Local reference images are converted to data URLs in memory and omitted from the persisted request log. They are still uploaded to the selected provider during submission.

## 4. Poll, Download, And Assemble

Query once:

```powershell
python scripts/poll_generation.py C:\path\minimax_generation_run.json
```

Wait for terminal states and download completed clips:

```powershell
python scripts/poll_generation.py C:\path\minimax_generation_run.json `
  --wait --timeout 1800 `
  --download-dir C:\path\clips
```

Once all jobs are terminal, polling or downloading the run does not require the API key; pending jobs still require the key for status queries.

Assemble downloaded clips in source order:

```powershell
python scripts/assemble_remake.py C:\path\minimax_generation_run.json `
  C:\path\remake.mp4
```

The assembler normalizes dimensions and frame rate, retains generated audio when present, inserts silence when absent, trims each generated clip to the requested source timing, and re-encodes an H.264/AAC MP4. It refuses to overwrite an existing output unless `--overwrite` is supplied.

OCR text and ASR dialogue are planning references; they are not automatically burned into the generated picture. Add reviewed typography, licensed music, voice, and final mix in a deterministic post-production stage.

## Provider Contracts

MiniMax v1:

- Create: `POST /v1/video_generation`
- Query: `GET /v1/query/video_generation?task_id=...`
- Resolve download: `GET /v1/files/retrieve?file_id=...`
- Official documentation: <https://platform.minimax.io/docs/api-reference/video-generation-t2v>
- Image-to-video: <https://platform.minimax.io/docs/api-reference/video-generation-i2v>

Seedance / Volcengine Ark:

- Create: `POST /api/v3/contents/generations/tasks`
- Query: `GET /api/v3/contents/generations/tasks/{task_id}`
- Successful response URL: `content.video_url`
- Official task documentation: <https://www.volcengine.com/docs/82379/1520757>
- Official Python SDK: <https://github.com/volcengine/volcengine-python-sdk>

Signed download URLs can expire. Download successful outputs promptly and keep the local generation run file private because it may temporarily contain those URLs.
