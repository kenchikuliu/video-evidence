# Benchmarking Reference

## Principle

Define visible or audible expectations before inspecting system outputs. Run every compared system on the same media and score the same manifest. Keep raw outputs and failures. Do not substitute qualitative impressions for the declared metric.

`score_manifest.py` reports weighted predeclared signal recall. It does not measure precision, false positives, temporal localization error, narrative understanding, advertising strategy, or edit quality.

## Manifest Format

```json
{
  "schema_version": 1,
  "name": "small-video-check",
  "cases": [
    {
      "id": "case-01",
      "analysis": "case-01/analysis.json",
      "signals": [
        {
          "id": "visible-export",
          "kind": "text",
          "source": "ocr",
          "any": ["export complete", "exported"]
        },
        {
          "id": "spoken-result",
          "kind": "text",
          "source": "asr",
          "any": ["the result is ready"]
        },
        {
          "id": "person-visible",
          "kind": "object",
          "any": ["person"]
        },
        {
          "id": "audible",
          "kind": "audio",
          "has_audio_stream": true,
          "silent": false
        },
        {
          "id": "multiple-scenes",
          "kind": "scenes",
          "min": 2,
          "max": 20,
          "weight": 0.5
        }
      ]
    }
  ]
}
```

Text alternatives use case-insensitive substring matching after whitespace normalization. Object alternatives match normalized COCO class names. Audio conditions are exact booleans. Scene counts are inclusive.

Paths in `analysis` are relative to `--analysis-root` when supplied, otherwise relative to the manifest. Without an explicit `analysis` field, the default is `<case-id>/analysis.json`.

## Run A Score

```powershell
python scripts/score_manifest.py benchmark.json `
  --analysis-root C:\path\system-results `
  --output C:\path\system-score.json
```

Use a separate results root for each system and the exact same manifest for each run.

## Required Reporting

Publish or retain:

- Source title, URL or local provenance, usage rights, file hash, and any trimming or transcoding.
- Manifest creation date, author, rationale, and immutable copy or hash.
- Tool, model, dependency, and hardware versions.
- Full commands and relevant thresholds.
- Per-case runtime and failures, including timeouts and missing optional models.
- Matched weight, total weight, recall, and per-signal details.
- Whether media, transcription, or model responses came from a cache.
- Metric limitations and any human adjudication procedure.

Do not publish copyrighted media merely to make a benchmark reproducible. Share lawful source links, hashes, time ranges, manifests, and derived aggregate measurements when redistribution rights are absent.

## Fair Comparison Checklist

- Give systems the same source media and time span.
- Keep expected signals independent of any one system's wording.
- Do not count a signal more than once unless its weight was declared in advance.
- Record optional components and unavailable stages separately.
- Do not call a component-level local tool an official end-to-end product result.
- Run repeated timing trials after warm-up when performance is part of the claim.
- Add a precision or false-positive study before claiming overall superiority from recall alone.
