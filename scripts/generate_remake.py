from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from video_providers import (
    DEFAULTS,
    ProviderError,
    build_payload,
    redact_payload,
    resolve_reference,
    submit_job,
)


def select_shots(spec: dict, shot_ids: list[str], all_shots: bool) -> list[dict]:
    shots = spec.get("shots", [])
    if not isinstance(shots, list) or not shots:
        raise ValueError("remake spec has no shots")
    if all_shots or not shot_ids:
        return shots
    requested = set(shot_ids)
    selected = [shot for shot in shots if str(shot.get("id")) in requested]
    missing = requested - {str(shot.get("id")) for shot in selected}
    if missing:
        raise ValueError(f"unknown shot ids: {sorted(missing)}")
    return selected


def validate_submission(
    shots: list[dict],
    submit: bool,
    confirm_paid_api: bool,
    explicit_selection: bool,
    reference_mode: str,
    allow_reference_upload: bool,
) -> None:
    if not submit:
        return
    if not confirm_paid_api:
        raise ValueError("paid submission requires --confirm-paid-api")
    if not explicit_selection:
        raise ValueError("paid submission requires --shot or --all-shots")
    unapproved = [str(shot.get("id")) for shot in shots if shot.get("approved") is not True]
    if unapproved:
        raise ValueError(f"unapproved shots cannot be submitted: {unapproved}")
    if reference_mode != "none" and not allow_reference_upload:
        raise ValueError("reference-guided submission requires --allow-reference-upload")


def write_run(path: Path, run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_resume_run(
    path: Path,
    provider: str,
    model: str,
    endpoint: str,
    api_key_env: str,
    submit: bool,
    reference_mode: str,
    allow_reference_upload: bool,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("kind") != "video_generation_run":
        raise ValueError("existing output is not a video_generation_run")
    expected_mode = "submitted" if submit else "dry_run"
    if run.get("provider") != provider or run.get("model") != model:
        raise ValueError("--resume requires the same provider and model as the existing run")
    if run.get("endpoint") != endpoint or run.get("api_key_env") != api_key_env:
        raise ValueError("--resume requires the same endpoint and API-key environment as the existing run")
    if run.get("execution", {}).get("mode") != expected_mode:
        raise ValueError("--resume requires the same dry-run or submitted mode as the existing run")
    if run.get("execution", {}).get("reference_upload_allowed") != allow_reference_upload:
        raise ValueError("--resume requires the same reference-upload setting as the existing run")
    if not isinstance(run.get("jobs"), list):
        raise ValueError("existing output has no valid jobs list")
    incompatible = [
        str(job.get("shot_id"))
        for job in run["jobs"]
        if not job.get("task_id")
        and job.get("reference_mode") not in (None, reference_mode)
    ]
    if incompatible:
        raise ValueError(
            f"--resume requires the same reference mode for incomplete jobs: {incompatible}"
        )
    return run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or submit MiniMax/Seedance generation jobs from a remake spec."
    )
    parser.add_argument("spec", type=Path)
    parser.add_argument("--provider", choices=tuple(DEFAULTS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--endpoint")
    parser.add_argument("--api-key-env")
    parser.add_argument("--resolution")
    parser.add_argument("--ratio")
    parser.add_argument("--shot", action="append", default=[])
    parser.add_argument("--all-shots", action="store_true")
    parser.add_argument("--reference-mode", choices=("none", "first-frame"), default="none")
    parser.add_argument("--allow-reference-upload", action="store_true")
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument("--watermark", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--confirm-paid-api", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run and skip jobs that already have a task id.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1 or spec.get("kind") != "video_remake_spec":
        raise ValueError("input is not a supported video_remake_spec")
    defaults = DEFAULTS[args.provider]
    provider_defaults = spec.get("provider_defaults", {}).get(args.provider, {})
    model = args.model or provider_defaults.get("model") or defaults["model"]
    endpoint = args.endpoint or defaults["endpoint"]
    resolution = (
        args.resolution or provider_defaults.get("resolution") or defaults["resolution"]
    )
    ratio = args.ratio or spec.get("output", {}).get("aspect_ratio", "16:9")
    api_key_env = args.api_key_env or defaults["api_key_env"]
    shots = select_shots(spec, args.shot, args.all_shots)
    validate_submission(
        shots,
        args.submit,
        args.confirm_paid_api,
        bool(args.shot or args.all_shots),
        args.reference_mode,
        args.allow_reference_upload,
    )

    output = (
        args.output or spec_path.with_name(f"{args.provider}_generation_run.json")
    ).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"output exists; pass --resume to continue it: {output}")
    run = (
        load_resume_run(
            output,
            args.provider,
            model,
            endpoint,
            api_key_env,
            args.submit,
            args.reference_mode,
            args.allow_reference_upload,
        )
        if args.resume
        else {
            "schema_version": 1,
            "kind": "video_generation_run",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_spec": str(spec_path),
            "provider": args.provider,
            "model": model,
            "endpoint": endpoint,
            "api_key_env": api_key_env,
            "output": spec.get("output", {}),
            "execution": {
                "mode": "submitted" if args.submit else "dry_run",
                "paid_api_submission": args.submit,
                "reference_upload_allowed": args.allow_reference_upload,
            },
            "jobs": [],
        }
    )
    existing_jobs = {
        str(job.get("shot_id")): job
        for job in run.get("jobs", [])
        if job.get("shot_id") is not None
    }
    existing_indexes = {
        str(job.get("shot_id")): index
        for index, job in enumerate(run.get("jobs", []))
        if job.get("shot_id") is not None
    }

    api_key = None
    if args.submit:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"missing API key environment variable: {api_key_env}")

    failures = 0
    for shot in shots:
        existing = existing_jobs.get(str(shot.get("id")))
        if args.resume and existing and existing.get("task_id"):
            continue
        if args.resume and existing and not args.submit and existing.get("status") == "planned":
            continue
        reference = None
        reference_note = None
        if args.reference_mode == "first-frame":
            if args.allow_reference_upload:
                reference_base = spec.get("source", {}).get("reference_base_dir")
                reference = resolve_reference(
                    shot.get("reference_frame"),
                    True,
                    Path(reference_base) if reference_base else spec_path.parent,
                )
            else:
                reference_note = "Reference omitted from dry run; upload was not allowed."
        payload = build_payload(
            args.provider,
            shot,
            model,
            resolution,
            ratio,
            reference,
            args.generate_audio,
            args.watermark,
        )
        job = {
            "shot_id": shot.get("id"),
            "order": shot.get("order"),
            "requested_duration": shot.get("requested_duration"),
            "generation_duration": payload.get("duration"),
            "reference_mode": args.reference_mode,
            "reference_note": reference_note,
            "request": redact_payload(payload),
            "status": "planned",
            "task_id": None,
            "download_url": None,
            "local_path": None,
        }
        if args.resume and str(shot.get("id")) in existing_indexes:
            run["jobs"][existing_indexes[str(shot.get("id"))]] = job
        else:
            run["jobs"].append(job)
        write_run(output, run)
        if not args.submit:
            continue
        try:
            submitted = submit_job(args.provider, endpoint, api_key or "", payload)
            job["status"] = "submitted"
            job["task_id"] = submitted["task_id"]
            job["submit_response"] = submitted["response"]
        except (ProviderError, OSError, ValueError) as error:
            failures += 1
            job["status"] = "submission_failed"
            job["error"] = str(error)
        write_run(output, run)

    summary = {
        "output": str(output),
        "provider": args.provider,
        "mode": run["execution"]["mode"],
        "jobs": len(run["jobs"]),
        "failures": failures,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
