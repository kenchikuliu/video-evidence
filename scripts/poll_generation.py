from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from video_providers import ProviderError, download_file, query_job


TERMINAL = {"succeeded", "failed", "submission_failed"}


def write_run(path: Path, run: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def pending_jobs(run: dict) -> list[dict]:
    return [
        job
        for job in run.get("jobs", [])
        if job.get("task_id") and job.get("status") not in TERMINAL
    ]


def poll_once(run: dict, api_key: str) -> None:
    provider = run["provider"]
    endpoint = run["endpoint"]
    for job in pending_jobs(run):
        try:
            result = query_job(provider, endpoint, api_key, str(job["task_id"]))
            job["status"] = result["status"]
            job["provider_status"] = result["provider_status"]
            job["download_url"] = result["download_url"]
            job["query_response"] = result["response"]
        except (ProviderError, OSError, ValueError) as error:
            job["last_query_error"] = str(error)


def download_ready(run: dict, directory: Path) -> int:
    failures = 0
    for job in run.get("jobs", []):
        if job.get("status") != "succeeded" or job.get("local_path"):
            continue
        url = job.get("download_url")
        if not url:
            failures += 1
            job["download_error"] = "provider reported success without a download URL"
            continue
        shot_id = str(job.get("shot_id", ""))
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", shot_id):
            failures += 1
            job["download_error"] = f"unsafe shot id for download filename: {shot_id!r}"
            continue
        output = directory / f"{shot_id}.mp4"
        try:
            download_file(str(url), output)
            job["local_path"] = str(output.resolve())
        except (ProviderError, OSError) as error:
            failures += 1
            job["download_error"] = str(error)
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poll submitted MiniMax/Seedance jobs and optionally download clips."
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--download-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_interval <= 0 or args.timeout <= 0:
        raise ValueError("poll interval and timeout must be positive")
    run_path = args.run.resolve()
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("kind") != "video_generation_run":
        raise ValueError("input is not a video_generation_run")
    api_key = None
    if pending_jobs(run):
        key_name = run.get("api_key_env")
        api_key = os.environ.get(str(key_name))
        if not api_key:
            raise ValueError(f"missing API key environment variable: {key_name}")

    started = time.monotonic()
    while True:
        remaining = pending_jobs(run)
        if not remaining:
            break
        poll_once(run, api_key or "")
        write_run(run_path, run)
        remaining = pending_jobs(run)
        if not args.wait or not remaining:
            break
        if time.monotonic() - started >= args.timeout:
            break
        time.sleep(min(args.poll_interval, max(0.0, args.timeout - (time.monotonic() - started))))

    download_failures = 0
    if args.download_dir:
        directory = args.download_dir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        download_failures = download_ready(run, directory)
        write_run(run_path, run)
    counts: dict[str, int] = {}
    for job in run.get("jobs", []):
        status = str(job.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    print(
        json.dumps(
            {
                "run": str(run_path),
                "statuses": counts,
                "download_failures": download_failures,
            },
            ensure_ascii=False,
        )
    )
    return 1 if download_failures or counts.get("failed") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
