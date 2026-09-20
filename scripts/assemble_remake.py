from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DIMENSIONS = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1": (1080, 1080),
    "4:3": (960, 720),
    "3:4": (720, 960),
}


def run_command(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def has_audio(path: Path) -> bool:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ]
    )
    return bool(result.stdout.strip())


def completed_jobs(run: dict) -> list[dict]:
    jobs = sorted(run.get("jobs", []), key=lambda item: int(item.get("order", 0)))
    if not jobs:
        raise ValueError("generation run has no jobs")
    incomplete = [job.get("shot_id") for job in jobs if not job.get("local_path")]
    if incomplete:
        raise ValueError(f"jobs have no downloaded clip: {incomplete}")
    return jobs


def build_ffmpeg_args(
    run: dict,
    output: Path,
    width: int | None = None,
    height: int | None = None,
    overwrite: bool = False,
) -> list[str]:
    jobs = completed_jobs(run)
    ratio = run.get("output", {}).get("aspect_ratio", "16:9")
    default_width, default_height = DIMENSIONS.get(ratio, DIMENSIONS["16:9"])
    width = width or default_width
    height = height or default_height
    if width <= 0 or height <= 0:
        raise ValueError("output dimensions must be positive")

    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n"]
    paths = []
    for job in jobs:
        path = Path(str(job["local_path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
        args.extend(["-i", str(path)])

    filters = []
    concat_inputs = []
    for index, (job, path) in enumerate(zip(jobs, paths)):
        duration = float(job.get("requested_duration") or 0)
        if duration <= 0:
            raise ValueError(f"job {job.get('shot_id')} has invalid requested_duration")
        filters.append(
            f"[{index}:v:0]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=30,"
            f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        if has_audio(path):
            filters.append(
                f"[{index}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,"
                f"atrim=duration={duration:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )
        else:
            filters.append(
                f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )
        concat_inputs.extend((f"[v{index}]", f"[a{index}]"))
    filters.append(
        "".join(concat_inputs) + f"concat=n={len(jobs)}:v=1:a=1[vout][aout]"
    )
    args.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble downloaded generation clips in source order with FFmpeg."
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run = json.loads(args.run.resolve().read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_ffmpeg_args(run, output, args.width, args.height, args.overwrite))
    print(json.dumps({"output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
