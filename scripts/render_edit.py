from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence


def build_ffmpeg_args(plan: dict, output_path: Path) -> list[str]:
    segments = plan.get("segments", [])
    if not segments:
        raise ValueError("edit plan has no segments")
    has_audio = bool(plan.get("source_has_audio"))
    filters = []
    concat_inputs = []
    for index, segment in enumerate(segments):
        start = float(segment["source_start"])
        end = float(segment["source_end"])
        filters.append(
            f"[0:v:0]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")
        if has_audio:
            filters.append(
                f"[0:a:0]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )
            concat_inputs.append(f"[a{index}]")

    if len(segments) == 1:
        filters.append("[v0]null[vout]")
        if has_audio:
            filters.append("[a0]anull[aout]")
    else:
        filters.append(
            "".join(concat_inputs)
            + f"concat=n={len(segments)}:v=1:a={1 if has_audio else 0}[vout]"
            + ("[aout]" if has_audio else "")
        )

    args = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(Path(plan["source"])),
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[vout]",
    ]
    if has_audio:
        args.extend(["-map", "[aout]"])
    args.extend(["-c:v", "libx264", "-crf", "18", "-preset", "medium"])
    if has_audio:
        args.extend(["-c:a", "aac", "-b:a", "192k"])
    args.extend(["-movflags", "+faststart", str(output_path)])
    return args


def run_command(args: Sequence[str]) -> None:
    subprocess.run(args, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an evidence-reel edit plan with FFmpeg.")
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_command(build_ffmpeg_args(plan, args.output.resolve()))
    print(json.dumps({"output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
