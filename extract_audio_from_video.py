#!/usr/bin/env python3
import argparse
import shutil
import subprocess
from pathlib import Path

VIDEO_EXTS_DEFAULT = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v", ".mpeg", ".mpg"
}

def extract_wav(ffmpeg: str, in_path: Path, out_path: Path, sr: int | None, mono: bool, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg]
    if overwrite:
        cmd.append("-y")

    cmd += [
        "-i", str(in_path),
        "-vn",              # no video
        "-c:a", "pcm_s16le" # PCM 16-bit for WAV
    ]

    if sr is not None:
        cmd += ["-ar", str(sr)]  # sample rate
    if mono:
        cmd += ["-ac", "1"]      # 1 channel

    cmd.append(str(out_path))

    # Capture stderr so you can see ffmpeg errors if something fails
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def main() -> int:
    ap = argparse.ArgumentParser(description="Extract WAV audio from all videos in a directory using ffmpeg.")
    ap.add_argument("--input_dir", type=Path, default="../datasets/ravdess/videos/Actor_01", help="Directory containing video files (searched recursively).")
    ap.add_argument("--output_dir", type=Path, default="../datasets/ravdess/wav", help="Directory to write .wav files into.")
    ap.add_argument("--sr", type=int, default=None, help="Optional sample rate (e.g., 16000).")
    ap.add_argument("--mono", action="store_true", help="Force mono output.")
    ap.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing WAVs.")
    ap.add_argument("--ext", action="append", default=None, help="Video extension to include (repeatable), e.g. --ext .mp4")

    args = ap.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found on PATH. Please install ffmpeg and try again.")

    exts = set(e.lower() for e in (args.ext or VIDEO_EXTS_DEFAULT))
    overwrite = not args.no_overwrite

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print(f"No video files found in {input_dir} (extensions: {sorted(exts)}).")
        return 0

    ok, failed = 0, 0
    for in_path in files:
        rel = in_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".wav")

        if out_path.exists() and not overwrite:
            continue

        try:
            extract_wav(ffmpeg, in_path, out_path, args.sr, args.mono, overwrite)
            ok += 1
        except subprocess.CalledProcessError as e:
            failed += 1
            err = e.stderr.decode("utf-8", errors="replace")
            print(f"[FAIL] {in_path}\n{err}\n")

    print(f"Done. Extracted: {ok}, Failed: {failed}")
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
