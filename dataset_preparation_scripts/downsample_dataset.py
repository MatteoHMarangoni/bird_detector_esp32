#!/usr/bin/env python3
"""
downsample_dataset.py

Run from VS Code: open this file and click ▶︎ “Run Python File”.

What this script does
---------------------
Clones an audio dataset while downsampling WAV files from (typically) 48 kHz to 16 kHz,
preserving:

- folder structure
- filenames
- number of channels
- non-audio files (copied verbatim)

Input/output folders
--------------------
This script assumes a repo layout like:

  <repo>/
    raw_data/          <-- INPUT dataset folder (you place your dataset here)
    downsampled/       <-- OUTPUT dataset folder (this script creates/writes here)
    resample_dataset.py

So you don’t pass command line arguments: it always works relative to the script.

Resampling engine
-----------------
- If ffmpeg is installed, it is used (fast and robust).
- If not, it falls back to Python (soundfile + soxr).

Dependencies for Python fallback:
  pip install soundfile soxr tqdm

Notes
-----
- By default, only WAV files are resampled.
- If a WAV file is not 48 kHz, it is copied as-is (unless FORCE=True).
"""

import concurrent.futures as cf
import os
import shutil
import subprocess
from pathlib import Path

# ================== CONFIG (TUNABLE SETTINGS) ==================

# Folders are resolved relative to this script file (repo-friendly).
INPUT_FOLDER_NAME = "data_raw"
OUTPUT_FOLDER_NAME = "downsampled"

# Parallel workers (ThreadPoolExecutor is OK here because ffmpeg is external work / IO bound)
JOBS = os.cpu_count() or 4

# If False: only resample WAVs that are actually ASSUMED_INPUT_SR (default 48k), otherwise copy.
# If True: resample ALL WAVs to TARGET_SR regardless of input sample rate.
FORCE = False

# If True: do not write anything, just print what would happen (if VERBOSE=True).
DRY_RUN = False

# If True: print every action and ffmpeg errors.
VERBOSE = False

# Target sample rate (downsample to this)
TARGET_SR = 16000

# "Expected" WAV input samplerate. Used to decide whether to resample or just copy.
ASSUMED_INPUT_SR = 48000

# WAV file extensions to resample
WAV_EXTS = {".wav", ".wave"}

# ===============================================================

# Skip macOS metadata and hidden junk
JUNK_NAMES = {".DS_Store"}


def is_junk(p: Path) -> bool:
    n = p.name
    return n.startswith("._") or n in JUNK_NAMES


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def ensure_dir(dst_file: Path):
    dst_file.parent.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, dry_run: bool, verbose: bool):
    ensure_dir(dst)
    if dry_run:
        if verbose:
            print(f"[DRY] COPY  {src} -> {dst}")
        return True
    shutil.copy2(src, dst)
    if verbose:
        print(f"[OK ] COPY  {src} -> {dst}")
    return True


def probe_samplerate_with_ffprobe(path: Path) -> int | None:
    try:
        cp = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate",
                "-of",
                "default=nw=1:nk=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        sr_str = cp.stdout.strip()
        return int(sr_str) if sr_str.isdigit() else None
    except Exception:
        return None


def resample_with_ffmpeg(src: Path, dst: Path, dry_run: bool, verbose: bool, force: bool):
    # Decide whether to resample or just copy (if not 48k and not forcing)
    in_sr = probe_samplerate_with_ffprobe(src)
    if in_sr is not None and in_sr != ASSUMED_INPUT_SR and not force:
        return copy_file(src, dst, dry_run, verbose)

    ensure_dir(dst)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ar",
        str(TARGET_SR),
        "-c:a",
        "pcm_s16le",  # 16-bit PCM WAV output; preserves channels
        str(dst),
    ]
    if dry_run:
        if verbose:
            print("[DRY] " + " ".join(cmd))
        return True

    cp = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        if verbose:
            print(f"[ERR] ffmpeg failed for {src} -> {dst}\n{cp.stderr}")
        return False

    if verbose:
        print(f"[OK ] FFMPEG {src} -> {dst}")
    return True


def resample_with_python(src: Path, dst: Path, dry_run: bool, verbose: bool, force: bool):
    """
    Fallback path using soundfile + soxr (streaming, memory-friendly).
    Only used if ffmpeg is not available or fails.
    """
    try:
        import soundfile as sf
        import soxr
    except Exception as e:
        if verbose:
            print("[ERR] Python fallback requires 'soundfile' and 'soxr'. Error:", e)
        return False

    # Check input SR; if not 48k and not forcing, copy it
    try:
        with sf.SoundFile(str(src), "r") as f:
            in_sr = f.samplerate
            channels = f.channels
            subtype = f.subtype or "PCM_16"
            fmt = f.format or "WAV"
    except Exception as e:
        if verbose:
            print(f"[ERR] Cannot open {src}: {e}")
        return False

    if in_sr != ASSUMED_INPUT_SR and not force:
        return copy_file(src, dst, dry_run, verbose)

    if dry_run:
        if verbose:
            print(f"[DRY] PYRESAMP {src} -> {dst}")
        return True

    try:
        ensure_dir(dst)
        # We keep output as WAV for consistency if input was WAV; for other formats we only reach here via WAV_EXTS anyway.
        out_format = "WAV"
        out_subtype = "PCM_16"  # safe and widely compatible
        with sf.SoundFile(str(src), "r") as fin:
            with sf.SoundFile(str(dst), "w", samplerate=TARGET_SR, channels=channels, format=out_format, subtype=out_subtype) as fout:
                io_ratio = TARGET_SR / float(fin.samplerate)
                resamp = soxr.Resampler(quality="VHQ", channels=channels, io_ratio=io_ratio, dtype="float32")

                BLOCK = 65536
                while True:
                    data = fin.read(frames=BLOCK, dtype="float32", always_2d=True)
                    if data.size == 0:
                        break
                    out = resamp.process(data)
                    if out.size:
                        fout.write(out)

                tail = resamp.flush()
                if tail.size:
                    fout.write(tail)

        if verbose:
            print(f"[OK ] PYRESAMP {src} -> {dst}")
        return True
    except Exception as e:
        if verbose:
            print(f"[ERR] Resample failed for {src}: {e}")
        return False


def handle_one(args_tuple):
    """
    Worker task:
    - If WAV: resample to TARGET_SR (if needed) using ffmpeg or python fallback.
    - Otherwise: copy file verbatim.
    """
    src, dst_root, use_ffmpeg, dry_run, verbose, force, src_root = args_tuple
    rel = src.relative_to(src_root)
    dst = dst_root / rel
    ext = src.suffix.lower()

    if ext in WAV_EXTS:
        if use_ffmpeg:
            ok = resample_with_ffmpeg(src, dst, dry_run, verbose, force)
            if not ok:  # try python fallback if ffmpeg failed
                ok = resample_with_python(src, dst, dry_run, verbose, force)
            return ok
        return resample_with_python(src, dst, dry_run, verbose, force)

    return copy_file(src, dst, dry_run, verbose)


def collect_files(src_root: Path):
    return [p for p in src_root.rglob("*") if p.is_file() and not is_junk(p)]


def main():
    # Resolve folders relative to this script (repo-friendly)
    script_dir = Path(__file__).resolve().parent
    src_root = (script_dir / INPUT_FOLDER_NAME).resolve()
    dst_root = (script_dir / OUTPUT_FOLDER_NAME).resolve()

    if not src_root.exists() or not src_root.is_dir():
        raise SystemExit(
            f"Input folder not found: {src_root}\n"
            f"Create it and put your dataset inside: {INPUT_FOLDER_NAME}/"
        )

    dst_root.mkdir(parents=True, exist_ok=True)

    use_ff = has_ffmpeg()
    if VERBOSE:
        print(f"Using ffmpeg+ffprobe: {use_ff}")
        print(f"Source: {src_root}")
        print(f"Dest:   {dst_root}")

    files = collect_files(src_root)
    if not files:
        print("No files found in input folder.")
        return

    try:
        from tqdm import tqdm
        iterator = tqdm
    except Exception:
        def iterator(x, **kwargs):
            return x

    tasks = [(f, dst_root, use_ff, DRY_RUN, VERBOSE, FORCE, src_root) for f in files]
    total = len(tasks)
    ok_count = 0

    with cf.ThreadPoolExecutor(max_workers=JOBS) as ex:
        for ok in iterator(ex.map(handle_one, tasks), total=total, desc="Processing"):
            ok_count += bool(ok)

    failed = total - ok_count
    print(f"\nDone. OK: {ok_count} / {total}  Failed: {failed}")
    if failed and not VERBOSE:
        print("Tip: set VERBOSE=True at the top to see error details.")


if __name__ == "__main__":
    main()
