#!/usr/bin/env python3
"""
split_into_1s_chunks.py

Run from VS Code: open this file and click ▶︎ “Run Python File”.

What this script does
---------------------
This script creates a *chunked copy* of a WAV dataset:

- Input folder:  ./data_raw     (next to this script)
- Output folder: ./data_split   (next to this script)

For every WAV file found in data_raw (recursively), it writes 1-second WAV chunks
to data_split while preserving:

- the same folder structure (subfolders are mirrored)
- the original filename stem (chunk index appended)

Non-audio files are copied verbatim so the output tree mirrors the input tree.

Speed notes
-----------
- For WAV files, this script uses Python's built-in `wave` module (fast, no ffmpeg).
- Uses a ThreadPoolExecutor to process many files concurrently (good for SSD + many small files).
- Writing many tiny files can still be limited by filesystem metadata overhead.

Tunable settings
----------------
See the CONFIG section below.
"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import sys
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, Tuple

from tqdm import tqdm

# ================== CONFIG (TUNABLE SETTINGS) ==================

INPUT_FOLDER_NAME = "data_raw"
# Folder containing dataset to segment (next to this script)

OUTPUT_FOLDER_NAME = "data_split"
# Folder where chunks will be written (next to this script)

CHUNK_SEC = 1.0
# Length of each chunk in seconds (> 0). The last chunk may be shorter.

# Threading: good for SSD + many small files
JOBS = min(16, (os.cpu_count() or 8))
# Suggested: 8–16 on Apple Silicon SSD.
# If you notice the system becoming sluggish, reduce this.

# What counts as audio for segmentation
WAV_EXTS: Set[str] = {".wav", ".wave"}

# If True, print per-file actions (slower / noisy)
VERBOSE = False

# ===============================================================

JUNK_NAMES = {".DS_Store"}


def is_junk_name(name: str) -> bool:
    return name.startswith("._") or name in JUNK_NAMES


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def iter_files(root: Path) -> Iterable[Path]:
    """Yield all files under root recursively, skipping common macOS junk."""
    for p in root.rglob("*"):
        if p.is_file() and not is_junk_name(p.name):
            yield p


def copy_verbatim(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def split_wav_with_wave(src_path: Path, dst_dir: Path, chunk_sec: float) -> int:
    """
    Split a WAV file into chunk_sec chunks using Python's built-in wave module.
    Returns the number of chunks written.
    """
    ensure_dir(dst_dir)
    stem = src_path.stem
    chunks_written = 0

    with contextlib.closing(wave.open(str(src_path), "rb")) as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        nf = wf.getnframes()

        frames_per_chunk = max(1, int(fr * chunk_sec))
        num_chunks = int(math.ceil(nf / frames_per_chunk))

        for i in range(num_chunks):
            start = i * frames_per_chunk
            wf.setpos(start)
            to_read = min(frames_per_chunk, nf - start)
            frames = wf.readframes(to_read)

            out_path = dst_dir / f"{stem}_chunk{i:03d}.wav"
            with contextlib.closing(wave.open(str(out_path), "wb")) as out:
                out.setnchannels(nch)
                out.setsampwidth(sw)
                out.setframerate(fr)
                out.writeframes(frames)

            chunks_written += 1

    return chunks_written


@dataclass
class Result:
    rel_path: str
    ok: bool
    chunks: int
    kind: str  # "wav" | "copy" | "skip"
    error: str


def process_one(src: Path, input_root: Path, output_root: Path, chunk_sec: float) -> Result:
    """
    Process one file:
    - WAV -> split into 1s chunks
    - Non-WAV -> copy verbatim
    """
    rel = src.relative_to(input_root)
    ext = src.suffix.lower()

    try:
        if ext in WAV_EXTS:
            dst_dir = output_root / rel.parent
            n = split_wav_with_wave(src, dst_dir, chunk_sec)
            return Result(str(rel), True, n, "wav", "")
        else:
            dst = output_root / rel
            copy_verbatim(src, dst)
            return Result(str(rel), True, 0, "copy", "")
    except Exception as e:
        return Result(str(rel), False, 0, "error", f"{type(e).__name__}: {e}")


def main():
    if CHUNK_SEC <= 0:
        print("CHUNK_SEC must be > 0", file=sys.stderr)
        return

    script_dir = Path(__file__).resolve().parent
    input_root = (script_dir / INPUT_FOLDER_NAME).resolve()
    output_root = (script_dir / OUTPUT_FOLDER_NAME).resolve()

    if not input_root.exists():
        print(f"Input folder not found: {input_root}", file=sys.stderr)
        print(f"Create it and put your dataset inside: {INPUT_FOLDER_NAME}/", file=sys.stderr)
        return

    ensure_dir(output_root)

    files = list(iter_files(input_root))
    if not files:
        print("No files found in input folder.")
        return

    print("=== Dataset Splitter (threaded, WAV-fast path) ===")
    print("Input: ", input_root)
    print("Output:", output_root)
    print("Chunk length (s):", CHUNK_SEC)
    print("Threads (JOBS):", JOBS)
    print("Total files scanned:", len(files))
    print()

    total_audio = 0
    total_other = 0
    total_chunks = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=JOBS) as ex:
        futures = [ex.submit(process_one, p, input_root, output_root, CHUNK_SEC) for p in files]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            r: Result = fut.result()

            if r.kind == "wav":
                total_audio += 1
                total_chunks += r.chunks
                if VERBOSE:
                    print(f"[OK ] {r.rel_path} -> {r.chunks} chunk(s)")
            elif r.kind == "copy":
                total_other += 1
                if VERBOSE:
                    print(f"[COPY] {r.rel_path}")
            else:
                errors += 1
                print(f"[ERR] {r.rel_path}: {r.error}", file=sys.stderr)

    print("\nDone.")
    print("Audio files processed:", total_audio)
    print("Non-audio files copied:", total_other)
    print("Chunks written:", total_chunks)
    print("Errors:", errors)


if __name__ == "__main__":
    main()
