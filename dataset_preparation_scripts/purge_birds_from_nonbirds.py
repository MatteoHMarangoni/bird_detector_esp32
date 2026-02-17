#!/usr/bin/env python3
"""
purge_birds_from_nonbirds.py

What this script does
---------------------
This script is meant to clean a folder of audio that is *supposed to be NON-BIRD* by
automatically separating chunks that contain bird detections from chunks that do not.

Workflow per input WAV file:
1) Run BirdNET ONCE on the full file to get time-stamped detections (selection table).
2) Split the WAV into fixed-length chunks (CHUNK_SEC).
3) For each chunk window, if any BirdNET detection overlaps the window and confidence >= MIN_CONF:
      -> COPY the chunk into BIRD_OUT
   else
      -> COPY the chunk into CLEAN_OUT
4) Temporary chunk files are deleted, and temporary folders are cleaned up.

Default folder layout (relative to this script)
----------------------------------------------
dataset_preparation_scripts/
  purge_birds_from_nonbirds.py
  data_raw/                  <-- input: put your "non-bird" recordings here (recursively)
  bird_out/                  <-- output: chunks that contain birds (COPIES)
  clean_out/                 <-- output: chunks with no birds (COPIES)
  _tmp_purge_work/           <-- temp working directory (auto-cleaned)

Requirements
------------
- Python 3
- birdnet-analyzer:  pip install birdnet-analyzer
- ffmpeg (optional but recommended) for robust chunk splitting

How to run
----------
Press "Run" / "Play" in VS Code, or:
python3 purge_birds_from_nonbirds.py
"""

import contextlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import set_start_method
from pathlib import Path
from shutil import which


# ================== CONFIG (ADJUST THESE DURING OPERATION) ==================

# Script folder (everything defaults relative to this)
SCRIPT_DIR = Path(__file__).resolve().parent

# INPUT: place source WAVs here (recursively)
INPUT_ROOT = SCRIPT_DIR / "data_raw"

# OUTPUTS: created next to the script
BIRD_OUT = SCRIPT_DIR / "bird_out"     # <-- chunks with bird detections
CLEAN_OUT = SCRIPT_DIR / "clean_out"   # <-- chunks without bird detections

# Temporary work (auto-cleaned)
TMP_ROOT = SCRIPT_DIR / "_tmp_purge_work"

# Chunking & detection behavior
CHUNK_SEC = 3        # <-- chunk length in seconds (common: 1, 2, 3)
MIN_CONF = 0.10      # <-- confidence threshold to classify a chunk as "bird"
SENSITIVITY = 1.0    # <-- BirdNET sensitivity (higher => more detections + more false positives)

# Parallelism (tune for your machine)
MAX_WORKERS = 4      # <-- number of source files processed in parallel
THREADS = "2"        # <-- BirdNET internal threads per process (usually 1–2)
BATCHSIZE = "1"      # <-- BirdNET batch size

# Location (improves BirdNET predictions) — adjust to your recording location
LAT = 52.070
LON = 4.300

# Optional: infer ISO week from filenames that start with YYYY-MM-DD (BirdNET can use this context)
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")

# ===========================================================================


def fmt_eta(seconds: float | None) -> str:
    """Format seconds as ETA: HH:MM:SS (or --:--:-- if unknown)."""
    if seconds is None or seconds <= 0 or not math.isfinite(seconds):
        return "ETA --:--:--"
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"ETA {h:02d}:{m:02d}:{s:02d}"


def infer_week_from_filename(filename: str):
    """If filename begins with YYYY-MM-DD, return ISO week number; else None."""
    m = DATE_RE.match(filename)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return datetime(y, mo, d).isocalendar()[1]
    except Exception:
        return None


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def has_ffmpeg() -> bool:
    return which("ffmpeg") is not None


def split_with_ffmpeg(src_path: Path, dst_dir: Path, base_stem: str, chunk_sec: int) -> list[Path]:
    """
    Split WAV into fixed-length segments with ffmpeg.

    Note:
    - We re-encode to PCM (pcm_s16le) to avoid edge cases with `-c copy` and odd WAV encodings.
    """
    ensure_dirs(dst_dir)
    pattern = dst_dir / f"{base_stem}_chunk%03d.wav"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_sec),
        "-segment_start_number",
        "0",
        "-c:a",
        "pcm_s16le",
        str(pattern),
    ]
    subprocess.run(cmd, check=True)

    chunks: list[Path] = []
    idx = 0
    while True:
        p = dst_dir / f"{base_stem}_chunk{idx:03d}.wav"
        if p.exists():
            chunks.append(p)
            idx += 1
        else:
            break
    return chunks


def split_with_wave(src_path: Path, dst_dir: Path, base_stem: str, chunk_sec: int) -> list[Path]:
    """Pure-Python WAV splitter (fallback if ffmpeg is not available)."""
    ensure_dirs(dst_dir)
    chunks: list[Path] = []

    with contextlib.closing(wave.open(str(src_path), "rb")) as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        nf = wf.getnframes()

        frames_per_chunk = int(fr * chunk_sec)
        num_chunks = int(math.ceil(nf / frames_per_chunk))

        for i in range(num_chunks):
            start = i * frames_per_chunk
            wf.setpos(start)
            to_read = min(frames_per_chunk, nf - start)
            frames = wf.readframes(to_read)

            out_path = dst_dir / f"{base_stem}_chunk{i:03d}.wav"
            with contextlib.closing(wave.open(str(out_path), "wb")) as out:
                out.setnchannels(nch)
                out.setsampwidth(sw)
                out.setframerate(fr)
                out.writeframes(frames)

            chunks.append(out_path)

    return chunks


def split_into_chunks(src_path: Path, tmp_dir: Path, chunk_sec: int) -> tuple[Path, list[Path]]:
    """
    Split src_path into chunks under a dedicated folder:
        tmp_dir/<base_stem>/
    Returns (chunk_dir, chunk_paths).
    """
    base_stem = src_path.stem
    chunk_dir = tmp_dir / base_stem
    ensure_dirs(chunk_dir)

    if has_ffmpeg():
        return chunk_dir, split_with_ffmpeg(src_path, chunk_dir, base_stem, chunk_sec)
    return chunk_dir, split_with_wave(src_path, chunk_dir, base_stem, chunk_sec)


def clean_selection_tables(dir_path: Path) -> None:
    """Remove BirdNET selection tables created in a directory tree."""
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith(".selection.table.txt") or f.lower().endswith(".csv"):
                try:
                    (Path(root) / f).unlink()
                except Exception:
                    pass


def parse_table_rows(table_path: Path):
    """
    Returns a list of detections:
      (begin_time, end_time, common_name, species_code, confidence)

    Parses header to find column indexes; supports tab or comma.
    Skips 'nocall'.
    """
    if not table_path.is_file():
        return []

    with table_path.open(errors="replace") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return []

    header = lines[0]
    rows = lines[1:]
    sep = "\t" if "\t" in header else ","
    cols = [c.strip() for c in header.split(sep)]

    def col_idx(name: str):
        name = name.lower()
        for i, c in enumerate(cols):
            if c.lower() == name:
                return i
        return None

    i_bt = col_idx("begin time (s)")
    i_et = col_idx("end time (s)")
    i_cn = col_idx("common name")
    i_sc = col_idx("species code")
    i_cf = col_idx("confidence")

    dets = []
    for ln in rows:
        parts = [p.strip() for p in ln.split(sep)]
        if any(idx is None or idx >= len(parts) for idx in (i_bt, i_et, i_cn, i_sc, i_cf)):
            continue

        cn = parts[i_cn]
        sc = parts[i_sc]
        if "nocall" in (cn + " " + sc).lower():
            continue

        try:
            bt = float(parts[i_bt])
            et = float(parts[i_et])
            cf = float(parts[i_cf].replace(",", "."))
        except ValueError:
            continue

        dets.append((bt, et, cn, sc, cf))

    return dets


def analyze_file_once(
    src_path: Path,
    out_dir: Path,
    lat: float,
    lon: float,
    week,
    min_conf: float,
    sensitivity: float,
    threads: str,
    batchsize: str,
):
    """
    Run BirdNET ONCE on the full source file, writing results into out_dir,
    and return (detections, table_path).
    """
    base = src_path.stem
    cmd = [
        sys.executable,
        "-m",
        "birdnet_analyzer.analyze",
        str(src_path),
        "-o",
        str(out_dir),
        "--min_conf",
        str(min_conf),
        "--rtype",
        "table",
        "--skip_existing_results",
        "--sensitivity",
        str(sensitivity),
        "--lat",
        str(lat),
        "--lon",
        str(lon),
        "-t",
        threads,
        "-b",
        batchsize,
    ]
    if week:
        cmd.extend(["--week", str(week)])

    subprocess.run(cmd, check=True)
    table_path = out_dir / f"{base}.BirdNET.selection.table.txt"
    return parse_table_rows(table_path), table_path


def route_chunks_by_overlap(
    chunk_paths: list[Path],
    detections,
    chunk_sec: int,
    bird_dir: Path,
    clean_dir: Path,
    min_conf: float,
) -> tuple[int, int]:
    """
    For each chunk window [k*chunk_sec, (k+1)*chunk_sec), classify as "bird" if
    any detection overlaps AND confidence >= min_conf.

    IMPORTANT:
    - This COPIES chunks to outputs (does not move), leaving raw inputs untouched.
    - Temp chunk files are deleted after copying to keep TMP_ROOT clean.
    """
    copied_bird = copied_clean = 0

    for idx, ch_path in enumerate(chunk_paths):
        s = idx * chunk_sec
        e = s + chunk_sec

        birdy = False
        for bt, et, cn, sc, cf in detections:
            if cf < min_conf:
                continue
            if (bt < e) and (et > s):  # time overlap
                birdy = True
                break

        dest_dir = bird_dir if birdy else clean_dir
        ensure_dirs(dest_dir)
        dest = dest_dir / ch_path.name

        # COPY (not move)
        shutil.copy2(ch_path, dest)

        # Remove temp chunk file after copying
        try:
            ch_path.unlink()
        except Exception:
            pass

        if birdy:
            copied_bird += 1
        else:
            copied_clean += 1

    return copied_bird, copied_clean


def process_one_file(args):
    """
    Worker: process a single source WAV (analyze once, split, route).
    Returns (analyzed_inc, bird_chunks, clean_chunks, msg)
    """
    (
        src_path_str,
        input_root_str,
        tmp_root_str,
        bird_out_str,
        clean_out_str,
        chunk_sec,
        lat,
        lon,
        min_conf,
        sensitivity,
        threads,
        batchsize,
    ) = args

    src_path = Path(src_path_str)
    input_root = Path(input_root_str)
    tmp_root = Path(tmp_root_str)
    bird_out = Path(bird_out_str)
    clean_out = Path(clean_out_str)

    try:
        # Preserve relative folder structure of input inside outputs
        rel = Path(os.path.relpath(src_path.parent, input_root))
        tmp_dir = tmp_root / rel
        bird_dir = bird_out / rel
        clean_dir = clean_out / rel
        ensure_dirs(tmp_dir, bird_dir, clean_dir)

        week = infer_week_from_filename(src_path.name)

        # 1) Analyze ONCE on the full file (table is written next to source)
        detections, table_path = analyze_file_once(
            src_path=src_path,
            out_dir=src_path.parent,
            lat=lat,
            lon=lon,
            week=week,
            min_conf=min_conf,
            sensitivity=sensitivity,
            threads=threads,
            batchsize=batchsize,
        )

        # 2) Split to chunks (in tmp_dir)
        chunk_dir, chunk_paths = split_into_chunks(src_path, tmp_dir, chunk_sec)

        # 3) Route by overlap (COPY to outputs; delete temp chunks)
        b, c = route_chunks_by_overlap(chunk_paths, detections, chunk_sec, bird_dir, clean_dir, min_conf)

        # 4) Cleanup BirdNET tables next to source + any tmp tables
        try:
            if table_path.is_file():
                table_path.unlink()
        except Exception:
            pass

        clean_selection_tables(tmp_dir)

        # 5) Remove now-empty chunk directory (and prune empty parents under tmp_dir)
        try:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        except Exception:
            pass

        # Try to remove empty tmp subfolders up the tree (stop at TMP_ROOT)
        try:
            p = tmp_dir
            while p != tmp_root and p.exists():
                if any(p.iterdir()):
                    break
                p.rmdir()
                p = p.parent
        except Exception:
            pass

        return (1, b, c, str(src_path))

    except subprocess.CalledProcessError as e:
        return (0, 0, 0, f"BirdNET error on {src_path}: {e}")
    except Exception as e:
        return (0, 0, 0, f"Worker error on {src_path}: {e}")


def main():
    input_root = INPUT_ROOT.resolve()
    bird_out = BIRD_OUT.resolve()
    clean_out = CLEAN_OUT.resolve()
    tmp_root = TMP_ROOT.resolve()

    print(f"INPUT_ROOT: {input_root}")
    print(f"BIRD_OUT:   {bird_out}")
    print(f"CLEAN_OUT:  {clean_out}")
    print(f"TMP_ROOT:   {tmp_root}")

    if not input_root.exists():
        print(f"\nERROR: INPUT_ROOT does not exist:\n  {input_root}\n")
        print("Create it and place WAV files inside, e.g.:")
        print(f"  {input_root}")
        return

    ensure_dirs(bird_out, clean_out, tmp_root)

    # Gather work items (absolute file paths)
    work = []
    for root, _, files in os.walk(input_root):
        for fn in files:
            if fn.startswith("._") or not fn.lower().endswith(".wav"):
                continue
            work.append(
                (
                    str(Path(root) / fn),
                    str(input_root),
                    str(tmp_root),
                    str(bird_out),
                    str(clean_out),
                    CHUNK_SEC,
                    LAT,
                    LON,
                    MIN_CONF,
                    SENSITIVITY,
                    THREADS,
                    BATCHSIZE,
                )
            )

    total = len(work)
    print(f"Found {total} WAV(s).")
    print(f"Launching {MAX_WORKERS} workers (BirdNET -t {THREADS} per process).")

    # macOS/Apple Silicon: explicit 'spawn' is safest for multiprocessing
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    if not work:
        print("Nothing to do.")
        return

    analyzed_files = copied_bird_chunks = copied_clean_chunks = 0
    done = 0
    start_ts = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_file, w) for w in work]
        for fut in as_completed(futures):
            a, b, c, msg = fut.result()
            analyzed_files += a
            copied_bird_chunks += b
            copied_clean_chunks += c
            done += 1

            elapsed = max(1e-6, time.time() - start_ts)
            rate = done / elapsed
            remaining = total - done
            eta_s = (remaining / rate) if rate > 0 else None

            if a == 1:
                print(
                    f"[{done}/{total}] ✓ {Path(msg).name} (+{b} bird, +{c} clean) | "
                    f"rate={rate:.2f} files/s | {fmt_eta(eta_s)}"
                )
            else:
                print(
                    f"[{done}/{total}] ✗ {msg} | "
                    f"rate={rate:.2f} files/s | {fmt_eta(eta_s)}"
                )

    # Final cleanup: remove TMP_ROOT if empty (or just keep it around)
    try:
        if tmp_root.exists() and not any(tmp_root.iterdir()):
            tmp_root.rmdir()
    except Exception:
        pass

    print("\nDone.")
    print(f"Analyzed original files: {analyzed_files}")
    print(f"Copied BIRD chunks:      {copied_bird_chunks}")
    print(f"Copied CLEAN chunks:     {copied_clean_chunks}")


if __name__ == "__main__":
    main()
