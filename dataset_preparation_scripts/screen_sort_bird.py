#!/usr/bin/env python3
"""
screen_sort_bird.py

What this script does
---------------------
1) Recursively scans an INPUT folder for .wav files.
2) Splits each input WAV into short chunks (CHUNK_SEC).
   - Uses ffmpeg if available (fast), otherwise falls back to Python's wave module.
3) Runs BirdNET (birdnet_analyzer) on batches of chunk files.
4) For each chunk, reads BirdNET’s selection table and finds the top non-"nocall" detection.
5) Copies the ORIGINAL chunk file into:
   - OUTPUT_ROOT/<prev_species_token>/   if confidence >= MIN_CONF
   - DISCARDED_ROOT/<prev_species_token>/ otherwise (low confidence / no hit)

Default folder layout (relative to this script)
----------------------------------------------
dataset_preparation_scripts/
  screen_sort_bird.py
  data_raw/                          <-- input (put your source WAVs here)
  output_sorted/                     <-- created automatically
  output_discarded_lowconf/          <-- created automatically
  _tmp_birdnet/                      <-- temp working area (created automatically)

Requirements
------------
- Python 3
- birdnet-analyzer:  pip install birdnet-analyzer
- ffmpeg (optional but recommended): for fast splitting

Run
---
python3 screen_sort_bird.py
"""

import contextlib
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import wave
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

# ================== CONFIG (ADJUST THESE DURING OPERATION) ==================

# Base directory is the folder containing THIS script:
SCRIPT_DIR = Path(__file__).resolve().parent

# Input folder (default): dataset_preparation_scripts/data_raw
# Put your original recordings here.
INPUT_ROOT = SCRIPT_DIR / "data_raw"

# Output folders (default): created next to the script
OUTPUT_ROOT = SCRIPT_DIR / "output_sorted"
DISCARDED_ROOT = SCRIPT_DIR / "output_discarded_lowconf"

# Temp working folders (kept local to repo for portability)
TMP_ROOT = SCRIPT_DIR / "_tmp_birdnet"
TMP_TABLE_DIR = TMP_ROOT / "birdnet_tables"
TMP_BATCH_ROOT = TMP_ROOT / "birdnet_batches"

# Thresholds / BirdNET behavior
MIN_CONF = 0.10        # <-- raise to be stricter (e.g. 0.2), lower to keep more (e.g. 0.05)
SENSITIVITY = 1.0      # <-- BirdNET sensitivity (1.0 neutral; higher can increase detections/noise)
CHUNK_SEC = 1          # <-- chunk length in seconds (common values: 1, 2, 3)

# Location used by BirdNET to improve predictions
LAT = 52.330           # <-- adjust to your recording location
LON = 4.891            # <-- adjust to your recording location

# Padding added to each chunk BEFORE BirdNET analysis (helps with very short chunks)
PAD_BEFORE_SEC = 1     # <-- set to 0 to disable
PAD_AFTER_SEC = 1      # <-- set to 0 to disable

# Parallelism / batching
MAX_WORKERS = 6        # <-- processes in parallel (try 4–10 depending on CPU/RAM)
THREADS = "1"          # <-- BirdNET threads per process (usually keep 1–2)
BATCHSIZE = "1"        # <-- BirdNET batch size; short clips rarely benefit from >1
GROUP_SIZE = 600       # <-- files per BirdNET invocation (try 300–1200)
MAX_INFLIGHT = MAX_WORKERS * 3

# Logging
QUIET_BIRDNET = True
REPORT_EVERY_SEC = 60
REPORT_EVERY_COUNT = 1000

# ===========================================================================


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def collect_files(root: Path):
    files = []
    for dirpath, _, fns in os.walk(root):
        for fn in fns:
            if fn.startswith("._"):  # macOS resource fork artifacts
                continue
            if fn.lower().endswith(".wav"):
                files.append(Path(dirpath) / fn)
    return files


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def split_with_ffmpeg(src_path: Path, dst_dir: Path, base_stem: str, chunk_sec: int):
    ensure_dir(dst_dir)
    pattern = str(dst_dir / f"{base_stem}_chunk%02d.wav")
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
        "1",
        "-c:a",
        "pcm_s16le",
        pattern,
    ]
    subprocess.run(cmd, check=True)

    out = []
    idx = 1
    while True:
        p = dst_dir / f"{base_stem}_chunk{idx:02d}.wav"
        if p.exists():
            out.append(p)
            idx += 1
        else:
            break
    return out


def split_with_wave(src_path: Path, dst_dir: Path, base_stem: str, chunk_sec: int):
    ensure_dir(dst_dir)
    out = []
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

            idx = i + 1
            out_path = dst_dir / f"{base_stem}_chunk{idx:02d}.wav"
            with contextlib.closing(wave.open(str(out_path), "wb")) as out_w:
                out_w.setnchannels(nch)
                out_w.setsampwidth(sw)
                out_w.setframerate(fr)
                out_w.writeframes(frames)

            out.append(out_path)
    return out


def split_into_chunks(src_path: Path, tmp_dir: Path, chunk_sec: int):
    base_stem = src_path.stem
    chunk_dir = tmp_dir / base_stem
    ensure_dir(chunk_dir)

    if has_ffmpeg():
        return split_with_ffmpeg(src_path, chunk_dir, base_stem, chunk_sec)
    return split_with_wave(src_path, chunk_dir, base_stem, chunk_sec)


def fmt_eta(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "ETA: --:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"ETA: {h:02d}:{m:02d}:{s:02d}"


def create_all_chunks(input_root: Path, tmp_chunks_root: Path, chunk_sec: int):
    ensure_dir(tmp_chunks_root)
    all_chunks = []

    files = collect_files(input_root)
    total = len(files)
    if total == 0:
        print(f"No input WAV files found in: {input_root}")
        return []

    start_ts = time.time()
    last_report_ts = start_ts
    processed = 0
    last_report_count = 0

    for src in files:
        processed += 1
        try:
            chs = split_into_chunks(src, tmp_chunks_root, chunk_sec)
            all_chunks.extend(chs)
            print(f"[{processed}/{total}] Split {src.name} -> {len(chs)} chunk(s)")
        except Exception as e:
            print(f"[{processed}/{total}] Failed to split {src.name}: {e}")

        now = time.time()
        time_ok = (now - last_report_ts) >= REPORT_EVERY_SEC
        count_ok = (processed - last_report_count) >= REPORT_EVERY_COUNT
        if time_ok or count_ok:
            elapsed = max(1e-6, now - start_ts)
            rate = processed / elapsed
            remaining = max(0, total - processed)
            eta = fmt_eta(remaining / rate if rate > 0 else None)
            print(
                f"Progress: {processed}/{total} files processed | "
                f"chunks so far={len(all_chunks)} | rate={rate:.2f} files/s | {eta}"
            )
            last_report_ts = now
            last_report_count = processed

    return all_chunks


def parse_table_rows(table_path: Path):
    """
    Returns list of detections:
      (begin_time, end_time, common_name, species_code, confidence)
    Skips 'nocall'. Supports tab or comma-separated tables.
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

    def col_idx(name):
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


def sanitize_species_folder(name: str) -> str:
    s = (name or "").strip().replace(" ", "_")
    return re.sub(r"[^\w\.-]+", "_", s) or "Unknown_Species"


def sanitize_for_filename(name: str) -> str:
    if not name:
        return "Unknown"
    s = name.strip().replace(" ", "_")
    return re.sub(r"[^\w\.-]+", "_", s)


def extract_prev_species_from_stem(stem: str) -> str:
    """
    Tries to infer the *previous* species token from the original filename.
    This script groups outputs into folders based on that previous token
    (useful for “screening” a pre-labeled dataset).

    Prefer token after 'bird_' in filename:
      ...bird_Eurasian_Magpie... -> Eurasian_Magpie

    Also strips trailing "_chunkNN" so folders don't include chunk suffixes.
    """
    stem = re.sub(r"_chunk\d+$", "", stem, flags=re.IGNORECASE)

    m = re.search(r"(?i)bird_([A-Za-z_]+)", stem)
    if m:
        return m.group(1)

    parts = re.split(r"[_\-]+", stem)
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if re.search(r"[A-Za-z]", a) and not re.search(r"\d", a) and re.search(r"[A-Za-z]", b) and not re.search(r"\d", b):
            j = i + 2
            while j < len(parts) and re.search(r"[A-Za-z]", parts[j]) and not re.search(r"\d", parts[j]):
                j += 1
            return "_".join(parts[i:j])

    for p in parts:
        if re.search(r"[A-Za-z]", p) and not re.search(r"\d", p):
            return p

    return "UnknownPrev"


def pad_with_zeros_to_path(src_path: Path, dst_path: Path, pad_before_sec: int, pad_after_sec: int):
    """
    Writes a padded WAV at dst_path:
      [silence pad_before] + [src audio] + [silence pad_after]
    """
    ensure_dir(dst_path.parent)
    with contextlib.closing(wave.open(str(src_path), "rb")) as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        nf = wf.getnframes()
        frames = wf.readframes(nf)

        nb_before = int(fr * pad_before_sec)
        nb_after = int(fr * pad_after_sec)
        bytes_per_frame = nch * sw

        silence_before = b"\x00" * (nb_before * bytes_per_frame)
        silence_after = b"\x00" * (nb_after * bytes_per_frame)

        tmp_path = dst_path.with_suffix(dst_path.suffix + ".pad.tmp")
        with contextlib.closing(wave.open(str(tmp_path), "wb")) as out:
            out.setnchannels(nch)
            out.setsampwidth(sw)
            out.setframerate(fr)
            out.writeframes(silence_before + frames + silence_after)

        tmp_path.replace(dst_path)


def analyze_batch(filepaths, out_root: Path, discarded_root: Path, tmp_tables: Path, batch_root: Path, min_conf: float):
    """
    Expects filepaths to be individual chunk WAV files (already split).
    Creates padded copies in a temp batch dir for BirdNET; saves ORIGINAL chunks to outputs.
    """
    kept = filtered = nohit = errors = 0
    completed = len(filepaths)

    ensure_dir(batch_root)
    tmp_batch_dir = Path(tempfile.mkdtemp(prefix="bn_batch_", dir=str(batch_root)))

    try:
        # Materialize padded copies for BirdNET analysis
        for src in filepaths:
            src = Path(src)
            target = tmp_batch_dir / src.name
            try:
                if PAD_BEFORE_SEC > 0 or PAD_AFTER_SEC > 0:
                    pad_with_zeros_to_path(src, target, PAD_BEFORE_SEC, PAD_AFTER_SEC)
                else:
                    shutil.copy2(src, target)
            except Exception:
                # If padding fails, try a plain copy and keep going
                try:
                    shutil.copy2(src, target)
                except Exception:
                    continue

        ensure_dir(tmp_tables)

        cmd = [
            sys.executable,
            "-m",
            "birdnet_analyzer.analyze",
            str(tmp_batch_dir),
            "-o",
            str(tmp_tables),
            "--min_conf",
            "0.0",
            "--rtype",
            "table",
            "--skip_existing_results",
            "--sensitivity",
            str(SENSITIVITY),
            "--lat",
            str(LAT),
            "--lon",
            str(LON),
            "-t",
            THREADS,
            "-b",
            BATCHSIZE,
        ]

        if QUIET_BIRDNET:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, check=True)

        for src in filepaths:
            src = Path(src)
            base = src.stem
            table = tmp_tables / f"{base}.BirdNET.selection.table.txt"

            detections = parse_table_rows(table)

            # cleanup table ASAP
            try:
                if table.is_file():
                    table.unlink()
            except Exception:
                pass

            best_species = None
            best_conf = None
            for bt, et, cn, sc, cf in detections:
                if best_conf is None or cf > best_conf:
                    best_conf = cf
                    best_species = cn

            original_filename = src.name
            prev_species_token = extract_prev_species_from_stem(src.stem)
            prev_folder = sanitize_species_folder(prev_species_token)

            if best_species is None:
                nohit += 1
                try:
                    dest_dir = discarded_root / prev_folder
                    ensure_dir(dest_dir)
                    conf_str = f"{(best_conf or 0.0):.3f}"
                    new_name = f"{conf_str}-NoHit-{original_filename}"
                    shutil.copy2(src, dest_dir / new_name)
                except Exception:
                    errors += 1
                continue

            pred_name_clean = sanitize_for_filename(best_species)
            conf_str = f"{(best_conf or 0.0):.3f}"

            if best_conf is not None and best_conf >= min_conf:
                try:
                    dest_dir = out_root / prev_folder
                    ensure_dir(dest_dir)
                    new_name = f"{conf_str}-{pred_name_clean}-{original_filename}"
                    shutil.copy2(src, dest_dir / new_name)
                    kept += 1
                except Exception:
                    errors += 1
            else:
                filtered += 1
                try:
                    dest_dir = discarded_root / prev_folder
                    ensure_dir(dest_dir)
                    new_name = f"{conf_str}-{pred_name_clean}-{original_filename}"
                    shutil.copy2(src, dest_dir / new_name)
                except Exception:
                    errors += 1

    except subprocess.CalledProcessError:
        errors += len(filepaths)
    except Exception:
        errors += len(filepaths)
    finally:
        try:
            shutil.rmtree(tmp_batch_dir, ignore_errors=True)
        except Exception:
            pass

    return (kept, filtered, nohit, errors, completed)


def main():
    # Normalize to Path objects (in case someone edits config to strings)
    input_root = Path(INPUT_ROOT).resolve()
    output_root = Path(OUTPUT_ROOT).resolve()
    discarded_root = Path(DISCARDED_ROOT).resolve()
    tmp_tables = Path(TMP_TABLE_DIR).resolve()
    tmp_batches = Path(TMP_BATCH_ROOT).resolve()

    print(f"INPUT_ROOT:     {input_root}")
    print(f"OUTPUT_ROOT:    {output_root}")
    print(f"DISCARDED_ROOT: {discarded_root}")
    print(f"TMP_ROOT:       {Path(TMP_ROOT).resolve()}")

    if CHUNK_SEC <= 0:
        raise ValueError("CHUNK_SEC must be > 0 seconds")

    ensure_dir(output_root)
    ensure_dir(discarded_root)
    ensure_dir(tmp_tables)
    ensure_dir(tmp_batches)

    if not input_root.exists():
        print(f"\nERROR: INPUT_ROOT does not exist:\n  {input_root}\n")
        print("Create it and place WAV files inside, e.g.:")
        print(f"  {input_root}")
        return

    # Dedicated temp chunks dir
    tmp_chunks_dir = Path(tempfile.mkdtemp(prefix="bn_chunks_", dir=str(tmp_batches)))

    try:
        chunk_files = create_all_chunks(input_root, tmp_chunks_dir, CHUNK_SEC)
    except Exception:
        chunk_files = []

    total_files = len(chunk_files)
    if total_files == 0:
        print("No chunk WAV files found after splitting.")
        shutil.rmtree(tmp_chunks_dir, ignore_errors=True)
        return

    batches = list(chunks(chunk_files, GROUP_SIZE))
    total_batches = len(batches)
    print(f"Found {total_files} chunk WAV(s) → {total_batches} batches of ~{GROUP_SIZE} each.")
    print(f"Launching {MAX_WORKERS} workers (BirdNET -t {THREADS}).")

    interrupted = {"flag": False}

    def handle_sigint(sig, frame):
        if not interrupted["flag"]:
            print("\n⚠️ Ctrl+C received — finishing in-flight batches and stopping new submissions…")
            interrupted["flag"] = True
        else:
            print("Force exit requested. Exiting now.")
            os._exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    submitted_batches = completed_batches = 0
    kept = filtered = nohit = errors = 0
    start_ts = time.time()
    last_report_ts = start_ts
    last_completed_for_count = 0

    def maybe_report(force=False):
        nonlocal last_report_ts, last_completed_for_count
        now = time.time()
        time_ok = (now - last_report_ts) >= REPORT_EVERY_SEC
        count_ok = ((completed_batches * GROUP_SIZE) - last_completed_for_count) >= REPORT_EVERY_COUNT

        if force or time_ok or count_ok:
            elapsed = max(1e-6, now - start_ts)
            completed_files = min(total_files, completed_batches * GROUP_SIZE)
            rate = completed_files / elapsed
            remaining_files = max(0, total_files - completed_files)
            eta = fmt_eta(remaining_files / rate if rate > 0 else None)
            inflight = submitted_batches - completed_batches

            print(
                f"Progress: {completed_files}/{total_files} done "
                f"(kept={kept}, filtered={filtered}, nohit={nohit}, errors={errors}) | "
                f"in-flight batches={inflight} | rate={rate:.2f} files/s | {eta}"
            )

            last_report_ts = now
            last_completed_for_count = completed_files

    batches_iter = iter(batches)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = set()
        try:
            while True:
                while not interrupted["flag"] and len(futures) < MAX_INFLIGHT:
                    try:
                        batch = next(batches_iter)
                    except StopIteration:
                        break

                    fut = ex.submit(
                        analyze_batch,
                        batch,
                        output_root,
                        discarded_root,
                        tmp_tables,
                        tmp_batches,
                        MIN_CONF,
                    )
                    futures.add(fut)
                    submitted_batches += 1

                    if submitted_batches % 10 == 0:
                        print(
                            f"…submitted {submitted_batches}/{total_batches} batches "
                            f"(~{submitted_batches * GROUP_SIZE}/{total_files} files), "
                            f"in-flight={len(futures)}"
                        )

                if not futures:
                    break

                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    k, f, n, e, _c = fut.result()
                    kept += k
                    filtered += f
                    nohit += n
                    errors += e
                    completed_batches += 1
                maybe_report()

            # Drain remaining
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    k, f, n, e, _c = fut.result()
                    kept += k
                    filtered += f
                    nohit += n
                    errors += e
                    completed_batches += 1
                maybe_report()

        except KeyboardInterrupt:
            print("\nInterrupted. Draining running batches…")

    maybe_report(force=True)
    print("Done.")

    # Cleanup chunk temp dir
    shutil.rmtree(tmp_chunks_dir, ignore_errors=True)


if __name__ == "__main__":
    main()