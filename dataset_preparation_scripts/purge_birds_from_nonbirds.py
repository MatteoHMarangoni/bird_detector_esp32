input#!/usr/bin/env python3
import contextlib
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import set_start_method
from shutil import which

# ================== CONFIG ==================
# Folders
INPUT_ROOT = "/Users/matteomarangoni/Desktop/input"
BIRD_OUT = "/Users/matteomarangoni/Desktop/bird_out"
CLEAN_OUT = "/Users/matteomarangoni/Desktop/clean_out"
TMP_ROOT = "/tmp/birdnet_chunk_work"

# Chunking & detection
CHUNK_SEC = 3  # 3s chunks
MIN_CONF = 0.10  # confidence threshold for routing to BIRD
SENSITIVITY = 1.0  # BirdNET sensitivity (1.0 is neutral; 1.2-1.5 more sensitive)

# Parallelism
MAX_WORKERS = 4  # how many files processed in parallel (good start for M1)
THREADS = "2"  # BirdNET internal threads per process (1-2 recommended)
BATCHSIZE = "1"  # BirdNET batch size

# Location (Den Haag) and week parsing
LAT = 52.070
LON = 4.300
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")
# ============================================


def infer_week_from_filename(filename: str):
    m = DATE_RE.match(filename)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    try:
        return datetime(y, mo, d).isocalendar()[1]
    except Exception:
        return None


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def has_ffmpeg():
    return which("ffmpeg") is not None


def split_with_ffmpeg(src_path, dst_dir, base_stem, chunk_sec=3):
    ensure_dirs(dst_dir)
    pattern = os.path.join(dst_dir, f"{base_stem}_chunk%03d.wav")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", src_path, "-f", "segment", "-segment_time", str(chunk_sec), "-c", "copy", pattern]
    subprocess.run(cmd, check=True)
    chunks = []
    idx = 0
    while True:
        p = os.path.join(dst_dir, f"{base_stem}_chunk{idx:03d}.wav")
        if os.path.exists(p):
            chunks.append(p)
            idx += 1
        else:
            break
    return chunks


def split_with_wave(src_path, dst_dir, base_stem, chunk_sec=3):
    ensure_dirs(dst_dir)
    chunks = []
    with contextlib.closing(wave.open(src_path, "rb")) as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        fr = wf.getframerate()
        nf = wf.getnframes()
        fpc = int(fr * chunk_sec)
        num_chunks = int(math.ceil(nf / fpc))
        for i in range(num_chunks):
            start = i * fpc
            wf.setpos(start)
            to_read = min(fpc, nf - start)
            frames = wf.readframes(to_read)
            out_path = os.path.join(dst_dir, f"{base_stem}_chunk{i:03d}.wav")
            with contextlib.closing(wave.open(out_path, "wb")) as out:
                out.setnchannels(nch)
                out.setsampwidth(sw)
                out.setframerate(fr)
                out.writeframes(frames)
            chunks.append(out_path)
    return chunks


def split_into_chunks(src_path, tmp_dir, chunk_sec=3):
    base_stem = os.path.splitext(os.path.basename(src_path))[0]
    chunk_dir = os.path.join(tmp_dir, base_stem)
    ensure_dirs(chunk_dir)
    if has_ffmpeg():
        return split_with_ffmpeg(src_path, chunk_dir, base_stem, chunk_sec)
    return split_with_wave(src_path, chunk_dir, base_stem, chunk_sec)


def clean_selection_tables(dir_path):
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith(".selection.table.txt") or f.lower().endswith(".csv"):
                try:
                    os.remove(os.path.join(root, f))
                except Exception:
                    pass


def parse_table_rows(table_path):
    """
    Returns a list of detections: (begin_time, end_time, common_name, species_code, confidence)
    Parses header to find column indexes; supports tab or comma.
    Skips 'nocall'.
    """
    if not os.path.isfile(table_path):
        return []

    with open(table_path, errors="replace") as f:
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


def analyze_file_once(src_path, out_dir, lat, lon, week, min_conf, sensitivity, threads, batchsize):
    """
    Run BirdNET ONCE on the full 15s file, write results next to it (out_dir),
    and return (detections, table_path).
    """
    base = os.path.splitext(os.path.basename(src_path))[0]
    cmd = [
        sys.executable,
        "-m",
        "birdnet_analyzer.analyze",
        src_path,
        "-o",
        out_dir,
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

    table_path = os.path.join(out_dir, f"{base}.BirdNET.selection.table.txt")
    return parse_table_rows(table_path), table_path


def route_chunks_by_overlap(chunks, detections, chunk_sec, bird_dir, clean_dir, min_conf):
    """
    For each chunk window [k*chunk_sec, (k+1)*chunk_sec), send to BIRD if any
    detection (non-nocall) with confidence >= min_conf overlaps the window.
    """
    moved_bird = moved_clean = 0
    for idx, ch_path in enumerate(chunks):
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
        dest = os.path.join(dest_dir, os.path.basename(ch_path))
        shutil.move(ch_path, dest)
        if birdy:
            moved_bird += 1
        else:
            moved_clean += 1
    return moved_bird, moved_clean


def count_wavs(root):
    n = 0
    for _, _, files in os.walk(root):
        for fn in files:
            if fn.startswith("._"):
                continue
            if fn.lower().endswith(".wav"):
                n += 1
    return n


def process_one_file(args):
    """
    Worker: process a single 15s WAV (analyze once, split, route).
    Returns (analyzed_inc, moved_bird_chunks, moved_clean_chunks, src_path)
    """
    (src_path, input_root, tmp_root, bird_out, clean_out, chunk_sec, lat, lon, min_conf, sensitivity, threads, batchsize) = args

    try:
        filename = os.path.basename(src_path)
        rel = os.path.relpath(os.path.dirname(src_path), input_root)
        tmp_dir = os.path.join(tmp_root, rel)
        bird_dir = os.path.join(bird_out, rel)
        clean_dir = os.path.join(clean_out, rel)
        ensure_dirs(tmp_dir, bird_dir, clean_dir)

        week = infer_week_from_filename(filename)

        # 1) Analyze ONCE on the full file
        detections, table_path = analyze_file_once(src_path, os.path.dirname(src_path), lat, lon, week, min_conf, sensitivity, threads, batchsize)

        # 2) Split to chunks (in tmp_dir)
        chunks = split_into_chunks(src_path, tmp_dir, chunk_sec)

        # 3) Route by overlap
        b, c = route_chunks_by_overlap(chunks, detections, chunk_sec, bird_dir, clean_dir, min_conf)

        # 4) Cleanup BirdNET tables next to source + any tmp tables
        try:
            if os.path.isfile(table_path):
                os.remove(table_path)
        except Exception:
            pass
        clean_selection_tables(tmp_dir)

        return (1, b, c, src_path)

    except subprocess.CalledProcessError as e:
        return (0, 0, 0, f"BirdNET error on {src_path}: {e}")
    except Exception as e:
        return (0, 0, 0, f"Worker error on {src_path}: {e}")


def main():
    print(f"INPUT_ROOT: {INPUT_ROOT}")
    print(f"BIRD_OUT:   {BIRD_OUT}")
    print(f"CLEAN_OUT:  {CLEAN_OUT}")
    print(f"TMP_ROOT:   {TMP_ROOT}")
    ensure_dirs(BIRD_OUT, CLEAN_OUT, TMP_ROOT)

    # Gather all work (absolute paths)
    work = []
    for root, _, files in os.walk(INPUT_ROOT):
        for fn in files:
            if fn.startswith("._") or not fn.lower().endswith(".wav"):
                continue
            work.append((os.path.join(root, fn), INPUT_ROOT, TMP_ROOT, BIRD_OUT, CLEAN_OUT, CHUNK_SEC, LAT, LON, MIN_CONF, SENSITIVITY, THREADS, BATCHSIZE))

    print(f"Found {len(work)} WAV(s). Launching {MAX_WORKERS} workers (BirdNET -t {THREADS} per process).")

    analyzed_files = moved_bird_chunks = moved_clean_chunks = 0

    # macOS/Apple Silicon: explicit 'spawn' is safest for multiprocessing
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    if not work:
        print("Nothing to do.")
        return

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(process_one_file, w) for w in work]
        for fut in as_completed(futures):
            a, b, c, msg = fut.result()
            analyzed_files += a
            moved_bird_chunks += b
            moved_clean_chunks += c
            if a == 1:
                print(f"✓ {os.path.basename(msg)}  (+{b} bird, +{c} clean)")
            else:
                print(f"✗ {msg}")

    print("\nDone.")
    print(f"Analyzed original files: {analyzed_files}")
    print(f"Moved BIRD chunks: {moved_bird_chunks}")
    print(f"Moved CLEAN chunks: {moved_clean_chunks}")


if __name__ == "__main__":
    main()
