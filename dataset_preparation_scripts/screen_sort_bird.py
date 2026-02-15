#!/usr/bin/env python3
# Short usage note:
# This script splits input WAV files into short chunks, runs BirdNET on batches of chunks,
# and sorts chunk files into OUTPUT_ROOT or FILTERED_ROOT (discarded low-confidence) based on top detection confidence.
# Configure paths and thresholds in the CONFIG section below. Requires birdnet_analyzer
# (pip install birdnet-analyzer) and optionally ffmpeg for faster splitting.
# Run: python3 screen_sort_bird.py

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import set_start_method
import contextlib
import math
import wave

# ================== CONFIG ==================
INPUT_ROOT = "/Users/matteomarangoni/Desktop/Bird_datasets/Amstelpark/original_raw_data"  # Input folder
OUTPUT_ROOT = "/Users/matteomarangoni/Desktop/birds_screened_sorted"  # Sorted results
DISCARDED_ROOT = "/Users/matteomarangoni/Desktop/birds_discarded(lowConf)"  # Discarded low-confidence results

TMP_TABLE_DIR = "/tmp/birdnet_tables"  # where BirdNET writes result tables
TMP_BATCH_ROOT = "/tmp/birdnet_batches"  # where we symlink batch inputs

MIN_CONF = 0.1  # keep only files with top non-nocall >= this
SENSITIVITY = 1.0  # 1.0 neutral

CHUNK_SEC = 1  # chunk size (seconds)

# Location latitude and longitude, this is used by BirdNET for more accurate predictions
LAT = 52.330
LON = 4.891

# Parallelism (Apple M1 sweet spot: more workers, 1 thread each)
MAX_WORKERS = 6  # processes in parallel (try 5–8)
THREADS = "1"  # BirdNET threads per process (keep 1–2)
BATCHSIZE = "1"  # model batchsize; short clips rarely benefit from >1

# Batch how many files per BirdNET invocation
GROUP_SIZE = 600  # try 400–1000; larger = fewer BirdNET launches

# Backpressure (don't flood the pool)
MAX_INFLIGHT = MAX_WORKERS * 3

QUIET_BIRDNET = True

# Progress cadence
REPORT_EVERY_SEC = 60
REPORT_EVERY_COUNT = 1000

# ============================================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# NEW: ffmpeg detection
def has_ffmpeg():
    return shutil.which("ffmpeg") is not None


# NEW: ffmpeg-based splitter
def split_with_ffmpeg(src_path, dst_dir, base_stem, chunk_sec=CHUNK_SEC):
    ensure_dir(dst_dir)
    # use 2-digit indices (01..99), start numbering at 1
    pattern = os.path.join(dst_dir, f"{base_stem}_chunk%02d.wav")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src_path,
        "-f",
        "segment",
        "-segment_time",
        str(chunk_sec),
        "-segment_start_number",
        "1",
        # force PCM WAV output to avoid container/codec quirks when segmenting
        "-c:a",
        "pcm_s16le",
        pattern,
    ]
    subprocess.run(cmd, check=True)
    chunks = []
    idx = 1
    while True:
        p = os.path.join(dst_dir, f"{base_stem}_chunk{idx:02d}.wav")
        if os.path.exists(p):
            chunks.append(p)
            idx += 1
        else:
            break
    return chunks


# NEW: wave-based splitter (fallback)
def split_with_wave(src_path, dst_dir, base_stem, chunk_sec=CHUNK_SEC):
    ensure_dir(dst_dir)
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
            idx = i + 1
            out_path = os.path.join(dst_dir, f"{base_stem}_chunk{idx:02d}.wav")
            with contextlib.closing(wave.open(out_path, "wb")) as out:
                out.setnchannels(nch)
                out.setsampwidth(sw)
                out.setframerate(fr)
                out.writeframes(frames)
            chunks.append(out_path)
    return chunks


# NEW: chunking orchestrator (mirrors purge_birds_from_nonbirds)
def split_into_chunks(src_path, tmp_dir, chunk_sec=CHUNK_SEC):
    base_stem = os.path.splitext(os.path.basename(src_path))[0]
    chunk_dir = os.path.join(tmp_dir, base_stem)
    ensure_dir(chunk_dir)
    if has_ffmpeg():
        return split_with_ffmpeg(src_path, chunk_dir, base_stem, chunk_sec)
    return split_with_wave(src_path, chunk_dir, base_stem, chunk_sec)


# NEW: create chunk files for all inputs and return list of chunk paths
def create_all_chunks(input_root: str, tmp_chunks_root: str, chunk_sec=CHUNK_SEC):
    ensure_dir(tmp_chunks_root)
    all_chunks = []
    files = collect_files(input_root)

    total = len(files)
    if total == 0:
        print("No input WAV files found to split.")
        return []

    start_ts = time.time()
    last_report_ts = start_ts
    processed = 0
    last_report_count = 0

    for src in files:
        processed += 1
        base = os.path.basename(src)
        try:
            chs = split_into_chunks(src, tmp_chunks_root, chunk_sec)
            all_chunks.extend(chs)
            print(f"[{processed}/{total}] Split {base} -> {len(chs)} chunk(s)")
        except Exception as e:
            print(f"[{processed}/{total}] Failed to split {base}: {e}")
            # skip problematic files but continue
        # Periodic summary (time- or count-based)
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


# NEW: parse full selection table into detections
def parse_table_rows(table_path):
    """
    Returns a list of detections: (begin_time, end_time, common_name, species_code, confidence)
    Skips 'nocall'. Supports tab or comma-separated tables.
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
        if any(
            idx is None or idx >= len(parts) for idx in (i_bt, i_et, i_cn, i_sc, i_cf)
        ):
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
    s = name.strip().replace(" ", "_")
    return re.sub(r"[^\w\.-]+", "_", s) or "Unknown_Species"

# NEW: safe filename sanitizer (keeps readable species names in filenames)
def sanitize_for_filename(name: str) -> str:
    if not name:
        return "Unknown"
    s = name.strip().replace(" ", "_")
    return re.sub(r"[^\w\.-]+", "_", s)

# NEW: extract a "previous species" token from the original chunk stem
def extract_prev_species_from_stem(stem: str) -> str:
    """
    Prefer species token after 'bird_' in original filename. Examples:
      ...bird_Eurasian_Magpie... -> 'Eurasian_Magpie'
      ...bird_Common_Wood_Pigeon... -> 'Common_Wood_Pigeon'

    Also: strip trailing "_chunkNNN" that comes from split files so folder names
    are derived from the original filename/species token and do NOT include
    the "_chunk" suffix.
    """
    # remove trailing _chunkNNN if present
    stem = re.sub(r'_chunk\d+$', '', stem, flags=re.IGNORECASE)

    # try to extract token after 'bird_'
    m = re.search(r'(?i)bird_([A-Za-z_]+)', stem)
    if m:
        return m.group(1)

    # fallback: original heuristic (avoid date/timestamp tokens containing digits)
    parts = re.split(r'[_\-]+', stem)
    # look for first contiguous run of >=2 letter-only parts
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if re.search(r'[A-Za-z]', a) and not re.search(r'\d', a) and re.search(r'[A-Za-z]', b) and not re.search(r'\d', b):
            j = i + 2
            while j < len(parts) and re.search(r'[A-Za-z]', parts[j]) and not re.search(r'\d', parts[j]):
                j += 1
            return "_".join(parts[i:j])
    # fallback: first alphabetic-only part
    for p in parts:
        if re.search(r'[A-Za-z]', p) and not re.search(r'\d', p):
            return p
    return "UnknownPrev"


def same_fs(path_a: str, path_b: str) -> bool:
    try:
        return os.stat(path_a).st_dev == os.stat(path_b).st_dev
    except:
        return False


def clone_or_link_or_copy(src: str, dst_dir: str, new_basename: str) -> str:
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, new_basename)
    # 1) hardlink if same filesystem
    try:
        if same_fs(src, dst_dir):
            os.link(src, dst)
            return dst
    except Exception:
        pass
    # 2) APFS clone
    try:
        subprocess.run(
            ["/bin/cp", "-c", src, dst],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return dst
    except Exception:
        pass
    # 3) plain copy
    shutil.copyfile(src, dst)
    return dst


def collect_files(root: str):
    files = []
    for dirpath, _, fns in os.walk(root):
        for fn in fns:
            if fn.startswith("._"):  # resource forks
                continue
            if fn.lower().endswith(".wav"):
                files.append(os.path.join(dirpath, fn))
    return files


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def parse_best_from_table(table_path: str):
    # previously a placeholder that returned undefined names.
    # Provide a safe implementation: pick best non-nocall detection by confidence.
    detections = parse_table_rows(table_path)
    best_species = None
    best_conf = None
    for bt, et, cn, sc, cf in detections:
        if best_conf is None or cf > best_conf:
            best_conf = cf
            best_species = cn
    return (best_species, best_conf)


# NEW: create a padded copy by replicating the chunk before and after the original
def pad_with_zeros_to_path(src_path: str, dst_path: str, pad_before_sec: int = 1, pad_after_sec: int = 1):
    """
    Create a WAV at dst_path containing pad_before_sec seconds of silence,
    then the original chunk, then pad_after_sec seconds of silence.
    """
    ensure_dir(os.path.dirname(dst_path) or ".")
    with contextlib.closing(wave.open(src_path, "rb")) as wf:
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

    tmp_path = dst_path + ".pad.tmp"
    with contextlib.closing(wave.open(tmp_path, "wb")) as out:
        out.setnchannels(nch)
        out.setsampwidth(sw)
        out.setframerate(fr)
        out.writeframes(silence_before + frames + silence_after)
    os.replace(tmp_path, dst_path)


def analyze_batch(filepaths, out_root, discarded_root, tmp_tables, batch_root, min_conf):
    """
    Now expects filepaths to be individual chunk files (already split).
    Create padded copies inside tmp_batch_dir for BirdNET; keep originals for final outputs.
    """
    kept = filtered = nohit = errors = 0
    completed = len(filepaths)

    ensure_dir(batch_root)
    tmp_batch_dir = tempfile.mkdtemp(prefix="bn_batch_", dir=batch_root)

    try:
        # Materialize padded copies into the batch dir for BirdNET to analyze
        for src in filepaths:
            link_name = os.path.join(tmp_batch_dir, os.path.basename(src))
            try:
                # create padded copy in batch dir by adding 1s silence before and after
                pad_with_zeros_to_path(src, link_name, pad_before_sec=1, pad_after_sec=1)
            except Exception:
                # fallback: try hardlink or plain copy if padding fails
                try:
                    if same_fs(src, tmp_batch_dir):
                        os.link(src, link_name)
                    else:
                        shutil.copy2(src, link_name)
                except Exception:
                    # skip file if we cannot materialize it
                    continue

        # Run BirdNET ONCE on the tmp batch dir
        ensure_dir(tmp_tables)
        cmd = [
            sys.executable,
            "-m",
            "birdnet_analyzer.analyze",
            tmp_batch_dir,
            "-o",
            tmp_tables,
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
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            subprocess.run(cmd, check=True)

        # For each chunk file in the batch, inspect its table and route
        for src in filepaths:
            base = os.path.splitext(os.path.basename(src))[0]
            table = os.path.join(tmp_tables, f"{base}.BirdNET.selection.table.txt")
            detections = parse_table_rows(table)
            # Always clean up table ASAP
            try:
                if os.path.isfile(table):
                    os.remove(table)
            except Exception:
                pass

            # Determine best single detection (highest confidence) for this chunk
            best_species = None
            best_conf = None
            for bt, et, cn, sc, cf in detections:
                if ("nocall" in (cn + " " + sc).lower()):
                    continue
                if (best_conf is None) or (cf > best_conf):
                    best_conf = cf
                    best_species = cn

            # For final saved outputs, use the ORIGINAL chunk file (src), not the padded batch copy
            stem, ext = os.path.splitext(os.path.basename(src))  # ensure stem/ext available for all branches
            original_filename = os.path.basename(src)
            prev_species_token = extract_prev_species_from_stem(stem)
            prev_folder = sanitize_species_folder(prev_species_token)

            if best_species is None:
                nohit += 1
                try:
                    dest_dir = os.path.join(discarded_root, prev_folder)
                    ensure_dir(dest_dir)
                    conf_str = f"{(best_conf or 0.0):.3f}"
                    pred = sanitize_for_filename(None)  # No prediction
                    new_name = f"{conf_str}-NoHit-{original_filename}"
                    dst = os.path.join(dest_dir, new_name)
                    shutil.copy2(src, dst)
                except Exception:
                    errors += 1
                continue

            pred_name_clean = sanitize_for_filename(best_species)
            conf_str = f"{best_conf:.3f}" if best_conf is not None else f"{0.0:.3f}"

            if best_conf is not None and best_conf >= min_conf:
                # sort into folder based on species token from previous filename
                dest_dir = os.path.join(out_root, prev_folder)
                ensure_dir(dest_dir)
                try:
                    new_name = f"{conf_str}-{pred_name_clean}-{original_filename}"
                    dst = os.path.join(dest_dir, new_name)
                    shutil.copy2(src, dst)
                    kept += 1
                except Exception:
                    errors += 1
            else:
                # Below-threshold: place into the discarded root under the previous-file-derived subfolder.
                filtered += 1
                try:
                    dest_dir = os.path.join(discarded_root, prev_folder)
                    ensure_dir(dest_dir)
                    new_name = f"{conf_str}-{pred_name_clean}-{original_filename}"
                    dst = os.path.join(dest_dir, new_name)
                    shutil.copy2(src, dst)
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


def fmt_eta(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "ETA: --:--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"ETA: {h:02d}:{m:02d}:{s:02d}"


def main():
    print(f"INPUT_ROOT:  {INPUT_ROOT}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"DISCARDED_ROOT: {DISCARDED_ROOT}")  # NEW
    # NEW: validate config early
    if CHUNK_SEC <= 0:
        raise ValueError("CHUNK_SEC must be > 0 seconds")
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(DISCARDED_ROOT)  # NEW
    ensure_dir(TMP_TABLE_DIR)
    ensure_dir(TMP_BATCH_ROOT)

    # NEW: create a dedicated temp-chunks dir and pre-split all inputs
    tmp_chunks_dir = tempfile.mkdtemp(prefix="bn_chunks_", dir=TMP_BATCH_ROOT)
    try:
        chunk_files = create_all_chunks(INPUT_ROOT, tmp_chunks_dir, CHUNK_SEC)
    except Exception:
        chunk_files = []
    total_files = len(chunk_files)
    if total_files == 0:
        print("No chunk WAV files found after splitting.")
        # clean up chunks dir
        try:
            shutil.rmtree(tmp_chunks_dir, ignore_errors=True)
        except Exception:
            pass
        return

    batches = list(chunks(chunk_files, GROUP_SIZE))
    total_batches = len(batches)
    print(f"Found {total_files} chunk WAV(s) → {total_batches} batches of ~{GROUP_SIZE} each.")
    print(f"Launching {MAX_WORKERS} workers (BirdNET -t {THREADS}).")

    # Graceful Ctrl+C
    interrupted = {"flag": False}

    def handle_sigint(sig, frame):
        if not interrupted["flag"]:
            print(
                "\n⚠️  Ctrl+C received — finishing in-flight batches and stopping new submissions…"
            )
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
        count_ok = (
            (completed_batches * GROUP_SIZE) - last_completed_for_count
        ) >= REPORT_EVERY_COUNT
        if force or time_ok or count_ok:
            elapsed = max(1e-6, now - start_ts)
            completed_files = min(total_files, completed_batches * GROUP_SIZE)
            rate = completed_files / elapsed  # files/sec
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

    # Producer/consumer with backpressure at batch level
    batches_iter = iter(batches)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = set()
        try:
            while True:
                # Submit up to MAX_INFLIGHT batches
                while not interrupted["flag"] and len(futures) < MAX_INFLIGHT:
                    try:
                        batch = next(batches_iter)
                    except StopIteration:
                        break
                    fut = ex.submit(
                        analyze_batch,
                        batch,
                        OUTPUT_ROOT,
                        DISCARDED_ROOT,  # NEW
                        TMP_TABLE_DIR,
                        TMP_BATCH_ROOT,
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
                    k, f, n, e, c = fut.result()
                    kept += k
                    filtered += f
                    nohit += n
                    errors += e
                    completed_batches += 1
                maybe_report()

                if interrupted["flag"]:
                    # stop submitting; keep draining
                    pass

            # Drain remaining
            while futures:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    k, f, n, e, c = fut.result()
                    kept += k
                    filtered += f
                    nohit += n
                    errors += e
                    completed_batches += 1
                maybe_report()

        except KeyboardInterrupt:
            print("\nInterrupted. Draining running batches…")

    # Final report
    maybe_report(force=True)
    print("Done.")

    # cleanup chunk temp dir
    try:
        shutil.rmtree(tmp_chunks_dir, ignore_errors=True)
    except Exception:
        pass


# Ensure script runs when invoked directly
if __name__ == "__main__":
    main()
