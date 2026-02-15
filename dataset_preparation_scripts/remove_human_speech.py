#!/usr/bin/env python3
"""
filter_human_clear_speech_vad_abs.py (FAST, absolute thresholds)

Run from VS Code: open this file and click ▶︎ “Run Python File”.

Folders (created next to this script):
  ./data_raw              (input)
  ./data_public           (safe-to-publish output)
  ./data_do_not_publish   (excluded: likely clear speech)

Goal:
- Keep crowd voices / distant talking / kids screams when they are fragmented.
- Exclude clear speech (conversations, speeches) using VAD-only absolute thresholds.
- Works across 1s / 2s / 3s / 15s clips without using fractions.

Method:
- Run Silero VAD -> get speech timestamps
- Compute:
    total_speech_s  = sum of speech segments
    max_seg_s       = longest continuous speech segment
- Quarantine if:
    max_seg_s >= MAX_CONTIG_SPEECH_S   (clear speech proxy)
    OR total_speech_s >= TOTAL_SPEECH_S (speech-dominant proxy)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import set_start_method

from silero_vad import get_speech_timestamps, load_silero_vad


# ===================== CONFIG =====================
INPUT_DIRNAME = "data_raw"
OUTPUT_PUBLIC_DIRNAME = "data_public"
OUTPUT_PRIVATE_DIRNAME = "data_do_not_publish"

WORKERS = 6

# VAD sensitivity (single pass)
VAD_THRESHOLD = 0.2
# Lower = more sensitive (catches far speech), but may flag more noise as speech-like.

# Absolute thresholds (seconds)
MAX_CONTIG_SPEECH_S = 1.
# If any *single* continuous speech-like segment lasts >= this many seconds,
# treat as likely clear speech and quarantine.

TOTAL_SPEECH_S = 3.0
# If total speech-like time in the file is >= this many seconds, quarantine.
# This mainly affects longer clips (e.g., 15s). It won't trigger on 1–3s.

# Ignore tiny VAD hits (seconds)
MIN_TOTAL_SPEECH_S_TO_CARE = 0.15
# Below this total speech-like duration, keep the file (helps ignore transients).

WRITE_LOG = True
LOG_FILENAME = "vad_abs_filter_log.tsv"
# ==================================================


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".wma"}


@dataclass
class Decision:
    rel_path: str
    exclude: bool
    reason: str
    dur_s: float
    total_speech_s: float
    max_seg_s: float


def iter_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("._"):
            yield p


def read_mono_16k(path: Path) -> Tuple[np.ndarray, int, float]:
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)

    dur_s = float(len(audio) / sr) if sr else 0.0

    if sr != 16000 and sr:
        x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
        new_len = int(len(audio) * 16000 / sr)
        if new_len <= 0:
            return np.zeros(0, dtype=np.float32), 16000, dur_s
        x_new = np.linspace(0, 1, num=new_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sr = 16000

    return audio, sr, dur_s


def vad_stats(audio_16k: np.ndarray, sr: int, vad_model, threshold: float) -> Tuple[float, float]:
    """
    Returns (total_speech_seconds, max_contiguous_segment_seconds)
    """
    ts = get_speech_timestamps(
        audio_16k,
        vad_model,
        sampling_rate=sr,
        threshold=threshold,
        min_speech_duration_ms=120,
        min_silence_duration_ms=120,
    )

    total = 0.0
    max_seg = 0.0
    for seg in ts:
        seg_s = (seg["end"] - seg["start"]) / sr
        total += seg_s
        if seg_s > max_seg:
            max_seg = seg_s
    return float(total), float(max_seg)


def copy_preserving_structure(src: Path, in_root: Path, out_root: Path) -> None:
    rel = src.relative_to(in_root)
    dst = out_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


_VAD_MODEL = None


def init_worker():
    global _VAD_MODEL
    _VAD_MODEL = load_silero_vad()


def process_one_file(src_path: str, in_root: str, public_root: str, private_root: str) -> Decision:
    global _VAD_MODEL

    src = Path(src_path)
    inr = Path(in_root)
    pub = Path(public_root)
    priv = Path(private_root)
    rel = str(src.relative_to(inr))

    try:
        audio16k, sr, dur_s = read_mono_16k(src)
        if audio16k.size == 0 or dur_s <= 0:
            copy_preserving_structure(src, inr, priv)
            return Decision(rel, True, "empty_or_invalid_audio", dur_s, 0.0, 0.0)
    except Exception:
        copy_preserving_structure(src, inr, priv)
        return Decision(rel, True, "decode_error", 0.0, 0.0, 0.0)

    total_speech_s, max_seg_s = vad_stats(audio16k, sr, _VAD_MODEL, threshold=VAD_THRESHOLD)

    if total_speech_s < MIN_TOTAL_SPEECH_S_TO_CARE:
        copy_preserving_structure(src, inr, pub)
        return Decision(rel, False, "vad_tiny_or_none", dur_s, total_speech_s, max_seg_s)

    # Absolute-threshold quarantine decision
    if (max_seg_s >= MAX_CONTIG_SPEECH_S) or (total_speech_s >= TOTAL_SPEECH_S):
        copy_preserving_structure(src, inr, priv)
        reason = f"quarantine(max_seg={max_seg_s:.2f}s,total={total_speech_s:.2f}s)"
        return Decision(rel, True, reason, dur_s, total_speech_s, max_seg_s)

    copy_preserving_structure(src, inr, pub)
    reason = f"keep(max_seg={max_seg_s:.2f}s,total={total_speech_s:.2f}s)"
    return Decision(rel, False, reason, dur_s, total_speech_s, max_seg_s)


def main() -> None:
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    script_dir = Path(__file__).resolve().parent
    in_root = (script_dir / INPUT_DIRNAME).resolve()
    public_root = (script_dir / OUTPUT_PUBLIC_DIRNAME).resolve()
    private_root = (script_dir / OUTPUT_PRIVATE_DIRNAME).resolve()

    in_root.mkdir(parents=True, exist_ok=True)
    public_root.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True)

    files = sorted(iter_audio_files(in_root))
    if not files:
        print("No audio files found in ./data_raw")
        return

    log_fh = None
    if WRITE_LOG:
        log_path = script_dir / LOG_FILENAME
        log_fh = open(log_path, "w", encoding="utf-8")
        log_fh.write("relative_path\texclude\tdur_s\ttotal_speech_s\tmax_seg_s\treason\n")
        print("Log:", log_path)

    kept = excluded = 0

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as ex:
        futures = [
            ex.submit(process_one_file, str(p), str(in_root), str(public_root), str(private_root))
            for p in files
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            d: Decision = fut.result()
            kept += 0 if d.exclude else 1
            excluded += 1 if d.exclude else 0

            if log_fh:
                log_fh.write(
                    f"{d.rel_path}\t{int(d.exclude)}\t{d.dur_s:.3f}\t{d.total_speech_s:.3f}\t{d.max_seg_s:.3f}\t{d.reason}\n"
                )

    if log_fh:
        log_fh.close()

    total = len(files)
    print("\nDone.")
    print("Total files:        ", total)
    print("Kept (data_public): ", kept)
    print("Excluded (private): ", excluded)
    if total:
        print("Excluded %:         ", f"{(excluded / total) * 100:.2f}%")
    print("\nPublish ONLY:", public_root)
    print("Do NOT publish:", private_root)


if __name__ == "__main__":
    main()
