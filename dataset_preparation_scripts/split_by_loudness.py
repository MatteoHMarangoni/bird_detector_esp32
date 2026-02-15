#!/usr/bin/env python3
"""
split_by_loudness.py

Run from VS Code: open and click ▶︎

Input folder (next to this script):
    ./data_raw

Output folders (created next to this script):
    ./data_below_threshold
    ./data_above_threshold

Splits audio files based on RMS loudness (dBFS).
"""

from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
from tqdm import tqdm


# ================= CONFIG =================
INPUT_DIRNAME = "data_raw"
OUTPUT_BELOW_DIRNAME = "data_below_threshold"
OUTPUT_ABOVE_DIRNAME = "data_above_threshold"

LOUDNESS_THRESHOLD_DB = -55.0  # RMS loudness threshold in dBFS (0 dBFS = digital full scale). -50 seems to work for a quiet park, increase to -40 to make it more aggressive, or -60 to keep quieter sounds

# ==========================================


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".aiff", ".aif", ".wma"}


def iter_audio_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("._"):
            yield p


def compute_rms_db(path: Path) -> float:
    audio, sr = sf.read(str(path), always_2d=False)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float32, copy=False)

    if len(audio) == 0:
        return -120.0  # treat empty as very quiet

    rms = np.sqrt(np.mean(audio ** 2))
    rms = max(rms, 1e-12)

    db = 20 * np.log10(rms)
    return float(db)


def copy_preserving_structure(src: Path, in_root: Path, out_root: Path):
    rel = src.relative_to(in_root)
    dst = out_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    script_dir = Path(__file__).resolve().parent

    in_root = (script_dir / INPUT_DIRNAME).resolve()
    below_root = (script_dir / OUTPUT_BELOW_DIRNAME).resolve()
    above_root = (script_dir / OUTPUT_ABOVE_DIRNAME).resolve()

    in_root.mkdir(parents=True, exist_ok=True)
    below_root.mkdir(parents=True, exist_ok=True)
    above_root.mkdir(parents=True, exist_ok=True)

    files = list(iter_audio_files(in_root))

    if not files:
        print("No audio files found in data_raw.")
        return

    print("=== Loudness Split ===")
    print("Threshold:", LOUDNESS_THRESHOLD_DB, "dBFS")
    print("Input:", in_root)
    print("Below:", below_root)
    print("Above:", above_root)
    print("Total files:", len(files))
    print()

    below_count = 0
    above_count = 0

    for p in tqdm(files, desc="Processing"):
        try:
            db = compute_rms_db(p)
        except Exception as e:
            print("Error reading:", p, e)
            continue

        if db < LOUDNESS_THRESHOLD_DB:
            copy_preserving_structure(p, in_root, below_root)
            below_count += 1
        else:
            copy_preserving_structure(p, in_root, above_root)
            above_count += 1

    print("\nDone.")
    print("Below threshold:", below_count)
    print("Equal or above:", above_count)


if __name__ == "__main__":
    main()
