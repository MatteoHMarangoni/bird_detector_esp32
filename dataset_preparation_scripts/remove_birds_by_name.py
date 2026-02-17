#!/usr/bin/env python3
"""
remove_birds_by_name.py

What this script does
---------------------
This script sorts .wav files into subfolders based on a "species token" extracted from
the filename. It is used for dataset cleanup / re-organization after other processing
steps that embed species names into filenames.

It scans INPUT_ROOT recursively for .wav files and for each file:
- Extracts a species token using heuristics (see extract_species()).
- Sanitizes the token into a safe folder name.
- Places the file into OUTPUT_ROOT/<species_folder>/ (move by default; copy optional).

Defaults (relative to this script)
----------------------------------
dataset_preparation_scripts/
  remove_birds_by_name.py
  data_raw/                 <-- input (put WAVs here)
  output_by_name/           <-- output (created automatically)

How to run
----------
- VS Code: press Run/Play (defaults will be used)
- Terminal: python3 remove_birds_by_name.py [input] [output] [--copy] [--dry]

Notes
-----
- By default, output is NOT in-place; it writes to output_by_name/ next to the script.
- If you want in-place reorganization, pass the same folder as both input and output.

"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path


# ================== CONFIG (ADJUST DURING OPERATION) ==================

# Base directory is the folder containing THIS script:
SCRIPT_DIR = Path(__file__).resolve().parent

# Default input folder (relative to script)
INPUT_ROOT_DEFAULT = SCRIPT_DIR / "data_raw"

# Default output folder (created next to the script)
OUTPUT_ROOT_DEFAULT = SCRIPT_DIR / "output_by_name"

# If True, skip files that already exist at destination (prevents accidental overwrite).
SKIP_IF_DEST_EXISTS = True

# =====================================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_folder_name(name: str) -> str:
    """
    Turn a raw species token into a safe folder name:
    - replace spaces/hyphens with underscore
    - remove any remaining non-word chars
    - drop purely-numeric or date-like parts
    """
    if not name:
        return "Unknown_Species"

    # normalize separators to underscores
    s = re.sub(r"[-\s]+", "_", name.strip())

    # remove anything that's not alnum or underscore
    s = re.sub(r"[^\w_]+", "", s)

    # split and drop numeric/date-like tokens (e.g. 2025, 07, 2025-07-07)
    parts = [
        p
        for p in s.split("_")
        if p
        and not re.match(r"^\d+$", p)
        and not re.match(r"^\d{4}-\d{2}-\d{2}$", p)
    ]

    if not parts:
        return "Unknown_Species"

    return "_".join(parts)


def extract_species(filename: str) -> str:
    """
    Extract a species token from a filename.

    Heuristic (priority order):
    1) Leading prediction like: '0.764-Rose-ringed_Parakeet-...'
    2) Token after 'bird_' like: '...bird_Eurasian_Magpie...'
    3) Fallback: last token that contains letters (split on '_' or '-')

    Returns a raw token (sanitization is done by sanitize_folder_name()).
    """
    stem = Path(filename).stem

    # strip trailing _chunkNNN if present
    stem = re.sub(r"_chunk\d+$", "", stem, flags=re.IGNORECASE)

    # 1) leading prediction: confidence-species-...
    m = re.match(r"^[0-9]+(?:\.[0-9]+)?-([A-Za-z0-9_\-]+)(?:-|$)", stem)
    if m:
        return m.group(1)

    # 2) token after bird_
    m2 = re.search(r"(?i)bird_([A-Za-z0-9_\-]+)", stem)
    if m2:
        return m2.group(1)

    # 3) fallback: pick last part that contains letters
    parts = re.split(r"[_\-]+", stem)
    for p in reversed(parts):
        if re.search(r"[A-Za-z]", p):
            return p

    return "Unknown"


def process(input_root: Path, output_root: Path, *, copy: bool = False, dry: bool = False) -> None:
    """
    Scan input_root recursively for .wav files and route each into
    output_root/<species_folder>/filename.wav.
    """
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    ensure_dir(output_root)

    moved = 0
    skipped = 0
    errors = 0

    for dirpath, _, files in os.walk(input_root):
        for fn in files:
            if fn.startswith("._"):
                continue
            if not fn.lower().endswith(".wav"):
                continue

            src = Path(dirpath) / fn

            species_token = extract_species(fn)
            folder = sanitize_folder_name(species_token)

            dest_dir = output_root / folder
            ensure_dir(dest_dir)
            dest = dest_dir / fn

            # prevent overwriting unless user explicitly wants it
            if SKIP_IF_DEST_EXISTS and dest.exists():
                skipped += 1
                continue

            try:
                if dry:
                    op = "COPY" if copy else "MOVE"
                    print(f"[DRY {op}] {src} -> {dest}")
                    moved += 1
                else:
                    if copy:
                        shutil.copy2(src, dest)
                    else:
                        # move preserves file on same FS; falls back to copy+delete across FS
                        shutil.move(str(src), str(dest))
                    moved += 1
            except Exception as e:
                print(f"ERROR processing {src} -> {dest}: {e}")
                errors += 1

    print(f"Done. moved={moved}, skipped={skipped}, errors={errors}")


def move_non_matching_folders(input_root: Path, other_name: str = "Other", *, dry: bool = False) -> None:
    """
    Move all subfolders of input_root that do NOT match the input_root basename
    into a folder named `other_name` under input_root.

    This is useful if you ran process() in-place (output == input) and you want to
    collect "everything else" into a single folder.

    Folders whose sanitized name equals the input basename are left in place.
    """
    input_root = input_root.resolve()

    if not input_root.is_dir():
        print(f"Input root not a directory: {input_root}")
        return

    basename = input_root.name
    expected_folder_name = sanitize_folder_name(basename)

    dest_root = input_root / other_name
    moved = 0
    skipped = 0
    errors = 0

    for entry in sorted(input_root.iterdir()):
        # skip files and the special target folder itself
        if not entry.is_dir():
            continue
        if entry.name == other_name:
            skipped += 1
            continue

        # if folder name equals the input basename (sanitized), skip it (leave in place)
        if entry.name == expected_folder_name:
            skipped += 1
            continue

        # non-matching folder -> move whole folder into Other
        try:
            if dry:
                print(f"[DRY MOVE] {entry} -> {dest_root}/")
                moved += 1
            else:
                dest_root.mkdir(parents=True, exist_ok=True)
                shutil.move(str(entry), str(dest_root))
                print(f"Moved: {entry.name} -> {other_name}/")
                moved += 1
        except Exception as e:
            print(f"ERROR moving {entry} -> {dest_root}: {e}")
            errors += 1

    print(f"Done. moved={moved}, skipped={skipped}, errors={errors}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sort WAVs into species folders by extracting a species token from filenames."
    )

    # Keep CLI optional (VS Code run uses defaults)
    p.add_argument(
        "input",
        nargs="?",
        default=str(INPUT_ROOT_DEFAULT),
        help=f"input root (default: {INPUT_ROOT_DEFAULT})",
    )
    p.add_argument(
        "output",
        nargs="?",
        default=str(OUTPUT_ROOT_DEFAULT),
        help=f"output root (default: {OUTPUT_ROOT_DEFAULT})",
    )

    p.add_argument("--copy", action="store_true", help="copy instead of move")
    p.add_argument("--dry", action="store_true", help="dry run (no file operations)")

    p.add_argument(
        "--group-other",
        action="store_true",
        help="move subfolders not matching the input basename into a folder named 'Other' (or --other-name)",
    )
    p.add_argument(
        "--other-name",
        default="Other",
        help="name of the folder to collect non-matching folders (default: 'Other')",
    )
    p.add_argument(
        "--no-process",
        action="store_true",
        help="when using --group-other, skip running the sorting step and only reorganize folders",
    )

    args = p.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    print(f"INPUT_ROOT:  {input_root.resolve()}")
    print(f"OUTPUT_ROOT: {output_root.resolve()}")
    print(f"MODE:        {'COPY' if args.copy else 'MOVE'}{' (dry-run)' if args.dry else ''}")

    if args.group_other:
        if not args.no_process:
            process(input_root, output_root, copy=args.copy, dry=args.dry)
        move_non_matching_folders(input_root, other_name=args.other_name, dry=args.dry)
    else:
        process(input_root, output_root, copy=args.copy, dry=args.dry)


if __name__ == "__main__":
    main()
