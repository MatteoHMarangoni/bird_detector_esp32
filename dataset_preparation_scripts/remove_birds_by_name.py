#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from pathlib import Path

# set default input/output folders here (edit these)
INPUT_ROOT = "/Users/matteomarangoni/Desktop/Bird_datasets/Combined_Zuiderpark/Common_Chaffinch"
OUTPUT_ROOT = INPUT_ROOT # or specify if required


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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
    parts = [p for p in s.split("_") if p and not re.match(r"^\d+$", p) and not re.match(r"^\d{4}-\d{2}-\d{2}$", p)]
    if not parts:
        return "Unknown_Species"
    return "_".join(parts)


def extract_species(filename: str) -> str:
    """
    Heuristic (priority order):
    1) Leading prediction like '0.764-Rose-ringed_Parakeet-...'
    2) Token after 'bird_'
    3) Fallback: first alphabetic run of tokens separated by _ or -
    Returns a raw token (sanitization done by sanitize_folder_name).
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    # strip trailing _chunkNNN if present
    stem = re.sub(r"_chunk\d+$", "", stem, flags=re.IGNORECASE)

    # 1) leading prediction: confidence-species-...
    m = re.match(r'^[0-9]+(?:\.[0-9]+)?-([A-Za-z0-9_\-]+)(?:-|$)', stem)
    if m:
        return m.group(1)

    # 2) token after bird_
    m2 = re.search(r"(?i)bird_([A-Za-z0-9_\-]+)", stem)
    if m2:
        return m2.group(1)

    # 3) fallback: pick first part (from end) that contains letters
    parts = re.split(r"[_\-]+", stem)
    for p in parts[::-1]:
        if re.search(r"[A-Za-z]", p):
            return p
    return "Unknown"


def process(input_root: Path, output_root: Path, copy: bool = False, dry: bool = False):
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    ensure_dir(output_root)
    moved = 0
    skipped = 0
    errors = 0
    for dirpath, _, files in os.walk(input_root):
        for fn in files:
            if not fn.lower().endswith(".wav"):
                continue
            src = Path(dirpath) / fn
            species_token = extract_species(fn)
            folder = sanitize_folder_name(species_token)
            dest_dir = output_root / folder
            ensure_dir(dest_dir)
            dest = dest_dir / fn
            try:
                if dry:
                    print(f"[DRY] -> {dest}")
                    moved += 1
                else:
                    if copy:
                        shutil.copy2(src, dest)
                    else:
                        # move preserves file if same FS, will fallback to copy if needed
                        shutil.move(str(src), str(dest))
                    moved += 1
            except Exception as e:
                print(f"ERROR moving {src} -> {dest}: {e}")
                errors += 1
    print(f"Done. moved={moved}, errors={errors}, skipped={skipped}")


def move_non_matching_folders(input_root: Path, other_name: str = "Other", dry: bool = False):
    """
    Move all subfolders of input_root that do NOT match the input_root basename
    into a folder named `other_name` under input_root.

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

        # If folder name equals the input basename (sanitized), skip it (leave in place)
        if entry.name == expected_folder_name:
            skipped += 1
            continue

        # Non-matching folder -> move whole folder into Other
        try:
            if dry:
                print(f"[DRY] move {entry} -> {dest_root}/")
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


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Split WAVs into species folders using 'bird_<Species>' token."
    )
    p.add_argument(
        "input",
        nargs="?",
        default=INPUT_ROOT,
        help=f"input root (default: {INPUT_ROOT})",
    )
    p.add_argument(
        "output",
        nargs="?",
        default=OUTPUT_ROOT,
        help=f"output root (default: {OUTPUT_ROOT})",
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
        help="when using --group-other, skip running the splitting/process step and only reorganize folders",
    )
    args = p.parse_args()

    # If grouping other, optionally run process first (unless --no-process)
    if args.group_other:
        if not args.no_process:
            process(Path(args.input), Path(args.output), copy=args.copy, dry=args.dry)
        move_non_matching_folders(Path(args.input), other_name=args.other_name, dry=args.dry)
    else:
        process(Path(args.input), Path(args.output), copy=args.copy, dry=args.dry)
