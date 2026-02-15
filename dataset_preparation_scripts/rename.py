import os

INPUT_DIR = "/Users/matteomarangoni/Desktop/Bird_datasets/Komorebi_samples/Cicada_tones_dev"
BASE_NAME = "cicada_tones_dev"
START_INDEX = 1
ZERO_PADDING = 3  # file001, file002, ...

files = sorted(
    f for f in os.listdir(INPUT_DIR)
    if os.path.isfile(os.path.join(INPUT_DIR, f))
)

for i, filename in enumerate(files, start=START_INDEX):
    old_path = os.path.join(INPUT_DIR, filename)

    name, ext = os.path.splitext(filename)
    new_name = f"{BASE_NAME}{str(i).zfill(ZERO_PADDING)}{ext}"
    new_path = os.path.join(INPUT_DIR, new_name)

    os.rename(old_path, new_path)

print(f"Renamed {len(files)} files.")
