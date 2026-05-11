import os
import shutil
import pandas as pd

# -----------------------------
# 1. Paths
# -----------------------------

ESC50_PATH = "../dataset/ESC-50-master"
AUDIO_PATH = os.path.join(ESC50_PATH, "audio")
META_PATH = os.path.join(ESC50_PATH, "meta", "esc50.csv")

OUTPUT_PATH = "../dataset/audio"

# -----------------------------
# 2. Create Output Folders
# -----------------------------

folders = [
    "chainsaw",
    "handsaw",
    "fire_crackling",
    "gunshot",
    "normal"
]

for folder in folders:
    os.makedirs(os.path.join(OUTPUT_PATH, folder), exist_ok=True)

# -----------------------------
# 3. Load Metadata
# -----------------------------

df = pd.read_csv(META_PATH)

print("Total files in ESC-50:", len(df))

# -----------------------------
# 4. Category Mapping
# -----------------------------

normal_categories = [
    "rain",
    "wind",
    "insects",
    "crickets",
    "sea_waves"
]

for index, row in df.iterrows():

    category = row["category"]
    filename = row["filename"]

    source_file = os.path.join(AUDIO_PATH, filename)

    if not os.path.exists(source_file):
        continue

    # Chainsaw
    if category == "chainsaw":
        destination = os.path.join(OUTPUT_PATH, "chainsaw", filename)

    # Hand saw
    elif category == "hand_saw":
        destination = os.path.join(OUTPUT_PATH, "handsaw", filename)

    # Crackling fire
    elif category == "crackling_fire":
        destination = os.path.join(OUTPUT_PATH, "fire_crackling", filename)

    # Fireworks used as gunshot-like sound
    elif category == "fireworks":
        destination = os.path.join(OUTPUT_PATH, "gunshot", filename)

    # Normal forest sounds
    elif category in normal_categories:
        destination = os.path.join(OUTPUT_PATH, "normal", filename)

    else:
        continue

    shutil.copy(source_file, destination)

print("Segregation Completed Successfully!")