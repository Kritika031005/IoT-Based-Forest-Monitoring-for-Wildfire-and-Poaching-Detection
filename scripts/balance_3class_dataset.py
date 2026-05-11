import os
import shutil
import random

BASE_PATH = "../dataset/audio"
OUTPUT_PATH = "../dataset/audio_balanced"

classes = ["gunshot_clean", "logging_clean", "normal_clean"]
LIMIT = 200

os.makedirs(OUTPUT_PATH, exist_ok=True)

for cls in classes:
    input_path = os.path.join(BASE_PATH, cls)
    output_class_path = os.path.join(OUTPUT_PATH, cls.replace("_clean", ""))

    os.makedirs(output_class_path, exist_ok=True)

    files = [f for f in os.listdir(input_path) if f.endswith(".wav")]
    random.shuffle(files)

    selected = files[:LIMIT]

    for file in selected:
        shutil.copy(
            os.path.join(input_path, file),
            os.path.join(output_class_path, file)
        )

    print(f"{cls}: {len(selected)} samples copied")

print("Balanced dataset ready!")