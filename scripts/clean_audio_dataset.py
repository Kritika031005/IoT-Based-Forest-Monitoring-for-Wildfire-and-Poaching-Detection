import os
import librosa
import soundfile as sf
import numpy as np

BASE_PATH = "../dataset/audio"
CLASSES = ["gunshot", "logging", "normal"]

TARGET_SR = 16000
TARGET_LENGTH = 3 * TARGET_SR  # 3 seconds

for cls in CLASSES:
    input_path = os.path.join(BASE_PATH, cls)
    output_path = os.path.join(BASE_PATH, cls + "_clean")

    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):
        if file.endswith(".wav"):
            file_path = os.path.join(input_path, file)

            try:
                audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

                # Trim or pad
                if len(audio) > TARGET_LENGTH:
                    audio = audio[:TARGET_LENGTH]
                else:
                    audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))

                # Normalize
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))

                sf.write(os.path.join(output_path, file), audio, TARGET_SR)

            except:
                print("Error processing:", file)

print("Cleaning completed!")