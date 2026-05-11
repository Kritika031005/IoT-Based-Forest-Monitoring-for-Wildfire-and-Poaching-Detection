import librosa
import soundfile as sf
import numpy as np
import os

INPUT_DIR = "../dataset/new_cleaned"
OUTPUT_DIR = "../dataset/new_augmented"

TARGET_SR = 16000

def augment(audio, sr):
    augmented = []

    for _ in range(10):  # increase data

        # 1. Original
        augmented.append(audio)

        # 2. Add noise
        noise = audio + 0.003 * np.random.randn(len(audio))
        augmented.append(noise)

        # 3. Pitch shift
        pitch = librosa.effects.pitch_shift(
            audio, sr=sr, n_steps=np.random.randint(-2, 3)
        )
        augmented.append(pitch)

        # 4. Time stretch
        stretch = librosa.effects.time_stretch(
            audio, rate=np.random.uniform(0.9, 1.1)
        )
        stretch = librosa.util.fix_length(stretch, size=len(audio))
        augmented.append(stretch)

        # 5. Volume change
        volume = audio * np.random.uniform(0.7, 1.3)
        augmented.append(volume)

    return augmented


for label in ["gunshot", "logging", "normal"]:

    in_folder = os.path.join(INPUT_DIR, label)
    out_folder = os.path.join(OUTPUT_DIR, label)

    os.makedirs(out_folder, exist_ok=True)

    for file in os.listdir(in_folder):

        path = os.path.join(in_folder, file)

        audio, sr = librosa.load(path, sr=TARGET_SR)

        augmented_list = augment(audio, sr)

        for i, aug_audio in enumerate(augmented_list):

            # Normalize
            aug_audio = aug_audio / (np.max(np.abs(aug_audio)) + 1e-6)

            sf.write(
                os.path.join(out_folder, f"{file.split('.')[0]}_{i}.wav"),
                aug_audio,
                sr
            )

print("Augmentation completed!")