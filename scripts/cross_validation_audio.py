import os
import numpy as np
import librosa
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# ==========================
# CONFIG
# ==========================

DATASET_PATH = "../dataset/audio"
CLASSES = ["gunshot_clean", "logging_clean", "normal_clean"]

TARGET_SR = 16000
DURATION = 3
SAMPLES = TARGET_SR * DURATION
N_MFCC = 40
MAX_LEN = 94

# ==========================
# FUNCTIONS
# ==========================

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = librosa.util.fix_length(audio, size=SAMPLES)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    # 🔥 SAME AS TRAINING
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    if mfcc.shape[0] > MAX_LEN:
        mfcc = mfcc[:MAX_LEN, :]
    else:
        mfcc = np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))

    return mfcc

# ==========================
# LOAD DATA
# ==========================

X = []
y = []

print("Loading dataset...")

for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        try:
            mfcc = extract_features(file_path)
            X.append(mfcc)
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)

print("Data shape:", X.shape)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# ==========================
# CROSS VALIDATION
# ==========================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):

    print(f"\nFold {fold+1}")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LEN, N_MFCC)),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print("Accuracy:", acc)
    accuracies.append(acc)

# ==========================
# RESULTS
# ==========================

print("\n📊 Cross Validation Results:")
print("Accuracies:", accuracies)
print("Mean Accuracy:", np.mean(accuracies))
print("Std Deviation:", np.std(accuracies))