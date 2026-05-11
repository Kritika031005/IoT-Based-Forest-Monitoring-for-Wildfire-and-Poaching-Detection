import os
import numpy as np
import librosa
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

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
# FEATURE EXTRACTION
# ==========================

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

    # Fix length (3 sec)
    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = librosa.util.fix_length(audio, size=SAMPLES)

    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = mfcc.T

    #  CRITICAL (same as inference)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    return mfcc

# ==========================
# PAD / TRIM
# ==========================

def pad_or_trim(mfcc):
    if mfcc.shape[0] > MAX_LEN:
        return mfcc[:MAX_LEN, :]
    else:
        return np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)), mode='constant')

# ==========================
# LOAD DATA
# ==========================

X = []
y = []

print(" Loading dataset...")

for label in CLASSES:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        try:
            mfcc = extract_features(file_path)
            mfcc = pad_or_trim(mfcc)

            X.append(mfcc)
            y.append(label)

        except Exception as e:
            print("Error:", file_path, e)

X = np.array(X)
y = np.array(y)
X = np.expand_dims(X, axis=-1)
print(" Data loaded:", X.shape)

# ==========================
# LABEL ENCODING
# ==========================

le = LabelEncoder()
y = le.fit_transform(y)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# MODEL (CNN)
# ==========================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN, N_MFCC, 1)),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),   #  increased

    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==========================
# TRAIN
# ==========================

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    callbacks=[early_stop]
)

# ==========================
# EVALUATION
# ==========================

# ==========================
# EVALUATION
# ==========================

loss, accuracy = model.evaluate(X_test, y_test)

print("\n Test Accuracy:", accuracy)

y_pred = np.argmax(model.predict(X_test), axis=1)

print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\n Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

import matplotlib.pyplot as plt

# Accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Learning Curve')
plt.show()

# Loss
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Learning Curve')
plt.show()

# ==========================
# SAVE MODEL
# ==========================

# os.makedirs("../saved_models", exist_ok=True)

# model.save("../saved_models/acoustic_model_final.h5")
# joblib.dump(le, "../saved_models/audio_label_encoder.pkl")

# print("\n Model Saved Successfully!")