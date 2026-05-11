import tensorflow as tf
import numpy as np
import librosa
import os

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "../saved_models/acoustic_model_final.h5"
DATASET_PATH = "../dataset/audio"

TARGET_SR = 16000
DURATION = 3
SAMPLES = TARGET_SR * DURATION

N_MFCC = 40
MAX_LEN = 94

# ==========================
# LOAD MODEL
# ==========================

model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# FEATURE EXTRACTION (FIXED)
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

    
    return mfcc


def pad_or_trim(mfcc):
    if mfcc.shape[0] > MAX_LEN:
        return mfcc[:MAX_LEN, :]
    else:
        return np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)), mode='constant')


# ==========================
# COLLECT REPRESENTATIVE DATA
# ==========================

sample_data = []

print(" Preparing representative dataset...")

for label in ["gunshot_clean", "logging_clean", "normal_clean"]:
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):

        try:
            path = os.path.join(folder, file)

            mfcc = extract_features(path)
            mfcc = pad_or_trim(mfcc)
            mfcc = np.expand_dims(mfcc, axis=-1)
            sample_data.append(mfcc)

        except:
            continue

sample_data = np.array(sample_data, dtype=np.float32)
np.random.shuffle(sample_data)

print(" Representative dataset ready:", sample_data.shape)


# ==========================
# REPRESENTATIVE FUNCTION
# ==========================

def representative_data_gen():
    for i in range(min(100, len(sample_data))):
        sample = sample_data[i]

        # Already (94, 40, 1)
        sample = np.expand_dims(sample, axis=0)

        yield [sample.astype(np.float32)]


# ==========================
# CONVERT TO INT8
# ==========================

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# ==========================
# SAVE MODEL
# ==========================

# os.makedirs("../saved_models", exist_ok=True)

# with open("../saved_models/acoustic_model_int8.tflite", "wb") as f:
#     f.write(tflite_model)

# print(" INT8 TFLite model saved successfully!")

# ==========================
# LOAD TFLITE MODEL
# ==========================

interpreter = tf.lite.Interpreter(model_path="../saved_models/acoustic_model_int8.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input:", input_details)
print("Output:", output_details)


# ==========================
# PREPARE TEST DATA
# ==========================

X_test = []
y_test = []
labels = ["gunshot_clean", "logging_clean", "normal_clean"]

print("\n Preparing test dataset...")

for idx, label in enumerate(labels):
    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):

        try:
            path = os.path.join(folder, file)

            mfcc = extract_features(path)
            mfcc = pad_or_trim(mfcc)
            mfcc = np.expand_dims(mfcc, axis=-1)

            X_test.append(mfcc)
            y_test.append(idx)

        except:
            continue

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test)

print(" Test data shape:", X_test.shape)


# ==========================
# RUN INT8 INFERENCE
# ==========================

input_scale, input_zero_point = input_details[0]['quantization']
output_scale, output_zero_point = output_details[0]['quantization']

y_pred = []

print("\n Running INT8 inference...")

for sample in X_test:

    sample = np.expand_dims(sample, axis=0)

    #  QUANTIZE INPUT
    sample_q = sample / input_scale + input_zero_point
    sample_q = np.clip(sample_q, -128, 127).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], sample_q)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    #  DEQUANTIZE OUTPUT
    output = (output.astype(np.float32) - output_zero_point) * output_scale

    pred = np.argmax(output)
    y_pred.append(pred)

y_pred = np.array(y_pred)


# ==========================
# CALCULATE ACCURACY
# ==========================

accuracy = np.mean(y_pred == y_test)

print("\n INT8 Model Accuracy:", accuracy)


# ==========================
# OPTIONAL: CONFUSION MATRIX
# ==========================

from sklearn.metrics import classification_report, confusion_matrix

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=labels))

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))