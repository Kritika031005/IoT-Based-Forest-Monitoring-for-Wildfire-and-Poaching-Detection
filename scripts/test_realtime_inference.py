# import numpy as np
# import librosa
# import tensorflow as tf
# import joblib

# # ==========================
# # CONFIG
# # ==========================

# MODEL_PATH = "../saved_models/acoustic_model_int8.tflite"
# LABEL_ENCODER_PATH = "../saved_models/audio_label_encoder.pkl"

# TARGET_SR = 16000
# DURATION = 3
# SAMPLES = TARGET_SR * DURATION
# N_MFCC = 40
# MAX_LEN = 94

# # ==========================
# # LOAD MODEL
# # ==========================

# interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
# interpreter.allocate_tensors()

# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# print("Input details:", input_details)
# print("Output details:", output_details)

# # Load label encoder
# le = joblib.load(LABEL_ENCODER_PATH)

# # ==========================
# # PAD FUNCTION
# # ==========================

# def pad_or_trim(mfcc):
#     if mfcc.shape[0] > MAX_LEN:
#         return mfcc[:MAX_LEN, :]
#     else:
#         return np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)), mode='constant')

# # ==========================
# # LOAD AUDIO
# # ==========================

# file_path = "E:/Documents/AI-Based Forest Monitoring/scripts/recorded_audio.wav"

# audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

# # Fix length
# if len(audio) > SAMPLES:
#     audio = audio[:SAMPLES]
# else:
#     audio = librosa.util.fix_length(audio, size=SAMPLES)

# audio = audio * 5
# audio = np.clip(audio, -1, 1)
# # ✅ Normalize audio (CRITICAL)
# audio = audio / (np.max(np.abs(audio)) + 1e-6)

# from scipy.signal import butter, lfilter

# def bandpass_filter(data, sr, low=100, high=4000):
#     nyq = 0.5 * sr
#     low /= nyq
#     high /= nyq
#     b, a = butter(4, [low, high], btype='band')
#     return lfilter(b, a, data)

# audio = bandpass_filter(audio, TARGET_SR)

# # ==========================
# # MFCC EXTRACTION
# # ==========================

# mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)
# mfcc = np.log(np.abs(mfcc) + 1e-6)

# # 🔥 CRITICAL FIX (MOST IMPORTANT)
# #mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

# # Pad
# mfcc = pad_or_trim(mfcc)

# # Add batch dimension
# mfcc = np.expand_dims(mfcc, axis=0).astype(np.float32)

# # Debug
# print("MFCC shape:", mfcc.shape)
# print("MFCC min:", np.min(mfcc))
# print("MFCC max:", np.max(mfcc))
# print("MFCC mean:", np.mean(mfcc))
# print("Audio max amplitude:", np.max(np.abs(audio)))

# # ==========================
# # QUANTIZE INPUT (INT8)
# # ==========================

# input_scale, input_zero_point = input_details[0]['quantization']

# mfcc_quantized =( mfcc / input_scale + input_zero_point).round()
# mfcc_quantized = np.clip(mfcc_quantized, -128, 127).astype(np.int8)

# # ==========================
# # INFERENCE (TFLITE)
# # ==========================

# interpreter.set_tensor(input_details[0]['index'], mfcc_quantized)
# interpreter.invoke()

# output = interpreter.get_tensor(output_details[0]['index'])

# # ==========================
# # DEQUANTIZE OUTPUT
# # ==========================

# output_scale, output_zero_point = output_details[0]['quantization']
# output = (output.astype(np.float32) - output_zero_point) * output_scale

# print("\nRaw output:", output)
# print("Sum:", np.sum(output))

# pred_class = np.argmax(output)
# label = le.inverse_transform([pred_class])[0]

# print("\n🎯 INT8 Prediction:", label)
# print("Confidence:", output)

# # ==========================
# # OPTIONAL: KERAS CHECK
# # ==========================

# model = tf.keras.models.load_model("../saved_models/acoustic_model_final.h5")

# pred = model.predict(mfcc.astype(np.float32))
# keras_label = le.inverse_transform([np.argmax(pred)])[0]

# print("\n🧠 Keras Prediction:", keras_label)

# prediction = model.predict(mfcc)
# print(prediction)

# import matplotlib.pyplot as plt
# import librosa.display

# plt.figure(figsize=(10, 4))
# librosa.display.specshow(mfcc[0].T, sr=16000)
# plt.colorbar()
# plt.title("MFCC of your audio")
# plt.show()

import numpy as np
import librosa
import tensorflow as tf
import joblib
from scipy.signal import butter, lfilter

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "../saved_models/acoustic_model_int8.tflite"
fire_model = joblib.load("../saved_models/fire_model.pkl")
LABEL_ENCODER_PATH = "../saved_models/audio_label_encoder.pkl"

# ==========================
# SENSOR INPUT (TEMP FIX)
# ==========================

temp = float(input("Enter Temperature: "))
pressure = float(input("Enter Pressure: "))

TARGET_SR = 16000
DURATION = 3
SAMPLES = TARGET_SR * DURATION
N_MFCC = 40
MAX_LEN = 94

# ==========================
# LOAD MODEL
# ==========================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("🔍 Model expects shape:", input_details[0]['shape'])

# Load label encoder
le = joblib.load(LABEL_ENCODER_PATH)

# ==========================
# BANDPASS FILTER
# ==========================

def bandpass_filter(data, sr, low=100, high=4000):
    nyq = 0.5 * sr
    low /= nyq
    high /= nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, data)

# ==========================
# PAD FUNCTION (FOR SHAPE: 94, 40)
# ==========================

def pad_or_trim(mfcc):
    if mfcc.shape[0] > MAX_LEN:
        return mfcc[:MAX_LEN, :]
    else:
        return np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)), mode='constant')

# ==========================
# LOAD AUDIO
# ==========================

file_path = "E:/Documents/AI-Based Forest Monitoring/scripts/recorded_audio-gunshot.wav"

audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

# Fix length to exactly 3 sec
if len(audio) > SAMPLES:
    audio = audio[:SAMPLES]
else:
    audio = librosa.util.fix_length(audio, size=SAMPLES)

# Amplify slightly
audio = audio * 5
audio = np.clip(audio, -1, 1)

# Normalize
audio = audio / (np.max(np.abs(audio)) + 1e-6)

# Bandpass filter
audio = bandpass_filter(audio, TARGET_SR)

print("🔊 Audio max amplitude:", np.max(np.abs(audio)))

# ==========================
# MFCC EXTRACTION
# ==========================

mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)

# ✅ IMPORTANT: transpose (model expects 94, 40)
mfcc = mfcc.T   # (time, features) → (94, 40)

# Pad/trim
mfcc = pad_or_trim(mfcc)

# Add batch dimension
mfcc = np.expand_dims(mfcc, axis=0)

# Add channel dimension if model expects 4D input
if len(input_details[0]['shape']) == 4:
    mfcc = np.expand_dims(mfcc, axis=-1)

mfcc = mfcc.astype(np.float32)

print("📐 Final MFCC shape:", mfcc.shape)

# ==========================
# QUANTIZATION (INT8)
# ==========================

input_scale, input_zero_point = input_details[0]['quantization']
print("\n🔍 Quantization Parameters:")
print("Input scale:", input_scale)
print("Input zero point:", input_zero_point)
mfcc_quantized = (mfcc / input_scale + input_zero_point).round()
mfcc_quantized = np.clip(mfcc_quantized, -128, 127).astype(np.int8)

# ==========================
# INFERENCE (TFLITE)
# ==========================

interpreter.set_tensor(input_details[0]['index'], mfcc_quantized)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])

# ==========================
# DEQUANTIZE OUTPUT
# ==========================

output_scale, output_zero_point = output_details[0]['quantization']
output = (output.astype(np.float32) - output_zero_point) * output_scale

print("\n📊 Raw output:", output)

pred_class = np.argmax(output)
label = le.inverse_transform([pred_class])[0]

print("\n🎯 INT8 Prediction:", label)
print("Confidence:", output)

# ==========================
# KERAS MODEL CHECK
# ==========================

model = tf.keras.models.load_model("../saved_models/acoustic_model_final.h5")

pred = model.predict(mfcc)
keras_label = le.inverse_transform([np.argmax(pred)])[0]

print("\n🧠 Keras Prediction:", keras_label)
print("Confidence:", pred)

# ==========================
# MFCC VISUALIZATION
# ==========================

import matplotlib.pyplot as plt
import librosa.display

plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc[0, :, :, 0].T, sr=TARGET_SR)
plt.colorbar()
plt.title("MFCC of your audio")
plt.show()

def estimate_humidity(temperature):
    humidity = 100 - (temperature * np.random.uniform(0.8, 1.2))
    humidity += np.random.normal(0, 5)
    humidity = np.clip(humidity, 10, 100)
    return humidity

humidity = estimate_humidity(temp)

print("\n🌡 Sensor Data:")
print("Temperature:", temp)
print("Pressure:", pressure)
print("Humidity:", humidity)

def predict_fire(temp, pressure, humidity):
    data = np.array([[temp, pressure, humidity]])
    pred = fire_model.predict(data)[0]
    prob = fire_model.predict_proba(data)[0][1]

    return pred, prob

fire_pred, fire_prob = predict_fire(temp, pressure, humidity)

print("🔥 Fire Prediction:", fire_pred)
print("🔥 Fire Probability:", fire_prob)

print("\n🚨 FINAL DECISION:")

if label == "gunshot_clean":
    print("🚨 POACHING DETECTED")

elif label == "logging_clean":
    print("🚨 ILLEGAL LOGGING DETECTED")

if fire_pred == 1:
    print("🔥 FIRE RISK DETECTED")

elif fire_pred == 2:
    print("🔥 HIGH FIRE RISK DETECTED")

else:
    print("✅ NORMAL FOREST ACTIVITY")