import serial
import numpy as np
import librosa
import tensorflow as tf
import joblib
import time
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter

from dotenv import load_dotenv
import os

load_dotenv()

sender = os.getenv("EMAIL")
password = os.getenv("APP_PASSWORD")
receiver = os.getenv("RECEIVER_EMAIL")

LATITUDE = 17.3850      # your location
LONGITUDE = 78.4867

# ==========================
# EMAIL ALERT FUNCTION
# ==========================

import smtplib
from email.mime.text import MIMEText
from datetime import datetime

def send_email_alert(message):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    map_link = f"https://www.google.com/maps?q={LATITUDE},{LONGITUDE}"

    full_message = f"""
Forest Monitoring Alert

Time: {timestamp}
Location: {LATITUDE}, {LONGITUDE}
{map_link}
{message}
"""

    msg = MIMEText(full_message)
    msg['Subject'] = "Forest Alert System"
    msg['From'] = sender
    msg['To'] = receiver

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()

        print(" Email sent successfully")

    except Exception as e:
        print(" Email failed:", e)



# ==========================
# LOGGING FUNCTION
# ==========================

import csv
from datetime import datetime
import os

LOG_FILE = "forest_alerts_log.csv"

def log_event(temp, pressure, humidity, sound_label, fire_pred, message):

    file_exists = os.path.isfile(LOG_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only once
        if not file_exists:
            writer.writerow([
                    "Timestamp",
                    "Temperature",
                    "Pressure",
                    "Humidity",
                    "Latitude",
                    "Longitude",
                    "Sound Prediction",
                    "Fire Prediction",
                    "Alert Message"

            ])

        

        writer.writerow([
            timestamp,
                temp,
                pressure,
                humidity,
                LATITUDE,
                LONGITUDE,
                sound_label,
                fire_pred,
                message
        ])

    print(" Event logged successfully")

# ==========================
# CONFIG
# ==========================

PORT = "COM8"
BAUD = 921600

SAMPLE_RATE = 16000
SECONDS = 3

MODEL_PATH = "../saved_models/acoustic_model_int8.tflite"
fire_model = joblib.load("../saved_models/fire_model.pkl")
LABEL_ENCODER_PATH = "../saved_models/audio_label_encoder.pkl"

TARGET_SR = 16000
SAMPLES = TARGET_SR * SECONDS
N_MFCC = 40
MAX_LEN = 94

# ==========================
# LOAD MODEL
# ==========================

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

le = joblib.load(LABEL_ENCODER_PATH)

# ==========================
# HELPERS (UNCHANGED)
# ==========================

def bandpass_filter(data, sr, low=100, high=4000):
    nyq = 0.5 * sr
    low /= nyq
    high /= nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, data)

def pad_or_trim(mfcc):
    if mfcc.shape[0] > MAX_LEN:
        return mfcc[:MAX_LEN, :]
    else:
        return np.pad(mfcc, ((0, MAX_LEN - mfcc.shape[0]), (0, 0)))

def estimate_humidity(temp):
    humidity = 100 - (temp * 0.9)
    return np.clip(humidity, 10, 100)

def predict_fire(temp, pressure, humidity):
    data = np.array([[temp, pressure, humidity]])
    pred = fire_model.predict(data)[0]
    prob = fire_model.predict_proba(data)[0][1]
    return pred, prob

# ==========================
# SERIAL CONNECTION
# ==========================

ser = serial.Serial(PORT, BAUD, timeout=10)
time.sleep(2)

print(" ESP32 Connected")

# ==========================
# STEP 1: REQUEST DATA
# ==========================

while True:

    print("\n==============================")
    print(" New Monitoring Cycle Started")
    print("==============================")

    # ==========================
    # STEP 1: REQUEST DATA
    # ==========================
    ser.write(b"GO\n")

    # ==========================
    # STEP 2: READ SENSOR
    # ==========================
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith("SENSOR:"):
            parts = line.replace("SENSOR:", "").split(",")
            temp = float(parts[0])
            pressure = float(parts[1])
            break

    print(f"\nTemperature : {temp}")
    print(f"Pressure    : {pressure}")

    # ==========================
    # STEP 3: ACK
    # ==========================
    ser.reset_input_buffer()
    ser.write(b"ACK\n")
    time.sleep(0.01)

    # ==========================
    # STEP 4: AUDIO MODE
    # ==========================
    print("\nChoose input mode:")
    print("1. Live ESP32 Audio")
    print("2. Saved Audio File")

    choice = input("Enter choice (1 or 2): ")

    if choice == "1":
        # ─── LIVE AUDIO ───

        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line == "AUDIO_START":
                break

        total_bytes = SAMPLE_RATE * SECONDS * 2
        raw_data = b""

        print("\n Recording audio...")

        while len(raw_data) < total_bytes:
            chunk = ser.read(min(1024, total_bytes - len(raw_data)))
            if len(chunk) == 0:
                print(" Timeout")
                break
            raw_data += chunk

        print("Bytes received:", len(raw_data))
        # Flush AUDIO_END marker
        # Read until AUDIO_END marker safely
        buffer = b""

        while True:
            chunk = ser.read(64)
            if not chunk:
                break
            buffer += chunk

            if b"AUDIO_END\n" in buffer:
                break

        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
        audio -= np.mean(audio)
        audio /= (np.max(np.abs(audio)) + 1e-6)
        audio = (audio * 32767).astype(np.int16)
        write("recorded_audio.wav", SAMPLE_RATE, audio)
        print("Saved: recorded_audio.wav")
        print("---")

    else:
        # ─── FILE MODE ───
        file_path = input("Enter audio file path: ")

        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        if len(audio) > SAMPLES:
            audio = audio[:SAMPLES]
        else:
            audio = librosa.util.fix_length(audio, size=SAMPLES)
        # WAIT FOR ESP32 TO FINISH AUDIO (IMPORTANT)
        # while True:
        #     line = ser.readline().decode(errors="ignore").strip()
        #     if line == "AUDIO_END":
        #         break

    # ==========================
    # AUDIO PROCESSING (UNCHANGED)
    # ==========================

    audio = audio * 5
    audio = np.clip(audio, -1, 1)

    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    audio = bandpass_filter(audio, TARGET_SR)

    # ==========================
    # MFCC (UNCHANGED)
    # ==========================

    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC)
    mfcc = mfcc.T
    mfcc = pad_or_trim(mfcc)

    mfcc = np.expand_dims(mfcc, axis=0)

    if len(input_details[0]['shape']) == 4:
        mfcc = np.expand_dims(mfcc, axis=-1)

    mfcc = mfcc.astype(np.float32)

    # ==========================
    # QUANTIZATION (UNCHANGED)
    # ==========================

    scale, zero = input_details[0]['quantization']
    mfcc_q = (mfcc / scale + zero).round()
    mfcc_q = np.clip(mfcc_q, -128, 127).astype(np.int8)

    # ==========================
    # AUDIO INFERENCE
    # ==========================

    interpreter.set_tensor(input_details[0]['index'], mfcc_q)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])

    scale, zero = output_details[0]['quantization']
    output = (output.astype(np.float32) - zero) * scale

    label = le.inverse_transform([np.argmax(output)])[0]

    print("\n Sound Prediction:", label)

    # ==========================
    # FIRE MODEL (UNCHANGED)
    # ==========================

    humidity = estimate_humidity(temp)

    fire_pred, fire_prob = predict_fire(temp, pressure, humidity)

    print("\n Fire Prediction:", fire_pred)
    print(" Probability:", fire_prob)

    # ==========================
    # FINAL DECISION
    # ==========================

    print("\n FINAL DECISION:")

    alerts = []

    if label == "gunshot_clean":
        alerts.append("POACHING DETECTED")

    if label == "logging_clean":
        alerts.append("ILLEGAL LOGGING DETECTED")

    # if fire_pred == 2:
    #     alerts.append("HIGH FIRE RISK DETECTED")

    if len(alerts) == 0:
        msg = "NORMAL FOREST ACTIVITY"
    else:
        msg = "\n".join(alerts)

    print(msg)
    # ==========================
    # SEND STATUS TO ESP32 (LED CONTROL)
    # ==========================
    ser.flush()
    
    print("Sending to ESP32:", 
      #"FIRE" if fire_pred == 2 else 
      "POACH" if label in ["gunshot_clean", "logging_clean"] else 
      "NORMAL")

    # if fire_pred == 2:
    #     ser.write(b"FIRE\n")   #  Fire → LED ON

    if label == "gunshot_clean" or label == "logging_clean":
        ser.write(b"POACH\n")  #  Poaching → Blink

    else:
        ser.write(b"NORMAL\n") #  Normal → LED OFF
    time.sleep(0.05)


    if len(alerts) > 0:
        send_email_alert(msg)

    log_event(temp, pressure, humidity, label, fire_pred, msg)

    print("\n Waiting for next cycle...\n")
    time.sleep(5)