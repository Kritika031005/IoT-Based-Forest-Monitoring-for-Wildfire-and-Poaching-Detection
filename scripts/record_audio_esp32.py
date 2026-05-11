import serial
import numpy as np
from scipy.io.wavfile import write
import time

PORT        = "COM8"
BAUD        = 921600
SAMPLE_RATE = 16000
SECONDS     = 3

ser = serial.Serial(PORT, BAUD, timeout=10)
time.sleep(2)

# Wait for READY
print("Waiting for ESP32...")

print("ESP32 ready.")

# while True:
    # ─── STEP 1: Tell ESP32 to start ───
ser.write(b"GO\n")

# ─── STEP 2: Read sensor line ───
while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line.startswith("SENSOR:"):
        parts = line.replace("SENSOR:", "").split(",")
        temp     = float(parts[0])
        pressure = float(parts[1])
        print(f"Temperature : {temp} C")
        print(f"Pressure    : {pressure} hPa")
        break

# ─── STEP 3: Acknowledge sensor received ───
ser.reset_input_buffer()   # 🔥 VERY IMPORTANT
ser.write(b"ACK\n")
time.sleep(0.05)           # 🔥 give ESP32 time
print("ACK sent")

# ─── STEP 4: Wait for AUDIO_START ───
while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line == "AUDIO_START":
        break

# ─── STEP 5: Read exact number of audio bytes ───
total_bytes = SAMPLE_RATE * SECONDS * 2
raw_data = b""
print("Recording audio...")

start_time = time.time()

while len(raw_data) < total_bytes:
    chunk = ser.read(min(1024, total_bytes - len(raw_data)))

    if len(chunk) == 0:
        print("⚠️ Timeout while receiving audio")
        break

    raw_data += chunk
    print("Bytes:", len(raw_data))


print("Total bytes received:", len(raw_data))
print("Expected bytes:", total_bytes)
# ─── STEP 7: Process and save audio ───
samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
samples -= np.mean(samples)
samples /= (np.max(np.abs(samples)) + 1e-6)
samples = (samples * 32767).astype(np.int16)
write("recorded_audio.wav", SAMPLE_RATE, samples)
print("Saved: recorded_audio.wav")
print("---")
time.sleep(1)