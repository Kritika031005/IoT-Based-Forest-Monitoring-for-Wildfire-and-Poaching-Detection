# 🌲 IoT-Based Forest Monitoring for Wildfire and Poaching Detection

An Edge-AI powered forest monitoring system designed to detect illegal poaching activities and early wildfire risks in real time using TinyML, acoustic analysis, and environmental sensing.

---

## 📌 Project Overview

Forest ecosystems are increasingly threatened by illegal poaching and rapidly spreading wildfires. Traditional monitoring systems rely heavily on cloud infrastructure and satellite surveillance, leading to:

- High detection latency
- Network dependency
- High power consumption
- Delayed response in remote regions

This project proposes an **IoT-based Edge AI monitoring system** capable of autonomous threat detection without continuous cloud dependency.

The system integrates:

- **ESP32-S3 Microcontroller**
- **INMP441 Digital Microphone**
- **BMP280 Environmental Sensor**
- **TinyML-based CNN model**
- **Real-time sensor fusion**

The proposed system performs on-device inference to achieve:

✅ Low latency  
✅ Low power consumption  
✅ Real-time threat detection  
✅ Reliable operation in remote forest areas  

---

# 🚀 Features

## 🔥 Wildfire Detection

- Detects early wildfire risk using:
  - Temperature
  - Humidity
  - Pressure
- Lightweight ML-based prediction system
- Real-time monitoring using BMP280 sensor

---

## 🚨 Poaching Detection

Acoustic event detection for:

- Gunshots
- Chainsaw / Illegal logging sounds
- Fire crackling sounds
- Normal forest activity

Uses:
- MFCC feature extraction
- Mel Spectrograms
- CNN-based audio classification

---

## 🧠 TinyML Deployment

- Quantized TensorFlow Lite INT8 model
- Optimized for ESP32-S3
- Model size < 400 KB
- Fast edge inference

---

## 📧 Alert System

- Sends real-time alerts for:
  - Gunshot detection
  - Illegal logging
  - High wildfire risk

---

## 📊 Logging System

- Detection events stored in CSV logs
- Continuous monitoring support

---

# 🏗️ System Architecture

```text
                ┌───────────────────────┐
                │     INMP441 MIC       │
                └──────────┬────────────┘
                           │
                    Audio Acquisition
                           │
                    MFCC / Spectrogram
                           │
                    TinyML CNN Model
                           │
┌─────────────┐      Threat Classification     ┌─────────────┐
│  BMP280     │ ─────────────────────────────► │  ESP32-S3   │
│  Sensor     │                                │ Edge Device │
└─────────────┘                                └──────┬──────┘
                                                       │
                                      ┌────────────────┴──────────────┐
                                      │                               │
                              Alert System                    CSV Logging
```

---

# 🧰 Hardware Components

| Component | Purpose |
|---|---|
| ESP32-S3 | Edge AI processing |
| INMP441 Microphone | Acoustic sensing |
| BMP280 Sensor | Temperature & pressure monitoring |
| Breadboard & Jumper Wires | Hardware connections |

---

# 💻 Software Requirements

- Python 3.x
- Arduino IDE / ESP-IDF
- TensorFlow
- TensorFlow Lite Micro
- Librosa
- NumPy
- SciPy

---


# 📊 Datasets Used

## 1️⃣ Wildfire Detection Dataset

Custom dataset containing:

- Temperature
- Humidity
- Pressure

Classes:
- Low
- Medium
- High fire risk

---

## 2️⃣ Chainsaw Detection Dataset

Sources:
- ESC-50 Dataset
- Freesound.org

Contains:
- Chainsaw sounds
- Background forest sounds

---

## 3️⃣ Gunshot Detection Dataset

Source:
- UrbanSound8K Dataset

Contains:
- Gunshot audio samples
- Environmental sound classes

---

# 🔗 Dataset Link

📁 Google Drive Dataset Link:

```text
https://drive.google.com/drive/folders/1yeAI1E1EFrxk76b1q0nzB1XkvIfanF76?usp=sharing
```

---

# ⚙️ Methodology

## 1. Data Collection

- Audio samples collected:
  - Gunshot
  - Chainsaw
  - Fire
  - Normal sounds
- Sensor data recorded using BMP280

---

## 2. Data Preprocessing

- Noise removal
- Audio segmentation
- WAV conversion (16 kHz)
- Feature normalization

---

## 3. Feature Extraction

- MFCC extraction
- Mel Spectrogram generation

---

## 4. Model Training

- Lightweight CNN model trained for audio classification
- Logistic Regression used for wildfire prediction

---

## 5. TinyML Optimization

- TensorFlow Lite conversion
- INT8 quantization for embedded deployment

---

# 🤖 Algorithms Used

## Convolutional Neural Network (CNN)

Used for:
- Gunshot detection
- Chainsaw detection
- Fire sound classification

Advantages:
- High accuracy
- Efficient feature extraction
- TinyML compatibility

---

## Logistic Regression

Used for:
- Wildfire risk prediction

Advantages:
- Lightweight
- Fast inference
- Suitable for embedded systems

---

# 📈 Experimental Results

| Module | Accuracy |
|---|---|
| Wildfire Detection | ~96% |
| Acoustic Detection | ~96% |
| Sensor Fusion | ~95% |

---

# ⚡ TinyML Performance

| Metric | Result |
|---|---|
| Model Size | < 400 KB |
| Inference | Real-time |
| Deployment | ESP32-S3 |

---

# 🔮 Future Enhancements

- LoRaWAN integration for long-range communication
- Solar-powered deployment
- GPS-based threat localization
- Multi-node forest monitoring network
- Mobile app integration
- Advanced deep learning models

---

# 📚 References

- TinyML for Edge AI
- Edge Impulse TinyML Projects
- UrbanSound8K Dataset
- ESC-50 Dataset
- IoT-based wildfire detection research papers

---

# 📜 License

This project is developed for academic and research purposes.

