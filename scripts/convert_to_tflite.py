import tensorflow as tf

MODEL_PATH = "../saved_models/acoustic_model_final.h5"
OUTPUT_PATH = "../saved_models/acoustic_model.tflite"

# Load model in inference mode
model = tf.keras.models.load_model(MODEL_PATH)

# Important: ensure no training behavior
model.trainable = False

# Convert
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# No optimizations for now (float model)
tflite_model = converter.convert()

# Save
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print("Fresh FLOAT TFLite model created successfully!")