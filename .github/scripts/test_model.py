"""
Model inference test script.
Loads the emotion recognition model and runs a test prediction.
"""

import json
from datetime import datetime

import numpy as np
import tensorflow as tf

# Load model (TF 2.10 SavedModel format)
print("Loading model...")
model = tf.keras.models.load_model("model/my_model")
model.load_weights("model/best_weights.h5")
print("Model loaded successfully")

# Simulate a blank 48x48 grayscale image
img = np.zeros((1, 48, 48, 1), dtype=np.float32)
result = model.predict(img)

label_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happiness", 4: "Sad", 5: "Surprise", 6: "Neutral"}
predicted_class = int(np.argmax(result))
predicted_emotion = label_dict[predicted_class]

assert result.shape == (1, 7), f"Expected output shape (1, 7), got {result.shape}"
assert abs(result.sum() - 1.0) < 0.05, f"Output probabilities do not sum to 1: {result.sum()}"

print(f"Model inference OK — output shape: {result.shape}")
print(f"Predicted emotion: {predicted_emotion}")
print(f"Probabilities: {result[0].tolist()}")

# Save results with timestamp
timestamp = datetime.now().strftime("%d%m%y")
output = {
    "timestamp": timestamp,
    "model": "my_model",
    "tensorflow_version": tf.__version__,
    "output_shape": list(result.shape),
    "probabilities": result[0].tolist(),
    "predicted_emotion": predicted_emotion,
    "status": "passed",
}

filename = f"inference_results_{timestamp}.json"
with open(filename, "w") as f:
    json.dump(output, f, indent=2)

print(f"Results saved to {filename}")
