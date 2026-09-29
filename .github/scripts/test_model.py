import keras
import numpy as np

model = keras.models.load_model("model/my_model")
model.load_weights("model/best_weights.h5")

# Simulate a blank 48x48 grayscale image
img = np.zeros((1, 48, 48, 1))
result = model.predict(img)

assert result.shape == (1, 7), f"Expected output shape (1, 7), got {result.shape}"
assert abs(result.sum() - 1.0) < 0.01, "Output probabilities do not sum to 1"
print("Model inference OK — output shape:", result.shape)
