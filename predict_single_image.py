from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
with open("model1.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("model1.weights.h5")
print("✅ Model loaded successfully")

# Define classes
classes = ['O', 'R']

# Single image path
img_path = r"C:\Users\DELL\OneDrive\Desktop\DNN1\DATASET\DATASET\TEST\O\O_12568.jpg"

# Load and preprocess image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
result = model.predict(img_array)
pred_idx = int(np.argmax(result, axis=1)[0])
pred_label = classes[pred_idx]
confidence = result[0][pred_idx]

print("\n📸 Image:", img_path)
print("🔮 Prediction:", pred_label)
print(f"Confidence: {confidence*100:.2f}%")
