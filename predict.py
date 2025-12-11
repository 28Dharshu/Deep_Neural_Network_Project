from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
import requests

# === Load trained model ===
with open('model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("model1.weights.h5")
print("✅ Model loaded successfully")

# === Load image for testing ===
img_path = r"C:\Users\DELL\OneDrive\Desktop\DNN1\DATASET\TEST\R\R_10001.jpg"
test_image = image.load_img(img_path, target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0

# === Predict ===
result = model.predict(test_image)
predicted_class = np.argmax(result, axis=1)[0]
classes = ['O', 'R']  # O = Organic, R = Recyclable

print("Prediction Probabilities:", result)
print("Predicted Class Index:", predicted_class)
print(f"The image belongs to class: {classes[predicted_class]}")

# === NodeMCU IP Address ===
nodemcu_ip = "http://192.168.1.105"  # ⚠ Change this to your NodeMCU IP

# === Send command to NodeMCU ===
try:
    if classes[predicted_class] == 'R':
        print("♻ Opening dustbin for recyclable waste...")
        requests.get(f"{nodemcu_ip}/open")
    else:
        print("🍂 Opening dustbin for organic waste...")
        requests.get(f"{nodemcu_ip}/close")
except Exception as e:
    print("⚠ Error communicating with NodeMCU:", e)
