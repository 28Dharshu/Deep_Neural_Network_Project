import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
with open("model1.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("model1.weights.h5")
print("✅ Model loaded successfully")

classes = ['O','R']

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Cannot open camera
cap.release()
cv2.destroyAllWindows()
")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    img = cv2.resize(frame, (128,128))
    img_array = np.expand_dims(img, axis=0) / 255.0

    result = model.predict(img_array)
    pred_idx = int(np.argmax(result, axis=1)[0])
    pred_label = classes[pred_idx]
    confidence = result[0][pred_idx]

    text = f"{pred_label} ({confidence*100:.1f}%)"
    cv2.putText(frame, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Live Waste Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
