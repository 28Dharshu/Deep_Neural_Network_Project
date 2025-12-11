from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np


with open('model1.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

model.load_weights("model1.weights.h5")
print(" Model loaded successfully")


img_path = r"C:\Users\DELL\OneDrive\Desktop\DNN1\DATASET\TEST"

test_image = image.load_img(img_path, target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0


result = model.predict(test_image)
predicted_class = np.argmax(result, axis=1)[0]


classes = ['O','R']  

print("Prediction Probabilities:", result)
print("Pedicted Class Index:", predicted_class)
print(f" The image belongs to class: {classes[predicted_class]}")
