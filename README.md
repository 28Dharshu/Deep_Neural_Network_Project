# AI-Powered Waste Classification System  
Using Deep Neural Networks, TensorFlow, Keras and OpenCV

---

## 1. Project Overview  

This AI-based Waste Classification System uses a Deep Neural Network (DNN) to classify waste images into:

- Recyclable Waste (R)
- Organic / Non-Recyclable Waste (O)

The project performs:
- Single image prediction
- Real-time webcam detection
- Dataset evaluation
- Accuracy graph generation
- CSV prediction report generation

The system helps automate waste segregation for smart environmental and recycling applications.

---

## 2. Features  

- Automated waste image classification  
- Real-time webcam-based detection  
- Dataset accuracy evaluation  
- Accuracy graph generation  
- CSV prediction report generation  
- Simple TensorFlow/Keras implementation  
- Lightweight and easy-to-run project  

---

## 3. Technologies Used  

- Python  
- TensorFlow  
- Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  

---

## 4. Model Architecture  

The project uses a Feed-Forward Deep Neural Network with:

- Input Layer (128 × 128 × 3 image)
- Hidden Layers with ReLU activation
- Output Layer with Sigmoid/Softmax activation
- Binary classification:
  - O → Organic / Non-Recyclable
  - R → Recyclable

---

## 5. Dataset Structure 



─ model1.json
─ model1.weights.h5
─ predict_single_image.py
─ live_camera_detection.py
─ evaluate_model_on_dataset.py
─ accuracy_plot.png
─ prediction_results.csv
─ README.md

─ DATASET/

    ─ TRAIN/
      ─ O/
      ─ R/
    
    ─ TEST/
      ─ O/
      ─ R/

---

## 6. Model Accuracy  

| Metric | Accuracy |
|--------|-----------|
| Overall Accuracy | 89.14% |
| Class O Accuracy | 89.51% |
| Class R Accuracy | 88.67% |

---

## 7. Output Types  

### 7.1 Single Image Prediction  

The system predicts whether the selected image belongs to:
- Recyclable Waste (R)
- Organic / Non-Recyclable Waste (O)

---

### 7.2 Real-Time Webcam Detection  

The webcam system:
- Captures live video
- Detects waste category
- Displays prediction label
- Shows confidence score in real time

---

### 7.3 Dataset Evaluation  

The evaluation system:
- Reads all TEST images
- Predicts each image
- Calculates overall accuracy
- Calculates class-wise accuracy
- Generates graph and CSV report

---

## 8. Output Results  

### 8.1 Accuracy Graph  

<p align="center">
  <img src="https://raw.githubusercontent.com/28Dharshu/Deep_Neural_Network_Project/main/accuracy_plot.png" width="500">
</p>

---

### 8.2 Evaluation Output  

```text
--- Evaluation Summary ---

Total images: 2513
Correct predictions: 2240
Overall accuracy: 89.14%

Class O:
1254 / 1401 correct -> 89.51%

Class R:
986 / 1112 correct -> 88.67%
