# AI-Powered Waste Classification Using Deep Neural Networks

## Classifying Recyclable and Organic Waste Using Machine Learning

---

## Project Overview

This project presents an AI-based waste classification system developed using a Deep Neural Network (DNN) with TensorFlow and Keras. The model automatically classifies waste images into two categories:

- R — Recyclable Waste
- O — Organic / Non-Recyclable Waste

The system is designed to support automated waste segregation for smart environmental management applications.

The project includes:
- Single image prediction
- Real-time waste detection using webcam
- Dataset evaluation and accuracy analysis
- Accuracy graph generation
- Prediction report generation in CSV format

---

## Dataset Structure

DNN-Project/
│
├── model1.json
├── model1.weights.h5
├── predict_single_image.py
├── live_camera_detection.py
├── evaluate_model_on_dataset.py
├── accuracy_plot.png
├── prediction_results.csv
├── README.md
│
└── DATASET/
    ├── TRAIN/
    │   ├── O/
    │   └── R/
    │
    └── TEST/
        ├── O/
        └── R/

---

## Model Description

The project uses a Feed-Forward Deep Neural Network for image classification.

### Architecture
- Input Layer: 128 × 128 × 3 image
- Hidden Layers with ReLU activation
- Output Layer with Softmax/Sigmoid activation
- Framework: TensorFlow / Keras

### Training Details
- Trained using labeled waste images
- Classes:
  - Organic / Non-Recyclable (O)
  - Recyclable (R)

---

## Model Performance

| Metric | Accuracy |
|--------|-----------|
| Overall Accuracy | 89.14% |
| Class O Accuracy | 89.51% |
| Class R Accuracy | 88.67% |

---

## Installation

Install the required Python libraries before running the project.

pip install tensorflow
pip install numpy
pip install opencv-python
pip install matplotlib
pip install pandas

---

## Running the Project

### 1. Single Image Prediction

Run the following script:

python predict_single_image.py

Set the image path inside the script:

img_path = r"C:\Users\DELL\Desktop\DNN1\DATASET\TEST\O\O_12568.jpg"

The model predicts whether the image belongs to:
- Recyclable Waste (R)
- Organic / Non-Recyclable Waste (O)

---

### 2. Real-Time Waste Detection Using Webcam

Run:

python live_camera_detection.py

Features:
- Opens webcam for live detection
- Predicts waste category in real time
- Displays prediction label and confidence score
- Press Q to exit the webcam window

---

### 3. Dataset Evaluation and Accuracy Graph

Run:

python evaluate_model_on_dataset.py

Functions Performed:
- Reads all images from TEST dataset
- Predicts each image
- Calculates:
  - Overall accuracy
  - Class-wise accuracy
- Generates:
  - Accuracy graph
  - CSV prediction report

Generated Files:
- accuracy_plot.png
- prediction_results.csv

Sample Output:

--- Evaluation Summary ---

Total images: 2513
Correct predictions: 2240
Overall accuracy: 89.14%

Class O:
1254 / 1401 correct -> 89.51%

Class R:
986 / 1112 correct -> 88.67%

---

## Features

- Automated waste image classification
- Real-time webcam-based prediction
- Dataset evaluation with performance metrics
- Accuracy visualization using graphs
- CSV report generation
- Simple and modular implementation
- Easy to extend for future improvements

---

## Applications

- Smart waste management systems
- Recycling centers
- Environmental monitoring systems
- Smart city applications
- AI-based educational projects
- IoT-enabled waste segregation systems

---

## Future Enhancements

- Deploy using TensorFlow Lite
- Add additional waste categories:
  - Plastic
  - Metal
  - Glass
  - Paper
- Improve performance using Convolutional Neural Networks (CNN)
- Raspberry Pi deployment for smart bin applications
- Cloud-based monitoring and analytics

---

## Technologies Used

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pandas

---

## Author

Dharshini M

Deep Neural Network Waste Classification Project
