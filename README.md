🧠 AI-Powered Waste Classification Using Deep Neural Networks

🔍 Classifying Recyclable (R) and Organic/Non-Recyclable (O) Waste Using Machine Learning

📌 Project Overview

This project uses a Deep Neural Network (DNN) built with TensorFlow/Keras to automatically classify waste images into two categories:

R — Recyclable Waste

O — Organic / Non-Recyclable Waste

The system can:
✔️ Predict a single image
✔️ Evaluate the entire dataset and generate accuracy graphs
✔️ Perform real-time detection using a live camera (webcam)
✔️ Save prediction reports and accuracy plots

This solution helps automate waste segregation, making it efficient and environmentally friendly.

📁 File Structure
DNN-Project/


1. model1.json                  # Saved model architecture

2. model1.weights.h5            # Model weights

3. predict_single_image.py      # Predicts one image

4. live_camera_detection.py     # Real-time waste detection using webcam

5. evaluate_model_on_dataset.py   # Dataset evaluation + accuracy graph

6. DATASET/                     # Training & testing dataset
     i)  TRAIN/ 
           -> O/
           -> R/
     ii) TEST/
           -> O/
           -> R/
7. accuracy_plot.png          # Generated accuracy graph

   
8. prediction_results.csv     # Per-image prediction report

    
10. README.md

🧠 Model Description

Deep Neural Network (DNN)

You used a feed-forward neural network with:

Input layer (128×128×3 image)

Hidden layers with ReLU activation

Dense output layer with softmax/sigmoid (2 classes: O, R)

Framework: TensorFlow/Keras
Training: On labeled waste dataset (O & R)
Evaluation Accuracy:

Overall: 89.14%

Class O: 89.51%

Class R: 88.67%

🔧 Installation

Make sure you have Python 3.8+ and install the required libraries:

pip install tensorflow
pip install numpy
pip install opencv-python
pip install matplotlib
pip install pandas

🖼 1. Predict a Single Image

Use the script predict_single_image.py:

python predict_single_image.py


Set your image path here:

img_path = r"C:\Users\DELL\Desktop\DNN1\DATASET\TEST\O\O_12568.jpg"

🎥 2. Real-Time Waste Detection (Webcam)

Run:

python live_camera_detection.py


This script:

Opens your webcam

Classifies each frame (R or O)

Displays prediction + confidence on screen

Press Q to exit

📊 3. Evaluate Dataset & Generate Graph

Run:

python evaluate_model_on_dataset.py


This script:
✔ Reads all TEST images (O & R)
✔ Predicts every image
✔ Calculates accuracy for each class
✔ Saves:

accuracy_plot.png

prediction_results.csv

Example output:

--- Evaluation Summary ---
Total images: 2513
Correct predictions: 2240
Overall accuracy: 89.14%
Class O: 1254/1401 correct -> 89.51%
Class R: 986/1112 correct -> 88.67%

🚀 Features

Accurate waste classification using DNN

Fully automated image preprocessing

Simple and clean prediction scripts

Real-time classification with webcam

Graphs & CSV reports for academic presentation

Easy to run and extend

🎯 Applications

✔ Smart waste bins
✔ Recycling centers
✔ Environmental monitoring
✔ Smart city IoT systems
✔ Educational AI projects

📌 Future Improvements

Convert to TensorFlow Lite for mobile deployment

Add more classes (metal, paper, plastic, glass)

Use Convolutional Neural Networks (CNN) for higher accuracy

Deploy on Raspberry Pi for real-time smart bin solutions

👨‍💻 Author

Dharshini M

Deep Neural Network (DNN) Waste Classification Project
