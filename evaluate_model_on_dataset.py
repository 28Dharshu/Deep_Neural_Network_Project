# evaluate_model_on_dataset.py
import os
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

MODEL_JSON = "model1.json"
MODEL_WEIGHTS = "model1.weights.h5"

DATASET_PATH = r"C:\Users\DELL\OneDrive\Desktop\DNN1\DATASET\DATASET\TEST\O\O_12568.jpg"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

def load_model(json_path, weights_path):
    with open(json_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)
    print("✅ Model loaded successfully.")
    return model

def list_class_folders(dataset_path):
    items = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    items_sorted = sorted(items)
    return items_sorted

def is_image_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in IMG_EXTS

def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr

def main():
    model = load_model(MODEL_JSON, MODEL_WEIGHTS)

    try:
        n_outputs = model.output_shape[-1]
    except Exception:
        n_outputs = model.predict(np.zeros((1,128,128,3))).shape[-1]

    class_folders = list_class_folders(DATASET_PATH)
    if len(class_folders) == 0:
        raise SystemExit(f"No class subfolders found in {DATASET_PATH}. Each class should be a folder.")

    if len(class_folders) == n_outputs:
        classes = class_folders
        print(f"Using class folders as labels (order = sorted): {classes}")
    else:
        print("⚠️ Warning: Number of class folders does not match model output size.")
        print(f"Found {len(class_folders)} folders: {class_folders}")
        print(f"Model predicts {n_outputs} outputs.")
        if n_outputs == 2:
            classes = ['O', 'R']
            print("Falling back to classes = ['O','R']. If this is wrong, please edit the script.")
        else:
            
            classes = [f"class_{i}" for i in range(n_outputs)]
            print(f"Falling back to generic classes: {classes}")

   
    total = 0
    correct = 0
    per_class_total = {c:0 for c in classes}
    per_class_correct = {c:0 for c in classes}
    results = []  
    try:
        input_shape = model.input_shape  # e.g. (None, 128, 128, 3)
        _, h, w, ch = input_shape
        if h is None or w is None:
            h, w = 128, 128
    except Exception:
        h, w = 128, 128

    
    for true_label in class_folders:
        folder = os.path.join(DATASET_PATH, true_label)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if not is_image_file(fname):
                continue
            img_path = os.path.join(folder, fname)
            try:
                x = preprocess_image(img_path, target_size=(h, w))
            except Exception as e:
                print(f"Failed to load {img_path}: {e}")
                continue
            probs = model.predict(x)  # shape (1, n_outputs)
            pred_idx = int(np.argmax(probs, axis=1)[0])
            # Map predicted index to label name if possible
            if pred_idx < len(classes):
                pred_label = classes[pred_idx]
            else:
                pred_label = f"idx_{pred_idx}"

            total += 1
            per_class_total[true_label] = per_class_total.get(true_label, 0) + 1
            if pred_label == true_label:
                correct += 1
                per_class_correct[true_label] = per_class_correct.get(true_label, 0) + 1

            results.append((img_path, true_label, pred_label, probs[0].tolist()))

    if total == 0:
        raise SystemExit("No images processed. Check your DATASET_PATH and file extensions.")

    overall_acc = correct / total * 100.0
    print("\n--- Evaluation Summary ---")
    print(f"Total images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Overall accuracy: {overall_acc:.2f}%")

    
    class_names = []
    class_accuracies = []
    for c in classes:
        t = per_class_total.get(c, 0)
        if t == 0:
            acc = None
        else:
            acc = per_class_correct.get(c, 0) / t * 100.0
        class_names.append(c)
        class_accuracies.append(acc if acc is not None else 0.0)
        print(f"Class {c}: {per_class_correct.get(c,0)}/{t} correct -> " +
              (f"{acc:.2f}%" if acc is not None else "N/A"))

    
    plt.figure(figsize=(8,5))
    plt.bar(class_names, class_accuracies)
    plt.xlabel("Class")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Per-class accuracy (Overall: {overall_acc:.2f}%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    out_plot = "accuracy_plot.png"
    plt.savefig(out_plot)
    print(f"\nSaved accuracy plot to: {out_plot}")
    try:
        plt.show()
    except Exception:
        pass

    import csv
    out_csv = "prediction_results.csv"
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "true_label", "pred_label", "probabilities"])
        for r in results:
            writer.writerow([r[0], r[1], r[2], r[3]])
    print(f"Saved per-image predictions to: {out_csv}")

if __name__ == "__main__":
    main()
