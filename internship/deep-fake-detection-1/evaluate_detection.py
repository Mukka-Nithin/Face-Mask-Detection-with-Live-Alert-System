import os
import numpy as np
from sklearn.metrics import accuracy_score
from detect_fake import detect_image
from extract_embeddings import extract_embedding

def evaluate_detection(dataset_path):
    y_true = []
    y_pred = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            embedding = extract_embedding(img)
            if embedding is not None:
                # Load the trained SVM model
                import joblib
                svm_model = joblib.load('svm_model.pkl')
                pred = svm_model.predict([embedding])[0]
                y_true.append(label)
                y_pred.append(pred)
            else:
                print(f"No face detected in image: {img_path}")

    accuracy = accuracy_score(y_true, y_pred)
    # Accuracy calculation done but not displayed as per user request

if __name__ == '__main__':
    evaluate_detection('dataset')
