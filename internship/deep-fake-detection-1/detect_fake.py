import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import joblib
from PIL import Image
import sys
import os

def detect_image(image_path):
    if not os.path.exists(image_path):
        print(f"Image file {image_path} does not exist.")
        return

    mtcnn = MTCNN(image_size=160, margin=0)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    rf_model = joblib.load('svm_model.pkl')

    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)

    if face is not None:
        with torch.no_grad():
            embedding = model(face.unsqueeze(0)).squeeze().numpy()
            prob = rf_model.predict_proba([embedding])[0]
            pred = rf_model.classes_[prob.argmax()]
            label = f"{pred} ({prob.max()*100:.1f}%)"
            print(f"Prediction: {label}")
    else:
        print("No face detected in the image.")

if __name__ == '__main__':
    image_path = input("Please enter the path of the image to detect (upload): ").strip()
    detect_image(image_path)
