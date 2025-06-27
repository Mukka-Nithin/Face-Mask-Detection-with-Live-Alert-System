import cv2
import torch
import joblib
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

def main():
    # Initialize face detector and embedding model
    mtcnn = MTCNN(image_size=160, margin=0)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    rf_model = joblib.load('svm_model.pkl')

    # Open webcam with fallback device indices
    cap = None
    for device_index in range(3):
        cap = cv2.VideoCapture(device_index)
        if cap.isOpened():
            print(f"Webcam opened successfully on device {device_index}")
            break
        else:
            cap.release()
            cap = None
    if cap is None or not cap.isOpened():
        print("Error: Could not open any webcam device. Please check if webcam is connected and not used by another application.")
        return

    label = "Press SPACE to capture image and detect, or Q to quit"
    last_label = label
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display last prediction label on frame
        cv2.putText(frame, last_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0) if 'Real' in last_label else (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('Webcam Deep Fake Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # SPACE key pressed
            # Convert frame to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face and get cropped face tensor
            face = mtcnn(img_rgb)
            if face is not None:
                with torch.no_grad():
                    embedding = model(face.unsqueeze(0)).squeeze().numpy()
                    prob = rf_model.predict_proba([embedding])[0]
                    pred = rf_model.classes_[prob.argmax()]
                    last_label = f"{pred} ({prob.max()*100:.1f}%)"
            else:
                last_label = "No face detected"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
