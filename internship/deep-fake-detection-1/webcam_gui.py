import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import torch
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

class WebcamApp:
    def __init__(self, window, window_title, threshold=0.95):
        self.window = window
        self.window.title(window_title)
        self.threshold = threshold

        # Load models
        self.mtcnn = MTCNN(image_size=160, margin=0)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.rf_model = joblib.load('svm_model.pkl')

        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.window.destroy()
            return

        # Create canvas for video frame
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Capture button
        self.btn_capture = tk.Button(window, text="Capture & Detect", width=20, command=self.capture_and_detect)
        self.btn_capture.pack(anchor=tk.CENTER, expand=True)

        # Label for prediction result
        self.label_result = tk.Label(window, text="Prediction: None", font=("Arial", 14))
        self.label_result.pack(anchor=tk.CENTER, expand=True)

        self.delay = 15
        self.update()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert frame to RGB
            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    def capture_and_detect(self):
        if hasattr(self, 'frame'):
            img = Image.fromarray(self.frame)
            face = self.mtcnn(img)
            if face is not None:
                with torch.no_grad():
                    embedding = self.model(face.unsqueeze(0)).squeeze().numpy()
                    prob = self.rf_model.predict_proba([embedding])[0]
                    max_prob = prob.max()
                    pred = self.rf_model.classes_[prob.argmax()]
                    if max_prob >= self.threshold:
                        label = f"{pred} ({max_prob*100:.1f}%)"
                    else:
                        label = f"Uncertain ({max_prob*100:.1f}%)"
                    self.label_result.config(text=f"Prediction: {label}")
            else:
                self.label_result.config(text="Prediction: No face detected")
        else:
            self.label_result.config(text="Prediction: No frame available")

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root, "Webcam Deep Fake Detection")
