# This is optional, a runner script for the whole pipeline
import os

print("Extracting embeddings...")
os.system("python extract_embeddings.py")

print("Training SVM model...")
os.system("python train_svm.py")

print("Live detection is disabled in this script. Please run 'python app.py' to use the web interface for image upload and detection.")
