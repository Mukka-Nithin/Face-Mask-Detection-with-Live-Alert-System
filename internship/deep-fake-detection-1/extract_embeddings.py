import os
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch

mtcnn = MTCNN(image_size=160, margin=0)
model = InceptionResnetV1(pretrained='vggface2').eval()

base_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor()
])

# Define augmentation transforms
augmentation_transforms = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(degrees=15),
]

def apply_augmentations(img):
    augmented_images = [img]
    for aug in augmentation_transforms:
        augmented_images.append(aug(img))
    return augmented_images

def extract_embedding(img):
    face = mtcnn(img)
    if face is not None:
        with torch.no_grad():
            return model(face.unsqueeze(0)).squeeze().numpy()
    return None

def process_directory(dataset_path, save_path):
    X, y = [], []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = Image.open(img_path).convert('RGB')
            augmented_imgs = apply_augmentations(img)
            for aug_img in augmented_imgs:
                embedding = extract_embedding(aug_img)
                if embedding is not None:
                    X.append(embedding)
                    y.append(label)
    X, y = np.array(X), np.array(y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, X=X, y=y)

if __name__ == '__main__':
    process_directory('dataset', 'embeddings/embeddings.npz')
