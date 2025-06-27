# Deep Fake Detection System

This project implements a Deep Fake Detection system using facial embeddings and a Support Vector Machine (SVM) classifier. The system is designed to detect whether an image is real or fake based on facial features.

## Project Structure

```
deep-fake-detection
├── main.py                # Entry point for the application
├── extract_embeddings.py   # Module for extracting facial embeddings
├── train_svm.py           # Module for training the SVM classifier
├── detect_fake.py         # Module for detecting fake images
├── dataset                # Directory containing the dataset
│   ├── real              # Subdirectory for real images
│   └── fake              # Subdirectory for fake images
├── embeddings             # Directory for storing extracted embeddings
├── svm_model.pkl         # Serialized SVM model
├── requirements.txt       # List of project dependencies
└── README.md              # Project documentation
```

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd deep-fake-detection
   ```

2. **Install Dependencies**
   Ensure you have Python installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   Place your images in the `dataset/real` and `dataset/fake` directories. Ensure that the images are properly labeled.

4. **Extract Embeddings**
   Run the `extract_embeddings.py` script to extract facial embeddings from the images in the dataset:
   ```bash
   python extract_embeddings.py
   ```

5. **Train the SVM Model**
   After extracting embeddings, train the SVM model using:
   ```bash
   python train_svm.py
   ```

6. **Detect Fake Images**
   Use the `detect_fake.py` script to check if an image is real or fake:
   ```bash
   python detect_fake.py --image <path_to_image>
   ```

## Usage Guidelines

- Ensure that your dataset is well-organized in the `dataset` directory.
- The `svm_model.pkl` file will be created after training the model and can be used for future predictions.
- Modify the scripts as needed to suit your specific requirements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.