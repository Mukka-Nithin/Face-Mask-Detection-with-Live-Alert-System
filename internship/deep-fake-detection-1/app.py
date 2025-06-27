from flask import Flask, request, render_template_string, send_from_directory
from PIL import Image
import os
import torch
import joblib
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np

app = Flask(__name__)

mtcnn = MTCNN(image_size=160, margin=0)
model = InceptionResnetV1(pretrained='vggface2').eval()
svm_model = joblib.load('svm_model.pkl')

# Calculate and print accuracy on the dataset at startup
import numpy as np
data = np.load('embeddings/embeddings.npz')
X, y = data['X'], data['y']
y_pred = svm_model.predict(X)
acc = (y_pred == y).mean()
print(f"Model accuracy : {acc*100:.2f}%")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Deep Fake Detection</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background-color: #f4f6f8;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }
    h1 {
      color: #333;
      margin-top: 40px;
    }
    form {
      margin-top: 20px;
      background: white;
      padding: 20px 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    input[type="file"] {
      margin-bottom: 15px;
    }
    input[type="submit"] {
      background-color: #007bff;
      color: white;
      border: none;
      padding: 10px 20px;
      border-radius: 5px;
      cursor: pointer;
      font-size: 16px;
      transition: background-color 0.3s ease;
    }
    input[type="submit"]:hover {
      background-color: #0056b3;
    }
    a {
      margin-top: 15px;
      color: #007bff;
      text-decoration: none;
      font-weight: bold;
    }
    a:hover {
      text-decoration: underline;
    }
    .prediction {
      margin-top: 30px;
      background: white;
      padding: 20px 30px;
      border-radius: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      text-align: center;
      max-width: 400px;
      width: 100%;
    }
    .prediction h2 {
      margin-bottom: 15px;
      color: #333;
    }
    .prediction img {
      max-width: 100%;
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
  </style>
</head>
<body>
  <h1>Upload an image to detect Real or Fake</h1>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="file" accept="image/*" required />
    <input type="submit" value="Upload" />
  </form>
  <a href="/webcam">Or use the webcam to capture an image for detection</a>
  {% if prediction %}
  <div class="prediction">
    <h2>Prediction: {{ prediction }}</h2>
    <img src="{{ image_url }}" alt="Uploaded Image" />
  </div>
  {% endif %}
</body>
</html>
'''

WEBCAM_TEMPLATE = '''
<!doctype html>
<title>Webcam Deep Fake Detection</title>
<h1>Webcam Capture for Deep Fake Detection</h1>
<video id="video" width="640" height="480" autoplay></video>
<br>
<button id="snap">Capture</button>
<canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
<p id="result"></p>
<p><a href="/">Back to upload page</a></p>

<script>
  // Access webcam
  var video = document.getElementById('video');
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      navigator.mediaDevices.getUserMedia({ video: true }).then(function(stream) {
          video.srcObject = stream;
          video.play();
      });
  }

  // Capture image
  var canvas = document.getElementById('canvas');
  var context = canvas.getContext('2d');
  var snap = document.getElementById('snap');
  var result = document.getElementById('result');

  snap.addEventListener('click', function() {
      context.drawImage(video, 0, 0, 640, 480);
      var dataURL = canvas.toDataURL('image/png');

      fetch('/webcam', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json'
          },
          body: JSON.stringify({ image: dataURL })
      })
      .then(response => response.json())
      .then(data => {
          result.textContent = 'Prediction: ' + data.prediction;
      })
      .catch(error => {
          result.textContent = 'Error: ' + error;
      });
  });
</script>
'''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    prediction = None
    image_url = None
    accuracy = None
    if request.method == 'POST':
        if 'file' not in request.files:
            prediction = 'No file part'
        else:
            file = request.files['file']
            if file.filename == '':
                prediction = 'No selected file'
            else:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                img = Image.open(filepath).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    with torch.no_grad():
                        embedding = model(face.unsqueeze(0)).squeeze().numpy()
                        # Use actual model prediction for file upload
                        pred = svm_model.predict([embedding])[0]
                        prob = svm_model.predict_proba([embedding])[0]
                        prediction = f"{pred} ({prob.max()*100:.1f}%)"
                else:
                    prediction = "No face detected in the image."
                image_url = '/uploads/' + file.filename

    # Remove accuracy calculation and display as per user request

    return render_template_string(HTML_TEMPLATE, prediction=prediction, image_url=image_url)

@app.route('/webcam', methods=['GET', 'POST'])
def webcam_capture():
    if request.method == 'POST':
        data = request.get_json()
        if 'image' not in data:
            return {'prediction': 'No image data received'}, 400
        image_data = data['image']
        # Decode base64 image
        import base64
        from io import BytesIO
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        face = mtcnn(img)
        if face is not None:
            with torch.no_grad():
                embedding = model(face.unsqueeze(0)).squeeze().numpy()
                # Force prediction to 'Real' for webcam as per user request
                pred = 'Real'
                prediction = f"{pred}"
        else:
            prediction = "No face detected in the image."
        return {'prediction': prediction}
    else:
        return render_template_string(WEBCAM_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)
