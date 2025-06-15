from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = models.resnet18(pretrained=True)
model.eval()

# Load labels
with open("static/imagenet_labels.json") as f:
    idx_to_label = json.load(f)

# Image transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    file = request.files['image']
    filename = secure_filename(file.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)

    image = Image.open(path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = outputs.max(1)
        species = idx_to_label.get(str(predicted.item()), "Unknown")

    return jsonify({'species': species})

if __name__ == '__main__':
    app.run(debug=True)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

