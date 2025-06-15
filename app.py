from flask import Flask, request, render_template
from PIL import Image
import torch
from torchvision import models, transforms
import json

app = Flask(__name__)

# Load MobileNetV2 instead of ResNet18 (less memory usage)
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Dummy labels — replace with real ImageNet labels if you want
with open("imagenet_labels.json") as f:
    idx_to_label = json.load(f)

# Define transform for image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route("/", methods=["GET", "POST"])
def index():
    label = None
    if request.method == "POST":
        try:
            image_file = request.files["image"]
            if image_file:
                image = Image.open(image_file).convert("RGB")
                image = transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(image)
                    _, predicted = outputs.max(1)
                    label = idx_to_label.get(str(predicted.item()), "Unknown")

        except Exception as e:
            label = f"Error: {str(e)}"
    return render_template("index.html", label=label)

if __name__ == "__main__":
    app.run()
