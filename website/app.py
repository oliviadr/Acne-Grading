from flask import Flask, request, render_template, redirect, jsonify, json
import torch
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn as nn
import base64

app = Flask(__name__)

# Define the device
device = torch.device("cpu")  # use "cuda:0" if you are using GPU

# Transformation definition remains the same
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Function to load a model
def load_model(model_path, model_type):
    if model_type == 'resnet':
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)  # Assuming you have 3 classes
    elif model_type == 'densenet':
        model = models.densenet169(pretrained=False)
    else:
        raise ValueError("Unsupported model type")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Load models
resnet_model = load_model('C:/Users/olivia/OneDrive/Desktop/a/models/finaltesthisbat45ep10last.pth', 'resnet')
densenet_model = load_model('C:/Users/olivia/OneDrive/Desktop/a/models/densenetfinaltest45b.pth', 'densenet')

def predict_image(image_bytes, model_type):
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    
    model = resnet_model if model_type == 'resnet' else densenet_model
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    class_names = ['low', 'medium', 'severe']  # Ensure this matches your class labels
    return class_names[predicted.item()]

@app.route('/upload_resnet', methods=['GET', 'POST'])
def upload_resnet():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        resnet_prediction = predict_image(img_bytes, 'resnet')
        return render_template('resultresnet.html', resnet_class=resnet_prediction)
    return render_template('indexresnet.html')


@app.route('/upload_densenet', methods=['GET', 'POST'])
def upload_densenet():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        densenet_prediction = predict_image(img_bytes, 'densenet')
        return render_template('resultdensenet.html', densenet_class=densenet_prediction)
    return render_template('indexdensenet.html')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        img_bytes = file.read()
        resnet_prediction = predict_image(img_bytes, 'resnet')
        densenet_prediction = predict_image(img_bytes, 'densenet')
        return render_template('result.html', resnet_class=resnet_prediction, densenet_class=densenet_prediction)

    return render_template('home.html')


@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/indexresnet', methods=['GET'])
def indexresnet():
    return render_template('indexresnet.html')

@app.route('/indexdensenet', methods=['GET'])
def indexdensenet():
    return render_template('indexdensenet.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)
    image_data = data['image_data']
    _, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    resnet_prediction = predict_image(image_bytes, 'resnet')
    densenet_prediction = predict_image(image_bytes, 'densenet')
    
    return jsonify(resnet_class=resnet_prediction, densenet_class=densenet_prediction)


@app.route('/predictresnet', methods=['POST'])
def predictresnet():
    data = json.loads(request.data)
    image_data = data['image_data']
    _, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    resnet_prediction = predict_image(image_bytes, 'resnet')
    
    return jsonify(resnet_class=resnet_prediction)

@app.route('/predictdensenet', methods=['POST'])
def predictdensenet():
    data = json.loads(request.data)
    image_data = data['image_data']
    _, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    densenet_prediction = predict_image(image_bytes, 'densenet')
    
    return jsonify(densenet_class=densenet_prediction)

@app.route('/predict_camera', methods=['POST'])
def predict_camera():
    data = json.loads(request.data)
    image_data = data['image_data']
    _, encoded = image_data.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    resnet_prediction = predict_image(image_bytes, 'resnet')
    densenet_prediction = predict_image(image_bytes, 'densenet')
    
    return jsonify(resnet_class=resnet_prediction, densenet_class=densenet_prediction)

@app.route('/result', methods=['GET'])
def result():
    resnet_class = request.args.get('resnet_class')
    densenet_class = request.args.get('densenet_class')
    return render_template('result.html', resnet_class=resnet_class, densenet_class=densenet_class)

@app.route('/resultresnet', methods=['GET'])
def resultresnet():
    resnet_class = request.args.get('resnet_class')
    densenet_class = request.args.get('densenet_class')
    return render_template('resultresnet.html', resnet_class=resnet_class, densenet_class=densenet_class)

@app.route('/resultdensenet', methods=['GET'])
def resultdensenet():
    # resnet_class = request.args.get('resnet_class')
    densenet_class = request.args.get('densenet_class')
    return render_template('resultdensenet.html',  densenet_class=densenet_class)


if __name__ == '__main__':
    app.run(debug=True)
