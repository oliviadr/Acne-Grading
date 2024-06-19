from flask import Flask, request, render_template, redirect, jsonify, json
import torch
from torchvision import models, transforms
from PIL import Image
import io
import torch.nn as nn
import base64

app = Flask(__name__)

# Define the device
device = torch.device("cpu")  

def pad_image_to_square(img):
    width, height = img.size
    max_side = max(width, height)
    padding = (
        (max_side - width) // 2,
        (max_side - height) // 2,
        (max_side - width + 1) // 2,
        (max_side - height + 1) // 2
    )
    return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')



transform = transforms.Compose([
    transforms.Lambda(pad_image_to_square),
    transforms.Resize(224),
    transforms.ToTensor(),
])


# Function to load a model
def load_model(model_path, model_type):
    if model_type == 'resnet':
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3) 
    elif model_type == 'densenet':
        model = models.densenet169(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 3)
    else:
        raise ValueError("Unsupported model type")
    
    # Load the state_dict for the model
    state_dict = torch.load(model_path, map_location=device)
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

# Load models
resnet_model = load_model('C:/Users/olivia/OneDrive/Desktop/a/Acne-Grading/models/model_resnet_50.pth', 'resnet')
densenet_model = load_model('C:/Users/olivia/OneDrive/Desktop/a/Acne-Grading/models/model_densenet_169.pth', 'densenet')

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
    
    class_names = ['low', 'medium', 'severe']  
    return class_names[predicted.item()]

def get_skincare_recommendations(severity):
    recommendations = {
        'low': [
            "Use a gentle cleanser twice daily.",
            "Apply a light moisturizer.",
            "Use sunscreen with SPF 30+.",
            "Avoid heavy makeup.",
            
            "1. Cetaphil Gentle Skin Cleanser",
            "2. Neutrogena Hydro Boost Water Gel",
            "3. La Roche-Posay Anthelios Melt-in Milk Sunscreen SPF 60"
        ],
        'medium': [
            "Use a cleanser with salicylic acid.",
            "Apply a moisturizer containing benzoyl peroxide.",
            "Consider using a topical retinoid.",
            "Use a gentle exfoliant once a week.",
            
            "1. Neutrogena Oil-Free Acne Wash",
            "2. Clean & Clear Persa-Gel 10 Acne Medication",
            "3. Differin Adapalene Gel 0.1% Acne Treatment"
        ],
        'severe': [
            "Consult with a dermatologist.",
            "Use prescription-strength treatments.",
            "Avoid harsh scrubs and exfoliants.",
            "Stay hydrated and maintain a healthy diet.",
           
            "1. CeraVe Hydrating Cleanser",
            "2. Tazorac (Tazarotene) Cream",
            "3. Aczone (Dapsone) Gel 7.5%"
        ]
    }
    return recommendations.get(severity, [])


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
        recommendations = get_skincare_recommendations(resnet_prediction)
        return render_template('resultresnet.html', resnet_class=resnet_prediction, recommendations=recommendations)
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
        recommendations = get_skincare_recommendations(densenet_prediction)
        return render_template('resultdensenet.html', densenet_class=densenet_prediction, recommendations=recommendations)
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
        resnet_recommendations = get_skincare_recommendations(resnet_prediction)
        densenet_recommendations = get_skincare_recommendations(densenet_prediction)
        return render_template('result.html', resnet_class=resnet_prediction, densenet_class=densenet_prediction, resnet_recommendations=resnet_recommendations, densenet_recommendations=densenet_recommendations)

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
    resnet_recommendations = get_skincare_recommendations(resnet_class)
    densenet_recommendations = get_skincare_recommendations(densenet_class)
    return render_template('result.html', resnet_class=resnet_class, densenet_class=densenet_class, resnet_recommendations=resnet_recommendations, densenet_recommendations=densenet_recommendations)

@app.route('/resultresnet', methods=['GET'])
def resultresnet():
    resnet_class = request.args.get('resnet_class')
    recommendations = get_skincare_recommendations(resnet_class)
    return render_template('resultresnet.html', resnet_class=resnet_class, recommendations=recommendations)

@app.route('/resultdensenet', methods=['GET'])
def resultdensenet():
    densenet_class = request.args.get('densenet_class')
    recommendations = get_skincare_recommendations(densenet_class)
    return render_template('resultdensenet.html', densenet_class=densenet_class, recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug=True)
