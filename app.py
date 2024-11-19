from flask import Flask, request, render_template, jsonify, send_from_directory
import torch
import os
import torch.nn as nn
import cv2
from torchvision import transforms
from werkzeug.utils import secure_filename

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Directory to save uploaded and predicted images
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16 MB

# Tumor labels
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
image_size = 150

# Define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (image_size // 16) * (image_size // 16), 256)
        self.fc2 = nn.Linear(256, len(labels))
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
model = CNNModel()
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"status": "Error", "message": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"status": "Error", "message": "No selected image"}), 400

    if file:
        try:
            # Save uploaded image
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"Uploaded file saved to: {filepath}")

            # Preprocess the uploaded image
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_tensor = transform(image).unsqueeze(0)

            # Predict tumor type
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)

            label_map = {
                'glioma_tumor': 'Glioma Tumor',
                'meningioma_tumor': 'Meningioma Tumor',
                'pituitary_tumor': 'Pituitary Tumor',
                'no_tumor': None
            }
            predicted_label = labels[predicted.item()]
            tumor_type = label_map.get(predicted_label)

            # Find the closest match
            training_image_paths = []
            for dirpath, _, filenames in os.walk('Brain Tumor/Training'):
                for file in filenames:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        training_image_paths.append(os.path.join(dirpath, file))

            min_distance = float('inf')
            closest_match_path = None

            for train_image_path in training_image_paths:
                train_img = cv2.imread(train_image_path)
                if train_img is None:
                    continue
                train_img = cv2.resize(train_img, (image_size, image_size))
                train_tensor = transform(train_img).unsqueeze(0)

                with torch.no_grad():
                    train_output = model(train_tensor)
                distance = torch.norm(output - train_output).item()
                if distance < min_distance:
                    min_distance = distance
                    closest_match_path = train_image_path

            # Save closest match image
            closest_match_filename = secure_filename(os.path.basename(closest_match_path))
            saved_match_path = os.path.join(app.config['UPLOAD_FOLDER'], closest_match_filename)
            cv2.imwrite(saved_match_path, cv2.imread(closest_match_path))
            print(f"Closest match saved to: {saved_match_path}")

            return jsonify({
                "status": "Tumor Detected!" if tumor_type else "No Tumor Detected!",
                "tumor_type": f"Tumor Type: {tumor_type}" if tumor_type else "",
                "closest_image": f"/uploads/{closest_match_filename}"
            })

        except Exception as e:
            return jsonify({"status": "Error", "message": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
