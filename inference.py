# Evaluating the model with Real world data
import os
import sys

if (len(sys.argv)<=1):
    print("Please attach the image as a command line argument: python3 inference.py path/to/image")
    exit()
if not os.path.exists(sys.argv[1]):
    print("Attached image does not exist!")
    exit()

import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
from PIL import Image
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.getcwd(), 'models')
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Linear(512, 1)
)

model.load_state_dict(torch.load(f"{model_path}/feature_extraction_model.pth", map_location=device))

# Replace last layer with Identity to output 512-dim features
model.fc = nn.Sequential(
    *list(model.fc.children())[:-1],
    nn.Identity()
)

model.eval()
model.to(device)

# Preprocessing the image before passing into Resnet Model
images_path = os.path.join(os.getcwd(), "images")
allowed_ext = ['.jpg' , '.jpeg' , '.png']

path_to_img = sys.argv[1]

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    models.ResNet50_Weights.DEFAULT.transforms()
])

img = Image.open(path_to_img).convert('RGB')
img_tensor = preprocess(img).unsqueeze(0)
img_tensor = img_tensor.to(device)
model.eval()

# Passing the image through the resnet model to infer the feature vectors
with torch.no_grad():
    features_512 = model(img_tensor)
features_np = features_512.cpu().numpy().reshape(1, -1) # Infered features

import joblib
loaded = joblib.load(f'{model_path}/xgb_model.joblib')

import xgboost as xgb

xgb_model = loaded.get('model')
dmatrix = xgb.DMatrix(features_np)
predictions = xgb_model.predict(dmatrix)
prediction = xgb_model.predict(dmatrix)
print(f'Prediction Percentage : {prediction}')

threshold = 0.5

class_label = "Organic" if prediction[0] >= 0.5 else "Not-Organic"

print(f"Predicted class: {class_label}")

