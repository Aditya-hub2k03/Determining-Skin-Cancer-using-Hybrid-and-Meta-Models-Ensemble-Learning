import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
import tensorflow as tf
from PIL import Image, ImageDraw
import os

# --- YOLOv5 Model ---
class YOLOv5(nn.Module):
    def __init__(self):
        super(YOLOv5, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 7)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.reshape(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load models
yolo_model = YOLOv5()
yolo_model.load_state_dict(torch.load('models/yolo_model.pth'))
yolo_model.eval()

cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
elastic_net_model = joblib.load('models/elastic_net_model.joblib')

hybrid_model = YOLOv5()
hybrid_model.load_state_dict(torch.load('models/hybrid_yolo_cnn_model.pth'))
hybrid_model.eval()

ensemble_model = tf.keras.models.load_model('models/ensemble_model.h5')
bnn_model = tf.keras.models.load_model('models/bnn_skin_lesion_model.h5')
dnn_model = tf.keras.models.load_model('models/dnn_skin_lesion_model.h5')
dnn_bnn_hybrid_model = tf.keras.models.load_model('models/dnn_bnn_hybrid_model.h5')
rcnn_model = tf.keras.models.load_model('models/rcnn_skin_lesion_model.h5')
ssd_model = tf.keras.models.load_model('models/ssd_skin_lesion_detection_model.h5')
hybrid_rcnn_ssd_model = tf.keras.models.load_model('models/hybrid_rcnn_ssd_skin_lesion_model.h5')

# Define prediction function
def predict_with_box(img, model_type):
    img_tensor = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2).float()
    
    if model_type == "yolo":
        output = yolo_model(img_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        x_center, y_center = 64, 64
    elif model_type == "hybrid":
        output = hybrid_model(img_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        x_center, y_center = 64, 64
    elif model_type == "cnn":
        img_array = np.expand_dims(img, axis=0) / 255.0
        output = cnn_model.predict(img_array)
        pred_class = np.argmax(output, axis=1)[0]
        x_center, y_center = 64, 64
    elif model_type == "elastic_net":
        img_flattened = img.flatten().reshape(1, -1)
        pred_class = int(np.clip(np.round(elastic_net_model.predict(img_flattened)), 0, 3)[0])
        x_center, y_center = 84, 84
    elif model_type == "ensemble":
        img_array = np.expand_dims(img, axis=0) / 255.0
        output = ensemble_model.predict(img_array)
        pred_class = np.argmax(output, axis=1)[0]
        x_center, y_center = 64, 64
    elif model_type == "bnn":
        img_array = np.expand_dims(img, axis=0) / 255.0
        output = bnn_model.predict(img_array)
        pred_class = np.argmax(output, axis=1)[0]
        x_center, y_center = 64, 64
    elif model_type == "dnn":
        img_array = np.expand_dims(img, axis=0) / 255.0
        output = dnn_model.predict(img_array)
        pred_class = np.argmax(output, axis=1)[0]
        x_center, y_center = 64, 64
    elif model_type == "dnn_bnn_hybrid":
        img_array = np.expand_dims(img, axis=0) / 255.0
        output = dnn_bnn_hybrid_model.predict(img_array)
        pred_class = np.argmax(output, axis=1)[0]
        x_center, y_center = 64, 64
    elif model_type == "rcnn":
        img_array = np.expand_dims(img, axis=0) / 255.0
        output = rcnn_model.predict(img_array)
        pred_class = np.argmax(output, axis=1)[0]
        x_center, y_center = 64, 64
    elif model_type == "ssd":
        img_array = np.expand_dims(img, axis=0) / 255.0
        output = ssd_model.predict(img_array)
        pred_class = np.argmax(output, axis=1)[0]
        x_center, y_center = 64, 64
    elif model_type == "hybrid_rcnn_ssd":
        img_array = np.expand_dims(img, axis=0) / 255.0
        output = hybrid_rcnn_ssd_model.predict(img_array)
        pred_class = np.argmax(output, axis=1)[0]
        x_center, y_center = 64, 64

    img_with_box = Image.fromarray(img)
    draw = ImageDraw.Draw(img_with_box)
    box_size = 20
    box = (x_center - box_size, y_center - box_size, x_center + box_size, y_center + box_size)
    draw.rectangle(box, outline="red", width=3)
    return img_with_box, pred_class

# Streamlit app
st.title("Skin Lesion Classification with Multiple Models")
st.write("Upload up to 10 images, and get predictions from each model with highlighted areas.")

uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
if uploaded_files:
    for file in uploaded_files[:10]:
        img = Image.open(file).convert("RGB")
        img = img.resize((128, 128))
        img_np = np.array(img)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Ensemble model predictions
        ensemble_img, ensemble_pred = predict_with_box(img_np, model_type="ensemble")
        st.image(ensemble_img, caption=f"Ensemble Prediction: {ensemble_pred}", use_container_width=True)

        # Other model predictions
        model_types = ["yolo", "cnn", "elastic_net", "hybrid", "bnn", "dnn", "dnn_bnn_hybrid", "rcnn", "ssd", "hybrid_rcnn_ssd"]
        cols = st.columns(5)
        for idx, model_type in enumerate(model_types):
            img_with_box, pred = predict_with_box(img_np, model_type=model_type)
            col = cols[idx % 5]
            with col:
                st.image(img_with_box, caption=f"{model_type.upper()} Prediction: {pred}", use_container_width=True)
