# 🩺 Skin Cancer Detection  
*Leveraging ensemble learning with hybrid and meta models for improved lesion prediction*

[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-ff4b4b)](https://streamlit.io/)  
[![Dataset: HAM10000](https://img.shields.io/badge/Dataset-HAM10000-orange)](https://dataverse.harvard.edu)  
[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)  
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-TensorFlow%2FKeras-green)](https://www.tensorflow.org/)

---

## 📜 Table of Contents

- [Overview](#overview)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
- [Algorithms & Features](#algorithms--features)  
- [Project Structure](#project-structure)  
- [Screenshots](#screenshots)  
- [Contributing](#contributing)  


---

## 📖 Overview  
The **Skin Cancer Detection** project applies **Hybrid** and **Meta‑Models** within an ensemble framework to classify skin lesions from dermatoscopic images.  
It combines deep learning architectures and traditional ML methods for enhanced predictive performance.

**Key Features:**
- 📊 Multi‑model ensemble with Hybrid and Meta models.
- 🖼️ Works with **HAM10000** dataset for real-world skin lesion analysis.
- 💻 Interactive **Streamlit** web application for live predictions.
- 🔍 Comparative evaluation of multiple ML/DL algorithms.

Repository: [GitHub Link](https://github.com/Aditya-hub2k03/Determining-Skin-Cancer-using-Hybrid-and-Meta-Models-Ensemble-Learning)

---

## 🚀 Getting Started

### ✅ Prerequisites
- **Python 3.x**
- **Streamlit**
- **Pip**

### 📦 Installation
```bash
# Clone the repository
git clone https://github.com/Aditya-hub2k03/Determining-Skin-Cancer-using-Hybrid-and-Meta-Models-Ensemble-Learning.git
cd Determining-Skin-Cancer-using-Hybrid-and-Meta-Models-Ensemble-Learning

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠 Usage
1. Run the **Jupyter Notebook** (`Main.ipynb`) to train and test models.
2. Start the **Streamlit** UI:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Upload an image → Get classification results instantly.

---

## 🤖 Algorithms & Features  
- **YOLOv5**
- **CNN (Convolutional Neural Network)**
- **Elastic Net**
- **R‑CNN**
- **SSD (Single Shot Detector)**
- **DNN (Deep Neural Network)**
- **BNN (Bayesian Neural Network)**
- **Hybrid Models**: Yolov5‑CNN, RNN‑SSD, DNN‑BNN
- **Meta Model**: Combines hybrid models for final decision making

---

## 📂 Project Structure
```
├── Main.ipynb           # Model training and evaluation
├── streamlit_app.py     # Streamlit UI
├── requirements.txt     # Dependencies
└── Dataset/             # Image data and metadata
```

---

## 📷 Screenshots  

### 🔹 Streamlit App Interface  
![Streamlit UI](screenshots/streamlit_ui.png)  

### 🔹 Sample Prediction  
![Prediction Example](screenshots/prediction_sample.png)  




---

## 🤝 Contributing
Contributions are welcome!  
- Open an **Issue**
- Submit a **Pull Request**

---

