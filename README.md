# 😷 FaceGuard — AI-Based Face Mask Detection System

<p align="center">
  <b>Real-Time Face Mask Detection Using Deep Learning</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.16.2-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv" />
  <img src="https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Model-MobileNetV2-blueviolet" />
</p>

## 📌 Overview

**FaceGuard** is an AI-powered Face Mask Detection System developed using Deep Learning and Computer Vision.

The system identifies whether a person is wearing a face mask or not. It supports image-based detection and real-time webcam detection through an interactive Streamlit web application.

The project uses a MobileNetV2-based transfer learning model to classify faces into two categories: **Mask** and **No Mask**.

## ✨ Key Features

- 📷 Image Upload – Detect masks from uploaded images.
- 🎥 Live Webcam Detection – Real-time mask detection using a webcam.
- 👥 Multiple Face Detection – Detect multiple faces in a single image.
- 🧠 AI-Based Classification – Predict Mask or No Mask.
- 📊 Prediction Confidence – Display confidence scores for predictions.
- 🎨 Interactive UI – Clean and professional Streamlit interface.
- ⚡ Fast Processing – MobileNetV2-based image classification.

## 🖥️ Application Screenshots

### 1. FaceGuard Dashboard
![Dashboard](screenshots/dashboard.png)

### 2. Image-Based Detection
![Image Detection](screenshots/image-detection.png)

### 3. Multiple Face Detection
![Multiple Face Detection](screenshots/multiple-faces.png)

### 4. Live Webcam Detection
![Live Webcam](screenshots/webcam-detection.png)

*Note: Add your actual screenshots to the `screenshots` folder using the filenames above.*

## 🎬 Screen Recording Demo

A demonstration video showcasing the complete FaceGuard application, including image upload, face detection, prediction results, and real-time webcam functionality.

**Demo Video:** Add your YouTube or Google Drive video link here.

## 🧠 Model Architecture

FaceGuard uses **MobileNetV2** with Transfer Learning.

- Base Model: MobileNetV2
- Input Image Size: 224 × 224 × 3
- Preprocessing: MobileNetV2 preprocess_input
- Feature Extraction: Global Average Pooling
- Hidden Layer: Dense (128 neurons, ReLU)
- Regularization: Dropout (0.5)
- Output Layer: Dense (2 neurons, Softmax)
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

### Prediction Classes

| Class | Description |
|---|---|
| Mask | Person wearing a face mask |
| No Mask | Person not wearing a face mask |

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| TensorFlow & Keras | Deep Learning |
| MobileNetV2 | Transfer Learning |
| OpenCV | Face Detection |
| Streamlit | Web Application |
| Streamlit-WebRTC | Live Webcam Streaming |
| NumPy | Numerical Operations |
| Pandas | Data Handling |

## 📂 Project Structure

```text
FaceGuard/
│
├── app.py
├── detect.py
├── train.py
├── mask_detector.h5
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
└── screenshots/
    ├── dashboard.png
    ├── image-detection.png
    ├── multiple-faces.png
    └── webcam-detection.png
```

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to Project Directory

```bash
cd FaceGuard
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## 💻 System Requirements

- Python 3.11
- Windows / Linux / macOS
- Minimum 4 GB RAM recommended
- Webcam for live detection
- Internet connection for initial dependency installation

## 📊 Project Highlights

- Implemented a Deep Learning model using Transfer Learning.
- Integrated OpenCV for face detection.
- Developed an interactive Streamlit application.
- Implemented image upload and real-time webcam detection.
- Added prediction confidence visualization.

## 🔮 Future Enhancements

- Improve detection under different lighting conditions.
- Deploy the application on a cloud platform.
- Add performance evaluation metrics.
- Improve face detection for different angles and distances.

## 👨‍💻 Author

**Ganesh Gokhale**

Aspiring Data Scientist | Machine Learning Enthusiast

---

<p align="center">
  <b>FaceGuard — Smart AI-Based Face Mask Detection</b><br/>
  Built with Python, Deep Learning & Computer Vision.
</p>
