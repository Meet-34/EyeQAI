# 👁️ EyeQAI — Real-Time Drowsiness & Attention Detection System

EyeQAI is a real-time AI-based monitoring system that detects user alertness using computer vision and deep learning techniques.

It classifies user state into:

* **Focused** — eyes open, head forward
* **Drowsy** — prolonged eye closure
* **Inattentive** — head turned away

---

## 🚀 Key Features

* Hybrid AI detection (CNN + EAR + Head Pose)
* Real-time webcam monitoring
* Frame smoothing to reduce false positives
* Attention score calculation
* REST API for metrics and logs

---

## 🧠 Architecture Overview

EyeQAI combines three signals for robust detection:

* **CNN Model (ResNet50)** → Eye state classification
* **EAR (Eye Aspect Ratio)** → Detects eye closure
* **Head Pose (Yaw Angle)** → Detects distraction

Final output is stabilized using **frame smoothing (3 consecutive frames)** to avoid sudden spikes.

---

## 📂 Project Structure

```bash
EyeQAI/
└── app/
    ├── app.py
    ├── detection.py
    ├── train_model.py
    ├── utils.py
    ├── requirements.txt
    ├── templates/
    ├── static/
```

⚠️ Note:

* `models/`, `data/`, and `venv/` are **excluded from GitHub** due to size limitations.

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment

```bash
cd app
python -m venv venv
venv\Scripts\activate     # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Train the Model

```bash
python train_model.py
```

This will generate the trained model inside the `models/` folder.

---

### 3. Run the Application

```bash
python app.py
```

Open in browser:
👉 http://localhost:5000

---

## 🌐 Deployment (Render)

* **Root Directory** → `app`
* **Build Command**

```bash
pip install -r requirements.txt
```

* **Start Command**

```bash
python app.py
```

⚠️ Render automatically manages the virtual environment. No need to activate `venv`.

---

## 📊 API Endpoints

| Endpoint      | Description              |
| ------------- | ------------------------ |
| `/video_feed` | Live webcam stream       |
| `/metrics`    | Real-time detection data |
| `/logs`       | Event history            |
| `/summary`    | Session summary          |
| `/health`     | System health            |

---

## 🎯 Detection Logic

| Condition  | Output      |
| ---------- | ----------- |
| EAR < 0.22 | Drowsy      |
| Yaw > 25°  | Inattentive |
| Otherwise  | Focused     |

Frame smoothing ensures stable predictions across multiple frames.

---

## ⚠️ Important Notes

* Large files (models, datasets) are excluded using `.gitignore`
* Model can be stored externally (Google Drive / cloud)
* System can run in EAR-only mode if model is not available

---

## 🛠️ Tech Stack

* Python
* OpenCV
* MediaPipe
* PyTorch / TensorFlow
* Flask

---

## 📌 Future Improvements

* Cloud-based model loading
* Multi-user analytics dashboard
* Mobile deployment
