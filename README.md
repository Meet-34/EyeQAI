# DrowsAI — Real-Time Drowsiness & Attention Detection System

A production-ready, real-time driver/user monitoring system that detects three states:
- **Focused** — alert, eyes open, head forward
- **Drowsy** — eyes closed, EAR below threshold
- **Inattentive** — head turned significantly (yaw angle exceeded)

---

## Architecture Overview

### Hybrid AI Approach

This system uses **three complementary signals** fused together for robust, low-false-positive detection:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  CNN Model   │     │    EAR       │     │   Head Pose      │
│ (ResNet50)   │  +  │  (geometry)  │  +  │   (yaw angle)    │
│ eye_state    │     │ < 0.22?      │     │  > 25°?          │
└──────────────┘     └──────────────┘     └──────────────────┘
         │                  │                       │
         └──────────────────┼───────────────────────┘
                            ▼
                   ┌─────────────────┐
                   │  Frame Smoother │ ← N consecutive frames
                   │  (N=3 frames)   │   must agree
                   └────────┬────────┘
                            ▼
               FOCUSED / DROWSY / INATTENTIVE
```

**Why hybrid?**
- CNN alone can misfire on partial occlusion or lighting changes
- EAR alone misses slow-closure patterns
- Combined: CNN confirms eye state, EAR validates geometry, yaw catches head-turn independently

**Frame Smoothing:**
- Status only changes after 3 consecutive frames agree → eliminates single-frame spikes

---

## Dataset

### Recommended: Driver Drowsiness Dataset (Kaggle)

1. Download from Kaggle:
   ```
   https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd
   ```
   or
   ```
   https://www.kaggle.com/datasets/rakibuleceruet/drowsiness-prediction-dataset
   ```

2. Unzip and organize as:
   ```
   backend/data/drowsiness_dataset/
   ├── open/          ← open-eye images
   └── closed/        ← closed-eye images
   ```

### For Inattentive Detection
- Head pose is computed live via MediaPipe FaceMesh (no separate dataset needed)
- Yaw angle is estimated using PnP solver on 6 canonical 3D face points

---

## Outlier Handling

The `clean_dataset()` function in `train_model.py` filters:

| Filter | Method | Threshold |
|--------|--------|-----------|
| Blurry images | Laplacian variance | `< 80.0` |
| Unreadable files | OpenCV decode check | Always removed |
| Multi-face images | Haar cascade (optional) | `!= 1 face` |

Uncomment the face-count check in `train_model.py` for stricter cleaning (slower).

---

## Setup

### 1. Python Environment

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download & Prepare Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (requires Kaggle API key at ~/.kaggle/kaggle.json)
kaggle datasets download ismailnasri20/driver-drowsiness-dataset-ddd
unzip driver-drowsiness-dataset-ddd.zip -d data/drowsiness_dataset
```

### 3. Train the Model

```bash
cd backend
python train_model.py
```

**Expected output:**
```
=== Step 1: Cleaning Dataset ===
[CLEAN] Total: 41793 | Kept: 38200 | Blur removed: 3593

=== Step 2: Building DataLoaders ===
[DATA] Classes: ['closed', 'open']
[DATA] Train: 30560 | Val: 7640

=== Step 3: Building ResNet50 Model ===
[MODEL] Trainable parameters: 26,621,954

=== Step 4: Training ===
Epoch [01/30] Train Loss: 0.4821  Acc: 0.7634 | Val Loss: 0.3108  Acc: 0.8892
Epoch [02/30] Train Loss: 0.2891  Acc: 0.8904 | Val Loss: 0.2014  Acc: 0.9231
...
[DONE] Best Validation Accuracy: 0.9487
```

**Expected accuracy: 85–95% on validation set**

### 4. Run the Application

```bash
cd backend
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Project Structure

```
drowsiness-ai/
├── backend/
│   ├── models/
│   │   └── drowsy_model.pth      ← Generated after training
│   ├── data/
│   │   ├── drowsiness_dataset/   ← Place raw Kaggle dataset here
│   │   └── cleaned/              ← Auto-generated after training
│   ├── train_model.py            ← Full training pipeline
│   ├── detection.py              ← Hybrid AI detection engine
│   ├── app.py                    ← Flask backend
│   ├── utils.py                  ← Shared utilities
│   └── requirements.txt
│
├── frontend/
│   ├── index.html                ← Live Monitor page
│   ├── dashboard.html            ← Analytics Dashboard
│   ├── index.css                 ← Monitor styles
│   ├── dashboard.css             ← Dashboard styles
│   ├── script.js                 ← Monitor logic
│   ├── dashboard.js              ← Dashboard charts & data
│   └── static/                   ← Additional static assets
│
└── README.md
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live Monitor page |
| `/dashboard` | GET | Analytics Dashboard |
| `/video_feed` | GET | MJPEG webcam stream |
| `/metrics` | GET | JSON: current frame metrics |
| `/logs` | GET | JSON: event history (up to 200) |
| `/summary` | GET | JSON: session aggregates |
| `/health` | GET | JSON: system health |

### `/metrics` Response Example

```json
{
  "status":          "FOCUSED",
  "ear":             0.312,
  "mar":             0.041,
  "yaw":             3.2,
  "eye_state":       "open",
  "eye_conf":        0.947,
  "attention_score": 94.5,
  "face_detected":   true,
  "fps":             28.4,
  "timestamp":       1706123456.789
}
```

---

## Accuracy Improvement Techniques Used

| Technique | Implementation |
|-----------|---------------|
| Transfer Learning | ResNet50 with ImageNet weights (frozen early layers) |
| Label Smoothing | `CrossEntropyLoss(label_smoothing=0.1)` |
| Class Balancing | `WeightedRandomSampler` based on class frequencies |
| Augmentation | Flip, rotation, brightness, color jitter |
| Cosine LR Schedule | `CosineAnnealingLR` for smooth decay |
| Early Stopping | Stops after 7 epochs of no val-acc improvement |
| Dropout | 0.4 + 0.3 in classification head |
| Fine-tuning | Only layer3, layer4, fc unfrozen |

---

## Thresholds (Tunable)

Edit in `backend/detection.py`:

```python
EAR_THRESHOLD  = 0.22   # Eye aspect ratio: below = closed
YAW_THRESHOLD  = 25.0   # Head yaw degrees: beyond = inattentive
MAR_THRESHOLD  = 0.65   # Mouth aspect ratio: above = yawning
CONSEC_FRAMES  = 3      # Frames to confirm state change
```

---

## System Requirements

- Python 3.10+
- Webcam
- 4GB RAM minimum (8GB recommended for training)
- GPU optional (CUDA accelerates training ~10x)
- Modern browser (Chrome/Firefox/Edge)

---

## Troubleshooting

**"Model not found" warning on startup:**
> Run `python train_model.py` first. The system will still work in EAR-only mode.

**Low FPS:**
> Reduce `max_width` in `utils.resize_frame()` to 480 or 320.

**False positives in dark environments:**
> Lower `EAR_THRESHOLD` slightly (e.g. 0.18) and increase `CONSEC_FRAMES` to 5.

**MediaPipe not installing:**
> Use Python 3.10 or 3.11. MediaPipe has limited support for Python 3.12+.
