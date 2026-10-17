# UrbanPulse — AI-Based Crowd Safety and Abnormal Movement Detection

UrbanPulse is a computer vision and deep learning prototype designed to monitor crowd safety. It tracks person movements using YOLO11n and ByteTrack, extracts robust frame-level movement features, classifies temporal windows using an LSTM sequence model, and emits debounced crowd alerts.

> **Research Prototype Disclaimer**: This project detects observable macroscopic crowd movement patterns (e.g. sudden dispersal, rapid directional divergence). It does **not** detect mental states, panic, or intent, and is not a substitute for certified life-safety or emergency systems.

---

## Architecture Overview

```text
Video / CCTV Footage
        │
        ▼
YOLO11n Person Detection (class 0)
        │
        ▼
ByteTrack Tracking (per-person IDs & trajectories)
        │
        ▼
Frame-Level Crowd Feature Aggregation
(person count, median speed, IQR dispersion, flow consistency via sin/cos)
        │
        ▼
Fixed Temporal Windows (30 frames / 1s, stride 5)
        │
        ▼
Compact LSTM Binary Classifier
(LSTM(32) → Dense(16, ReLU) → Dropout(0.2) → Sigmoid)
        │
        ▼
Temporal Alert Smoothing (Debounce: 3-window persistence / clear)
        │
        ▼
Streamlit Monitoring Dashboard
```

---

## Project Structure

```text
UrbanPulse/
├── app/
│   └── app.py                  # Streamlit monitoring dashboard
├── data/
│   ├── README.md               # Dataset download instructions
│   └── umn_annotations.csv     # Frozen UMN clip annotations (v1.0)
├── models/
│   └── README.md               # Model artifact specifications
├── notebooks/
│   └── README.md               # Notebook pipeline mapping
├── src/
│   ├── __init__.py
│   ├── alert.py                # Temporal debounce / persistence state machine
│   ├── annotations.py          # Ground truth clip boundaries & onsets
│   ├── evaluate.py             # Validation threshold selection & test metrics
│   ├── inference.py            # Sliding-window inference runner
│   ├── sequences.py            # Leakage-safe sequence windowing & clip splits
│   ├── train.py                # LSTM model definition & training routine
│   └── video_features.py       # YOLO+ByteTrack crowd feature extraction
├── tests/
│   ├── test_alert_logic.py     # Debounce unit tests
│   ├── test_annotations.py     # Annotation coverage & boundary tests
│   └── test_sequences.py       # Windowing & leakage tests
├── requirements.txt
└── .gitignore
```

---

## Dataset: UMN Unusual Crowd Activity

- **Source**: [UMN Unusual Crowd Activity](https://www.crcv.ucf.edu/projects/Abnormal_Crowd/) (11 clips across 3 scenes: Lawn-1, Indoor-2, Plaza-3).
- **Total Frames**: 7,739 frames @ 30 FPS.
- **Leakage Prevention**: Evaluated strictly using clip-held-out splits:
  - **Train**: Clips `[1, 3, 5, 7, 8, 11]`
  - **Validation**: Clips `[4, 9]`
  - **Test**: Clips `[2, 6, 10]`
- **Unresolved Clips**: Clips 5 & 6 (Indoor-2) have visually ambiguous onsets and are flagged as `unresolved` (excluded from supervised ground truth to avoid noise).

---

## Getting Started

### 1. Installation
Install core requirements:
```bash
pip install -r requirements.txt
```

### 2. Running Unit Tests
Validate all modules (annotations, sequence windowing, alert logic):
```bash
python -m pytest tests
```

### 3. Launching the Dashboard
Launch the interactive Streamlit monitoring interface:
```bash
streamlit run app/app.py
```
*Note: If Streamlit is not yet installed, the dashboard script safely informs the user.*

---

## GPU Training & Heavy Extraction (Google Colab)

Feature extraction with YOLO11n + ByteTrack over high-resolution video is computationally intensive:
1. Run heavy extraction and model training on Google Colab GPU using the notebook workflow (`../Untitled0.ipynb`).
2. Train via `python -m src.train` to generate `urbanpulse_lstm.keras` and `urbanpulse_scaler.joblib`.
3. Transfer the lightweight trained artifacts into `models/` for low-latency CPU inference on laptops.
