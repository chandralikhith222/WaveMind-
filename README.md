# AMC Backend — README

## What this backend does

This is the FastAPI backend for a **Deep Learning-based Automatic Modulation Classification (AMC)** system with integrated **SNR Estimation**.

Given a raw I/Q signal (128 samples per channel), the backend:
1. Estimates the SNR level (low / medium / high) using a trained Keras model.
2. Routes the signal to the appropriate AMC model based on the SNR.
3. Returns the modulation class (e.g. QPSK, QAM16) and prediction confidence.

---

## Requirements

- **Python**: 3.10, 3.11, or 3.12 (recommended: 3.11)
- **OS**: Windows, macOS, or Linux

---

## Environment setup

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Model files

Place the trained `.keras` models and label encoder in `backend/models/`:

```
backend/
└── models/
    ├── snr_model.keras            ← SNR Estimator
    ├── high_snr_amc_model.keras   ← High-SNR AMC classifier
    ├── medium_snr_amc_model.keras ← Medium-SNR AMC classifier
    └── label_encoder.pkl          ← scikit-learn LabelEncoder
```

The backend will fail to start if any of these files are missing.

---

## Starting the backend

```bash
# From inside the backend/ directory
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Or, without live-reload (recommended for production):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Swagger / API documentation

Once running, open your browser and go to:

```
http://127.0.0.1:8000/docs
```

You can test all three endpoints interactively from the Swagger UI.

---

## Available endpoints

### `GET /health`

Confirms the server is running and all models are loaded.

**Response:**
```json
{
  "status": "ok",
  "models_loaded": true,
  "models": {
    "snr_estimator": true,
    "high_snr_amc": true,
    "medium_snr_amc": true
  }
}
```

---

### `POST /predict/snr`

Estimates the SNR category of an I/Q signal.

**Request body:**
```json
{
  "signal": [/* 256 floats: first 128 = I channel, next 128 = Q channel */]
}
```

**Response:**
```json
{
  "snr_class": "high",
  "snr_class_index": 2,
  "confidence": 0.973
}
```

SNR class index mapping: `0 = low`, `1 = medium`, `2 = high`

---

### `POST /predict/amc`

Classifies the modulation scheme of an I/Q signal.

**Request body:**
```json
{
  "signal": [/* 256 floats: first 128 = I, next 128 = Q */]
}
```

**Response:**
```json
{
  "modulation": "QPSK",
  "confidence": 0.9411,
  "snr_class": "high",
  "all_probabilities": {
    "8PSK": 0.001,
    "AM-DSB": 0.0,
    "BPSK": 0.003,
    "CPFSK": 0.0,
    "GFSK": 0.0,
    "PAM4": 0.001,
    "QAM16": 0.002,
    "QAM64": 0.003,
    "QPSK": 0.9411,
    "WBFM": 0.0
  }
}
```

**Supported modulation classes:**
`8PSK`, `AM-DSB`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `QAM16`, `QAM64`, `QPSK`, `WBFM`

---

## Input format (signal)

- The `signal` field is a **flat list of 256 floats**.
- The first 128 values are the **I (in-phase)** samples.
- The next 128 values are the **Q (quadrature)** samples.
- Values are raw I/Q — **do not normalize** before sending; the models handle this internally.
- This corresponds to a `(2, 128)` array flattened row-major (C order).

**Python example:**
```python
import numpy as np
import requests

# iq_signal has shape (2, 128)
iq_signal = np.random.randn(2, 128).astype(np.float32)
payload = {"signal": iq_signal.flatten().tolist()}

response = requests.post("http://127.0.0.1:8000/predict/amc", json=payload)
print(response.json())
```

---

## SNR routing logic

```
Signal (256 floats)
       ↓
  SNR Estimator
       ↓
  SNR bucket (low / medium / high)
       ↓
 ┌─────┴──────┬──────────────┐
 ↓            ↓              ↓
low        medium           high
 ↓            ↓              ↓
(medium    Medium-SNR     High-SNR
 fallback)  AMC Model     AMC Model
 └─────┬──────┘──────────────┘
       ↓
  Modulation + Confidence
```

> **Note:** Until a dedicated Low-SNR model is provided, low-SNR signals fall back to the Medium-SNR AMC model.

---

## How to add future low / medium SNR models

1. Place your new model in `backend/models/`:
   - `low_snr_amc_model.keras`

2. In `app.py`, uncomment:
   ```python
   LOW_SNR_MODEL_PATH = MODELS_DIR / "low_snr_amc_model.keras"
   ```
   and the corresponding block in the `lifespan()` function:
   ```python
   models["low_amc"] = tf.keras.models.load_model(str(LOW_SNR_MODEL_PATH))
   ```

3. In `_route_amc_model()`, update the `"low"` branch:
   ```python
   elif snr_class == "low":
       return models["low_amc"], preprocess_for_low_snr_amc
   ```

4. Add the corresponding `preprocess_for_low_snr_amc()` function in `preprocessing.py` if the new model uses different features.

---

## Frontend connection (future)

The backend exposes clean JSON APIs that a frontend can call via HTTP.
The expected architecture:

```
Browser (React / plain HTML+JS)
        |
        |  POST /predict/amc  { signal: [...256 floats...] }
        ↓
FastAPI Backend (this repo)
        ↓
    Preprocessing
        ↓
    Keras Model
        ↓
    JSON response  →  back to browser
```

CORS is already configured. For production, set the `ALLOWED_ORIGINS` environment variable:

```bash
ALLOWED_ORIGINS=https://your-frontend.vercel.app uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Production start command

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

The `PORT` environment variable is respected automatically (defaults to 8000 if not set).

---

## Project structure

```
backend/
├── app.py             ← FastAPI application, endpoints, model loading
├── preprocessing.py   ← Signal preprocessing (mirrors Colab pipeline exactly)
├── requirements.txt   ← Python dependencies
├── README.md          ← This file
└── models/
    ├── snr_model.keras
    ├── high_snr_amc_model.keras
    ├── medium_snr_amc_model.keras
    └── label_encoder.pkl
```
