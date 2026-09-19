"""
app.py
------
FastAPI backend for the Automatic Modulation Classification (AMC) project.

Endpoints
---------
GET  /health          → confirm server + models are up
POST /predict         → unified prediction from uploaded .npy file (FormData)
POST /predict/snr     → estimate SNR bucket from a raw I/Q signal (JSON)
POST /predict/amc     → classify modulation using the appropriate AMC model (JSON)

Startup sequence
----------------
1. Load SNR estimator     (models/snr_model.keras)
2. Load High-SNR AMC      (models/high_snr_amc_model.keras)
3. Load Medium-SNR AMC    (models/medium_snr_amc_model.keras)
4. Load label encoder     (models/label_encoder.pkl)
5. Server becomes ready
"""

import io
import os
import pickle
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

try:
    from .preprocessing import (
        preprocess_for_snr,
        preprocess_for_high_snr_amc,
        preprocess_for_medium_snr_amc,
        SIGNAL_LENGTH,
    )
except ImportError:
    from preprocessing import (
        preprocess_for_snr,
        preprocess_for_high_snr_amc,
        preprocess_for_medium_snr_amc,
        SIGNAL_LENGTH,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — relative to this file, so they work anywhere
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

SNR_MODEL_PATH        = MODELS_DIR / "snr_model.keras"
HIGH_SNR_MODEL_PATH   = MODELS_DIR / "high_snr_amc_model.keras"
MED_SNR_MODEL_PATH    = MODELS_DIR / "medium_snr_amc_model.keras"
LOW_SNR_MODEL_PATH    = MODELS_DIR / "low_snr_amc_model.keras"
LABEL_ENCODER_PATH    = MODELS_DIR / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# SNR Class Labels: index 0 = low, 1 = medium, 2 = high
# ---------------------------------------------------------------------------
SNR_CLASS_LABELS = ["low", "medium", "high"]

# ---------------------------------------------------------------------------
# Global model store
# ---------------------------------------------------------------------------
models: dict = {
    "snr":        None,
    "high_amc":   None,
    "medium_amc": None,
    "low_amc":    None,
    "label_encoder": None,
    "loaded":     False,
}


# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading models...")

    # --- SNR Estimator ---
    if not SNR_MODEL_PATH.exists():
        raise FileNotFoundError(f"SNR model not found: {SNR_MODEL_PATH}")
    models["snr"] = tf.keras.models.load_model(str(SNR_MODEL_PATH))
    log.info("SNR model loaded  ✓")

    # --- High-SNR AMC ---
    if not HIGH_SNR_MODEL_PATH.exists():
        raise FileNotFoundError(f"High-SNR AMC model not found: {HIGH_SNR_MODEL_PATH}")
    models["high_amc"] = tf.keras.models.load_model(str(HIGH_SNR_MODEL_PATH))
    log.info("High-SNR AMC model loaded  ✓")

    # --- Medium-SNR AMC ---
    if not MED_SNR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Medium-SNR AMC model not found: {MED_SNR_MODEL_PATH}")
    models["medium_amc"] = tf.keras.models.load_model(str(MED_SNR_MODEL_PATH))
    log.info("Medium-SNR AMC model loaded  ✓")

    # --- Low-SNR AMC (if exists) ---
    if LOW_SNR_MODEL_PATH.exists():
        models["low_amc"] = tf.keras.models.load_model(str(LOW_SNR_MODEL_PATH))
        log.info("Low-SNR AMC model loaded  ✓")
    else:
        models["low_amc"] = None
        log.info("Low-SNR AMC model offline (not present)  ℹ")

    # --- Label Encoder ---
    if not LABEL_ENCODER_PATH.exists():
        raise FileNotFoundError(f"Label encoder not found: {LABEL_ENCODER_PATH}")
    with open(LABEL_ENCODER_PATH, "rb") as f:
        models["label_encoder"] = pickle.load(f)
    log.info("Label encoder loaded  ✓  Classes: %s", models["label_encoder"].classes_.tolist())

    models["loaded"] = True
    log.info("All models ready. Server is up.")

    yield

    log.info("Shutting down — releasing models.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AMC Backend API",
    description="Automatic Modulation Classification with SNR Estimation.",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class SignalInput(BaseModel):
    signal: list[float]

    @field_validator("signal")
    @classmethod
    def check_length(cls, v):
        expected = 2 * SIGNAL_LENGTH   # 256
        if len(v) == 0:
            raise ValueError("signal is empty.")
        if len(v) != expected:
            raise ValueError(
                f"signal must contain exactly {expected} values "
                f"({SIGNAL_LENGTH} I + {SIGNAL_LENGTH} Q). Got {len(v)}."
            )
        return v

    def to_iq_array(self) -> np.ndarray:
        flat = np.array(self.signal, dtype=np.float32)
        return flat.reshape(2, SIGNAL_LENGTH)


class SNRResponse(BaseModel):
    snr_class:       str
    snr_class_index: int
    confidence:      float


class AMCResponse(BaseModel):
    modulation:      str
    confidence:      float
    snr_class:       str
    all_probabilities: dict[str, float]


def _extract_iq_from_array(arr: np.ndarray) -> np.ndarray:
    """Normalize any loaded numpy array shape to canonical (2, 128)."""
    arr = arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape == (2, SIGNAL_LENGTH):
        return arr
    elif arr.ndim == 3 and arr.shape == (1, 2, SIGNAL_LENGTH):
        return arr[0]
    elif arr.ndim == 1 and arr.shape == (2 * SIGNAL_LENGTH,):
        return arr.reshape(2, SIGNAL_LENGTH)
    elif arr.ndim == 2 and arr.shape == (SIGNAL_LENGTH, 2):
        return arr.T
    else:
        raise ValueError(
            f"Invalid array shape {arr.shape}. Expected (2, {SIGNAL_LENGTH}) "
            f"or ({SIGNAL_LENGTH}, 2) or ({2*SIGNAL_LENGTH},)."
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "models_loaded": models["loaded"],
        "models": {
            "snr_estimator":    models["snr"]        is not None,
            "high_snr_amc":     models["high_amc"]   is not None,
            "medium_snr_amc":   models["medium_amc"] is not None,
            "low_snr_amc":      models["low_amc"]    is not None,
        },
    }


@app.post("/predict", summary="Unified prediction from uploaded .npy file")
async def predict_unified(file: UploadFile = File(...)):
    """
    Unified prediction endpoint taking a .npy file upload (multipart/form-data).
    Performs SNR estimation, SNR region classification, and AMC classification.
    """
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    if not file.filename.lower().endswith(".npy"):
        raise HTTPException(
            status_code=400,
            detail="INVALID FILE TYPE. Only .npy signal files are accepted."
        )

    try:
        content = await file.read()
        raw_array = np.load(io.BytesIO(content), allow_pickle=False)
        iq = _extract_iq_from_array(raw_array)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"INVALID NPY SIGNAL DATA: {str(e)}")

    # Step 1: SNR Estimation
    try:
        snr_x = preprocess_for_snr(iq)
        snr_probs = models["snr"].predict(snr_x, verbose=0)[0]  # [p_low, p_med, p_high]
        snr_idx = int(np.argmax(snr_probs))
        snr_label = SNR_CLASS_LABELS[snr_idx]
        snr_conf = float(snr_probs[snr_idx])
    except Exception as e:
        log.error("SNR prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="SNR estimation inference failed.")

    snr_prob_dist = {
        "HIGH":   round(float(snr_probs[2]), 4),
        "MEDIUM": round(float(snr_probs[1]), 4),
        "LOW":    round(float(snr_probs[0]), 4),
    }

    # Step 2: Route to AMC Expert Model
    amc_model = None
    preprocess_fn = None
    if snr_label == "high":
        amc_model = models["high_amc"]
        preprocess_fn = preprocess_for_high_snr_amc
    elif snr_label == "medium":
        amc_model = models["medium_amc"]
        preprocess_fn = preprocess_for_medium_snr_amc
    elif snr_label == "low":
        amc_model = models["low_amc"]
        # If low model exists, use corresponding preprocess; otherwise None
        preprocess_fn = preprocess_for_medium_snr_amc if amc_model else None

    modulation_available = (amc_model is not None)
    modulation_class = None
    modulation_confidence = None
    all_probabilities = {}
    message = None

    if modulation_available:
        try:
            amc_x = preprocess_fn(iq)
            amc_probs = amc_model.predict(amc_x, verbose=0)[0]
            class_idx = int(np.argmax(amc_probs))
            le = models["label_encoder"]
            modulation_class = str(le.inverse_transform([class_idx])[0])
            modulation_confidence = round(float(amc_probs[class_idx]), 4)
            all_probabilities = {
                str(le.inverse_transform([i])[0]): round(float(p), 4)
                for i, p in enumerate(amc_probs)
            }
        except Exception as e:
            log.error("AMC prediction failed: %s", e)
            raise HTTPException(status_code=500, detail="AMC classification failed.")
    else:
        message = f"No AMC expert model is currently available for this SNR region ({snr_label.upper()} SNR)."

    return {
        "snr_category": snr_label.upper(),
        "snr_confidence": round(snr_conf, 4),
        "snr_probabilities": snr_prob_dist,
        "modulation_available": modulation_available,
        "modulation_class": modulation_class,
        "modulation_confidence": modulation_confidence,
        "all_probabilities": all_probabilities,
        "message": message,
        "waveform": {
            "i": iq[0].tolist(),
            "q": iq[1].tolist(),
        },
    }


@app.post("/predict/snr", response_model=SNRResponse, summary="Estimate SNR from I/Q signal")
def predict_snr(body: SignalInput):
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    try:
        iq = body.to_iq_array()
        x  = preprocess_for_snr(iq)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        probs = models["snr"].predict(x, verbose=0)[0]
    except Exception as e:
        log.error("SNR model inference failed: %s", e)
        raise HTTPException(status_code=500, detail="SNR model inference error.")

    class_index = int(np.argmax(probs))
    class_label = SNR_CLASS_LABELS[class_index]
    confidence  = float(probs[class_index])

    return SNRResponse(
        snr_class=class_label,
        snr_class_index=class_index,
        confidence=round(confidence, 4),
    )


@app.post("/predict/amc", response_model=AMCResponse, summary="Classify modulation of I/Q signal")
def predict_amc(body: SignalInput):
    if not models["loaded"]:
        raise HTTPException(status_code=503, detail="Models are not loaded yet.")

    try:
        iq = body.to_iq_array()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Step 1: SNR routing
    try:
        snr_x     = preprocess_for_snr(iq)
        snr_probs = models["snr"].predict(snr_x, verbose=0)[0]
        snr_index = int(np.argmax(snr_probs))
        snr_class = SNR_CLASS_LABELS[snr_index]
    except Exception as e:
        log.error("SNR routing failed: %s", e)
        raise HTTPException(status_code=500, detail="SNR estimation step failed.")

    # Step 2: Route to AMC model
    if snr_class == "high":
        amc_model = models["high_amc"]
        preprocess_fn = preprocess_for_high_snr_amc
    elif snr_class == "medium":
        amc_model = models["medium_amc"]
        preprocess_fn = preprocess_for_medium_snr_amc
    else:
        # Fallback to medium if low-SNR model not present
        amc_model = models["low_amc"] or models["medium_amc"]
        preprocess_fn = preprocess_for_medium_snr_amc

    try:
        amc_x     = preprocess_fn(iq)
        amc_probs = amc_model.predict(amc_x, verbose=0)[0]
    except Exception as e:
        log.error("AMC model inference failed: %s", e)
        raise HTTPException(status_code=500, detail="AMC model inference error.")

    class_index  = int(np.argmax(amc_probs))
    le           = models["label_encoder"]
    modulation   = str(le.inverse_transform([class_index])[0])
    confidence   = float(amc_probs[class_index])

    all_probs = {
        str(le.inverse_transform([i])[0]): round(float(p), 6)
        for i, p in enumerate(amc_probs)
    }

    return AMCResponse(
        modulation=modulation,
        confidence=round(confidence, 4),
        snr_class=snr_class,
        all_probabilities=all_probs,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
