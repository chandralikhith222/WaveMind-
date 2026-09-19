# SNR-Aware Automatic Modulation Classification

A deep-learning based signal intelligence system for **SNR estimation, SNR-region identification, and Automatic Modulation Classification (AMC)** using I/Q signal data.

The system accepts `.npy` signal files through a web interface, sends them to a FastAPI backend, performs inference using trained TensorFlow/Keras models, and displays the prediction results along with signal visualizations.

---

## Project Overview

The system follows this pipeline:

```text
                .NPY I/Q SIGNAL
                       │
                       ▼
              ┌─────────────────┐
              │    FRONTEND     │
              │ HTML/CSS/JS     │
              └────────┬────────┘
                       │
                       │ HTTP Request
                       ▼
              ┌─────────────────┐
              │     FASTAPI     │
              │     BACKEND     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  SNR ESTIMATOR  │
              │  Keras Model    │
              └────────┬────────┘
                       │
                       ▼
                  SNR REGION
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        LOW          MEDIUM        HIGH
          │            │            │
          ▼            ▼            ▼
       Low-SNR      Medium-SNR    High-SNR
       AMC Model    AMC Model     AMC Model
          │            │            │
          └────────────┼────────────┘
                       ▼
             MODULATION CLASS
                       │
                       ▼
                  CONFIDENCE
                       │
                       ▼
                 JSON RESPONSE
                       │
                       ▼
                  FRONTEND
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        SNR         CLASS       WAVEFORM
      RESULTS      RESULTS    VISUALIZATION
