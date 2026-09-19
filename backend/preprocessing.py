"""
preprocessing.py
----------------
All signal preprocessing logic for the AMC backend.

IMPORTANT: This file reproduces the EXACT same preprocessing that was used
during Colab training/testing. Nothing has been added or changed.

Pipeline overview
-----------------
SNR Estimator  → preprocess_for_snr(iq_array)
    Input:  raw I/Q array of shape (2, 128)  – row 0 = I, row 1 = Q
    Output: numpy array of shape (1, 2, 128, 1)   [batch, channels, time, 1]

High-SNR AMC   → preprocess_for_high_snr_amc(iq_array)
    Input:  raw I/Q array of shape (2, 128)
    Output: numpy array of shape (1, 2, 128, 1)

Medium-SNR AMC → preprocess_for_medium_snr_amc(iq_array)
    Input:  raw I/Q array of shape (2, 128)
    Output: numpy array of shape (1, 4, 128, 1)
    Extra:  phase (unwrapped, normalised to [-1, 1]) and
            magnitude (normalised per sample) are computed from I/Q
            using add_phase_and_magnitude() — identical to feature_engineering.py.

Signal length: 128 samples (per I or Q channel).
Data type    : float32 throughout.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helper – identical to feature_engineering.py / add_phase_and_magnitude
# ---------------------------------------------------------------------------

def _add_phase_and_magnitude(X: np.ndarray) -> np.ndarray:
    """
    Extends a raw I/Q tensor with two derived channels: phase and magnitude.

    Parameters
    ----------
    X : np.ndarray, shape (N, 2, 128, 1)
        Batch of raw I/Q signals.
        X[:, 0, :, 0] = I channel
        X[:, 1, :, 0] = Q channel

    Returns
    -------
    np.ndarray, shape (N, 4, 128, 1)
        Four channels per sample: [I, Q, phase_norm, magnitude_norm]

    Why each step:
    - arctan2(Q, I)   : instantaneous phase in radians [-π, π]
    - unwrap(phase)   : removes 2π discontinuities so the phase is continuous
    - / π             : normalises phase to [-1, 1] range
    - magnitude       : envelope of the signal  √(I² + Q²)
    - / (max + 1e-8)  : per-sample normalisation; 1e-8 prevents division by zero
    """
    I = X[:, 0, :, 0]   # shape (N, 128)
    Q = X[:, 1, :, 0]   # shape (N, 128)

    # Instantaneous phase, unwrapped and normalised
    phase = np.arctan2(Q, I)                               # (N, 128)
    phase_unwrapped = np.unwrap(phase, axis=-1)            # (N, 128)
    phase_norm = phase_unwrapped / np.pi                   # (N, 128)  → [-1, 1]

    # Signal envelope, normalised per sample
    magnitude = np.sqrt(I ** 2 + Q ** 2)                  # (N, 128)
    magnitude = magnitude / (np.max(magnitude, axis=-1, keepdims=True) + 1e-8)

    # Stack all four channels along axis=1: [I, Q, phase, magnitude]
    stacked = np.stack([I, Q, phase_norm, magnitude], axis=1)  # (N, 4, 128)

    # Add the trailing singleton dimension required by Conv2D (channels_last → here "1")
    return stacked[..., np.newaxis].astype(np.float32)     # (N, 4, 128, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SIGNAL_LENGTH = 128   # number of I/Q samples expected by every model


def _validate_iq(iq_array: np.ndarray) -> None:
    """
    Raises ValueError with a clear message if the I/Q array is malformed.
    Called by every preprocess_* function so validation is centralised.
    """
    if iq_array.ndim != 2:
        raise ValueError(
            f"Expected a 2-D array of shape (2, {SIGNAL_LENGTH}), "
            f"got shape {iq_array.shape}."
        )
    if iq_array.shape[0] != 2:
        raise ValueError(
            f"First dimension must be 2 (I and Q channels), "
            f"got {iq_array.shape[0]}."
        )
    if iq_array.shape[1] != SIGNAL_LENGTH:
        raise ValueError(
            f"Signal length must be {SIGNAL_LENGTH} samples, "
            f"got {iq_array.shape[1]}."
        )


def preprocess_for_snr(iq_array: np.ndarray) -> np.ndarray:
    """
    Prepare an I/Q signal for the SNR Estimator model.

    The SNR model expects raw I/Q — no additional feature engineering.

    Parameters
    ----------
    iq_array : np.ndarray, shape (2, 128)

    Returns
    -------
    np.ndarray, shape (1, 2, 128, 1), dtype float32
        Ready to pass directly to snr_model.predict().
    """
    _validate_iq(iq_array)

    # Shape: (2, 128) → (1, 2, 128) → (1, 2, 128, 1)
    x = iq_array.astype(np.float32)
    x = x[np.newaxis, ...]          # add batch dimension
    x = x[..., np.newaxis]          # add trailing "image" dimension
    return x                         # (1, 2, 128, 1)


def preprocess_for_high_snr_amc(iq_array: np.ndarray) -> np.ndarray:
    """
    Prepare an I/Q signal for the High-SNR AMC model.

    The high-SNR model also uses raw I/Q only (same shape as SNR model).

    Parameters
    ----------
    iq_array : np.ndarray, shape (2, 128)

    Returns
    -------
    np.ndarray, shape (1, 2, 128, 1), dtype float32
    """
    _validate_iq(iq_array)

    x = iq_array.astype(np.float32)
    x = x[np.newaxis, ...]
    x = x[..., np.newaxis]
    return x                         # (1, 2, 128, 1)


def preprocess_for_medium_snr_amc(iq_array: np.ndarray) -> np.ndarray:
    """
    Prepare an I/Q signal for the Medium-SNR AMC model.

    The medium-SNR model requires FOUR channels:
        [I, Q, unwrapped_phase_normalised, magnitude_normalised]

    This is computed by _add_phase_and_magnitude(), which is the exact
    same function as add_phase_and_magnitude() in feature_engineering.py.

    Parameters
    ----------
    iq_array : np.ndarray, shape (2, 128)

    Returns
    -------
    np.ndarray, shape (1, 4, 128, 1), dtype float32
    """
    _validate_iq(iq_array)

    # First convert to (1, 2, 128, 1) so _add_phase_and_magnitude can index it
    x = iq_array.astype(np.float32)
    x = x[np.newaxis, ..., np.newaxis]    # (1, 2, 128, 1)

    # Add phase + magnitude → (1, 4, 128, 1)
    return _add_phase_and_magnitude(x)
