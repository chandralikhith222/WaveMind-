
import numpy as np

def add_phase_and_magnitude(X):
    I = X[:, 0, :, 0]
    Q = X[:, 1, :, 0]

    phase = np.arctan2(Q, I)
    phase_unwrapped = np.unwrap(phase, axis=-1)
    phase_norm = phase_unwrapped / np.pi

    magnitude = np.sqrt(I ** 2 + Q ** 2)
    magnitude = magnitude / (np.max(magnitude, axis=-1, keepdims=True) + 1e-8)

    stacked = np.stack([I, Q, phase_norm, magnitude], axis=1)
    return stacked[..., np.newaxis].astype(np.float32)
