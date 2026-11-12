"""pose section: K (3,3) intrinsics and T (4,4) c2w. A (3,3)/(4,4) input
broadcasts to every frame; stored raw (no gzip — measured BIGGER with gzip
for short sequences: break-even ~106 frames for K, ~64 for T)."""
import numpy as np


def broadcast_matrix(m, T, shape):
    """(shape) broadcast to (T, *shape) float32, or (T, *shape) stored as-is
    (K and T alike)."""
    a = np.asarray(m, dtype=np.float32)
    if a.shape == shape:
        return np.broadcast_to(a, (T, *shape)).copy()
    if a.shape != (T, *shape):
        raise ValueError(f"matrix must be {shape} or (T, *{shape}), got {a.shape}")
    return a.copy()
