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


# ---- convention checks (eval request E3; the convention itself is documented
# in docs/schema.md: OpenCV pinhole, pixel (u, v) centre at (u, v)) ----
def check_K(K):
    """(T, 3, 3) (or (3, 3)) OpenCV PINHOLE intrinsics: fx, fy > 0,
    K[2,2] == 1 (within 1e-6), zero skew (K[1,0]) and zero K[2,1] / K[2,0].
    Returns K unchanged."""
    a = np.asarray(K, dtype=np.float64)
    single = a.ndim == 2
    m = a[None] if single else a
    for i in range(m.shape[0]):
        fx, fy = m[i, 0, 0], m[i, 1, 1]
        if not (fx > 0 and fy > 0):
            raise ValueError(f"K frame {i}: fx / fy must be > 0, got {fx} / {fy}")
        if abs(m[i, 2, 2] - 1.0) > 1e-6:
            raise ValueError(f"K frame {i}: K[2,2] must be 1 (pinhole), "
                             f"got {m[i, 2, 2]}")
        if abs(m[i, 1, 0]) > 1e-6:
            raise ValueError(f"K frame {i}: K[1,0] (skew) must be 0, "
                             f"got {m[i, 1, 0]}")
        if abs(m[i, 2, 1]) > 1e-6:
            raise ValueError(f"K frame {i}: K[2,1] must be 0, got {m[i, 2, 1]}")
        if abs(m[i, 2, 0]) > 1e-6:
            raise ValueError(f"K frame {i}: K[2,0] must be 0, got {m[i, 2, 0]}")
    return m[0] if single else a


def check_pose(pose_c2w):
    """(T, 4, 4) (or (4, 4)) camera-to-world poses: last row [0, 0, 0, 1] and
    an orthonormal R with |det(R) - 1| <= 1e-3 (1e-6 on the rows). Returns
    the input unchanged."""
    a = np.asarray(pose_c2w, dtype=np.float64)
    single = a.ndim == 2
    m = a[None] if single else a
    for i in range(m.shape[0]):
        p = m[i]
        if not np.allclose(p[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"pose frame {i}: last row must be [0, 0, 0, 1], "
                             f"got {p[3]}")
        R = p[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
            raise ValueError(f"pose frame {i}: R = T[:3,:3] must be orthonormal")
        if abs(np.linalg.det(R) - 1.0) > 1e-3:
            raise ValueError(f"pose frame {i}: det(R) must be 1 "
                             f"(a c2w rotation), got {np.linalg.det(R)}")
    return m[0] if single else a
