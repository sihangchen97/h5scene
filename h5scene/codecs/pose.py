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
def K_row_errors(m):
    """The per-frame K convention errors for (T, 3, 3) float64 m (the one
    shared rule set: check_K and h5scene.verify). An empty list = OK."""
    errs = []
    for i in range(m.shape[0]):
        if not np.all(np.isfinite(m[i])):
            errs.append(f"K frame {i}: intrinsics must be finite (no NaN/inf)")
            continue
        fx, fy = m[i, 0, 0], m[i, 1, 1]
        if not (fx > 0 and fy > 0):
            errs.append(f"K frame {i}: fx / fy must be > 0, got {fx} / {fy}")
        if abs(m[i, 2, 2] - 1.0) > 1e-6:
            errs.append(f"K frame {i}: K[2,2] must be 1 (pinhole), "
                        f"got {m[i, 2, 2]}")
        for r, c, name in ((1, 0, "K[1,0] (skew)"), (2, 1, "K[2,1]"),
                           (2, 0, "K[2,0]")):
            if abs(m[i, r, c]) > 1e-6:
                errs.append(f"K frame {i}: {name} must be 0, got {m[i, r, c]}")
    return errs


def check_K(K):
    """(T, 3, 3) (or (3, 3)) OpenCV PINHOLE intrinsics: FINITE (no NaN/inf),
    fx, fy > 0, K[2,2] == 1 (within 1e-6), zero skew (K[1,0]) and zero
    K[2,1] / K[2,0]. Returns K unchanged (the input, not a float64 copy —
    the storage dtype is the SECTIONS spec's); the first convention error
    raises (K_row_errors lists them all)."""
    a = np.asarray(K, dtype=np.float64)
    m = a if a.ndim == 3 else a[None]
    errs = K_row_errors(m)
    if errs:
        raise ValueError(errs[0])
    return K


def pose_row_errors(m):
    """The per-frame c2w convention errors for (T, 4, 4) float64 m (the one
    shared rule set: check_pose and h5scene.verify). An empty list = OK."""
    errs = []
    for i in range(m.shape[0]):
        p = m[i]
        if not np.all(np.isfinite(p)):
            errs.append(f"pose frame {i}: must be finite (no NaN/inf)")
            continue
        if not np.allclose(p[3], [0, 0, 0, 1], atol=1e-6):
            errs.append(f"pose frame {i}: last row must be [0, 0, 0, 1], "
                        f"got {p[3]}")
        R = p[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-3):
            errs.append(f"pose frame {i}: R = T[:3,:3] must be orthonormal")
        if abs(np.linalg.det(R) - 1.0) > 1e-3:
            errs.append(f"pose frame {i}: det(R) must be 1 "
                        f"(a c2w rotation), got {np.linalg.det(R)}")
    return errs


def check_pose(pose_c2w):
    """(T, 4, 4) (or (4, 4)) camera-to-world poses: FINITE (no NaN/inf),
    last row [0, 0, 0, 1] and an orthonormal R with |det(R) - 1| <= 1e-3
    (1e-6 on the rows). Returns the input unchanged (not a float64 copy)."""
    a = np.asarray(pose_c2w, dtype=np.float64)
    m = a if a.ndim == 3 else a[None]
    errs = pose_row_errors(m)
    if errs:
        raise ValueError(errs[0])
    return pose_c2w
