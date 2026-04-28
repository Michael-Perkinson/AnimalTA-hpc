try:
    import cupy as cp
    cp.array([0])  # probe — import alone succeeds even without a GPU
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False
