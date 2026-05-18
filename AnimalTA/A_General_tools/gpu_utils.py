try:
    import cupy as cp
    CUPY_AVAILABLE = True
    GPU_FALLBACK_REASON = None
except Exception as e:
    CUPY_AVAILABLE = False
    GPU_FALLBACK_REASON = str(e)
