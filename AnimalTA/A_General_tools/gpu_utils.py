try:
    import cupy as cp
    cp.array([1], dtype=cp.uint32).astype(cp.float32)  # forces NVRTC JIT compile
    CUPY_AVAILABLE = True
    GPU_FALLBACK_REASON = None
except Exception as e:
    CUPY_AVAILABLE = False
    GPU_FALLBACK_REASON = str(e)
