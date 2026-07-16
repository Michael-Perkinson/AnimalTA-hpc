try:
    import cupy as cp
    try:
        cp.array([1], dtype=cp.uint32).astype(cp.float32)  # forces NVRTC JIT compile
    except cp.cuda.memory.OutOfMemoryError:
        pass  # GPU available but under memory pressure at startup; JIT compiles on first use
    CUPY_AVAILABLE = True
    GPU_FALLBACK_REASON = None
except Exception as e:
    CUPY_AVAILABLE = False
    GPU_FALLBACK_REASON = str(e)
