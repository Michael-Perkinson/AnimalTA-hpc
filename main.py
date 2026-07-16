from AnimalTA.Main_interface import start_mainframe
from AnimalTA import compat
import multiprocessing
import os
import sys

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    multiprocessing.freeze_support()
    compat.startup_debug(f"cli entry; platform={sys.platform} DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    from AnimalTA.A_General_tools import gpu_utils
    gpu_status = "available" if gpu_utils.CUPY_AVAILABLE else "not available (CPU fallback)"
    print(f"[AnimalTA] GPU acceleration: {gpu_status}", file=sys.stderr, flush=True)
    if not gpu_utils.CUPY_AVAILABLE and gpu_utils.GPU_FALLBACK_REASON:
        print(f"[AnimalTA] GPU fallback reason: {gpu_utils.GPU_FALLBACK_REASON}", file=sys.stderr, flush=True)
        import tkinter as tk
        import tkinter.messagebox as mb
        _root = tk.Tk()
        _root.withdraw()
        mb.showwarning(
            "GPU unavailable",
            f"GPU acceleration could not be initialised. Running on CPU.\n\n{gpu_utils.GPU_FALLBACK_REASON}"
        )
        _root.destroy()
    start_mainframe()
