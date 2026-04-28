from AnimalTA.Main_interface import start_mainframe
from AnimalTA import compat
import multiprocessing
import os
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    compat.startup_debug(f"cli entry; platform={sys.platform} DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    from AnimalTA.A_General_tools import gpu_utils
    gpu_status = "available" if gpu_utils.CUPY_AVAILABLE else "not available (CPU fallback)"
    print(f"[AnimalTA] GPU acceleration: {gpu_status}", file=sys.stderr, flush=True)
    start_mainframe()
