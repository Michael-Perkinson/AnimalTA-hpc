from AnimalTA.Main_interface import start_mainframe
from AnimalTA import compat
import multiprocessing
import os
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    compat.startup_debug(f"cli entry; platform={sys.platform} DISPLAY={os.environ.get('DISPLAY', '<unset>')}")
    start_mainframe()
