# security_settings_track.py
import psutil
from pathlib import Path

# Global variables
stop_threads = False
activate_protection = False
activate_super_protection = False
capture = None
ID_kepts = None

def init():
    global stop_threads
    global activate_protection
    global activate_super_protection
    global capture
    global ID_kepts

    stop_threads = False
    activate_protection = False
    activate_super_protection = False
    capture = None
    ID_kepts = None

def _get_memory_usage_percent():
    '''Get memory usage percentage, preferring cgroup-aware reading when available.

    Tries cgroup v2 and v1 first (for HPC/containerized environments), then falls back
    to node-wide psutil reading for non-cgroup environments (local dev, Windows, etc).
    Returns percentage (0-100) or falls back to psutil on any error.
    '''
    # Try cgroup v2
    try:
        cgroup_v2_max = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        cgroup_v2_current = Path("/sys/fs/cgroup/memory.current").read_text().strip()

        # If limit is "max", treat as unlimited (no cgroup constraint)
        if cgroup_v2_max != "max":
            limit = int(cgroup_v2_max)
            current = int(cgroup_v2_current)
            if limit > 0:
                return 100.0 * current / limit
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        pass

    # Try cgroup v1
    try:
        cgroup_v1_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text().strip()
        cgroup_v1_current = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes").read_text().strip()

        limit = int(cgroup_v1_limit)
        current = int(cgroup_v1_current)

        # Cgroup v1 uses sentinel values for "unlimited"; treat very large values as no limit
        # A finite limit should be much smaller than 2**62
        if 0 < limit < (1 << 62):
            return 100.0 * current / limit
    except (FileNotFoundError, PermissionError, ValueError, OSError):
        pass

    # Fall back to node-wide memory usage
    return psutil.virtual_memory()._asdict()["percent"]

def check_memory_overload():
    '''Some problems of memory leak were encountered with decord, these problems have been fixed but this is a security to control potential loss of memory.'''
    mem=_get_memory_usage_percent()
    if mem > 99:#Too much memory of the computer is used, we trigger a security that will stop immediatly the program
        return 1
    elif mem > 97.5: #The computer is reaching it's limit, it could come from a memory leak of decord, we add a security that will limity decord's memory leaks
        return 0
    else:
        return -1 #The memory usage is back to normal, we remove previous security as it slow down the rocess a bit

