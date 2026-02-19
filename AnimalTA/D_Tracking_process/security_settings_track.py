# security_settings_track.py
import psutil

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

def check_memory_overload():
    '''Some problems of memory leak were encountered with decord, these problems have been fixed but this is a security to control potential loss of memory.'''
    mem=psutil.virtual_memory()._asdict()["percent"]
    if mem > 99:#Too much memory of the computer is used, we trigger a security that will stop immediatly the program
        return 1
    elif mem > 97.5: #The computer is reaching it's limit, it could come from a memory leak of decord, we add a security that will limity decord's memory leaks
        return 0
    else:
        return -1 #The memory usage is back to normal, we remove previous security as it slow down the rocess a bit

