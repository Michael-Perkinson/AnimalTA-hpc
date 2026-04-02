# AnimalTA Linux Port — Platform Audit

Generated as part of the Linux/Apptainer port for Open OnDemand (NeSI).

---

## 1. Platform-Specific Findings

### BLOCKER — `from ctypes import windll`
**File:** `AnimalTA/Main_interface.py:15`
```python
from ctypes import windll
```
**Also:** `Main_interface.py:213-217` — `set_appwindow()` uses `windll.user32` to manipulate the Windows taskbar entry (cosmetic only).
```python
def set_appwindow(root):
    hwnd = windll.user32.GetParent(root.winfo_id())
    style = windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ...
```
**Fix:** Guard the import and the `set_appwindow()` call with `if sys.platform == "win32"`.

---

### BLOCKER — Auto-updater downloads and runs a Windows `.exe` installer

**Files and lines:**

| File | Line | Code |
|------|------|------|
| `AnimalTA/Main_interface.py` | 126–193 | Checks for new version, downloads `.exe`, runs `/SILENT` |
| `AnimalTA/A_General_tools/Diverse_functions.py` | 31–32 | `urllib.request.urlretrieve(... ".exe", ...)` |
| `AnimalTA/A_General_tools/Interface_info.py` | 155, 164, 183, 193 | Downloads and runs `last_update.exe` |

**Fix:** Wrap all auto-update logic with `if os.environ.get("ANIMALTA_DISABLE_UPDATE") != "1" and sys.platform == "win32":`. Set `ANIMALTA_DISABLE_UPDATE=1` in the container environment.

---

### BLOCKER — Hardcoded `ffmpeg.exe` path

**File:** `AnimalTA/A_General_tools/Class_converter.py:13`
```python
ffmpeg_path = os.path.join(File_folder, "ffmpeg", "ffmpeg.exe")
```
The converter assumes a bundled Windows `ffmpeg.exe`. On Linux, `ffmpeg` is a system package.

**Fix:** Detect platform and fall back to system `ffmpeg`:
```python
if sys.platform == "win32":
    ffmpeg_path = os.path.join(File_folder, "ffmpeg", "ffmpeg.exe")
else:
    ffmpeg_path = "ffmpeg"  # use system ffmpeg on Linux/Mac
```

---

### BLOCKER — `simsun.ttc` Windows font

**Files:**
- `AnimalTA/A_General_tools/Class_stabilise.py:117`
- `AnimalTA/A_General_tools/Class_Lecteur.py:614, 624`

```python
fontpath = os.path.join(".", "simsun.ttc")
font = ImageFont.truetype(fontpath, 20)
```
SimSun (`simsun.ttc`) is a Chinese font bundled with Windows. The file is **not present in the repo** and no try/except wraps these calls — on Linux they will raise `FileNotFoundError` and crash the stabilisation preview and video reader overlay.

**Fix:** Replace with a font available on Linux. `DejaVuSans.ttf` ships with most Linux distros and is also used by matplotlib. Fall back to PIL's built-in default if unavailable:
```python
import shutil
fontpath = shutil.which("DejaVuSans.ttf") or "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
try:
    font = ImageFont.truetype(fontpath, size)
except (OSError, IOError):
    font = ImageFont.load_default()
```

---

### MINOR — `import ntpath` (Windows path module)

**Files:** `AnimalTA/Class_Video.py:2`, `AnimalTA/A_General_tools/Interface_Change_Folder.py:5`, `AnimalTA/B_Project_organisation/Interface_pretracking.py:2`, `AnimalTA/A_General_tools/Interface_Vids_for_convert.py:7`

Used only as `ntpath.basename(path)` to extract filenames. `ntpath` is part of Python's standard library on all platforms and handles both Windows (`\`) and Unix (`/`) separators. This is actually a **correct cross-platform approach** for reading stored project files that may contain Windows-style paths.

**Severity:** Harmless — no change needed.

---

### MINOR — `subprocess.STARTUPINFO` / `CREATE_NO_WINDOW`

**File:** `AnimalTA/A_General_tools/Class_converter.py:96–100`
```python
if hasattr(subprocess, "STARTUPINFO"):
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

process = subprocess.Popen(cmd, ..., startupinfo=startupinfo, creationflags=subprocess.CREATE_NO_WINDOW, ...)
```
`STARTUPINFO` is already guarded with `hasattr`. However `subprocess.CREATE_NO_WINDOW` (value `0x08000000`) is passed unconditionally — on Linux this flag is silently ignored by `subprocess`, so it is **harmless** but untidy.

**Fix:** Guard `creationflags` with `if sys.platform == "win32"`.

---

### HARMLESS — `multiprocessing.freeze_support()`

**File:** `cli.py:5`
```python
multiprocessing.freeze_support()
```
No-op on non-Windows/non-frozen environments. Safe to leave as-is.

---

### HARMLESS — PyInstaller splash screen (`pyi_splash`)

**File:** `AnimalTA/Main_interface.py:2–3, 24–25`
```python
if getattr(sys, 'frozen', False):
    import pyi_splash
```
Already guarded. Not present in a container run.

---

### SUSPICIOUS — Django imports in a desktop tkinter app

**File:** `AnimalTA/Class_Video.py:3`
```python
from django.db.models.functions import Repeat
```
**Severity:** High — this import is **completely unused** in the file. Django is a web framework with ~15 transitive dependencies; there is no reason for it to be here. Looks like a stray development artifact.

**File:** `AnimalTA/B_Project_organisation/Interface_pretracking.py:26`
```python
from django.utils.crypto import get_random_string
```
**Used at:** lines 753, 1158 — to generate a 10-character random project ID.
**Severity:** Medium — functionally correct but an unnecessarily heavy dependency. Python's `secrets` module provides `secrets.token_urlsafe()` for this purpose without requiring Django.

**Fix (both):** Replace with standard library equivalents and remove django from dependencies entirely.

---

## 2. Complete Third-Party Dependency List

| Install name | Import name | Files |
|---|---|---|
| `opencv-python-headless` | `cv2` | 54 files (core — video/image processing) |
| `Pillow` | `PIL` | 23 files (image display in tkinter) |
| `numpy` | `numpy` | 57 files (numerical arrays — core) |
| `pandas` | `pandas` | 4 files (data analysis) |
| `scipy` | `scipy` | 9 files (signal processing, spatial) |
| `scikit-learn` | `sklearn` | 2 files (KMeans clustering) |
| `scikit-image` | `skimage` | 5 files (morphology, graph) |
| `decord` | `decord` | 8 files (video frame decoding) |
| `h5py` | `h5py` | 1 file (HDF5 import) |
| `pymediainfo` | `pymediainfo` | 1 file (video metadata) |
| `pandastable` | `pandastable` | 1 file (table widget) |
| `psutil` | `psutil` | 3 files (process priority, memory) |
| `pyautogui` | `pyautogui` | 3 files (mouse position) |
| `pymsgbox` | `pymsgbox` | 1 file (message dialogs) |
| `matplotlib` | `matplotlib` | 3 files (analysis plots) |
| `tksheet` | `tksheet` | 1 file (spreadsheet widget) |
| `django` | `django` | 2 files (**suspicious — see above**) |
| `pytest` | `pytest` | 1 file (tests only) |
| `hypothesis` | `hypothesis` | 1 file (tests only) |

> **Note:** Use `opencv-python-headless` (not `opencv-python`) in the container to avoid Qt/GTK conflicts with tkinter.

---

## 3. Config/Data Path Assumptions

- **Settings file:** Written to `AnimalTA/Files/Settings` (relative to install dir, resolved by `UserMessages.resource_path()`). Uses `tempfile.gettempdir()` for a temporary copy during updates — this is portable.
- **`Last_downloaded` file:** Written to `AnimalTA/Files/Last_downloaded` — same mechanism, portable.
- **Update installer:** Written to `AnimalTA/Files/last_update.exe` and `last_update.crdownload` — Windows-only, irrelevant in container.
- **Logo/icon:** `AnimalTA/Files/Logo.ico` — used by `root.iconbitmap()`. `.ico` format works on Linux tkinter. Non-blocking.
- No `AppData`, `%USERPROFILE%`, or registry references found.
- No hardcoded drive letters (`C:\`, `D:\`) found.
- All internal paths use `os.path.join()` — portable.

---

## 4. Compiled/Binary Components

- `AnimalTA/Files/ffmpeg/ffmpeg.exe` — Windows ffmpeg binary bundled with the app. Not used in the container (system ffmpeg replaces it).
- No `.pyd`, `.dll`, or `.so` files found in the Python source tree.

---

## Summary

| Category | Count | Action |
|---|---|---|
| Blockers | 3 | Must fix before Linux works |
| Minor | 1 | Fix for cleanliness |
| Harmless | 2 | Leave as-is |
| Suspicious (non-platform) | 2 | Fix to remove Django dependency |
