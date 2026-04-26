# GPU Acceleration Plan

Target: replace CPU-bound per-frame operations in the tracking pipeline with CUDA equivalents.
Expected speedup: 5–10x. Current baseline: ~12 min per video on a laptop CPU.

---

## 1. Apptainer.def — change base image

**Current:**
```
Bootstrap: docker
From: python:3.11-slim-bookworm
```

**Change to** an NVIDIA CUDA runtime image so the container has access to CUDA libs:
```
Bootstrap: docker
From: nvidia/cuda:12.4-runtime-ubuntu22.04
```

Then in `%post`, replace `pip install opencv-python-headless` with a CUDA-enabled OpenCV.
Options (pick one):
- Build OpenCV from source with `-D WITH_CUDA=ON` (slow build, ~30 min, but full control)
- Use `opencv-contrib-python` wheels from `cv2-cuda` third-party index if available for the target Python version
- Install via apt: `python3-opencv` from Ubuntu repos sometimes includes CUDA support depending on the Ubuntu version

Also replace `decord` with the GPU build:
```bash
# In %post, after CUDA base is set up:
pip install decord  # CPU fallback only — GPU build needs source compile
# OR: build from source inside the container:
git clone --recursive https://github.com/dmlc/decord
cd decord && mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../python && pip install .
```

Add to `%environment`:
```
export CUDA_VISIBLE_DEVICES=0
```

---

## 2. ood/template/script.sh.erb — expose GPU to container

**Current:**
```bash
apptainer exec \
    --bind /home/$USER:/home/$USER \
    --bind /tmp:/tmp \
    ...
```

**Add `--nv` flag** (passes NVIDIA drivers from host into container):
```bash
apptainer exec \
    --nv \
    --bind /home/$USER:/home/$USER \
    --bind /tmp:/tmp \
    ...
```

---

## 3. ood/form.yml — add GPU resource field

Add a GPU selector so users can request a GPU node:
```yaml
  num_gpus:
    widget: "number_field"
    label: "GPUs"
    value: 1
    min: 0
    max: 1
```
And add `- num_gpus` to the `form:` list.
Then wire it into the SLURM/PBS header in `script.sh.erb` via `<%= num_gpus %>`.

---

## 4. GPU availability check — add once at startup

Add a module-level helper, e.g. in `AnimalTA/compat.py` or a new `AnimalTA/gpu_utils.py`:
```python
def cuda_available():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False

CUDA = cuda_available()
```

Import `CUDA` everywhere that needs it and use it to branch between GPU and CPU paths.

---

## 5. decord.VideoReader calls — switch to GPU context

Every `decord.VideoReader(path)` call should become `decord.VideoReader(path, ctx=decord.gpu(0) if CUDA else decord.cpu(0))`.

Files and lines to update:

| File | Line(s) | Current |
|------|---------|---------|
| `AnimalTA/D_Tracking_process/Do_the_track.py` | 87, 90 | `decord.VideoReader(Vid.Fusion[...][1])` |
| `AnimalTA/D_Tracking_process/Do_the_track_multi.py` | 83 | `decord.VideoReader(Vid.Fusion[...][1])` |
| `AnimalTA/D_Tracking_process/Function_prepare_images.py` | 42, 48 | `decord.VideoReader(Vid.Fusion[...][1])` |
| `AnimalTA/G_Specials/Test_specific_parts_track.py` | 44 | `decord.VideoReader(Vid.Fusion[...][1])` |
| `AnimalTA/E_Post_tracking/b_Analyses/Body_part_functions.py` | 63 | `decord.VideoReader(Vid.Fusion[...][1])` |
| `AnimalTA/B_Project_organisation/Interface_pretracking.py` | 1762, 1773 | Already explicit `ctx=decord.cpu(0)` — easy swap |
| `AnimalTA/A_General_tools/Video_loader.py` | 78 | `decord.VideoReader(File)` in background thread |

---

## 6. Function_prepare_images.py — the hot loop (main work)

File: `AnimalTA/D_Tracking_process/Function_prepare_images.py`

This is the frame-by-frame loop (line 31 onward). The pattern for CUDA ops is:
1. Upload numpy frame to GPU: `gpu_mat = cv2.cuda_GpuMat(); gpu_mat.upload(frame)`
2. Run CUDA op on `gpu_mat`
3. Download only when needed for CPU-only ops: `frame = gpu_mat.download()`

### Line 78 — cvtColor BGR→GRAY
```python
# CPU:
Timg = cv2.cvtColor(Timg, cv2.COLOR_BGR2GRAY)
# GPU:
gpu_Timg = cv2.cuda_GpuMat(); gpu_Timg.upload(Timg)
gpu_Timg = cv2.cuda.cvtColor(gpu_Timg, cv2.COLOR_BGR2GRAY)
Timg = gpu_Timg.download()
```

### Lines 108–112 — background subtraction (absdiff / subtract)
```python
# CPU:
img = cv2.absdiff(TMP_back, img)
img = cv2.subtract(TMP_back, img)
img = cv2.subtract(img, TMP_back)
# GPU (upload both, run op, download):
gpu_img = cv2.cuda_GpuMat(); gpu_img.upload(img)
gpu_back = cv2.cuda_GpuMat(); gpu_back.upload(TMP_back)
gpu_img = cv2.cuda.absdiff(gpu_back, gpu_img)   # or subtract variants
img = gpu_img.download()
```
Tip: cache `gpu_back` outside the loop — `TMP_back` only changes for dynamic background.

### Line 126 — threshold (binary)
```python
# CPU:
_, img = cv2.threshold(img, Vid.Track[1][0], 255, cv2.THRESH_BINARY)
# GPU:
gpu_img = cv2.cuda_GpuMat(); gpu_img.upload(img)
_, gpu_img = cv2.cuda.threshold(gpu_img, Vid.Track[1][0], 255, cv2.THRESH_BINARY)
img = gpu_img.download()
```

### Line 135 — adaptiveThreshold (no CUDA equivalent)
Keep on CPU. Upload/download cost is still worth it because all surrounding ops are GPU.

### Lines 143, 147 — erode / dilate
```python
# CPU:
img = cv2.erode(img, kernel, iterations=Vid.Track[1][1])
img = cv2.dilate(img, kernel, iterations=Vid.Track[1][2])
# GPU — create filter objects once OUTSIDE the loop:
erode_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_ERODE, cv2.CV_8UC1, kernel)
dilate_filter = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel)
# Inside loop:
gpu_img = erode_filter.apply(gpu_img)
gpu_img = dilate_filter.apply(gpu_img)
img = gpu_img.download()
```
Creating the filter objects inside the loop is expensive — move them above `for frame in ...`.

### Line 139 — bitwise_and (mask)
```python
# CPU:
img = cv2.bitwise_and(img, img, mask=mask)
# GPU:
gpu_mask = cv2.cuda_GpuMat(); gpu_mask.upload(mask)  # upload mask once outside loop
cv2.cuda.bitwise_and(gpu_img, gpu_img, dst=gpu_img, mask=gpu_mask)
```

### Line 151 — findContours (stays CPU)
```python
# No CUDA equivalent. Download before this line.
img = gpu_img.download()
cnts, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
```

---

## 7. Function_prepare_images_multi.py — multiprocess version

File: `AnimalTA/D_Tracking_process/Function_prepare_images_multi.py`

Uses `cv2.VideoCapture` (lines 61, 81) rather than decord, so no decord change needed here.
Same OpenCV CUDA op replacements as above apply.
Note: CUDA context is per-process, so each worker process gets its own GPU context — this is fine.

---

## Summary of effort

| Task | Difficulty | Impact |
|------|-----------|--------|
| Apptainer base image + CUDA OpenCV | High (build complexity) | Required for everything |
| `--nv` in script.sh.erb | Trivial | Required for everything |
| decord GPU context (all VideoReader calls) | Low (search/replace) | Medium — faster decoding |
| CUDA ops in Function_prepare_images.py | Medium | High — this is the hot path |
| Move filter creation outside loop | Trivial | Small but free win |
| Cache gpu_back outside loop | Trivial | Small but free win |
| GPU availability fallback (CUDA flag) | Low | Required for non-GPU nodes |

Start with the Apptainer rebuild — nothing else works without it.
