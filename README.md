# AnimalTA — Linux / HPC Fork

> Forked from [VioletteChiara/AnimalTA](https://github.com/VioletteChiara/AnimalTA).
> This fork adds native Linux support, an Apptainer container for HPC clusters, and
> Open OnDemand deployment. All core tracking functionality is Violette Chiara's work.

AnimalTA is an easy-to-use GUI program for tracking and analysing animal movement in video.
It supports multiple arenas, flexible background subtraction, Kalman filtering, and a full
post-tracking analysis suite.

---

## Using on an HPC cluster (Linux)

The recommended path is the pre-built Apptainer container, which bundles all dependencies.

### 1. Get the container

Download `animalta.sif` from the [latest release](../../releases/latest), or build it yourself:

```bash
apptainer build animalta.sif Apptainer.def
```

### 2. Run

```bash
apptainer run animalta.sif
```

A display is required — run this inside a VNC session or via Open OnDemand.

### 3. Open OnDemand deployment

See [ood/DEPLOYMENT.md](ood/DEPLOYMENT.md) for the full step-by-step guide for HPC sysadmins.

---

## Using on Windows

Download the installer from the [original repo's releases](https://github.com/VioletteChiara/AnimalTA/releases).
The Windows version has a one-click installer and does not require this fork.

---

## Installing from source (Linux / macOS)

```bash
pip install -r requirements.txt
python main.py
```

Requires Python 3.10+ and a working display (X11 or Wayland via XWayland).
Use `opencv-python-headless` (already specified in `requirements.txt`) to avoid
conflicts between OpenCV's Qt backend and tkinter.

---

## Roadmap

- **GPU acceleration** *(in progress)* — CUDA-accelerated background subtraction and contour extraction
  to cut tracking time on GPU-equipped HPC nodes.

---

## Citation

If you use AnimalTA in your research, please cite the original paper:

> Chiara, V., & Kim, S.-Y. (2023). AnimalTA: A highly flexible and easy-to-use program
> for tracking and analyzing animal movement in different environments.
> *Methods in Ecology and Evolution*, 14, 1699–1707.
> [https://doi.org/10.1111/2041-210X.14115](https://doi.org/10.1111/2041-210X.14115)

---

## License

MIT — see [LICENSE](LICENSE). Original copyright © 2022 Violette Chiara.
