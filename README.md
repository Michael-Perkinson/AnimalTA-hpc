# AnimalTA - Linux / HPC Fork

> Based on [AnimalTA 4.0.0](https://github.com/VioletteChiara/AnimalTA), with
> Linux, Apptainer, Open OnDemand, and HPC tracking-throughput adaptations.
> Upstream 4.2.0 changes have not yet been integrated. All core AnimalTA
> tracking and analysis functionality is Violette Chiara's work.

AnimalTA is a graphical application for tracking and analysing animal movement
in video. It supports multiple arenas, flexible background subtraction, Kalman
filtering, tracking correction, and a full post-tracking analysis suite.

## What this fork adds

- Native Linux and macOS compatibility while retaining Windows support.
- An Apptainer container and Open OnDemand deployment for HPC clusters.
- Tracking workers sized to the CPU and memory allocated by the scheduler.
- A bounded shared-memory frame pool to avoid serialising full frames between
  processes.
- Parallel video readers designed to reduce contention on shared filesystems.
- OpenCV as the default tracking reader, with Decord backends available for
  testing and workload-specific tuning.
- Tracking diagnostics written to standard error, including the selected
  reader and worker configuration.
- Optional CUDA support with a safe CPU fallback. The per-frame tracking path
  remains CPU-based because GPU transfer overhead was slower for this workload.

## Using on an HPC cluster

The recommended approach is the Apptainer container, which bundles the Python
and system dependencies.

### Build the container

Download `animalta.sif` from the [latest fork release](../../releases/latest),
or build it locally:

```bash
apptainer build animalta.sif Apptainer.def
```

### Run

```bash
apptainer run animalta.sif
```

AnimalTA requires a graphical display. Run it inside a VNC session or through
Open OnDemand.

### Open OnDemand deployment

See [ood/DEPLOYMENT.md](ood/DEPLOYMENT.md) for the cluster-administrator setup
guide.

## Installing from source

Python 3.11 or newer and a working Tk display are required. On Linux this is
normally X11 or Wayland through XWayland.

```bash
python -m pip install .
python main.py
```

Installing the project also provides an `animalta` command:

```bash
animalta
```

The project deliberately uses `opencv-python-headless` so OpenCV's Qt backend
does not conflict with Tkinter.

## Windows

Windows users who do not need the HPC adaptations can use the installer from
the [upstream releases](https://github.com/VioletteChiara/AnimalTA/releases).

## Tracking reader selection

The default OpenCV reader is the best-tested option for the HPC workflow. The
reader can be selected before launching AnimalTA:

```bash
export ANIMALTA_READER_BACKEND=opencv
```

Supported values are `opencv`, `auto`, `decord`, and `decord-gpu`. Advanced
reader process, batching, and buffering controls are available through the
`ANIMALTA_READER_*` and `ANIMALTA_DECORD_*` environment variables. Normal runs
should not need to set them.

## Citation

If you use AnimalTA in your research, please cite the original paper:

> Chiara, V., & Kim, S.-Y. (2023). AnimalTA: A highly flexible and easy-to-use
> program for tracking and analyzing animal movement in different environments.
> *Methods in Ecology and Evolution*, 14, 1699-1707.
> [https://doi.org/10.1111/2041-210X.14115](https://doi.org/10.1111/2041-210X.14115)

## License

MIT - see [LICENSE](LICENSE). Original copyright (c) 2022 Violette Chiara.
