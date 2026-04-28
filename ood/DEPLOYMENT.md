# AnimalTA — Open OnDemand Deployment Guide

## What you need

- An Apptainer `.sif` container built from the repo
- This `ood/` directory deployed as an OOD interactive app
- A VNC-capable OOD installation (standard for interactive desktop apps)

---

## Step 1: Get the container

Download `animalta.sif` from the GitHub Actions artifacts:
> https://github.com/Michael-Perkinson/AnimalTA-linux/actions

Pick the latest successful run of "Build Apptainer container" and download the `animalta-sif` artifact.
Place the `.sif` somewhere stable on your cluster, e.g. `/apps/animalta/animalta.sif`.

Alternatively, build it yourself:
```bash
apptainer build animalta.sif Apptainer.def
```

---

## Step 2: Deploy the OOD app

Copy the `ood/` directory to your OOD app location:
```bash
cp -r ood/ /var/www/ood/apps/sys/animalta
```
(Path may differ — use your site's OOD sys app directory.)

---

## Step 3: Edit the two CHANGEME items

### `form.yml` — set your cluster name
```yaml
cluster: "your-cluster-name"   # line 2
```

### `template/script.sh.erb` — set the path to the .sif
```bash
/apps/animalta/animalta.sif    # replace /path/to/animalta.sif
```

---

## Step 4: Wire up SLURM resource requests

The form collects `hours`, `num_cpus`, and `memory` from the user but the script
doesn't automatically pass them to the scheduler. Add `#SBATCH` directives at the
top of `template/script.sh.erb` (after the shebang line):

```bash
#SBATCH --time=<%= hours %>:00:00
#SBATCH --cpus-per-task=<%= num_cpus %>
#SBATCH --mem=<%= memory %>G
#SBATCH --job-name=animalta
```

Adjust directive names for your scheduler (PBS/SLURM/etc.).

---

## Step 5: Load the Apptainer module (if needed)

If your cluster uses a module system, uncomment and edit this line in
`template/script.sh.erb`:
```bash
# module load Apptainer  # CHANGEME: uncomment if module system is used
```

---

## Step 6: Add bind mounts for shared storage

The script binds `/home/$USER` and `/tmp` by default. If users' video files
live on shared storage (e.g. `/nfs/...`, `/scratch/...`), add those paths:

```bash
apptainer exec \
    --nv \                              # remove if no GPU
    --bind /home/$USER:/home/$USER \
    --bind /tmp:/tmp \
    --bind /your/shared/storage:/your/shared/storage \   # add this
    ...
```

---

## User data location

AnimalTA stores projects and settings at:
```
/home/$USER/.animalta/
```
This is writable user-local storage inside the container's home bind. No shared
writable directory is needed.

---

## GPU (optional)

GPU acceleration is under active development and not yet available in a release container.
When ready:
- The `--nv` flag is already in `script.sh.erb` — ensure the job lands on a GPU node
- Add a GPU SLURM directive: `#SBATCH --gres=gpu:1`
- Add a GPU field to `form.yml` (see the commented example in that file)

Without `--nv` or on a CPU-only node, remove the `--nv` flag — Apptainer will
warn but still run.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| App launches but GUI is blank | VNC display not set; check `DISPLAY=:<%= display %>` in script |
| `apptainer: command not found` | Module not loaded; add `module load Apptainer` |
| `FileNotFoundError: animalta.sif` | Wrong path in script.sh.erb |
| App hangs on startup | Container can't write to `~/.animalta`; check home bind mount |
| Font/text looks wrong | Expected — minor rendering difference between Windows and Linux; no action needed |
