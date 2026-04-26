# AnimalTA Open OnDemand App

## Building the container
```bash
apptainer build animalta.sif Apptainer.def
```

## Installing the OOD app
Copy this `ood/` directory to your OOD apps directory
(typically `/var/www/ood/apps/sys/` or use the OOD dev sandbox).

## Configuration
Edit the following files and replace CHANGEME markers:
- `form.yml`: cluster name
- `template/script.sh.erb`: path to .sif file, module loads

The launcher seeds a writable runtime area at `$HOME/.animalta/Projects`.
If a bundled or read-only `.ata` project is opened, AnimalTA now creates a
writable working copy there before conversion, tracking, or coordinate export.
