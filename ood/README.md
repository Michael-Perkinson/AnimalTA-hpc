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
