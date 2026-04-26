# AnimalTA — Open OnDemand App

This directory contains the OOD interactive app configuration for AnimalTA.

For full deployment instructions see [DEPLOYMENT.md](DEPLOYMENT.md).

## Quick summary

1. Build or download `animalta.sif` (see DEPLOYMENT.md Step 1)
2. Copy this `ood/` directory to your OOD sys apps directory
3. Set your cluster name in `form.yml`
4. Set the `.sif` path in `template/script.sh.erb`
5. Wire `#SBATCH` directives from the form fields (see DEPLOYMENT.md Step 4)
