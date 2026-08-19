#!/bin/bash
# Production 3-tier pipeline — step3, assemble C_ℓ^GR from the HDF5 build.
#
# Streaming mode: peak RAM ~15-20 GB even though the on-disk HDF5 is
# ~150 GB; the integrals are read tile-by-tile.  Edit the tracer/output
# paths below to match your tomographic bin configuration.
#
# Submit:  sbatch examples/production/step3_compute_cl.slurm

#SBATCH -J PF_step3_cl
#SBATCH -o production/logs/step3_cl.o%j
#SBATCH -e production/logs/step3_cl.e%j
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=40G
#SBATCH -t 02:00:00

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JULIA=~/.juliaup/bin/julia
cd "$REPO"
mkdir -p production/logs

# Inputs:
#   $META   : step2's HDF5 meta file (the .h5 with the part-file index)
#   $TRACER : tomographic tracer config (auto-correlations and source pairs)
#   $OUTPUT : per-pair C_ℓ result file
META="$REPO/production/ClGR_production_meta.h5"
TRACER="$REPO/examples/tracer_sample3_binmid.h5"
OUTPUT="$REPO/examples/Cl_sample3_binmid_auto.h5"

$JULIA --project -t 16,1 src/compute_ClGR.jl \
    "$META" "$OUTPUT" \
    --tracer-1="$TRACER" \
    --cosmo-funcr="$REPO/data/cosmo_funcr.txt" \
    --fNL=1.0

echo "step3 (compute C_ℓ) DONE"
