#!/bin/bash
# Production 3-tier pipeline — step1, high-ell tier (ell=201-500).
#
# Computes the TwoFAST radial integrand cache for the high-ell tier
# (dlnR=0.0005, nR=2049).  Output: jld2 files under
# ${OUTDIR}/production/results/tier_high/.
#
# Submit:  sbatch examples/production/step1_tier_high.slurm

#SBATCH -J PF_step1_high
#SBATCH -o production/logs/step1_high.o%j
#SBATCH -e production/logs/step1_high.e%j
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=400G
#SBATCH -t 03:00:00

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JULIA=~/.juliaup/bin/julia
cd "$REPO"
mkdir -p production/logs production/results/tier_high

JULIA_NUM_THREADS=6 $JULIA --project -p 8 src/run_twofast.jl \
    --Nr=4096 --nR=2049 --dlnR=0.0005 \
    --ellmin=201 --ellmax=500 --ellmax-margin=510 \
    --matterpower="$REPO/data/planck_base_plikHM_TTTEEE_lowTEB_lensing_post_BAO_H070p6_JLA_matterpower.dat" \
    --cosmo-funcr="$REPO/data/cosmo_funcr.txt" \
    --outdir="$REPO/production/results/tier_high" \
    --streaming-mlcache

echo "step1 (tier_high) DONE"
