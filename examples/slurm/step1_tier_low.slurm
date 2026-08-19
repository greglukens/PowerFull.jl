#!/bin/bash
# Production 3-tier pipeline — step1, low-ell tier (ell=2-50).
#
# Computes the TwoFAST radial integrand cache for the low-ell tier
# (dlnR=0.002, nR=4097).  Output: jld2 files under
# ${OUTDIR}/production/results/tier_low/.
#
# Submit:  sbatch examples/production/step1_tier_low.slurm
#
# Resource sizing assumes a single 64-core node with 200 GB RAM (8 Julia
# workers × 6 threads).  Adjust -p / JULIA_NUM_THREADS for other nodes.

#SBATCH -J PF_step1_low
#SBATCH -o production/logs/step1_low.o%j
#SBATCH -e production/logs/step1_low.e%j
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=200G
#SBATCH -t 02:00:00

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JULIA=~/.juliaup/bin/julia
cd "$REPO"
mkdir -p production/logs production/results/tier_low

JULIA_NUM_THREADS=6 $JULIA --project -p 8 src/run_twofast.jl \
    --Nr=4096 --nR=4097 --dlnR=0.002 \
    --ellmin=2 --ellmax=50 --ellmax-margin=60 \
    --matterpower="$REPO/data/planck_base_plikHM_TTTEEE_lowTEB_lensing_post_BAO_H070p6_JLA_matterpower.dat" \
    --cosmo-funcr="$REPO/data/cosmo_funcr.txt" \
    --outdir="$REPO/production/results/tier_low" \
    --streaming-mlcache

echo "step1 (tier_low) DONE"
