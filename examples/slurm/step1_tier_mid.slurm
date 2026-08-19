#!/bin/bash
# Production 3-tier pipeline — step1, mid-ell tier (ell=51-200).
#
# Computes the TwoFAST radial integrand cache for the mid-ell tier
# (dlnR=0.001, nR=2049).  Output: jld2 files under
# ${OUTDIR}/production/results/tier_mid/.
#
# Submit:  sbatch examples/production/step1_tier_mid.slurm

#SBATCH -J PF_step1_mid
#SBATCH -o production/logs/step1_mid.o%j
#SBATCH -e production/logs/step1_mid.e%j
#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem=300G
#SBATCH -t 02:00:00

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JULIA=~/.juliaup/bin/julia
cd "$REPO"
mkdir -p production/logs production/results/tier_mid

JULIA_NUM_THREADS=6 $JULIA --project -p 8 src/run_twofast.jl \
    --Nr=4096 --nR=2049 --dlnR=0.001 \
    --ellmin=51 --ellmax=200 --ellmax-margin=210 \
    --matterpower="$REPO/data/planck_base_plikHM_TTTEEE_lowTEB_lensing_post_BAO_H070p6_JLA_matterpower.dat" \
    --cosmo-funcr="$REPO/data/cosmo_funcr.txt" \
    --outdir="$REPO/production/results/tier_mid" \
    --streaming-mlcache

echo "step1 (tier_mid) DONE"
