#!/bin/bash
# Production 3-tier pipeline — step2, build HDF5 base + integrated arrays.
#
# Combines the three step1 tier outputs (tier_low, tier_mid, tier_high)
# into a single HDF5 build covering ell=2-500 with per-tier (dlnR, nR).
# Output: ${REPO}/production/ClGR_production_{meta,part_NNN}.h5.
#
# Submit:  sbatch examples/production/step2_build_3tier.slurm

#SBATCH -J PF_step2_build
#SBATCH -o production/logs/step2_build.o%j
#SBATCH -e production/logs/step2_build.e%j
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=400G
#SBATCH -t 02:00:00

set -euo pipefail
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
JULIA=~/.juliaup/bin/julia
NTHREADS=${SLURM_CPUS_PER_TASK:-24}
export JULIA_NUM_THREADS=$NTHREADS

cd "$REPO"
mkdir -p production/logs

# Stage the three tiers' jld2 files via symlinks into one directory so
# build_and_export.jl can index them with one --datadir argument.
DATADIR="$REPO/production/data_3tier"
OUTNAME="$REPO/production/ClGR_production"

mkdir -p "$DATADIR"
rm -f "$DATADIR"/TwoFAST_*.jld2

for d in "$REPO/production/results/tier_low" \
         "$REPO/production/results/tier_mid" \
         "$REPO/production/results/tier_high"; do
    for f in "$d"/TwoFAST_*.jld2; do
        [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "$DATADIR/$(basename "$f")"
    done
done

echo "Staged jld2 files in $DATADIR:"
ls "$DATADIR" | head -30

$JULIA --project -t $NTHREADS src/build_and_export.jl \
    --tier=0.002,2,50,4097 \
    --tier=0.001,51,200,2049 \
    --tier=0.0005,201,500,2049 \
    --datadir="$DATADIR" \
    --outname="$OUTNAME" \
    --cosmo-funcr="$REPO/data/cosmo_funcr.txt"

echo "step2 (3-tier build) DONE"
