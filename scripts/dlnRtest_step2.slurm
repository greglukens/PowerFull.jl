#!/bin/bash
#SBATCH -J PF_dlnRbld
#SBATCH -o production/logs/dlnRtest_step2_%j.out
#SBATCH -e production/logs/dlnRtest_step2_%j.err
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=60G
#SBATCH -t 04:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
DATADIR=$REPO/production/results/test_dlnR_jump
OUTDIR=$REPO/production/results/test_dlnR_jump/builds

# Required env: DLNR, ELLMIN, ELLMAX, OUTNAME
# Optional   : RCUT_MIN, RCUT_MAX (defaults: 0.0 / Inf = no cut)
: "${DLNR:?need DLNR}"
: "${ELLMIN:?need ELLMIN}"
: "${ELLMAX:?need ELLMAX}"
: "${OUTNAME:?need OUTNAME}"
RCUT_MIN="${RCUT_MIN:-0.0}"
RCUT_MAX="${RCUT_MAX:-1e10}"

cd "$REPO"
mkdir -p "$OUTDIR" production/logs

NTHREADS=${SLURM_CPUS_PER_TASK:-24}
export JULIA_NUM_THREADS=$NTHREADS

echo "Host=$(hostname) Job=$SLURM_JOB_ID  CPUs=$SLURM_CPUS_ON_NODE  threads=$NTHREADS"
echo "DLNR=$DLNR  ell=$ELLMIN..$ELLMAX  outname=$OUTNAME"
echo "Rcut_min=$RCUT_MIN  Rcut_max=$RCUT_MAX"
date

$JULIA --project -t $NTHREADS src/build_and_export.jl \
    --Nr=4096 --nR=2049 \
    --dlnR=$DLNR --ellmin=$ELLMIN --ellmax=$ELLMAX \
    --datadir="$DATADIR" \
    --outname="$OUTDIR/$OUTNAME" \
    --Rcut-min=$RCUT_MIN \
    --Rcut-max=$RCUT_MAX \
    --no-base

date
echo "step2 DONE  ($OUTNAME)"
