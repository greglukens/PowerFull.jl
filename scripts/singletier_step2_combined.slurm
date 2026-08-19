#!/bin/bash
# Combined step2 + step2b: build_and_export with default flags (does both
# integrated AND base in one call, ~8 min vs split 2+6 min).
# Use this for new builds; the dlnRtest split was historical.
#SBATCH -J PF_s2_comb
#SBATCH -o production/logs/singletier_step2_comb_%j.out
#SBATCH -e production/logs/singletier_step2_comb_%j.err
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=120G
#SBATCH -t 06:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia

# Required env: NR_VAL, DLNR, ELLMIN, ELLMAX, DATADIR, OUTNAME
: "${NR_VAL:?need NR_VAL}"
: "${DLNR:?need DLNR}"
: "${ELLMIN:?need ELLMIN}"
: "${ELLMAX:?need ELLMAX}"
: "${DATADIR:?need DATADIR}"
: "${OUTNAME:?need OUTNAME}"

OUTDIR=${OUTDIR:-$DATADIR/builds}

cd "$REPO"
mkdir -p "$OUTDIR" production/logs

NTHREADS=${SLURM_CPUS_PER_TASK:-24}
export JULIA_NUM_THREADS=$NTHREADS

echo "Host=$(hostname)  Job=$SLURM_JOB_ID  CPUs=$SLURM_CPUS_ON_NODE  threads=$NTHREADS"
echo "nR=$NR_VAL  dlnR=$DLNR  ell=$ELLMIN..$ELLMAX  outname=$OUTNAME"
date

$JULIA --project -t $NTHREADS src/build_and_export.jl \
    --Nr=4096 --nR=$NR_VAL \
    --dlnR=$DLNR --ellmin=$ELLMIN --ellmax=$ELLMAX \
    --datadir="$DATADIR" \
    --outname="$OUTDIR/$OUTNAME"

date
echo "step2_combined DONE  ($OUTNAME)"
