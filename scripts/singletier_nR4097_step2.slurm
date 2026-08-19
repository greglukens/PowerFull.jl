#!/bin/bash
#SBATCH -J PF_4097_s2
#SBATCH -o production/logs/singletier_nR4097_step2_%j.out
#SBATCH -e production/logs/singletier_nR4097_step2_%j.err
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=120G
#SBATCH -t 12:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
DATADIR=$REPO/production/results/test_singletier_nR4097
OUTDIR=$REPO/production/results/test_singletier_nR4097/builds

DLNR=0.002
ELLMIN=2
ELLMAX=210
OUTNAME=ClGR_nR4097

cd "$REPO"
mkdir -p "$OUTDIR" production/logs

NTHREADS=${SLURM_CPUS_PER_TASK:-24}
export JULIA_NUM_THREADS=$NTHREADS

echo "Host=$(hostname) Job=$SLURM_JOB_ID  CPUs=$SLURM_CPUS_ON_NODE  threads=$NTHREADS"
echo "DLNR=$DLNR  ell=$ELLMIN..$ELLMAX  outname=$OUTNAME  nR=4097"
date

$JULIA --project -t $NTHREADS src/build_and_export.jl \
    --Nr=4096 --nR=4097 \
    --dlnR=$DLNR --ellmin=$ELLMIN --ellmax=$ELLMAX \
    --datadir="$DATADIR" \
    --outname="$OUTDIR/$OUTNAME" \
    --no-base

date
echo "singletier_nR4097 step2 DONE  ($OUTNAME)"
