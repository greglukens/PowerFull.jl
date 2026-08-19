#!/bin/bash
#SBATCH -J PF_s1_all
#SBATCH -o logs/step1_all_%A_%a.o
#SBATCH -e logs/step1_all_%A_%a.e
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=120G
#SBATCH -t 24:00:00
#SBATCH -p standard
#SBATCH -A duj13_cr_default
#SBATCH --array=0-17
# =============================================================================
# Step-1 tier builder for ALL scenarios:
#   scenario_id 0..17 = {Om,Ok,w0,wa,ns,as,h,omega_ch2,omega_bh2} x {p,n}
# Run this THREE times, once per tier, via --export STEP1_TIER=low|mid|high
# (or edit the export below). Mirror the mem/partition of your existing
# low/mid/high jobs: low uses standard/90G, mid+high use himem/300G.
# =============================================================================
set -euo pipefail
module purge
module load julia/1.11.2

REPO="${REPO:-/storage/group/duj13/default/PowerFull.jl}"
OUTROOT="${OUTROOT:-/storage/home/gql5196/scratch/powerfull_sweep}"
POWER_SPEC_DIR="${POWER_SPEC_DIR:-/storage/home/gql5196/work/tamred/data/power_spec}"

export PK_MODE="${PK_MODE:-wiggle}"
export STEP1_KIND=array
export STEP1_TIER="${STEP1_TIER:-low}"     # set low|mid|high at submit time
export STEP1_WORKERS="${STEP1_WORKERS:-6}"

export REPO OUTROOT POWER_SPEC_DIR
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
export PYTHON=/storage/icds/RISE/sw8/anaconda/anaconda3/bin/python

cd "$REPO"
mkdir -p production/logs "$OUTROOT"

echo "Host $(hostname) Job ${SLURM_JOB_ID} tier=${STEP1_TIER} task=${SLURM_ARRAY_TASK_ID} PK_MODE=${PK_MODE}"
date
julia --project=. -t 1 scripts/run_powerfull_step1_tier.jl ${SLURM_ARRAY_TASK_ID}
date
echo "DONE PF_s1_all tier=${STEP1_TIER} task=${SLURM_ARRAY_TASK_ID}"

