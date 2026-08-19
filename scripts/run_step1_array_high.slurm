#!/bin/bash
#SBATCH -J PF_s1_arr_high
#SBATCH -o logs/step1_array_high.o%A_%a
#SBATCH -e logs/step1_array_high.e%A_%a
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=300G
#SBATCH -t 24:00:00
#SBATCH -p himem
#SBATCH -A duj13_cr_default
#SBATCH --array=0-11

set -euo pipefail

module purge
module load julia/1.11.2

REPO="${REPO:-/storage/group/duj13/default/PowerFull.jl}"
OUTROOT="${OUTROOT:-/storage/home/gql5196/scratch/powerfull_sweep}"
POWER_SPEC_DIR="${POWER_SPEC_DIR:-/storage/home/gql5196/work/tamred/data/power_spec}"

# wiggle or nobao. Override with:
#   sbatch --export=ALL,PK_MODE=nobao <this_script>
export PK_MODE="${PK_MODE:-wiggle}"

export STEP1_KIND=array
export STEP1_TIER=high
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

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  CPUs: ${SLURM_CPUS_ON_NODE:-unknown}"
echo "STEP1_KIND=${STEP1_KIND}"
echo "STEP1_TIER=${STEP1_TIER}"
echo "PK_MODE=${PK_MODE}"
echo "STEP1_WORKERS=${STEP1_WORKERS}"
echo "OUTROOT=${OUTROOT}"
echo "POWER_SPEC_DIR=${POWER_SPEC_DIR}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
date

julia --project=. -t 1 scripts/run_powerfull_step1_tier.jl ${SLURM_ARRAY_TASK_ID}

date
echo "DONE PF_s1_arr_high"
