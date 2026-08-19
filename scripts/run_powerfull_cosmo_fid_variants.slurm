#!/bin/bash
#SBATCH -J pf_fidvar
#SBATCH -p himem
#SBATCH -A duj13_cr_default
#SBATCH -c 10
#SBATCH --mem=100G
#SBATCH --array=0-6
#SBATCH -t 1-00:00:00
#SBATCH -o logs/pf_fidvar_%A_%a.out
#SBATCH -e logs/pf_fidvar_%A_%a.err

module purge
module load julia/1.11.2

# ============================================================
# Environment (SET EVERYTHING HERE)
# ============================================================

export REPO=/storage/group/duj13/default/PowerFull.jl
export OUTROOT=/storage/home/gql5196/scratch/powerfull_sweep
export POWER_SPEC_DIR=/storage/home/gql5196/work/tamred/data/power_spec
export TRACER_LIST=${REPO}/examples/tracer_list_prod.txt
export PAIRS_FILE=${REPO}/examples/pairs_full.txt

# wiggle or nobao. Override at submit time if desired:
#   sbatch --export=ALL,PK_MODE=nobao scripts/run_powerfull_cosmo_sweep.slurm
export PK_MODE=${PK_MODE:-wiggle}

# Step 2/3 use threads. Step 1 is run separately by the tier scripts.
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Avoid threaded BLAS oversubscription.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
export PYTHON=/storage/icds/RISE/sw8/anaconda/anaconda3/bin/python

cd ${REPO}

echo "PK_MODE=${PK_MODE}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"

julia --project=. -t ${JULIA_NUM_THREADS} scripts/run_powerfull_cosmo_fid_variants.jl ${SLURM_ARRAY_TASK_ID}
