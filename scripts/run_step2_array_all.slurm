#!/bin/bash
#SBATCH -J PF_s2_all
#SBATCH -o logs/step2_all_%A_%a.o
#SBATCH -e logs/step2_all_%A_%a.e
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=300G
#SBATCH -t 24:00:00
#SBATCH -p himem
#SBATCH -A duj13_cr_default
#SBATCH --array=0-17
# =============================================================================
# STEP-2-ONLY array: builds ClGR_output_meta.h5 (+ part files) per scenario.
#   array index 0..17 = scenario id (one task per scenario, NO variant fan-out)
#   0..17 = {Om,Ok,w0,wa,ns,as,h,omega_ch2,omega_bh2} x {p,n}
#   (mu0/Sigma0 excluded; they reuse the fiducial Step 2.)
#
# Submit once per PK_MODE:
#   sbatch                              run_step2_array_all.slurm   # wiggle
#   sbatch --export=ALL,PK_MODE=nobao   run_step2_array_all.slurm   # nobao
#
# Requires Step 1 complete for every scenario+mode. Step 3 (the Cl_* variants)
# is run afterwards with run_cont_array_vars_all.slurm.
# =============================================================================
set -euo pipefail
module purge
module load julia/1.11.2

export REPO=/storage/group/duj13/default/PowerFull.jl
export OUTROOT=/storage/home/gql5196/scratch/powerfull_sweep
export POWER_SPEC_DIR=/storage/home/gql5196/work/tamred/data/power_spec
export PK_MODE=${PK_MODE:-wiggle}

# Portable multi-arch precompile cache (avoids znver4 "incompatible target" errors)
export JULIA_CPU_TARGET="${JULIA_CPU_TARGET:-generic;znver4,clone_all;znver3,clone_all;x86-64-v3,clone_all}"

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
export PYTHON=/storage/icds/RISE/sw8/anaconda/anaconda3/bin/python

cd ${REPO}
mkdir -p logs ${OUTROOT}

echo "Host $(hostname) Job ${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} PK_MODE=${PK_MODE}"
date
julia --project=. -t ${JULIA_NUM_THREADS} \
    scripts/run_powerfull_step2.jl ${SLURM_ARRAY_TASK_ID}
date
echo "DONE PF_s2_all task=${SLURM_ARRAY_TASK_ID} PK_MODE=${PK_MODE}"

