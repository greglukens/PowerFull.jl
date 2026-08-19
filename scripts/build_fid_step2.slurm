#!/bin/bash
#SBATCH -J pf_fid_step2
#SBATCH -p himem
#SBATCH -A duj13_cr_default
#SBATCH -c 10
#SBATCH --mem=100G
#SBATCH -t 1-00:00:00
#SBATCH -o logs/pf_fid_step2_%j.out
#SBATCH -e logs/pf_fid_step2_%j.err
# =============================================================================
# build_fid_step2.slurm
#
# Builds the FIDUCIAL Step-2 (cosmo table + ClGR_output_meta.h5) for one PK_MODE.
# The array driver run_step2_array_all.slurm builds only the 18 SHIFTED scenarios
# (fid is excluded by design), so this fills that gap. Single task (no array).
# Regenerates the fid cosmo_funcr.txt with the FIXED generator, then builds
# Step 2, reusing fid_<mode>/step1.
#
# Submit once per PK_MODE:
#   sbatch                              build_fid_step2.slurm   # wiggle
#   sbatch --export=ALL,PK_MODE=nobao   build_fid_step2.slurm   # nobao
#
# Run this BEFORE any Step-3 (run_cont_fid_vars.slurm) — Step 3 needs the meta.
# =============================================================================

module purge
module load julia/1.11.2

export REPO=/storage/group/duj13/default/PowerFull.jl
export OUTROOT=/storage/home/gql5196/scratch/powerfull_sweep
export POWER_SPEC_DIR=/storage/home/gql5196/work/tamred/data/power_spec

# wiggle or nobao:
export PK_MODE=${PK_MODE:-wiggle}

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_PKG_PRECOMPILE_AUTO=0
export PYTHON=/storage/icds/RISE/sw8/anaconda/anaconda3/bin/python

cd ${REPO}
mkdir -p logs

echo "PK_MODE=${PK_MODE}"
echo "OUTROOT=${OUTROOT}"
echo "JULIA_NUM_THREADS=${JULIA_NUM_THREADS}"
date

julia --project=. -t ${JULIA_NUM_THREADS} scripts/build_fid_step2.jl

date
echo "DONE pf_fid_step2 PK_MODE=${PK_MODE}"

