#!/bin/bash
# Lucas direct patch for w^0_{ell=2, j=2, jp=2}(r1, r2) on the (1155, 1155) physical grid.
# Uses 9-term decomp via quad_jar_jbt_log (Lucas algorithm), parallelized.
# Output: standalone HDF5 file. Can be injected into any build's base/w_0_2_2[ell=2] slice.
#SBATCH -J PF_lucas_w22
#SBATCH -o production/logs/lucas_w22_ell2_%j.out
#SBATCH -e production/logs/lucas_w22_ell2_%j.err
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=32G
#SBATCH -t 12:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
LUCAS_REPO="/gpfs/djeong/TwoFAST/Lucas"   # has PkSpectra/Splines/Quad_jar_jbt project
JULIA=~/.juliaup/bin/julia

# Use Option A's r-grid as reference (1155 points). Same grid shared by all builds.
META=$REPO/production/results/test_singletier_nR4097/builds/ClGR_nR4097_meta.h5
OUTDIR=$REPO/production/results/lucas_patches
OUTPATH=$OUTDIR/w_0_2_2_ell2_lucas.h5

cd "$LUCAS_REPO"   # use Lucas project that has PkSpectra etc.
mkdir -p "$OUTDIR" "$REPO/production/logs"

NTHREADS=${SLURM_CPUS_PER_TASK:-24}
export JULIA_NUM_THREADS=1   # Use distributed (pmap), not threads

echo "Host=$(hostname)  Job=$SLURM_JOB_ID  CPUs=$SLURM_CPUS_ON_NODE"
echo "PWD=$(pwd)  using project $(pwd)/Project.toml"
echo "Workers: $((NTHREADS - 1))"
date

$JULIA --project -p $((NTHREADS - 1)) "$REPO/scripts/lucas_w22_ell2_patch.jl" "$META" "$OUTPATH"

date
echo "lucas_w22_ell2 DONE"
