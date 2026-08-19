#!/bin/bash
# dlnRtest step2b: append base/* arrays to the 10 existing builds in-place.
# Required because step2 was run with --no-base, so bin-to-bin Cl assembly
# fails (compute_ClGR reads base/w_0_0_0).
#SBATCH -J PF_dlnRbase
#SBATCH -o production/logs/dlnRtest_step2b_%j.out
#SBATCH -e production/logs/dlnRtest_step2b_%j.err
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=60G
#SBATCH -t 04:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
DATADIR=$REPO/production/results/test_dlnR_jump
OUTDIR=$REPO/production/results/test_dlnR_jump/builds

cd "$REPO"
mkdir -p production/logs

NTHREADS=${SLURM_CPUS_PER_TASK:-24}
export JULIA_NUM_THREADS=$NTHREADS

echo "Host=$(hostname)  Job=$SLURM_JOB_ID  CPUs=$SLURM_CPUS_ON_NODE  threads=$NTHREADS"
date

run_one() {
    local outname="$1"
    local meta="$OUTDIR/${outname}_meta.h5"
    local part1="$OUTDIR/${outname}_part_001.h5"
    if [ ! -f "$meta" ]; then
        echo "SKIP (no meta): $outname"
        return
    fi
    # Skip if base already populated (22 keys means base_only already ran).
    local nbase
    nbase=$($JULIA --project -e "using HDF5; n=h5open(\"$part1\",\"r\") do f; length(keys(f[\"base\"])); end; print(n)" 2>/dev/null || echo "0")
    if [ "$nbase" = "22" ]; then
        echo "SKIP (base already complete, 22 keys): $outname"
        return
    fi
    echo "=== base_only: $outname ==="
    date
    $JULIA --project -t $NTHREADS src/build_and_export.jl \
        --datadir="$DATADIR" \
        --outname="$OUTDIR/$outname" \
        --base-only
    date
    echo ""
}

# T0 set (5 builds, ell=2..30)
run_one ClGR_d0p004           # already done by test job 199899
run_one ClGR_d0p005_full
run_one ClGR_d0p005_cut
run_one ClGR_d0p006_full
run_one ClGR_d0p006_cut

# Mid set (5 builds, ell=2..210)
run_one ClGR_d0p0015
run_one ClGR_d0p002_full
run_one ClGR_d0p002_cut
run_one ClGR_d0p0025_full
run_one ClGR_d0p0025_cut

date
echo "step2b DONE"
