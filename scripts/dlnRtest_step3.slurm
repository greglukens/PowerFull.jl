#!/bin/bash
# dlnR test step3: compute bin-to-bin Cl for each of 10 builds.
# variant=full, fNL=5 (matches Cl_hierarchy_full.h5 setup).
#SBATCH -J PF_dlnRcl
#SBATCH -o production/logs/dlnRtest_step3_%j.out
#SBATCH -e production/logs/dlnRtest_step3_%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -t 03:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
BUILDS=$REPO/production/results/test_dlnR_jump/builds
CLOUT=$REPO/production/results/test_dlnR_jump/cl
# Tests use planck_base_plikHM (matches step1 P(k)); main paper = astropy_planck_2018.
COSMO_FUNCR=$REPO/data/cosmo_funcr.txt

cd "$REPO"
mkdir -p "$CLOUT" production/logs
echo "Host=$(hostname)  Job=$SLURM_JOB_ID  CPUs=$SLURM_CPUS_ON_NODE"
date

run_one() {
    local name="$1"
    local meta="$BUILDS/ClGR_${name}_meta.h5"
    local out="$CLOUT/Cl_${name}.h5"
    if [ ! -f "$meta" ]; then
        echo "SKIP: $meta missing"
        return
    fi
    echo "=== $name ==="
    date
    $JULIA --project -t 4,1 src/compute_ClGR.jl \
        "$meta" "$out" \
        --tracer-list=examples/tracer_list_prod.txt \
        --pairs-file=examples/pairs_hierarchy.txt \
        --cosmo-funcr="$COSMO_FUNCR" \
        --variant=full --fNL=5
    date
    echo ""
}

# T0 set (5 builds)
run_one d0p004
run_one d0p005_full
run_one d0p005_cut
run_one d0p006_full
run_one d0p006_cut

# Mid set (5 builds)
run_one d0p0015
run_one d0p002_full
run_one d0p002_cut
run_one d0p0025_full
run_one d0p0025_cut

date
echo "step3 DONE"
