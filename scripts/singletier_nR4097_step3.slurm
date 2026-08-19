#!/bin/bash
# Option A Cl assembly: nR=4097 single-tier candidate.
#SBATCH -J PF_4097_s3
#SBATCH -o production/logs/singletier_nR4097_step3_%j.out
#SBATCH -e production/logs/singletier_nR4097_step3_%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -t 03:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
BUILDS=$REPO/production/results/test_singletier_nR4097/builds
CLOUT=$REPO/production/results/test_singletier_nR4097/cl
COSMO_FUNCR=$REPO/data/cosmo_funcr.txt

cd "$REPO"
mkdir -p "$CLOUT" production/logs
echo "Host=$(hostname)  Job=$SLURM_JOB_ID  CPUs=$SLURM_CPUS_ON_NODE"
date

NAME=nR4097
META=$BUILDS/ClGR_${NAME}_meta.h5
OUT=$CLOUT/Cl_${NAME}.h5

if [ ! -f "$META" ]; then
    echo "ERROR: $META missing — step2 not finished?"
    exit 1
fi

echo "=== $NAME ==="
$JULIA --project -t 4,1 src/compute_ClGR.jl \
    "$META" "$OUT" \
    --tracer-list=examples/tracer_list_prod.txt \
    --pairs-file=examples/pairs_hierarchy.txt \
    --cosmo-funcr="$COSMO_FUNCR" \
    --variant=full --fNL=5

date
echo "singletier_nR4097 step3 DONE"
