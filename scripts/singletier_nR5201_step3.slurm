#!/bin/bash
#SBATCH -J PF_5201_s3
#SBATCH -o production/logs/singletier_nR5201_step3_%j.out
#SBATCH -e production/logs/singletier_nR5201_step3_%j.err
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -t 03:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
BUILDS=$REPO/production/results/test_singletier_nR5201/builds
CLOUT=$REPO/production/results/test_singletier_nR5201/cl
COSMO_FUNCR=$REPO/data/cosmo_funcr.txt

cd "$REPO"
mkdir -p "$CLOUT" production/logs
echo "Host=$(hostname)  Job=$SLURM_JOB_ID"
date

NAME=nR5201
META=$BUILDS/ClGR_${NAME}_meta.h5
OUT=$CLOUT/Cl_${NAME}.h5

[ -f "$META" ] || { echo "ERROR: $META missing"; exit 1; }

$JULIA --project -t 4,1 src/compute_ClGR.jl \
    "$META" "$OUT" \
    --tracer-list=examples/tracer_list_prod.txt \
    --pairs-file=examples/pairs_hierarchy.txt \
    --cosmo-funcr="$COSMO_FUNCR" \
    --variant=full --fNL=5

date
echo "singletier_nR5201 step3 DONE"
