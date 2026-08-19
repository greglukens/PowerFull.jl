#!/bin/bash
#SBATCH -J PF_dlnRtest
#SBATCH -o production/logs/dlnRtest_step1_%j.out
#SBATCH -e production/logs/dlnRtest_step1_%j.err
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -t 04:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
# Tests use planck_base_plikHM (= raw Planck 2018, matches PkFull/Lucas).
# Main paper results use astropy_planck_2018 — separate run.
MATTERPOWER=$REPO/data/planck_base_plikHM_TTTEEE_lowTEB_lensing_post_BAO_H070p6_JLA_matterpower.dat
COSMO_FUNCR=$REPO/data/cosmo_funcr.txt
OUTDIR=$REPO/production/results/test_dlnR_jump

# Required env: DLNR, ELLMIN, ELLMAX, ELLMAX_MARGIN
: "${DLNR:?need DLNR}"
: "${ELLMIN:?need ELLMIN}"
: "${ELLMAX:?need ELLMAX}"
: "${ELLMAX_MARGIN:?need ELLMAX_MARGIN}"

cd "$REPO"
mkdir -p "$OUTDIR" production/logs
echo "Host=$(hostname) Job=$SLURM_JOB_ID DLNR=$DLNR ell=$ELLMIN..$ELLMAX margin=$ELLMAX_MARGIN CPUs=$SLURM_CPUS_ON_NODE"
date

$JULIA --project -p 22 src/run_twofast.jl \
    --Nr=4096 --nR=2049 \
    --dlnR=$DLNR --ellmin=$ELLMIN --ellmax=$ELLMAX --ellmax-margin=$ELLMAX_MARGIN \
    --matterpower="$MATTERPOWER" \
    --cosmo-funcr="$COSMO_FUNCR" \
    --outdir="$OUTDIR"

date
echo "dlnRtest step1 DONE  (DLNR=$DLNR)"
