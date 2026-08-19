#!/bin/bash
#SBATCH -J PF_5201_s1
#SBATCH -o production/logs/singletier_nR5201_step1_%j.out
#SBATCH -e production/logs/singletier_nR5201_step1_%j.err
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=400G
#SBATCH -t 24:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
MATTERPOWER=$REPO/data/planck_base_plikHM_TTTEEE_lowTEB_lensing_post_BAO_H070p6_JLA_matterpower.dat
COSMO_FUNCR=$REPO/data/cosmo_funcr.txt
OUTDIR=$REPO/production/results/test_singletier_nR5201

# nR=5201, dlnR=0.002 → R log range = 5200*0.002 = 10.40 → R ∈ [0.0055, 181]
# (similar coverage to T0 d0p005_full [0.006, 167] but with finer dlnR.
#  Tests R-tail contribution outside Option A's [0.017, 60] cleanly.)
DLNR=0.002
ELLMIN=2
ELLMAX=210
ELLMAX_MARGIN=10

cd "$REPO"
mkdir -p "$OUTDIR" production/logs
echo "Host=$(hostname) Job=$SLURM_JOB_ID DLNR=$DLNR ell=$ELLMIN..$ELLMAX margin=$ELLMAX_MARGIN nR=5201"
date

$JULIA --project -p 22 src/run_twofast.jl \
    --Nr=4096 --nR=5201 \
    --dlnR=$DLNR --ellmin=$ELLMIN --ellmax=$ELLMAX --ellmax-margin=$ELLMAX_MARGIN \
    --matterpower="$MATTERPOWER" \
    --cosmo-funcr="$COSMO_FUNCR" \
    --outdir="$OUTDIR"

date
echo "singletier_nR5201 step1 DONE"
