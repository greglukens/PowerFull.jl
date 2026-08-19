#!/bin/bash
#SBATCH -J PF_4097_s1
#SBATCH -o production/logs/singletier_nR4097_step1_%j.out
#SBATCH -e production/logs/singletier_nR4097_step1_%j.err
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -t 12:00:00

set -euo pipefail

REPO="${REPO:-/gpfs/djeong/PowerFull.jl}"
JULIA=~/.juliaup/bin/julia
# Tests use planck_base_plikHM (= raw Planck 2018, matches PkFull/Lucas).
MATTERPOWER=$REPO/data/planck_base_plikHM_TTTEEE_lowTEB_lensing_post_BAO_H070p6_JLA_matterpower.dat
COSMO_FUNCR=$REPO/data/cosmo_funcr.txt
OUTDIR=$REPO/production/results/test_singletier_nR4097

# Single-tier candidate config:
# nR=4097, dlnR=0.002 → R log range = 4096*0.002 = 8.192 → R ∈ [0.0166, 60.34]
# (T0 wide R AND Mid fine dlnR; eliminates T0/Mid boundary kink)
DLNR=0.002
ELLMIN=2
ELLMAX=210
ELLMAX_MARGIN=10

cd "$REPO"
mkdir -p "$OUTDIR" production/logs
echo "Host=$(hostname) Job=$SLURM_JOB_ID DLNR=$DLNR ell=$ELLMIN..$ELLMAX margin=$ELLMAX_MARGIN CPUs=$SLURM_CPUS_ON_NODE"
date

$JULIA --project -p 22 src/run_twofast.jl \
    --Nr=4096 --nR=4097 \
    --dlnR=$DLNR --ellmin=$ELLMIN --ellmax=$ELLMAX --ellmax-margin=$ELLMAX_MARGIN \
    --matterpower="$MATTERPOWER" \
    --cosmo-funcr="$COSMO_FUNCR" \
    --outdir="$OUTDIR"

date
echo "singletier_nR4097 step1 DONE"
