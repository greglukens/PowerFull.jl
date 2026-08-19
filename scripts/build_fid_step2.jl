#!/usr/bin/env julia
# =============================================================================
# build_fid_step2.jl — build the FIDUCIAL Step-2 (cosmo table + meta) for one
# PK_MODE.  The array driver (run_powerfull_step2.jl) excludes fid by design, so
# this fills that gap.  Mirrors build_step2_if_needed exactly: same NR, TIERS,
# --datadir, --outname, --cosmo-funcr, --max-size-gb.  Reuses fid_<mode>/step1.
#
#   PK_MODE=wiggle julia --project=$REPO scripts/build_fid_step2.jl
#   PK_MODE=nobao  julia --project=$REPO scripts/build_fid_step2.jl
#
# Regenerates the fid cosmo_funcr.txt with the FIXED generator (fiducial cosmo),
# then builds ClGR_output_meta.h5.  Deletes any stale table/meta first so the
# isfile guards can't hand back broken files.
# =============================================================================

const REPO    = get(ENV, "REPO", "/storage/group/duj13/default/PowerFull.jl")
const OUTROOT = get(ENV, "OUTROOT", "/storage/home/gql5196/scratch/powerfull_sweep")
const POWER_SPEC_DIR = get(ENV, "POWER_SPEC_DIR",
    "/storage/home/gql5196/work/tamred/data/power_spec")
const PK_MODE = get(ENV, "PK_MODE", "wiggle")
const NUM_THREADS = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "8"))

# fiducial cosmology
const OM_FID, OK_FID, H_FID = 0.3111, 0.0, 0.6766
const NS_FID, AS_FID, W0_FID, WA_FID = 0.9665, 0.0, -1.0, 0.0
const H0_FID = 100.0 * H_FID

# must match run_powerfull_step2.jl exactly
const NR = 4096
const TIERS = [
    (nR=4097, dlnR=0.002,  ellmin=2,   ellmax=50,  cache_ellmax=60),
    (nR=2049, dlnR=0.001,  ellmin=51,  ellmax=200, cache_ellmax=210),
    (nR=2049, dlnR=0.0005, ellmin=201, ellmax=500, cache_ellmax=510),
]

run_cmd(c) = (println("\n>> ", c); flush(stdout); run(c))

pk_suffix = PK_MODE == "wiggle" ? "_v2.dat" : "_noBAO_v2.dat"
pk = joinpath(POWER_SPEC_DIR, "planck_2018_cosmology_power_hires_fid$(pk_suffix)")
isfile(pk) || error("Missing fiducial P(k): $pk")

outdir = joinpath(OUTROOT, "fid_$(PK_MODE)")
mkpath(outdir)
cosmo_funcr = joinpath(outdir, "cosmo_funcr.txt")
meta_file   = joinpath(outdir, "ClGR_output_meta.h5")
datadir     = joinpath(outdir, "step1")

isdir(datadir) || error("Missing fid Step-1 cache: $datadir (Step 1 is reused; must exist).")

# back up & clear stale derived files so isfile-guards can't return broken ones
if isfile(cosmo_funcr)
    bak = cosmo_funcr * ".OLD"
    println(">> backing up old table -> $bak"); mv(cosmo_funcr, bak; force=true)
end
for f in [meta_file; filter(p->occursin("ClGR_output_part_", p), readdir(outdir; join=true))]
    isfile(f) && (println(">> rm stale $f"); rm(f; force=true))
end

println("="^60)
println("FIDUCIAL Step 2  PK_MODE=$PK_MODE")
println("  outdir = $outdir")
println("  P(k)   = $pk")
println("="^60)

# 1) fiducial cosmo table (fixed generator honors these flags)
run_cmd(Cmd([
    "julia", "--project=$REPO", "-t", string(NUM_THREADS),
    joinpath(REPO, "scripts", "make_powerfull_cosmo_table.jl"),
    outdir, "--matterpower=$pk",
    "--Om=$OM_FID", "--Ok=$OK_FID", "--h=$H_FID", "--H0=$H0_FID",
    "--ns=$NS_FID", "--as=$AS_FID", "--w0=$W0_FID", "--wa=$WA_FID",
]))
isfile(cosmo_funcr) || error("fid cosmo table not created: $cosmo_funcr")

# 2) fiducial step2 — identical args to build_step2_if_needed
build_args = [
    "julia", "--project=$REPO", "-t", string(NUM_THREADS),
    joinpath(REPO, "src", "build_and_export.jl"),
    "--Nr=$NR",
    "--datadir=$datadir",
    "--outname=$(joinpath(outdir, "ClGR_output"))",
    "--cosmo-funcr=$cosmo_funcr",
    "--max-size-gb=5.0",
]
for t in TIERS
    push!(build_args, "--tier=$(t.dlnR),$(t.ellmin),$(t.ellmax),$(t.nR)")
end
run_cmd(Cmd(build_args))
isfile(meta_file) || error("fid Step 2 finished but meta missing: $meta_file")

println("\nDONE fiducial Step 2: $meta_file")
