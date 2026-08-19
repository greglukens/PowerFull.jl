#!/usr/bin/env julia

using Printf
using Base.Threads

# ============================================================
# Step-1-only tier builder for PowerFull sweeps.
#
# This builds only the scenario-specific TwoFAST files in:
#   OUTROOT/<scenario>_<PK_MODE>/step1/
#
# It reuses existing repo-level Cacheout_* Ml/F21 caches and refuses
# to run if the required Cacheout folder is missing.
#
# Control with environment:
#   STEP1_KIND = fid or array
#   STEP1_TIER = low, mid, or high
#   PK_MODE    = wiggle or nobao
#
# For STEP1_KIND=array, pass task_id as ARGS[1] or SLURM_ARRAY_TASK_ID.
# ============================================================

const REPO = get(ENV, "REPO", "/storage/group/duj13/default/PowerFull.jl")
const OUTROOT = get(ENV, "OUTROOT", "/storage/home/gql5196/scratch/powerfull_sweep")
const POWER_SPEC_DIR = get(ENV, "POWER_SPEC_DIR", "/storage/home/gql5196/work/tamred/data/power_spec")

const STEP1_KIND = get(ENV, "STEP1_KIND", "fid")
const STEP1_TIER = get(ENV, "STEP1_TIER", "low")
const PK_MODE = get(ENV, "PK_MODE", "wiggle")

STEP1_KIND in ("fid", "array") || error("STEP1_KIND must be fid or array. Got: $STEP1_KIND")
PK_MODE in ("wiggle", "nobao") || error("PK_MODE must be wiggle or nobao. Got: $PK_MODE")

# Cache policy for the shared Ml/F21 Cacheout.  run_twofast.jl builds the
# cache on the fly whenever the F21EllCache dir is absent, so by DEFAULT we
# let a missing geometry build itself.  If you want the old strict behavior
# (refuse and stop when the cache is missing — useful to guarantee a scenario
# array never triggers an expensive rebuild), set STEP1_REQUIRE_CACHE=1.
const STEP1_REQUIRE_CACHE = get(ENV, "STEP1_REQUIRE_CACHE", "0") in ("1", "true", "yes")

# Threads used only for tiny helper commands like the cosmology table.
const NUM_THREADS = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(Threads.nthreads())))

# Distributed workers used by src/run_twofast.jl.
const STEP1_WORKERS = parse(Int, get(ENV, "STEP1_WORKERS", "6"))

# Optional MlCache redirect (big speedup if I/O-bound): point MlCache at a fast
# node-local disk like /dev/shm instead of the (slow, shared) Cacheout on gpfs.
# STEP1_MLCACHE_DIR="" (default) keeps MlCache alongside Cacheout.
# STEP1_MLCACHE_CLEANUP=1 deletes each base's MlCache after use — REQUIRED when
# using /dev/shm so the RAM disk doesn't fill (MlCache is 100+ GB per geometry).
# NOTE: this only moves the transient MlCache; the reusable F21EllCache still
# lands in the shared Cacheout dir, so cross-scenario reuse is preserved.
const STEP1_MLCACHE_DIR = get(ENV, "STEP1_MLCACHE_DIR", "")
const STEP1_MLCACHE_CLEANUP = get(ENV, "STEP1_MLCACHE_CLEANUP", "0") in ("1", "true", "yes")

const NR = 4096
const N_CASES = 7

# IMPORTANT: saved ell ranges are the production ranges. cache_ellmax is the
# pre-existing buffered Cacheout ceiling in REPO.
#
# Three-tier strategy: as ℓ increases the integrand support narrows, so the
# R range contracts and the dlnR spacing refines from tier to tier.  nR is
# therefore tier-specific (the low tier needs the widest R range and the
# most points; the lensing projections carry the ℓ(ℓ+1) weight and set the
# resolution requirement, motivating finer dlnR at high ℓ).
#   low : ℓ∈[2,50]    nR=4097 dlnR=0.002  → R∈[0.0166, 60.34]
#   mid : ℓ∈[51,200]  nR=2049 dlnR=0.001  → R∈[0.36, 2.79]
#   high: ℓ∈[201,500] nR=2049 dlnR=0.0005 → R∈[0.60, 1.67] around the R=1 peak
const TIER_MAP = Dict(
    "low"  => (nR=4097, dlnR=0.002,  ellmin=2,   ellmax=50,  cache_ellmax=60),
    "mid"  => (nR=2049, dlnR=0.001,  ellmin=51,  ellmax=200, cache_ellmax=210),
    "high" => (nR=2049, dlnR=0.0005, ellmin=201, ellmax=500, cache_ellmax=510),
)
haskey(TIER_MAP, STEP1_TIER) || error("STEP1_TIER must be low, mid, or high. Got: $STEP1_TIER")
const TIER = TIER_MAP[STEP1_TIER]

# Fiducials / shifts.
const OM_FID = parse(Float64, get(ENV, "OM_FID", "0.3111"))
const OK_FID = parse(Float64, get(ENV, "OK_FID", "0.0"))
const H_FID  = parse(Float64, get(ENV, "H_FID",  "0.6766"))
const NS_FID = parse(Float64, get(ENV, "NS_FID", "0.9665"))
const AS_FID = parse(Float64, get(ENV, "AS_FID", "0.0"))
const W0_FID = parse(Float64, get(ENV, "W0_FID", "-1.0"))
const WA_FID = parse(Float64, get(ENV, "WA_FID", "0.0"))

# Physical density fiducials for the h / omega_ch2 / omega_bh2 shifts.
const OBH2_FID = parse(Float64, get(ENV, "OBH2_FID", "0.02242"))
const OCH2_FID = parse(Float64, get(ENV, "OCH2_FID", "0.11933"))

# Fixed physical matter not contained in omega_b h^2 or omega_c h^2.
#
# This retains the fiducial total Omega_m exactly while h, omega_ch2, or
# omega_bh2 are shifted. It includes the fixed massive-neutrino matter sector
# and the small closure residual from the rounded fiducial parameters.
const OMH2_OTHER_FID = parse(
    Float64,
    get(
        ENV,
        "OMH2_OTHER_FID",
        string(OM_FID * H_FID^2 - OBH2_FID - OCH2_FID),
    ),
)

const OMH2_TOTAL_FID =
    OBH2_FID +
    OCH2_FID +
    OMH2_OTHER_FID

if !(OMH2_OTHER_FID >= 0.0)
    error(
        "OMH2_OTHER_FID must be non-negative; got " *
        string(OMH2_OTHER_FID)
    )
end

if !isapprox(
    OMH2_TOTAL_FID / H_FID^2,
    OM_FID;
    rtol=0.0,
    atol=1e-14,
)
    error(
        "Physical-density closure failure: " *
        "(omega_b+omega_c+omega_other)/h^2 = " *
        string(OMH2_TOTAL_FID / H_FID^2) *
        ", expected OM_FID=" *
        string(OM_FID)
    )
end

const DELTAS = Dict(
    "Om" => parse(Float64, get(ENV, "DELTA_OM", "0.001")),
    "Ok" => parse(Float64, get(ENV, "DELTA_OK", "0.001")),
    "w0" => parse(Float64, get(ENV, "DELTA_W0", "0.005")),
    "wa" => parse(Float64, get(ENV, "DELTA_WA", "0.005")),
    "ns" => parse(Float64, get(ENV, "DELTA_NS", "0.0005")),
    "as" => parse(Float64, get(ENV, "DELTA_AS", "0.0005")),
    "h"         => parse(Float64, get(ENV, "DELTA_H",    string(0.005 * 0.6766))),
    "omega_ch2" => parse(Float64, get(ENV, "DELTA_OCH2", "0.001")),
    "omega_bh2" => parse(Float64, get(ENV, "DELTA_OBH2", "0.0001")),
)

const SCENARIOS = [
    ("Om_p", "Om", +1),
    ("Om_n", "Om", -1),
    ("Ok_p", "Ok", +1),
    ("Ok_n", "Ok", -1),
    ("w0_p", "w0", +1),
    ("w0_n", "w0", -1),
    ("wa_p", "wa", +1),
    ("wa_n", "wa", -1),
    ("ns_p", "ns", +1),
    ("ns_n", "ns", -1),
    ("as_p", "as", +1),
    ("as_n", "as", -1),
    # --- new (indices 12..17) ---
    ("h_p",         "h",         +1),
    ("h_n",         "h",         -1),
    ("omega_ch2_p", "omega_ch2", +1),
    ("omega_ch2_n", "omega_ch2", -1),
    ("omega_bh2_p", "omega_bh2", +1),
    ("omega_bh2_n", "omega_bh2", -1),
]

function run_cmd(cmd::Cmd)
    println("\n>> ", cmd)
    flush(stdout)
    run(cmd)
end

function selected_cosmology(param::Union{Nothing,String}=nothing, sign::Int=0)
    Om = OM_FID; Ok = OK_FID; h = H_FID; ns = NS_FID; as = AS_FID; w0 = W0_FID; wa = WA_FID
    if param !== nothing
        Δ = DELTAS[param]
        if param == "Om"
            Om += sign * (Δ / 2.0)
        elseif param == "Ok"
            Ok += sign * (Δ / 2.0)
        elseif param == "w0"
            w0 += sign * (Δ / 2.0)
        elseif param == "wa"
            wa += sign * (Δ / 2.0)
        elseif param == "ns"
            ns += sign * (Δ / 2.0)
        elseif param == "as"
            as += sign * (Δ / 2.0)
        elseif param == "h"
            h  = H_FID + sign * (Δ / 2.0)
            Om = (OBH2_FID + OCH2_FID + OMH2_OTHER_FID) / h^2
        elseif param == "omega_ch2"
            och2 = OCH2_FID + sign * (Δ / 2.0)
            Om = (OBH2_FID + och2 + OMH2_OTHER_FID) / H_FID^2
        elseif param == "omega_bh2"
            obh2 = OBH2_FID + sign * (Δ / 2.0)
            Om = (obh2 + OCH2_FID + OMH2_OTHER_FID) / H_FID^2
        else
            error("Unknown shifted parameter: $param")
        end
    end
    return (; Om, Ok, h, H0=100.0*h, ns, as, w0, wa)
end

function cosmo_cli_args(c)
    return [
        "--Om=$(c.Om)", "--Ok=$(c.Ok)", "--h=$(c.h)", "--H0=$(c.H0)",
        "--ns=$(c.ns)", "--as=$(c.as)", "--w0=$(c.w0)", "--wa=$(c.wa)",
    ]
end

function fid_pk(mode::String)
    suffixes = mode == "wiggle" ? ["_v2.dat"] : ["_noBAO_v2.dat", "_noBAOv2.dat"]
    candidates = [joinpath(POWER_SPEC_DIR, "planck_2018_cosmology_power_hires_fid$(s)") for s in suffixes]
    for p in candidates
        isfile(p) && return p
    end
    error("Missing fiducial P(k), mode=$mode. Tried:\n  " * join(candidates, "\n  "))
end

function shifted_pk(mode::String, scenario::String)
    suffixes = mode == "wiggle" ? ["_v2.dat", "v2.dat"] : ["_noBAO_v2.dat", "_noBAOv2.dat"]
    candidates = [joinpath(POWER_SPEC_DIR, "planck_2018_cosmology_power_hires_$(scenario)$(s)") for s in suffixes]
    for p in candidates
        isfile(p) && return p
    end
    error("Missing shifted P(k), scenario=$scenario, mode=$mode. Tried:\n  " * join(candidates, "\n  "))
end

function scenario_from_env()
    if STEP1_KIND == "fid"
        return ("fid", nothing, 0, selected_cosmology(), fid_pk(PK_MODE))
    end

    task_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
    if task_id < 0 || task_id >= length(SCENARIOS)
        error("Invalid task_id=$task_id. Valid task IDs are 0:$(length(SCENARIOS)-1).")
    end
    scen, param, sign = SCENARIOS[task_id + 1]
    return (scen, param, sign, selected_cosmology(param, sign), shifted_pk(PK_MODE, scen))
end

function cosmo_stamp(scenario::String, param, sign, c)
    return join([
        "scenario=$scenario", "pk_mode=$PK_MODE", "shift_param=$(param)", "shift_sign=$(sign)",
        "Om=$(c.Om)", "Ok=$(c.Ok)", "h=$(c.h)", "H0=$(c.H0)",
        "ns=$(c.ns)", "as=$(c.as)", "w0=$(c.w0)", "wa=$(c.wa)",
        "omh2_other=$(OMH2_OTHER_FID)",
        "omh2_total_fid=$(OMH2_TOTAL_FID)",
    ], "\n") * "\n"
end

# Extract only the PHYSICAL cosmology lines (Om/Ok/h/H0/ns/as/w0/wa) from a
# stamp, ignoring bookkeeping (scenario/pk_mode/shift_*). Two stamps differing
# only in bookkeeping describe the SAME cosmology, so a tier re-run in an
# existing folder must not be blocked by such a difference.
function _physical_cosmo_lines(stamp::String)
    ks = ("Om=", "Ok=", "h=", "H0=", "ns=", "as=", "w0=", "wa=")
    return Set(filter(l -> any(k -> startswith(l, k), ks),
                      split(strip(stamp), "\n")))
end

# Set STEP1_FORCE_STAMP=1 to overwrite the stamp even on a genuine physical
# cosmology mismatch (only if you KNOW the existing products are consistent).
const STEP1_FORCE_STAMP = get(ENV, "STEP1_FORCE_STAMP", "0") in ("1", "true", "yes")

function any_step1_products(outdir::String)
    datadir = joinpath(outdir, "step1")
    isdir(datadir) || return false
    return any(f -> startswith(f, "TwoFAST_w_integrand_") || startswith(f, "TwoFAST_output_"), readdir(datadir))
end

function verify_or_write_stamp!(outdir::String, scenario::String, param, sign, c)
    mkpath(outdir)
    stamp_path = joinpath(outdir, "selected_cosmology.txt")
    new_stamp = cosmo_stamp(scenario, param, sign, c)

    if !isfile(stamp_path)
        write(stamp_path, new_stamp)
        return
    end

    old = read(stamp_path, String)
    old == new_stamp && return   # exact match

    physical_changed = _physical_cosmo_lines(old) != _physical_cosmo_lines(new_stamp)

    if !physical_changed
        # Same cosmology, only bookkeeping differs -> refresh stamp, keep products.
        @info "Refreshing stamp (same physical cosmology, bookkeeping differs); keeping Step-1 products" outdir
        write(stamp_path, new_stamp)
        return
    end

    if any_step1_products(outdir) && !STEP1_FORCE_STAMP
        error("Physical cosmology in selected_cosmology.txt does not match this run in $outdir, " *
              "and Step-1 products already exist.\nExisting products were built for a DIFFERENT " *
              "cosmology — wipe that scenario folder before rebuilding Step 1, or set " *
              "STEP1_FORCE_STAMP=1 if you are certain the existing products are consistent.")
    end

    if STEP1_FORCE_STAMP
        @warn "STEP1_FORCE_STAMP=1: overwriting stamp despite a physical cosmology mismatch" outdir
    else
        @warn "Replacing stale selected_cosmology.txt because no Step-1 products exist yet" outdir
    end
    rm(joinpath(outdir, "cosmo_funcr.txt"), force=true)
    write(stamp_path, new_stamp)
end

function make_cosmo_table_if_needed(outdir::String, pk::String, c)
    cosmo_funcr = joinpath(outdir, "cosmo_funcr.txt")
    isfile(cosmo_funcr) && return cosmo_funcr

    lockdir = joinpath(outdir, ".cosmo_funcr.lock")
    got_lock = false
    try
        mkdir(lockdir)
        got_lock = true
    catch e
        got_lock = false
    end

    if got_lock
        try
            # Another process may have created it before we acquired the lock.
            if !isfile(cosmo_funcr)
                args = [
                    "julia", "--project=$(REPO)", "-t", string(max(1, min(NUM_THREADS, 8))),
                    joinpath(REPO, "scripts", "make_powerfull_cosmo_table.jl"),
                    outdir,
                    "--matterpower=$(pk)",
                ]
                append!(args, cosmo_cli_args(c))
                println(">> Making cosmology table: $cosmo_funcr")
                run_cmd(Cmd(args))
            end
        finally
            rm(lockdir; force=true, recursive=true)
        end
    else
        println(">> Waiting for another tier job to create $cosmo_funcr")
        waited = 0
        while !isfile(cosmo_funcr)
            sleep(10)
            waited += 10
            waited > 7200 && error("Timed out waiting for cosmology table: $cosmo_funcr")
        end
    end

    isfile(cosmo_funcr) || error("Cosmology table was not created: $cosmo_funcr")
    return cosmo_funcr
end

function step1_filename(tier, case::Int)
    return @sprintf("TwoFAST_w_integrand_nr=%d_nR=%d_dlnR=%g_ell=%d-%d_%d.jld2",
                    NR, tier.nR, tier.dlnR, tier.ellmin, tier.ellmax, case)
end

function tier_complete(datadir::String, tier)
    isdir(datadir) || return false
    for case in 1:N_CASES
        isfile(joinpath(datadir, step1_filename(tier, case))) || return false
    end
    return true
end

function repo_cache_dir(tier)
    return joinpath(REPO, "Cacheout_nR=$(tier.nR)_dlnR=$(tier.dlnR)_ellmax=$(tier.cache_ellmax)")
end

# Inspect the actual CONTENTS of a Cacheout dir, not just its existence.
# run_twofast.jl writes, per q-value (one per base), a directory
# `F21EllCache_q=<v>[ _oddprimes]` and a `MlCache_q=<v>[ _oddprimes]/MlCache.bin`.
# A bare/empty dir (left by a killed build) must NOT count as a cache hit.
#
# Returns one of:
#   :missing  — dir absent, or present but has no F21EllCache* at all
#   :partial  — has some F21EllCache* dirs but ≥1 lacks its MlCache*/MlCache.bin
#   :complete — every F21EllCache* dir has a matching non-empty MlCache.bin
function cache_status(tier)
    d = repo_cache_dir(tier)
    isdir(d) || return :missing
    entries = readdir(d)
    f21 = filter(e -> startswith(e, "F21EllCache"), entries)
    isempty(f21) && return :missing
    for f in f21
        # F21EllCache_q=1.3  ->  MlCache_q=1.3   (same suffix after the prefix)
        suffix = f[length("F21EllCache")+1:end]
        mlbin = joinpath(d, "MlCache" * suffix, "MlCache.bin")
        (isfile(mlbin) && filesize(mlbin) > 0) || return :partial
    end
    return :complete
end

function check_repo_cacheout!(tier)
    d = repo_cache_dir(tier)
    st = cache_status(tier)
    if st === :complete
        println(">> Found complete repo Cacheout: $d")
        return
    end
    if STEP1_REQUIRE_CACHE
        error("Repo Cacheout is $(st) (not complete): $d\n" *
              "STEP1_REQUIRE_CACHE=1 is set, so this job refuses to build it.\n" *
              "Unset STEP1_REQUIRE_CACHE to let run_twofast.jl build/finish it on this run.")
    end
    if st === :partial
        @warn "Repo Cacheout is PARTIAL; run_twofast.jl will fill in the missing q-caches (reuses what's present)." dir=d
    else
        @warn "Repo Cacheout missing/empty; run_twofast.jl will BUILD it on this run (slow, esp. at nR=4097)." dir=d
    end
    println(">> Will build/finish repo Cacheout: $d  (status: $st)")
end

function build_tier!(outdir::String, pk::String, cosmo_funcr::String, tier)
    datadir = joinpath(outdir, "step1")
    mkpath(datadir)

    if tier_complete(datadir, tier)
        println(">> Step 1 tier already complete: $STEP1_TIER in $datadir")
        return
    end

    check_repo_cacheout!(tier)

    println(">> Building Step 1 tier=$STEP1_TIER")
    println(">> Output step1 dir: $datadir")
    println(">> P(k): $pk")
    println(">> cosmo_funcr: $cosmo_funcr")
    println(">> saved ell range: $(tier.ellmin):$(tier.ellmax)")
    println(">> nR / dlnR: $(tier.nR) / $(tier.dlnR)")
    println(">> cache ellmax: $(tier.cache_ellmax)")
    println(">> STEP1_WORKERS: $STEP1_WORKERS")
    if !isempty(STEP1_MLCACHE_DIR)
        println(">> MlCache dir: $STEP1_MLCACHE_DIR  (cleanup: $STEP1_MLCACHE_CLEANUP)")
    end

    twofast_args = String[
        "julia", "--project=$(REPO)", "-t", "1", "-p", string(STEP1_WORKERS),
        joinpath(REPO, "src", "run_twofast.jl"),
        "--Nr=$(NR)",
        "--nR=$(tier.nR)",
        "--dlnR=$(tier.dlnR)",
        "--ellmin=$(tier.ellmin)",
        "--ellmax=$(tier.ellmax)",
        "--ellmax-margin=$(tier.cache_ellmax)",
        "--outdir=$(datadir)",
        "--matterpower=$(pk)",
        "--cosmo-funcr=$(cosmo_funcr)",
    ]
    if !isempty(STEP1_MLCACHE_DIR)
        push!(twofast_args, "--mlcache-dir=$(STEP1_MLCACHE_DIR)")
    end
    if STEP1_MLCACHE_CLEANUP
        push!(twofast_args, "--mlcache-cleanup")
    end

    run_cmd(Cmd(Cmd(twofast_args); dir=REPO))

    tier_complete(datadir, tier) || error("Step 1 tier finished but required w-integrand files are incomplete: $datadir, tier=$STEP1_TIER")
end

# ============================================================
# Main
# ============================================================

scenario, param, sign, cosmo, pk = scenario_from_env()
outdir = joinpath(OUTROOT, "$(scenario)_$(PK_MODE)")
mkpath(outdir)

println("======================================================")
println("PowerFull Step-1-only tier builder")
println("  STEP1_KIND   = $STEP1_KIND")
println("  scenario     = $scenario")
println("  PK_MODE      = $PK_MODE")
println("  STEP1_TIER   = $STEP1_TIER")
println("  REQUIRE_CACHE= $STEP1_REQUIRE_CACHE")
println("  OUTDIR       = $outdir")
println("  REPO         = $REPO")
println("  workers      = $STEP1_WORKERS")
println("  Ωm Ok h      = $(cosmo.Om), $(cosmo.Ok), $(cosmo.h)")
println("  ns as w0 wa  = $(cosmo.ns), $(cosmo.as), $(cosmo.w0), $(cosmo.wa)")
println("======================================================")
flush(stdout)

verify_or_write_stamp!(outdir, scenario, param, sign, cosmo)
cosmo_funcr = make_cosmo_table_if_needed(outdir, pk, cosmo)
build_tier!(outdir, pk, cosmo_funcr, TIER)

println("DONE Step 1 tier=$STEP1_TIER for $(scenario)_$(PK_MODE)")
flush(stdout)

