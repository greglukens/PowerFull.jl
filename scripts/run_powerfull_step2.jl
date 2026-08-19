#!/usr/bin/env julia

using Printf
using Base.Threads
using Dates
using Sockets

# ============================================================
# STEP-2-ONLY array driver for shifted cosmology scenarios.
#
#   julia run_powerfull_cosmo_step2.jl TASK_ID
#
# where TASK_ID = SCENARIO index (0-based) into SCENARIOS below.
# ONE array task == ONE scenario == ONE ClGR_output_meta.h5 build.
#
# This script:
#   - verifies the scenario-local step1/ cache is complete,
#   - builds the cosmology table (Step 0) if missing,
#   - runs src/build_and_export.jl to produce Step 2
#       ClGR_output_meta.h5 + ClGR_output_part_*.h5,
#   - then STOPS. It does NOT compute any Cl_* variants (Step 3).
#
# Run Step 3 afterwards with run_powerfull_cosmo_array_variants.jl
# (the Step-3-only variant array).
#
# mu0 / Sigma0 are intentionally NOT in SCENARIOS: they reuse the
# fiducial Step 2 and are handled elsewhere.
# ============================================================

const REPO           = get(ENV, "REPO", "/storage/group/duj13/default/PowerFull.jl")
const OUTROOT        = ENV["OUTROOT"]
const POWER_SPEC_DIR = ENV["POWER_SPEC_DIR"]
const PK_MODE        = ENV["PK_MODE"]   # wiggle or nobao

if !(PK_MODE in ["wiggle", "nobao"])
    error("run_powerfull_cosmo_step2.jl expects PK_MODE=wiggle or nobao. Got: $PK_MODE")
end

# ============================================================
# Fiducials / shifts  (must match the P(k) generator and Fisher)
# ============================================================

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
# The stored CAMB P(k) files were generated with CAMB's default massive-neutrino
# sector while holding that sector fixed during h / omega_bh2 / omega_ch2
# shifts. OMH2_OTHER_FID also absorbs the tiny closure difference caused by
# rounded Planck fiducials, so the unshifted background remains exactly
# Omega_m = OM_FID.
const OMH2_OTHER_FID = parse(
    Float64,
    get(
        ENV,
        "OMH2_OTHER_FID",
        string(OM_FID * H_FID^2 - OBH2_FID - OCH2_FID),
    ),
)

const OMH2_TOTAL_FID = OBH2_FID + OCH2_FID + OMH2_OTHER_FID

if !(OMH2_OTHER_FID >= 0.0)
    error("OMH2_OTHER_FID must be non-negative; got $OMH2_OTHER_FID")
end

if !isapprox(OMH2_TOTAL_FID / H_FID^2, OM_FID; rtol=0.0, atol=1e-14)
    error(
        "Physical-density closure failure: " *
        "(omega_b+omega_c+omega_other)/h^2 = " *
        "$(OMH2_TOTAL_FID / H_FID^2), expected OM_FID=$OM_FID"
    )
end

# TOTAL Delta per parameter; each side is fid +- Delta/2 (matches P(k) + Fisher).
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

# Scenario index == SLURM array task id. 18 scenarios (no mu0/Sigma0).
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
    ("h_p",         "h",         +1),
    ("h_n",         "h",         -1),
    ("omega_ch2_p", "omega_ch2", +1),
    ("omega_ch2_n", "omega_ch2", -1),
    ("omega_bh2_p", "omega_bh2", +1),
    ("omega_bh2_n", "omega_bh2", -1),
]

# ============================================================
# Read SLURM array index -> scenario
# ============================================================

task_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
if task_id < 0 || task_id >= length(SCENARIOS)
    error("Invalid task_id=$task_id. Valid scenario IDs are 0:$(length(SCENARIOS)-1).")
end

const SELECTED      = SCENARIOS[task_id + 1]
const SCENARIO_NAME = SELECTED[1]
const SHIFT_PARAM   = SELECTED[2]
const SHIFT_SIGN    = SELECTED[3]

# ============================================================
# Run settings (mirror the production tier system)
# ============================================================

const NR = 4096

const TIERS = [
    (nR=4097, dlnR=0.002,  ellmin=2,   ellmax=50,  cache_ellmax=60),
    (nR=2049, dlnR=0.001,  ellmin=51,  ellmax=200, cache_ellmax=210),
    (nR=2049, dlnR=0.0005, ellmin=201, ellmax=500, cache_ellmax=510),
]

const ELLMIN_TOTAL = 2
const ELLMAX_TOTAL = 500
const TIER_ELL_LIST = reduce(vcat, [collect(t.ellmin:t.ellmax) for t in TIERS])
TIER_ELL_LIST == collect(ELLMIN_TOTAL:ELLMAX_TOTAL) || error("TIERS do not exactly cover ell=$(ELLMIN_TOTAL):$(ELLMAX_TOTAL)")

const NUM_THREADS = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(Threads.nthreads())))
const N_CASES = 7

# ============================================================
# Helpers
# ============================================================

function run_cmd(cmd::Cmd)
    println("\n>> ", cmd)
    flush(stdout)
    run(cmd)
end

function with_lock(f::Function, lockdir::String; sleep_s::Float64=20.0, stale_hours::Float64=36.0)
    while true
        try
            mkpath(dirname(lockdir))
            mkdir(lockdir)
            write(joinpath(lockdir, "owner.txt"), "pid=$(getpid())\ntime=$(Dates.now())\nhost=$(Sockets.gethostname())\n")
            break
        catch e
            if e isa Base.IOError
                marker = joinpath(lockdir, "owner.txt")
                if isfile(marker)
                    age_hours = (time() - stat(marker).mtime) / 3600
                    if age_hours > stale_hours
                        println(">> Removing stale lock: $lockdir")
                        rm(lockdir; recursive=true, force=true)
                        continue
                    end
                end
                println(">> Waiting for lock: $lockdir")
                flush(stdout)
                sleep(sleep_s)
            else
                rethrow(e)
            end
        end
    end
    try
        return f()
    finally
        rm(lockdir; recursive=true, force=true)
    end
end

function selected_cosmology(param::String, sign::Int)
    Om = OM_FID; Ok = OK_FID; h = H_FID; ns = NS_FID; as = AS_FID; w0 = W0_FID; wa = WA_FID

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
        Om   = (OBH2_FID + och2 + OMH2_OTHER_FID) / H_FID^2
    elseif param == "omega_bh2"
        obh2 = OBH2_FID + sign * (Δ / 2.0)
        Om   = (obh2 + OCH2_FID + OMH2_OTHER_FID) / H_FID^2
    else
        error("Unknown shifted parameter: $param")
    end

    H0 = 100.0 * h
    return (; Om=Om, Ok=Ok, h=h, H0=H0, ns=ns, as=as, w0=w0, wa=wa)
end

const COSMO = selected_cosmology(SHIFT_PARAM, SHIFT_SIGN)

function print_selected_cosmology(cosmo)
    println("Selected cosmology:")
    println("  scenario = $SCENARIO_NAME  (shift $SHIFT_PARAM, sign $SHIFT_SIGN, Δ=$(DELTAS[SHIFT_PARAM]))")
    println("  Om=$(cosmo.Om)  Ok=$(cosmo.Ok)  h=$(cosmo.h)  H0=$(cosmo.H0)")
    println("  ns=$(cosmo.ns)  as=$(cosmo.as)  w0=$(cosmo.w0)  wa=$(cosmo.wa)")
end

function cosmo_cli_args(cosmo)
    return [
        "--Om=$(cosmo.Om)",
        "--Ok=$(cosmo.Ok)",
        "--h=$(cosmo.h)",
        "--H0=$(cosmo.H0)",
        "--ns=$(cosmo.ns)",
        "--as=$(cosmo.as)",
        "--w0=$(cosmo.w0)",
        "--wa=$(cosmo.wa)",
    ]
end

function cosmo_stamp(cosmo)
    return join([
        "scenario=$SCENARIO_NAME",
        "shift_param=$SHIFT_PARAM",
        "shift_sign=$SHIFT_SIGN",
        "shift_delta=$(DELTAS[SHIFT_PARAM])",
        "Om=$(cosmo.Om)",
        "Ok=$(cosmo.Ok)",
        "h=$(cosmo.h)",
        "H0=$(cosmo.H0)",
        "ns=$(cosmo.ns)",
        "as=$(cosmo.as)",
        "w0=$(cosmo.w0)",
        "wa=$(cosmo.wa)",
        "omh2_other=$(OMH2_OTHER_FID)",
        "omh2_total_fid=$(OMH2_TOTAL_FID)",
    ], "\n") * "\n"
end

function shifted_pk(mode::String, scenario::String)
    suffix = mode == "wiggle" ? "v2.dat" : "_noBAOv2.dat"
    p = joinpath(POWER_SPEC_DIR, "planck_2018_cosmology_power_hires_$(scenario)$(suffix)")
    isfile(p) || error("Missing shifted P(k): $p")
    return p
end

step1_datadir(outdir::String) = joinpath(outdir, "step1")

function step1_filename(tier, case::Int)
    return @sprintf(
        "TwoFAST_w_integrand_nr=%d_nR=%d_dlnR=%g_ell=%d-%d_%d.jld2",
        NR, tier.nR, tier.dlnR, tier.ellmin, tier.ellmax, case
    )
end

function check_step1_complete!(datadir::String)
    if !isdir(datadir)
        error("Missing Step 1 directory: $datadir. Run the Step-1 tier SLURM jobs first.")
    end
    missing = String[]
    for tier in TIERS, case in 1:N_CASES
        f = joinpath(datadir, step1_filename(tier, case))
        isfile(f) || push!(missing, f)
    end
    if !isempty(missing)
        preview = join(missing[1:min(end, 20)], "\n  ")
        extra = length(missing) > 20 ? "\n  ... and $(length(missing)-20) more" : ""
        error("Step 1 incomplete in $datadir; missing:\n  $preview$extra")
    end
    println(">> Step 1 cache OK: $datadir")
    return datadir
end

function write_stamp_if_needed!(outdir::String, cosmo)
    stamp_path = joinpath(outdir, "selected_cosmology.txt")
    new_stamp  = cosmo_stamp(cosmo)
    if !isfile(stamp_path)
        write(stamp_path, new_stamp)
    elseif read(stamp_path, String) != new_stamp
        println(">> NOTE: existing selected_cosmology.txt differs; overwriting with current stamp.")
        write(stamp_path, new_stamp)
    end
end

function make_cosmo_table_if_needed(outdir::String, pk::String, cosmo)
    cosmo_funcr = joinpath(outdir, "cosmo_funcr.txt")
    if isfile(cosmo_funcr)
        println(">> Cosmology table already exists: $cosmo_funcr")
        return cosmo_funcr
    end
    args = ["julia", "--project=$(REPO)", "-t", string(NUM_THREADS),
            joinpath(REPO, "scripts", "make_powerfull_cosmo_table.jl"),
            outdir, "--matterpower=$(pk)"]
    append!(args, cosmo_cli_args(cosmo))
    println(">> Building cosmology table")
    run_cmd(Cmd(args))
    isfile(cosmo_funcr) || error("Cosmology table was not created: $cosmo_funcr")
    return cosmo_funcr
end

function build_step2_if_needed(outdir::String, pk::String, cosmo_funcr::String)
    meta_file = joinpath(outdir, "ClGR_output_meta.h5")
    if isfile(meta_file)
        println(">> Step 2 already exists; skipping build: $meta_file")
        return
    end

    datadir = check_step1_complete!(step1_datadir(outdir))

    build_args = [
        "julia", "--project=$(REPO)", "-t", string(NUM_THREADS),
        joinpath(REPO, "src", "build_and_export.jl"),
        "--Nr=$(NR)",
        "--datadir=$(datadir)",
        "--outname=$(joinpath(outdir, "ClGR_output"))",
        "--cosmo-funcr=$(cosmo_funcr)",
        "--max-size-gb=5.0",
    ]
    # Per-tier nR: --tier=dlnR,ellmin,ellmax,nR
    for tier in TIERS
        push!(build_args, "--tier=$(tier.dlnR),$(tier.ellmin),$(tier.ellmax),$(tier.nR)")
    end
    run_cmd(Cmd(build_args))

    isfile(meta_file) || error("Step 2 finished but meta file was not created: $meta_file")
end

# ============================================================
# Main
# ============================================================

println("======================================================")
println("STEP 2 build :: scenario=$SCENARIO_NAME  PK_MODE=$PK_MODE  task=$task_id")
print_selected_cosmology(COSMO)
println("======================================================")
flush(stdout)

pk = shifted_pk(PK_MODE, SCENARIO_NAME)
outdir = joinpath(OUTROOT, "$(SCENARIO_NAME)_$(PK_MODE)")
mkpath(outdir)

println("Output dir : $outdir")
println("P(k)       : $pk")

write_stamp_if_needed!(outdir, COSMO)
cosmo_funcr = make_cosmo_table_if_needed(outdir, pk, COSMO)
build_step2_if_needed(outdir, pk, cosmo_funcr)

println("DONE STEP 2: $(SCENARIO_NAME)_$(PK_MODE)")
flush(stdout)

