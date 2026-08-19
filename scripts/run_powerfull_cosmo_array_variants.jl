#!/usr/bin/env julia

using Printf
using Base.Threads
using Dates
using Sockets

# ============================================================
# Array driver for shifted cosmology scenarios.
# Usage:
#   julia run_powerfull_cosmo_array.jl TASK_ID
# where TASK_ID indexes SCENARIOS below, e.g. SLURM_ARRAY_TASK_ID.
# ============================================================

# ============================================================
# Paths / environment
# ============================================================

const REPO = get(ENV, "REPO", "/storage/group/duj13/default/PowerFull.jl")

const OUTROOT        = ENV["OUTROOT"]
const POWER_SPEC_DIR = ENV["POWER_SPEC_DIR"]
const TRACER_LIST    = ENV["TRACER_LIST"]
const PAIRS_FILE     = ENV["PAIRS_FILE"]
const PK_MODE        = ENV["PK_MODE"]   # wiggle or nobao for this array task

if !(PK_MODE in ["wiggle", "nobao"])
    error("run_powerfull_cosmo_array.jl expects PK_MODE=wiggle or nobao. Got: $PK_MODE")
end

# ============================================================
# Fiducials / shifts
# These are the source values used to construct the selected shifted cosmology.
# Override with ENV if desired.
# ============================================================

const OM_FID = parse(Float64, get(ENV, "OM_FID", "0.3111"))
const OK_FID = parse(Float64, get(ENV, "OK_FID", "0.0"))
const H_FID  = parse(Float64, get(ENV, "H_FID",  "0.6766"))
const NS_FID = parse(Float64, get(ENV, "NS_FID", "0.9665"))
const AS_FID = parse(Float64, get(ENV, "AS_FID", "0.0"))
const W0_FID = parse(Float64, get(ENV, "W0_FID", "-1.0"))
const WA_FID = parse(Float64, get(ENV, "WA_FID", "0.0"))

# Physical density fiducials, needed for the h / omega_ch2 / omega_bh2 shifts.
# These are the same values used by build_dewiggled_spectra_h_ocdm_obar.py.
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

const DELTAS = Dict(
    "Om" => parse(Float64, get(ENV, "DELTA_OM", "0.001")),
    "Ok" => parse(Float64, get(ENV, "DELTA_OK", "0.001")),
    "w0" => parse(Float64, get(ENV, "DELTA_W0", "0.005")),
    "wa" => parse(Float64, get(ENV, "DELTA_WA", "0.005")),
    "ns" => parse(Float64, get(ENV, "DELTA_NS", "0.0005")),
    "as" => parse(Float64, get(ENV, "DELTA_AS", "0.0005")),
    # New: TOTAL Delta, split as +-Delta/2 (matches the P(k) generator).
    "h"         => parse(Float64, get(ENV, "DELTA_H",    string(0.005 * 0.6766))),
    "omega_ch2" => parse(Float64, get(ENV, "DELTA_OCH2", "0.001")),
    "omega_bh2" => parse(Float64, get(ENV, "DELTA_OBH2", "0.0001")),
)

# Task IDs are 0-based.
# Each scenario is one finite-difference side for one parameter.
# NEW scenarios (h, omega_ch2, omega_bh2) are appended at the BACK so the
# existing task-id -> scenario mapping for indices 0..11 is unchanged.
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

# ============================================================
# Variants: one SLURM array task computes exactly one output file.
# ============================================================

const CL_VARIANTS = [
    ("gaussian",     "Cl_f0.h5"),
    ("fi",           "Cl_fi.h5"),
    ("ff",           "Cl_ff.h5"),
    ("fi_kaiser",    "Cl_fi_kaiser.h5"),
    ("fi_newtonian", "Cl_fi_newtonian.h5"),
    ("newtonian",    "Cl_newtonian.h5"),
    ("kaiser",       "Cl_kaiser.h5"),
]
const N_VARIANTS = length(CL_VARIANTS)

# ============================================================
# Read flattened SLURM array index
# ============================================================

task_id = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
const N_TASKS = length(SCENARIOS) * N_VARIANTS
if task_id < 0 || task_id >= N_TASKS
    error("Invalid task_id=$task_id. Valid task IDs are 0:$(N_TASKS-1).")
end

const SCENARIO_ID = div(task_id, N_VARIANTS)
const VARIANT_ID  = mod(task_id, N_VARIANTS)
const SELECTED_SCENARIO = SCENARIOS[SCENARIO_ID + 1]
const SCENARIO_NAME = SELECTED_SCENARIO[1]
const SHIFT_PARAM = SELECTED_SCENARIO[2]
const SHIFT_SIGN = SELECTED_SCENARIO[3]
const SELECTED_VARIANT = CL_VARIANTS[VARIANT_ID + 1]
const VARIANT_NAME = SELECTED_VARIANT[1]
const VARIANT_OUTFILE = SELECTED_VARIANT[2]

# ============================================================
# Run settings
# ============================================================

const NR = 4096

const TIERS = [
    # Current Step-1/Step-2 tier system, matching the non-variant drivers.
    # These cover ell=2:500 exactly and use per-tier nR.
    (nR=4097, dlnR=0.002,  ellmin=2,   ellmax=50,  cache_ellmax=60),
    (nR=2049, dlnR=0.001,  ellmin=51,  ellmax=200, cache_ellmax=210),
    (nR=2049, dlnR=0.0005, ellmin=201, ellmax=500, cache_ellmax=510),
]

const ELLMIN_TOTAL = 2
const ELLMAX_TOTAL = 500
const TIER_ELL_LIST = reduce(vcat, [collect(t.ellmin:t.ellmax) for t in TIERS])
TIER_ELL_LIST == collect(ELLMIN_TOTAL:ELLMAX_TOTAL) || error("TIERS do not exactly cover ell=$(ELLMIN_TOTAL):$(ELLMAX_TOTAL)")



# Number of Julia threads to give each launched command.
const NUM_THREADS = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(Threads.nthreads())))

# Step 1 / TwoFAST uses distributed workers (-p), not threads.
const STEP1_WORKERS = parse(Int, get(ENV, "STEP1_WORKERS", "22"))

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
    Om = OM_FID
    Ok = OK_FID
    h  = H_FID
    ns = NS_FID
    as = AS_FID
    w0 = W0_FID
    wa = WA_FID

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
        # Shift h at fixed physical densities; H0 and derived Om both move.
        h  = H_FID + sign * (Δ / 2.0)
        Om = (OBH2_FID + OCH2_FID + OMH2_OTHER_FID) / h^2
    elseif param == "omega_ch2"
        # Shift CDM density at fixed h; only derived Om moves.
        och2 = OCH2_FID + sign * (Δ / 2.0)
        Om   = (OBH2_FID + och2 + OMH2_OTHER_FID) / H_FID^2
    elseif param == "omega_bh2"
        # Shift baryon density at fixed h; only derived Om moves.
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
    println("Selected cosmology used everywhere:")
    println("  scenario = $SCENARIO_NAME")
    println("  shifted  = $SHIFT_PARAM, sign = $SHIFT_SIGN, Δ = $(DELTAS[SHIFT_PARAM])")
    println("  Ω_m,0    = $(cosmo.Om)")
    println("  Ω_k,0    = $(cosmo.Ok)")
    println("  h        = $(cosmo.h)")
    println("  H0       = $(cosmo.H0) km/s/Mpc")
    println("  n_s      = $(cosmo.ns)")
    println("  α_s      = $(cosmo.as)")
    println("  w0       = $(cosmo.w0)")
    println("  wa       = $(cosmo.wa)")
    println("  omega_other h^2 (fixed) = $(OMH2_OTHER_FID)")
    println("  omega_m h^2 fid total   = $(OMH2_TOTAL_FID)")
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

function cosmo_table_cli_args(cosmo)
    side_str = SHIFT_SIGN > 0 ? "p" : "n"
    return vcat([
        "--param=$(SHIFT_PARAM)",
        "--side=$(side_str)",
    ], cosmo_cli_args(cosmo))
end

function verify_cosmo_table_matches!(cosmo_funcr::String, cosmo; atol=5e-6)
    if !isfile(cosmo_funcr)
        error("Cosmology table was not created: $cosmo_funcr")
    end
    first_data = nothing
    for line in eachline(cosmo_funcr)
        t = strip(line)
        if isempty(t) || startswith(t, "#") || startswith(lowercase(t), "z")
            continue
        end
        first_data = split(t)
        break
    end
    first_data === nothing && error("Could not find data rows in $cosmo_funcr")

    # Columns: z, r, H_over_c, Omega_m, f, D, a
    Om_table = parse(Float64, first_data[4])
    if abs(Om_table - cosmo.Om) > atol
        error("Cosmology table mismatch: selected Om=$(cosmo.Om), table Om(z≈0)=$(Om_table). make_powerfull_cosmo_table.jl is ignoring the passed cosmology.")
    end
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

function reset_if_cosmology_changed!(outdir::String, cosmo)
    # Variant-array mode: if Step 2 already exists, do NOT acquire .reset_lock
    # and do NOT delete Step 2 / Step 3 products. We are only reusing Step 2
    # and computing one Cl_* variant per array task.
    meta_file  = joinpath(outdir, "ClGR_output_meta.h5")
    stamp_path = joinpath(outdir, "selected_cosmology.txt")
    new_stamp  = cosmo_stamp(cosmo)

    if isfile(meta_file)
        if !isfile(stamp_path)
            println(">> Step 2 meta exists; writing missing selected_cosmology.txt without reset.")
            write(stamp_path, new_stamp)
        elseif read(stamp_path, String) != new_stamp
            println(">> WARNING: Step 2 meta exists but selected_cosmology.txt differs.")
            println(">> Reusing existing Step 2 because this is variant-array mode.")
            println(">> To force a rebuild, manually remove ClGR_output_meta.h5, ClGR_output*.h5, and selected_cosmology.txt.")
        else
            println(">> Cosmology stamp OK; reusing existing Step 2.")
        end
        return
    end

    # Only lock/reset if Step 2 is absent and this scenario genuinely needs setup.
    with_lock(joinpath(outdir, ".reset_lock")) do
        if isfile(meta_file)
            println(">> Step 2 appeared while waiting; no reset needed.")
            return
        end

        old_stamp = isfile(stamp_path) ? read(stamp_path, String) : nothing
        if old_stamp == new_stamp
            return
        end

        println(old_stamp === nothing ? ">> No selected_cosmology.txt found. Initializing stamp in $outdir" : ">> Cosmology changed and Step 2 is absent. Cleaning stale products in $outdir")
        rm(joinpath(outdir, "cosmo_funcr.txt"), force=true)
        for f in readdir(outdir; join=true)
            startswith(basename(f), "ClGR_output") && rm(f, force=true, recursive=true)
        end
        for (_, outfile_name) in CL_VARIANTS
            rm(joinpath(outdir, outfile_name), force=true)
        end
        write(stamp_path, new_stamp)
    end
end

function shifted_pk(mode::String, scenario::String)
    suffix = mode == "wiggle" ? "v2.dat" : "_noBAOv2.dat"
    candidates = [
        joinpath(POWER_SPEC_DIR, "planck_2018_cosmology_power_hires_$(scenario)$(suffix)"),
    ]
    for p in candidates
        if isfile(p)
            return p
        end
    end
    error("Missing shifted P(k) for scenario=$scenario, mode=$mode. Tried:\n  " * join(candidates, "\n  "))
end

function make_cosmo_table_if_needed(outdir::String, pk::String, cosmo)
    cosmo_funcr = joinpath(outdir, "cosmo_funcr.txt")
    isfile(cosmo_funcr) && (println(">> Skipping cosmology table; already exists: $cosmo_funcr"); return cosmo_funcr)
    with_lock(joinpath(outdir, ".cosmo_table_lock")) do
        isfile(cosmo_funcr) && (println(">> Skipping cosmology table; already exists: $cosmo_funcr"); return)
        args = ["julia", "--project=$(REPO)", "-t", string(NUM_THREADS), joinpath(REPO, "scripts", "make_powerfull_cosmo_table.jl"), outdir, "--matterpower=$(pk)"]
        append!(args, cosmo_cli_args(cosmo))
        println(">> Making cosmology table with selected scenario cosmology")
        run_cmd(Cmd(args))
        isfile(cosmo_funcr) || error("Cosmology table was not created: $cosmo_funcr")
    end
    return cosmo_funcr
end

const N_CASES = 7

function step1_filename(tier, case::Int)
    return @sprintf(
        "TwoFAST_w_integrand_nr=%d_nR=%d_dlnR=%g_ell=%d-%d_%d.jld2",
        NR, tier.nR, tier.dlnR, tier.ellmin, tier.ellmax, case
    )
end

step1_datadir(outdir::String) = joinpath(outdir, "step1")

function step1_complete(datadir::String)
    isdir(datadir) || return false
    for tier in TIERS, case in 1:N_CASES
        isfile(joinpath(datadir, step1_filename(tier, case))) || return false
    end
    return true
end

# Step 1 is now handled by separate low/mid/high SLURM jobs.
# Main fid/array scripts must NOT run src/run_twofast.jl anymore; they only
# verify that the scenario-local step1/ folder is complete before Step 2.
function check_step1_complete!(datadir::String)
    if !isdir(datadir)
        error("Missing Step 1 directory: $datadir. Run the separate low/mid/high Step-1 SLURM jobs first.")
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

function build_integrals_if_needed(outdir::String, pk::String, cosmo_funcr::String)
    meta_file = joinpath(outdir, "ClGR_output_meta.h5")

    if isfile(meta_file)
        println(">> Step 2 meta exists; Step-3-only variant job will reuse: $meta_file")
        return
    end

    error("""
Missing Step 2 meta file: $meta_file

This variant driver is intentionally Step-3-only.
Run/build Step 2 first with the non-variant driver, then rerun this variant-array job.
""")
end
function compute_selected_cl_if_needed(outdir::String, cosmo_funcr::String, cosmo)
    meta_file = joinpath(outdir, "ClGR_output_meta.h5")
    isfile(meta_file) || error("Missing Step 2 meta file: $meta_file")
    variant = VARIANT_NAME
    outfile = joinpath(outdir, VARIANT_OUTFILE)
    if isfile(outfile)
        println(">> Skipping Step 3 variant=$variant; already exists: $outfile")
        return
    end
    fNL_arg = variant in ("fi", "fi_kaiser", "fi_newtonian", "ff") ? "--fNL=1.0" : "--fNL=0.0"
    println(">> Computing Step 3 variant=$variant")
    println(">> Output: $outfile")
    println(">> fNL argument: $fNL_arg")
    flush(stdout)
    run_cmd(Cmd(["julia", "--project=$(REPO)", "-t", string(NUM_THREADS), joinpath(REPO, "src", "compute_ClGR.jl"), meta_file, outfile, "--tracer-list=$(TRACER_LIST)", "--pairs-file=$(PAIRS_FILE)", "--cosmo-funcr=$(cosmo_funcr)", fNL_arg, "--Omm0=$(cosmo.Om)", "--H0=$(cosmo.H0)", "--variant=$(variant)"]))
end

# Main
# ============================================================

println("======================================================")
println("Running array scenario: $SCENARIO_NAME, PK_MODE=$PK_MODE")
println("Task ID: $task_id  |  scenario_id=$SCENARIO_ID  variant_id=$VARIANT_ID")
println("Variant: $VARIANT_NAME -> $VARIANT_OUTFILE")
println("Threads per launched Julia command: $NUM_THREADS")
println("Target ell range: $ELLMIN_TOTAL:$ELLMAX_TOTAL ($(ELLMAX_TOTAL - ELLMIN_TOTAL + 1) multipoles)")
print_selected_cosmology(COSMO)
println("======================================================")
flush(stdout)

pk = shifted_pk(PK_MODE, SCENARIO_NAME)
outdir = joinpath(OUTROOT, "$(SCENARIO_NAME)_$(PK_MODE)")
mkpath(outdir)

println("Output: $outdir")
println("P(k):   $pk")

reset_if_cosmology_changed!(outdir, COSMO)
cosmo_funcr = make_cosmo_table_if_needed(outdir, pk, COSMO)
build_integrals_if_needed(outdir, pk, cosmo_funcr)
compute_selected_cl_if_needed(outdir, cosmo_funcr, COSMO)

println("DONE: $(SCENARIO_NAME)_$(PK_MODE) variant=$VARIANT_NAME")
flush(stdout)

