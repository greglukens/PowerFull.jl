#!/usr/bin/env julia

using Printf
using Base.Threads
using Dates
using Sockets

# ============================================================
# Paths / environment
# ============================================================

const REPO = get(ENV, "REPO", "/storage/group/duj13/default/PowerFull.jl")

const OUTROOT = get(ENV, "OUTROOT",
    "/storage/home/gql5196/scratch/powerfull_sweep")

const POWER_SPEC_DIR = get(ENV, "POWER_SPEC_DIR",
    "/storage/home/gql5196/work/tamred/data/power_spec")

const TRACER_LIST = get(ENV, "TRACER_LIST",
    "/storage/group/duj13/default/PowerFull.jl/examples/tracer_list_prod.txt")

const PAIRS_FILE = get(ENV, "PAIRS_FILE",
    "/storage/group/duj13/default/PowerFull.jl/examples/pairs_full.txt")

const PK_MODE = get(ENV, "PK_MODE", "wiggle")

if !(PK_MODE in ["wiggle", "nobao"])
    error("run_powerfull_cosmo_array.jl expects PK_MODE=wiggle or nobao.")
end


# Fiducial cosmology.  Every scenario below is built by explicitly
# constructing the actual cosmology values first; those exact values are then
# passed to the background-table generator and to Step 3.
const OM_FID      = parse(Float64, get(ENV, "OM_FID", "0.3111"))
const OK_FID      = parse(Float64, get(ENV, "OK_FID", "0.0"))
const H_FID       = parse(Float64, get(ENV, "H_FID", "0.6766"))
const H0_FID      = 100.0 * H_FID
const NS_FID      = parse(Float64, get(ENV, "NS_FID", "0.9665"))
const AS_FID      = parse(Float64, get(ENV, "AS_FID", "0.0"))
const W0_FID      = parse(Float64, get(ENV, "W0_FID", "-1.0"))
const WA_FID      = parse(Float64, get(ENV, "WA_FID", "0.0"))

const DELTAS = Dict(
    "Om" => parse(Float64, get(ENV, "DELTA_OM", "0.001")),
    "Ok" => parse(Float64, get(ENV, "DELTA_OK", "0.001")),
    "w0" => parse(Float64, get(ENV, "DELTA_W0", "0.005")),
    "wa" => parse(Float64, get(ENV, "DELTA_WA", "0.005")),
    "ns" => parse(Float64, get(ENV, "DELTA_NS", "0.0005")),
    "as" => parse(Float64, get(ENV, "DELTA_AS", "0.0005")),
)


# Number of Julia threads to give each launched worker command.
# In SLURM, set this automatically from --cpus-per-task; otherwise fall back
# to the runner's current JULIA thread count, then 1.
const NUM_THREADS = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", string(Threads.nthreads())))

# Step 1 / TwoFAST uses distributed workers (-p), not threads.
const STEP1_WORKERS = parse(Int, get(ENV, "STEP1_WORKERS", "22"))

# ============================================================
# Sweep settings
# ============================================================

# Fiducial-only driver: do not run parameter +/- shifts.
# Keep PARAMS/SIDES defined as empty arrays so any accidental sweep loop is inert.
const PARAMS = String[]
const SIDES  = String[]

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


const N_CASES = 7

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

const VARIANT_ID = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : parse(Int, get(ENV, "SLURM_ARRAY_TASK_ID", "0"))
if VARIANT_ID < 0 || VARIANT_ID >= N_VARIANTS
    error("Invalid VARIANT_ID=$VARIANT_ID. Valid IDs are 0:$(N_VARIANTS-1).")
end
const SELECTED_VARIANT = CL_VARIANTS[VARIANT_ID + 1]
const VARIANT_NAME = SELECTED_VARIANT[1]
const VARIANT_OUTFILE = SELECTED_VARIANT[2]

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

function step1_filename(tier, case::Int)
    return @sprintf(
        "TwoFAST_w_integrand_nr=%d_nR=%d_dlnR=%g_ell=%d-%d_%d.jld2",
        NR, tier.nR, tier.dlnR, tier.ellmin, tier.ellmax, case
    )
end

function check_inputs!()
    if !isdir(POWER_SPEC_DIR)
        error("POWER_SPEC_DIR does not exist: $POWER_SPEC_DIR")
    end

    if !isfile(TRACER_LIST)
        error("TRACER_LIST not found: $TRACER_LIST")
    end

    if !isfile(PAIRS_FILE)
        error("PAIRS_FILE not found: $PAIRS_FILE")
    end

    if !(PK_MODE in ["wiggle", "nobao", "both"])
        error("PK_MODE must be wiggle, nobao, or both. Got: $PK_MODE")
    end

    println("\nInput summary:")
    println("  REPO           = $REPO")
    println("  OUTROOT        = $OUTROOT")
    println("  POWER_SPEC_DIR = $POWER_SPEC_DIR")
    println("  TRACER_LIST    = $TRACER_LIST")
    println("  PAIRS_FILE     = $PAIRS_FILE")
    println("  PK_MODE        = $PK_MODE")
    println("  NUM_THREADS    = $NUM_THREADS")
    println("  fid Om, Ok     = $OM_FID, $OK_FID")
    println("  fid h, H0      = $H_FID, $H0_FID")
    println("  fid ns, as     = $NS_FID, $AS_FID")
    println("  fid w0, wa     = $W0_FID, $WA_FID")
end

function pk_file(param::String, side::String, mode::String)
    suffix = mode == "wiggle" ? "_v2.dat" : "_noBAO_v2.dat"
    return joinpath(
        POWER_SPEC_DIR,
        "planck_2018_cosmology_power_hires_$(param)_$(side)$(suffix)"
    )
end

function fid_pk(mode::String)
    suffix = mode == "wiggle" ? "_v2.dat" : "_noBAO_v2.dat"
    return joinpath(
        POWER_SPEC_DIR,
        "planck_2018_cosmology_power_hires_fid$(suffix)"
    )
end


function selected_cosmology(param=nothing, side=nothing)
    Om = OM_FID
    Ok = OK_FID
    h  = H_FID
    ns = NS_FID
    as = AS_FID
    w0 = W0_FID
    wa = WA_FID

    if param !== nothing
        sgn = side == "p" ? +1.0 : side == "m" ? -1.0 : error("side must be p or m; got $side")
        d = DELTAS[String(param)]
        if param == "Om"
            Om += sgn*d
        elseif param == "Ok"
            Ok += sgn*d
        elseif param == "w0"
            w0 += sgn*d
        elseif param == "wa"
            wa += sgn*d
        elseif param == "ns"
            ns += sgn*d
        elseif param == "as"
            as += sgn*d
        else
            error("Unknown parameter: $param")
        end
    end

    return (; Om, Ok, h, H0=100.0*h, ns, as, w0, wa)
end

function print_selected_cosmology(cosmo)
    println("Selected cosmology used everywhere:")
    println("  Ω_m,0 = $(cosmo.Om)")
    println("  Ω_k,0 = $(cosmo.Ok)")
    println("  h     = $(cosmo.h)")
    println("  H0    = $(cosmo.H0) km/s/Mpc")
    println("  n_s   = $(cosmo.ns)")
    println("  α_s   = $(cosmo.as)")
    println("  w0    = $(cosmo.w0)")
    println("  wa    = $(cosmo.wa)")
end

function cosmo_cli_args(cosmo)
    # These are intentionally explicit: the background-table script must use
    # these values, not its own internal fiducials/defaults.  Keep the option
    # names in sync with scripts/make_powerfull_cosmo_table.jl.
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
        "Om=$(cosmo.Om)",
        "Ok=$(cosmo.Ok)",
        "h=$(cosmo.h)",
        "H0=$(cosmo.H0)",
        "ns=$(cosmo.ns)",
        "as=$(cosmo.as)",
        "w0=$(cosmo.w0)",
        "wa=$(cosmo.wa)",
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

function make_cosmo_table_if_needed(outdir::String, pk::String, cosmo; param=nothing, side=nothing)
    cosmo_funcr = joinpath(outdir, "cosmo_funcr.txt")
    isfile(cosmo_funcr) && (println(">> Skipping cosmology table; already exists: $cosmo_funcr"); return cosmo_funcr)
    with_lock(joinpath(outdir, ".cosmo_table_lock")) do
        isfile(cosmo_funcr) && (println(">> Skipping cosmology table; already exists: $cosmo_funcr"); return)
        println(">> Making cosmology table: $cosmo_funcr")
        args = ["julia", "--project=$(REPO)", "-t", string(NUM_THREADS), joinpath(REPO, "scripts", "make_powerfull_cosmo_table.jl"), outdir, "--matterpower=$(pk)"]
        append!(args, cosmo_cli_args(cosmo))
        if param !== nothing
            push!(args, "--param=$(param)"); push!(args, "--side=$(side)")
        end
        run_cmd(Cmd(args))
        isfile(cosmo_funcr) || error("Cosmology table was not created: $cosmo_funcr")
    end
    return cosmo_funcr
end

function step1_datadir(outdir::String)
    return joinpath(outdir, "step1")
end

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
    isfile(cosmo_funcr) || error("Missing scenario cosmology table: $cosmo_funcr")
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
    run_cmd(Cmd(["julia", "--project=$(REPO)", "-t", string(NUM_THREADS), joinpath(REPO, "src", "compute_ClGR.jl"), meta_file, outfile, "--tracer-list=$(TRACER_LIST)", "--pairs-file=$(PAIRS_FILE)", "--cosmo-funcr=$(cosmo_funcr)", fNL_arg, "--Omm0=$(cosmo.Om)", "--H0=$(cosmo.H0)", "--variant=$(variant)"]))
end

function run_one_scenario(scen::String, pk::String; param=nothing, side=nothing)
    outdir = joinpath(OUTROOT, scen)
    mkpath(outdir)

    println("\n============================================================")
    println("Scenario: $scen")
    println("Output:   $outdir")
    println("P(k):     $pk")
    println("============================================================")

    cosmo = selected_cosmology(param, side)
    print_selected_cosmology(cosmo)
    reset_if_cosmology_changed!(outdir, cosmo)

    if !isfile(pk)
        error("Missing power spectrum file: $pk")
    end

    cosmo_funcr = make_cosmo_table_if_needed(outdir, pk, cosmo; param=param, side=side)

    build_integrals_if_needed(outdir, pk, cosmo_funcr)

    compute_selected_cl_if_needed(outdir, cosmo_funcr, cosmo)
end

# ============================================================
# Main
# ============================================================

check_inputs!()

mode = PK_MODE
run_one_scenario("fid_$(mode)", fid_pk(mode))

println("\nDone variant=$VARIANT_NAME.")
flush(stdout)
