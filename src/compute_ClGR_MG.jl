#!/usr/bin/env -S julia --project -t auto
# =============================================================================
#
<
#
# Assemble the observed relativistic angular power spectrum C_ℓ^{ij}
# from pre-computed TwoFAST integrals (Step 3 of the PowerFull pipeline).
#
# Cross-correlation is the primary interface:
#
#   julia -t N --project src/compute_ClGR.jl <integrals.h5> <output.h5> \
#       --tracer-1=<tracer_i.h5> --tracer-2=<tracer_j.h5> \
#       [--cosmo-funcr=<path>] [--fNL=<float>] [--delta-c=<float>]
#
# For an auto-spectrum, pass the same tracer file twice (or omit
# --tracer-2, which then defaults to --tracer-1).
#
# Each tracer HDF5 holds the per-sample bias + selection in z-space.
# See examples/spherex_paper_example.jl for the expected keys.
#
# =============================================================================

using HDF5
using Base.Threads

println("Starting compute_ClGR.jl with $(nthreads()) threads")

include(joinpath(@__DIR__, "calcClGR_MG.jl"))
using .CalcClGR

include(joinpath(@__DIR__, "cosmofns.jl"))
using .cosmofns: cosmofn

# =============================================================================
# CLI parsing
# =============================================================================

function _parse_args()
    opts = Dict{Symbol,Any}(
        :input        => nothing,
        :output       => nothing,
        :tracer_1     => nothing,
        :tracer_2     => nothing,
        :tracer_list  => nothing,
        :pairs        => nothing,
        :cosmo_funcr  => joinpath(@__DIR__, "..", "data", "cosmo_funcr_astropy_planck2018.txt"),
        :fNL          => 0.0,
        :delta_c      => 1.686,
        :Omm0         => nothing,
        :H0           => nothing,
        :variant      => :full,
        :mu0          => 0.0,
        :Sigma0       => 0.0,
    )
    positional = String[]
    for arg in ARGS
        if startswith(arg, "--tracer-1=")
            opts[:tracer_1] = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tracer-2=")
            opts[:tracer_2] = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--tracer-list=")
            opts[:tracer_list] = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--pairs=")
            opts[:pairs] = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--pairs-file=")
            pf = String(split(arg, "=", limit=2)[2])
            isfile(pf) || error("pairs-file not found: $pf")
            opts[:pairs] = String(strip(read(pf, String)))
        elseif startswith(arg, "--cosmo-funcr=")
            opts[:cosmo_funcr] = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--fNL=")
            opts[:fNL] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--delta-c=")
            opts[:delta_c] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--Omm0=")
            opts[:Omm0] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--H0=")
            opts[:H0] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--mu0=")
            opts[:mu0] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--Sigma0=")
            opts[:Sigma0] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--variant=")
            v = String(split(arg, "=", limit=2)[2])
            opts[:variant] = Symbol(v)
            opts[:variant] in (:full, :gaussian, :kaiser, :newtonian, :fi, :fi_kaiser, :fi_newtonian, :ff) ||
                error("--variant must be one of full, gaussian, kaiser, newtonian, fi, fi_kaiser, fi_newtonian, ff (got '$v')")
        elseif startswith(arg, "--")
            @warn "Unknown argument: $arg"
        else
            push!(positional, arg)
        end
    end
    opts[:input]  = length(positional) >= 1 ? positional[1] : "ClGR_integrals.h5"
    opts[:output] = length(positional) >= 2 ? positional[2] : "ClGR_result.h5"

    opts[:Omm0] === nothing && error("--Omm0=<Ω_m,0> is required; pass the value from the selected cosmology")
    opts[:H0]   === nothing && error("--H0=<km/s/Mpc> is required; pass the value from the selected cosmology")

    multi = opts[:tracer_list] !== nothing
    if multi
        opts[:pairs] === nothing &&
            error("--tracer-list requires --pairs=i1-j1,i2-j2,... (1-based)")
        (opts[:tracer_1] !== nothing || opts[:tracer_2] !== nothing) &&
            error("--tracer-list is exclusive with --tracer-1/--tracer-2")
    else
        if opts[:tracer_1] === nothing
            error("""
            compute_ClGR.jl requires either:
              --tracer-1=<path>.h5  [optional --tracer-2=<path>.h5]  (single pair)
            or
              --tracer-list=<file>.txt --pairs=i1-j1,i2-j2,...       (multi-pair)

            See examples/spherex_paper_example.jl for how to build tracer h5 files.""")
        end
        if opts[:tracer_2] === nothing
            opts[:tracer_2] = opts[:tracer_1]
            @info "--tracer-2 not provided; running auto-correlation with tracer-1."
        end
    end
    return opts
end

"""
    _read_tracer_list(path) -> Vector{String}

Read tracer h5 paths from a text file, one per line.  Blank lines and
lines beginning with `#` are skipped.
"""
function _read_tracer_list(path::String)::Vector{String}
    isfile(path) || error("tracer list file not found: $path")
    paths = String[]
    for (ln_idx, ln) in enumerate(eachline(path))
        s = strip(ln)
        (isempty(s) || startswith(s, "#")) && continue
        push!(paths, String(s))
    end
    isempty(paths) && error("tracer list $path is empty")
    for p in paths
        isfile(p) || error("tracer h5 in list not found: $p")
    end
    return paths
end

"""
    _parse_pairs(spec, n_tracers) -> Vector{Tuple{Int,Int}}

Parse `--pairs=1-1,1-2,2-2,...` (1-based) into a Vector of (i, j) index tuples,
validating against `n_tracers`.
"""
function _parse_pairs(spec::String, n_tracers::Int)::Vector{Tuple{Int,Int}}
    pairs = Tuple{Int,Int}[]
    for tok in split(spec, ",")
        s = strip(tok)
        isempty(s) && continue
        parts = split(s, "-")
        length(parts) == 2 || error("invalid pair spec '$s' (expected 'i-j')")
        i = parse(Int, parts[1]); j = parse(Int, parts[2])
        (1 ≤ i ≤ n_tracers) && (1 ≤ j ≤ n_tracers) ||
            error("pair ($i, $j) out of range 1:$n_tracers")
        push!(pairs, (i, j))
    end
    isempty(pairs) && error("--pairs parsed to empty list")
    return pairs
end

# Component outputs store coefficients, not values at fiducial fNL.
const _FNL_COMPONENT_VARIANTS = Set([:fi, :fi_kaiser, :fi_newtonian, :ff])

_effective_fNL_for_variant(variant::Symbol, fNL::Float64) =
    (variant in _FNL_COMPONENT_VARIANTS) ? 1.0 : fNL

# =============================================================================
# Main
# =============================================================================

function main()
    opts = _parse_args()
    fNL_compute = _effective_fNL_for_variant(opts[:variant], opts[:fNL])

    multi = opts[:tracer_list] !== nothing
    println("\n" * "="^64)
    println("Computing observed C_ℓ^{ij}" * (multi ? "  (multi-pair mode)" : ""))
    println("="^64)
    println("Integrals (input) : $(opts[:input])")
    println("Output            : $(opts[:output])")
    if multi
        println("Tracer list       : $(opts[:tracer_list])")
        println("Pairs             : $(opts[:pairs])")
    else
        println("Tracer 1          : $(opts[:tracer_1])")
        println("Tracer 2          : $(opts[:tracer_2])")
    end
    println("Cosmology table   : $(opts[:cosmo_funcr])")
    println("variant           : $(opts[:variant])")
    println("f_NL input        : $(opts[:fNL])")
    println("f_NL used         : $(fNL_compute)")
    if opts[:variant] in _FNL_COMPONENT_VARIANTS && opts[:fNL] != fNL_compute
        println("  component mode: writing coefficient file, so f_NL is forced to 1.0")
    end
    println("δ_c (bPhi default): $(opts[:delta_c])")
    println("Ω_m,0             : $(opts[:Omm0])")
    println("H_0 [km/s/Mpc]    : $(opts[:H0])")
    if opts[:mu0] != 0.0 || opts[:Sigma0] != 0.0
        println("MODIFIED GRAVITY  : μ₀ = $(opts[:mu0]), Σ₀ = $(opts[:Sigma0])  (local limit)")
    else
        println("MODIFIED GRAVITY  : off (GR; μ₀=0, Σ₀=0)")
    end
    println()

    println("Loading cosmology...")
    cf = cosmofn(opts[:cosmo_funcr])

    println("Loading integrals metadata (streaming mode — integrals read on-demand)...")
    meta_rr, meta_ell_values = h5open(opts[:input], "r") do f
        (Float64.(read(f, "grid/rr")), Int.(read(f, "grid/ell_values")))
    end
    println("  nr = $(length(meta_rr)), nell = $(length(meta_ell_values))")
    println("  r ∈ [$(round(meta_rr[1], digits=1)), $(round(meta_rr[end], digits=1))] Mpc/h")
    println("  ℓ ∈ [$(meta_ell_values[1]), $(meta_ell_values[end])]")

    # z(r) and dz/dr from cfns.  cfns.fHr stores physical H(z)/c in h/Mpc,
    # so dz/dr = H(z)/c.  calcClGR forms conformal aH as a(r)*cfns.fHr(r).
    z_of_r    = r -> cf.fzr(r)
    dzdr_of_r = r -> cf.fHr(r)

    if multi
        tracer_paths = _read_tracer_list(opts[:tracer_list])
        n_tracers    = length(tracer_paths)
        pair_indices = _parse_pairs(opts[:pairs], n_tracers)
        n_pairs      = length(pair_indices)

        println("\nTracer list: $n_tracers unique tracer h5 files")
        for (k, p) in enumerate(tracer_paths)
            println("  [$k] $p")
        end
        println("Pairs: $n_pairs")
        for (k, (ti, tj)) in enumerate(pair_indices)
            println("  [$k] ($ti, $tj)")
        end

        println("\nLoading tracers...")
        tracers = [load_tracer_h5(p) for p in tracer_paths]
        for (k, t) in enumerate(tracers)
            println("  [$k] z ∈ [$(t.zmin), $(t.zmax)]")
        end

        println("\nComputing C_ℓ^{ij} for $n_pairs pairs × $(length(meta_ell_values)) ells (streaming, shared I/O)...")
        Cl_obs = @time compute_Cl_observed_multi(opts[:input], tracers, pair_indices,
            cf, z_of_r, dzdr_of_r, meta_ell_values;
            fNL=fNL_compute, Omm0=opts[:Omm0], H0=opts[:H0],
            delta_c=opts[:delta_c], variant=opts[:variant],
            mu0=opts[:mu0], Sigma0=opts[:Sigma0], verbose=true)

        println("\nC_ℓ range: [$(minimum(Cl_obs)), $(maximum(Cl_obs))]")
        for p in 1:n_pairs
            (ti, tj) = pair_indices[p]
            println("  pair ($ti, $tj):  ℓ=$(meta_ell_values[1]) → $(Cl_obs[1, p]),   ℓ=$(meta_ell_values[end]) → $(Cl_obs[end, p])")
        end

        println("\nSaving to $(opts[:output])...")
        h5open(opts[:output], "w") do f
            f["ell"] = meta_ell_values
            create_group(f, "tracers")
            for (k, p) in enumerate(tracer_paths)
                f["tracers/$(k)"] = p
            end
            create_group(f, "pairs")
            for p in 1:n_pairs
                (ti, tj) = pair_indices[p]
                g = create_group(f, "pairs/$(ti)_$(tj)")
                g["ell"] = meta_ell_values
                g["Cl"]  = Cl_obs[:, p]
                g["tracer_1_index"] = ti
                g["tracer_2_index"] = tj
            end
            # Combined Cl matrix for convenience: [nell, n_pairs]
            f["Cl_all"] = Cl_obs
            create_group(f, "provenance")
            f["provenance/tracer_list"]  = opts[:tracer_list]
            f["provenance/pairs"]        = opts[:pairs]
            f["provenance/integrals"]    = opts[:input]
            f["provenance/cosmo_funcr"]  = opts[:cosmo_funcr]
            f["provenance/fNL_input"]    = opts[:fNL]
            f["provenance/fNL_used"]     = fNL_compute
            f["provenance/variant"]      = string(opts[:variant])
            f["provenance/delta_c"]      = opts[:delta_c]
            f["provenance/Omm0"]         = opts[:Omm0]
            f["provenance/H0_kmsMpc"]    = opts[:H0]
            f["provenance/mu0"]          = opts[:mu0]
            f["provenance/Sigma0"]       = opts[:Sigma0]
        end
        println("Done. Wrote $n_pairs pairs × $(length(meta_ell_values)) C_ℓ values.")
        println("="^64)
        return Cl_obs
    end

    # Single-pair path (original)
    println("\nLoading tracer 1...")
    t1 = load_tracer_h5(opts[:tracer_1])
    println("  z range: [$(t1.zmin), $(t1.zmax)]")

    t2 = if opts[:tracer_2] == opts[:tracer_1]
        println("Tracer 2 = tracer 1 (auto-correlation)")
        t1
    else
        println("Loading tracer 2...")
        t2_ = load_tracer_h5(opts[:tracer_2])
        println("  z range: [$(t2_.zmin), $(t2_.zmax)]")
        t2_
    end

    println("\nBuilding ClGRParams (tracer × cosmology)...")
    mg = (opts[:mu0] != 0.0 || opts[:Sigma0] != 0.0) ?
        build_mg_model(cf; mu0=opts[:mu0], Sigma0=opts[:Sigma0]) : nothing
    if mg !== nothing
        println("  [MG] local-limit modified gravity active: μ₀=$(opts[:mu0]), Σ₀=$(opts[:Sigma0])")
    end
    params_1 = tracer_to_clgr_params(t1, cf;
        fNL=fNL_compute, Omm0=opts[:Omm0], H0=opts[:H0], delta_c=opts[:delta_c], mg=mg)
    params_2 = if t2 === t1
        params_1
    else
        tracer_to_clgr_params(t2, cf;
            fNL=fNL_compute, Omm0=opts[:Omm0], H0=opts[:H0], delta_c=opts[:delta_c], mg=mg)
    end

    println("\nComputing C_ℓ^{ij} for $(length(meta_ell_values)) ells (streaming)...")
    Cl_obs = @time compute_Cl_observed(opts[:input], params_1, params_2,
                                       t1.phi, t2.phi, z_of_r, dzdr_of_r,
                                       meta_ell_values; verbose=true)

    println("\nC_ℓ range: [$(minimum(Cl_obs)), $(maximum(Cl_obs))]")
    println("Sample ℓ=$(meta_ell_values[1])  →  C_ℓ = $(Cl_obs[1])")
    println("Sample ℓ=$(meta_ell_values[end]) →  C_ℓ = $(Cl_obs[end])")

    println("\nSaving to $(opts[:output])...")
    h5open(opts[:output], "w") do f
        f["ell"] = meta_ell_values
        f["Cl"]  = Cl_obs
        create_group(f, "provenance")
        f["provenance/tracer_1"]   = opts[:tracer_1]
        f["provenance/tracer_2"]   = opts[:tracer_2]
        f["provenance/integrals"]  = opts[:input]
        f["provenance/cosmo_funcr"] = opts[:cosmo_funcr]
        f["provenance/fNL"]        = opts[:fNL]
        f["provenance/delta_c"]    = opts[:delta_c]
        f["provenance/Omm0"]       = opts[:Omm0]
        f["provenance/H0_kmsMpc"]  = opts[:H0]
        f["provenance/mu0"]        = opts[:mu0]
        f["provenance/Sigma0"]     = opts[:Sigma0]
    end

    println("Done. Wrote $(length(Cl_obs)) C_ℓ values.")
    println("="^64)
    return Cl_obs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

