#!/usr/bin/env -S julia --project
# =============================================================================
# compute_bsig8_fsig8_derivs_tiled.jl
#
# Nuisance derivative generator for EXCLUSIVE observed-z tiled SPHEREx tracers.
# Reuses fiducial tapered Step-2 integrals.
#
# Modes:
#   sample_poly : Khek-style shared fsigma8 polynomial + one bsigma8 polynomial
#                 per SPHEREx redshift-accuracy sample.
#   per_tile    : one independent multiplicative galaxy-bias amplitude per tile,
#                 closer to Dore et al. 2014's one-bias-per-z-cell treatment.
# =============================================================================

include(joinpath(@__DIR__, "calcClGR.jl"))
include(joinpath(@__DIR__, "cosmofns.jl"))

using .CalcClGR
using .cosmofns: cosmofn
using HDF5
using Printf

parse_bool(x::AbstractString) = lowercase(strip(x)) in ("1", "true", "yes", "on")

function parse_args(args)
    o = Dict{Symbol,Any}(
        :input => nothing,
        :tracer_list => nothing,
        :pairs => nothing,
        :cosmo_funcr => nothing,
        :outdir => "bsig8_fsig8_derivs_tiled",
        :poly_order => 2,
        :eps => 1e-3,
        :fNL => 0.0,
        :H0 => nothing,
        :Omm0 => nothing,
        :delta_c => 1.686,
        :coeff_index => nothing,
        :bias_mode => "sample_poly",
        :include_fsig8 => true,
    )

    for a in args
        if startswith(a, "--input=")
            o[:input] = String(split(a, "=", limit=2)[2])
        elseif startswith(a, "--tracer-list=")
            o[:tracer_list] = String(split(a, "=", limit=2)[2])
        elseif startswith(a, "--pairs=")
            o[:pairs] = String(split(a, "=", limit=2)[2])
        elseif startswith(a, "--pairs-file=")
            pf = String(split(a, "=", limit=2)[2])
            isfile(pf) || error("pairs-file not found: $pf")
            o[:pairs] = String(strip(read(pf, String)))
        elseif startswith(a, "--cosmo-funcr=")
            o[:cosmo_funcr] = String(split(a, "=", limit=2)[2])
        elseif startswith(a, "--outdir=")
            o[:outdir] = String(split(a, "=", limit=2)[2])
        elseif startswith(a, "--poly-order=")
            o[:poly_order] = parse(Int, split(a, "=", limit=2)[2])
        elseif startswith(a, "--eps=")
            o[:eps] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--fNL=")
            o[:fNL] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--H0=")
            o[:H0] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--Omm0=")
            o[:Omm0] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--delta-c=")
            o[:delta_c] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--coeff-index=")
            o[:coeff_index] = parse(Int, split(a, "=", limit=2)[2])
        elseif startswith(a, "--bias-mode=")
            o[:bias_mode] = lowercase(strip(split(a, "=", limit=2)[2]))
        elseif startswith(a, "--include-fsig8=")
            o[:include_fsig8] = parse_bool(split(a, "=", limit=2)[2])
        elseif startswith(a, "--")
            error("unknown argument: $a")
        end
    end

    for k in (:input, :tracer_list, :pairs, :cosmo_funcr, :H0, :Omm0)
        o[k] === nothing && error("missing required --$(replace(string(k), '_' => '-'))")
    end

    o[:poly_order] >= 0 || error("poly-order must be nonnegative")
    o[:eps] > 0 || error("eps must be positive")
    o[:bias_mode] in ("sample_poly", "per_tile") ||
        error("bias-mode must be sample_poly or per_tile")

    return o
end

function parse_pairs(s::AbstractString)
    out = Tuple{Int,Int}[]
    for tok in split(strip(s), ",")
        t = strip(tok)
        isempty(t) && continue
        ij = split(t, "-")
        length(ij) == 2 || error("bad pair token: $t")
        push!(out, (parse(Int, ij[1]), parse(Int, ij[2])))
    end
    isempty(out) && error("parsed zero pairs")
    return out
end

function read_tracer_list(path::String)
    isfile(path) || error("tracer list not found: $path")
    paths = String[]
    for ln in eachline(path)
        s = strip(ln)
        (isempty(s) || startswith(s, "#")) && continue
        p = isabspath(s) ? normpath(s) : normpath(joinpath(dirname(path), s))
        isfile(p) || error("tracer in list not found: $p")
        push!(paths, p)
    end
    isempty(paths) && error("tracer list is empty: $path")
    return paths
end

function tracer_sample_from_file(path::String)
    h5open(path, "r") do f
        haskey(f, "sample") && return Int(read(f["sample"]))
    end
    m = match(r"_s(\d+)_", basename(path))
    m === nothing && error("cannot determine sample for $path")
    return parse(Int, m.captures[1])
end

function tracer_zobs_from_file(path::String)
    h5open(path, "r") do f
        for key in ("z_obs", "zmid")
            haskey(f, key) && return Float64(read(f[key]))
        end
    end
    return NaN
end

function main()
    o = parse_args(ARGS)
    mkpath(o[:outdir])

    N = o[:poly_order]
    eps = o[:eps]
    pairs = parse_pairs(o[:pairs])

    println("="^100)
    println("TILED nuisance derivative generator")
    println("="^100)
    println("input       = $(o[:input])")
    println("tracers     = $(o[:tracer_list])")
    println("outdir      = $(o[:outdir])")
    println("bias_mode   = $(o[:bias_mode])")
    println("poly_order  = $N")
    println("eps         = $eps")
    println("WINDOW_QUAD = $(get(ENV, "WINDOW_QUAD", "unset"))")
    println("WINDOW_NZ   = $(get(ENV, "WINDOW_NZ", "unset"))")
    println("WINDOW_ZSCAN= $(get(ENV, "WINDOW_ZSCAN", "unset"))")
    flush(stdout)

    cf = cosmofn(o[:cosmo_funcr])
    z_of_r = r -> cf.fzr(r)
    dzdr_of_r = r -> cf.fHr(r)

    meta_ell_values = h5open(o[:input], "r") do f
        Int.(read(f, "grid/ell_values"))
    end

    tracer_paths = read_tracer_list(o[:tracer_list])
    tracers = [load_tracer_h5(p) for p in tracer_paths]
    ntr = length(tracers)

    tracer_sample = [tracer_sample_from_file(p) for p in tracer_paths]
    tracer_zobs = [tracer_zobs_from_file(p) for p in tracer_paths]
    samples = sort(unique(tracer_sample))
    counts = [count(==(s), tracer_sample) for s in samples]

    maximum(maximum(p) for p in pairs) <= ntr ||
        error("pair file references tracer beyond ntr=$ntr")

    println("ntracers=$ntr samples=$samples counts=$counts npairs=$(length(pairs))")
    flush(stdout)

    affected_pair_indices_for_sample(s::Int) =
        findall(p -> tracer_sample[p[1]] == s || tracer_sample[p[2]] == s, pairs)
    affected_pair_indices_for_tile(t::Int) =
        findall(p -> p[1] == t || p[2] == t, pairs)

    function contract_subset(pair_ids; f_scale=nothing, bg_scales=nothing)
        subpairs = pairs[pair_ids]
        t0 = time()
        println("      contracting $(length(subpairs))/$(length(pairs)) pairs")
        flush(stdout)
        result = compute_Cl_observed_multi(
            o[:input], tracers, subpairs, cf,
            z_of_r, dzdr_of_r, meta_ell_values;
            fNL=o[:fNL], Omm0=o[:Omm0], H0=o[:H0], delta_c=o[:delta_c],
            variant=:full, f_scale=f_scale, bg_scales=bg_scales, verbose=true,
        )
        println("      contraction done in $(round(time()-t0, digits=1)) s")
        flush(stdout)
        return result
    end

    all_pair_ids = collect(eachindex(pairs))
    plus_poly(i) = z -> 1.0 + eps * z^i
    minus_poly(i) = z -> 1.0 - eps * z^i
    plus_amp = z -> 1.0 + eps
    minus_amp = z -> 1.0 - eps
    one_fn = z -> 1.0

    pair_string = join(["$(i)-$(j)" for (i,j) in pairs], ",")

    function write_deriv(outp::String, dC::AbstractMatrix, attrs::Dict)
        size(dC, 1) == length(meta_ell_values) || error("dC ell dimension mismatch")
        size(dC, 2) == length(pairs) || error("dC pair dimension mismatch")

        h5open(outp, "w") do f
            f["Cl_all"] = dC
            f["ell"] = meta_ell_values
            prov = create_group(f, "provenance")
            prov["pairs"] = pair_string
            prov["tracer_list"] = o[:tracer_list]
            prov["cosmo_funcr"] = o[:cosmo_funcr]
            prov["input_meta"] = o[:input]
            prov["window_quad"] = get(ENV, "WINDOW_QUAD", "")
            prov["window_nz"] = parse(Int, get(ENV, "WINDOW_NZ", "0"))
            prov["window_zscan"] = parse(Int, get(ENV, "WINDOW_ZSCAN", "0"))
            for (k,v) in attrs
                attributes(f)[String(k)] = v
            end
        end
    end

    jobs = Tuple[]
    if o[:include_fsig8]
        for i in 0:N
            push!(jobs, (:fsig8_poly, i))
        end
    end
    if o[:bias_mode] == "sample_poly"
        for s in samples, i in 0:N
            push!(jobs, (:bsig8_sample_poly, s, i))
        end
    else
        for t in 1:ntr
            push!(jobs, (:bias_tile, t))
        end
    end

    njobs = length(jobs)

    run_indices = if o[:coeff_index] === nothing
        0:(njobs - 1)
    else
        ci = o[:coeff_index]
        0 <= ci < njobs || error("coeff-index=$ci outside 0:$(njobs - 1)")
        ci:ci
    end

    println("total jobs=$njobs; running $(first(run_indices)):$(last(run_indices))")
    flush(stdout)

    function embed_subset(sub::AbstractMatrix, pair_ids)
        full = zeros(Float64, size(sub,1), length(pairs))
        full[:, pair_ids] .= sub
        return full
    end

    for ci in run_indices
        job = jobs[ci+1]
        kind = job[1]

        if kind == :fsig8_poly
            i = job[2]
            println("\n[$ci/$((njobs-1))] fsig8 polynomial order $i")
            cp = contract_subset(all_pair_ids; f_scale=plus_poly(i))
            cm = contract_subset(all_pair_ids; f_scale=minus_poly(i))
            dC = (cp .- cm) ./ (2eps)
            outp = joinpath(o[:outdir], @sprintf("Cl_dfsig8_a%d.h5", i))
            write_deriv(outp, dC, Dict("kind"=>"fsigma8_shared_polynomial", "order"=>i, "eps"=>eps, "coeff_index"=>ci))
            println("wrote $outp")

        elseif kind == :bsig8_sample_poly
            s, i = job[2], job[3]
            pair_ids = affected_pair_indices_for_sample(s)
            bgp = [tracer_sample[k] == s ? plus_poly(i) : one_fn for k in 1:ntr]
            bgm = [tracer_sample[k] == s ? minus_poly(i) : one_fn for k in 1:ntr]
            println("\n[$ci/$((njobs-1))] bsig8 sample=$s polynomial order=$i")
            cp = contract_subset(pair_ids; bg_scales=bgp)
            cm = contract_subset(pair_ids; bg_scales=bgm)
            dC = embed_subset((cp .- cm) ./ (2eps), pair_ids)
            outp = joinpath(o[:outdir], @sprintf("Cl_dbsig8_s%d_b%d.h5", s, i))
            write_deriv(outp, dC, Dict("kind"=>"bsigma8_per_sample_polynomial", "sample"=>s, "order"=>i, "eps"=>eps, "coeff_index"=>ci))
            println("wrote $outp")

        elseif kind == :bias_tile
            t = job[2]
            pair_ids = affected_pair_indices_for_tile(t)
            bgp = [k == t ? plus_amp : one_fn for k in 1:ntr]
            bgm = [k == t ? minus_amp : one_fn for k in 1:ntr]
            println("\n[$ci/$((njobs-1))] independent tile bias t=$t sample=$(tracer_sample[t]) zobs=$(tracer_zobs[t])")
            cp = contract_subset(pair_ids; bg_scales=bgp)
            cm = contract_subset(pair_ids; bg_scales=bgm)
            dC = embed_subset((cp .- cm) ./ (2eps), pair_ids)
            outp = joinpath(o[:outdir], @sprintf("Cl_dbias_tile%04d.h5", t))
            write_deriv(outp, dC, Dict("kind"=>"independent_tile_bias_amplitude", "tracer_index"=>t, "sample"=>tracer_sample[t], "z_obs"=>tracer_zobs[t], "eps"=>eps, "coeff_index"=>ci))
            println("wrote $outp")
        end
        flush(stdout)
    end

    if o[:coeff_index] === nothing || o[:coeff_index] == 0
        meta_path = joinpath(o[:outdir], "derivs_meta.h5")
        h5open(meta_path, "w") do f
            f["bias_mode"] = o[:bias_mode]
            f["poly_order"] = N
            f["eps"] = eps
            f["include_fsig8"] = o[:include_fsig8] ? 1 : 0
            f["njobs"] = njobs
            f["ntracers"] = ntr
            f["sample"] = tracer_sample
            f["z_obs"] = tracer_zobs
            f["pairs_i"] = [p[1] for p in pairs]
            f["pairs_j"] = [p[2] for p in pairs]
            f["tracer_paths"] = tracer_paths
            f["window_nz"] = parse(Int, get(ENV, "WINDOW_NZ", "0"))
            f["window_zscan"] = parse(Int, get(ENV, "WINDOW_ZSCAN", "0"))
        end
        println("wrote $meta_path")
    end

    println("\nDone.")
end

main()

