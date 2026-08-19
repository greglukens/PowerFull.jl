# =============================================================================
# compute_Qbias_derivs.jl
#
# d C_l / d(polynomial coefficient) derivative cubes for a magnification-bias
# 𝒬(z) nuisance scheme, in the angular-Cl pipeline.  Direct analogue of
# compute_bsig8_fsig8_derivs.jl (Khek et al. polynomial scheme), but for 𝒬(z).
#
# Scheme (mirrors the SHARED fσ8 polynomial, NOT the per-sample bσ8 one):
#     Q(z) = Q_fid(z) * sum_{i=0}^{N} c_i z^i      (shared across ALL samples)
#   fiducial: c_0 = 1, higher = 0.
#
# 𝒬(z) is read in from the tracer files (/Q) and splined.  In the SphereX mock
# (and this paper's setup) every sample is drawn from the SAME luminosity
# function and the SAME flux limit, with samples split by redshift uncertainty
# σ_z/(1+z), NOT by flux cut.  Since 𝒬 = -d ln n̄_g(>L_min)/d ln L_min depends
# only on the LF + flux limit, 𝒬(z) is IDENTICAL across samples.  It is
# therefore a SHARED nuisance, exactly like fσ8 — one set of coefficients c_i
# modulates 𝒬(z) for every tracer at once.  (Contrast bσ8, which is per-sample.)
#
# Modulating the polynomial == multiplicatively rescaling 𝒬(z).  At fiducial:
#   d(Q)/dc_i  => rescale 𝒬 by (1 + eps z^i) for ALL tracers,  central diff.
#
# Why this is cheap (same logic as bσ8/fσ8): the w/u/v/s/t/l base integrals
# depend only on P(k)+geometry, NOT on 𝒬, so they are streamed/reused.  𝒬 only
# re-weights the contraction.  Each derivative is a fast re-contraction via
# compute_Cl_observed_multi(...; Q_scales=).
#
# Where 𝒬 flows (all handled inside tracer_to_clgr_params once Q_scale is set):
#   - 𝒞 coefficient  (C_fn:  the (1-𝒬) and -2𝒬 terms)
#   - 𝒜 coefficient  (A_fn:  the -2𝒬 term)
#   - the (1-𝒬) prefactors on the time-delay and lensing contributions
#     (via the stored ClGRParams.Q used at contraction).
# 𝒬 does NOT enter bPhi (PNG bias), so there is no spurious f_NL coupling.
#
# REQUIRES the Q_scales= hook in compute_Cl_observed_multi / tracer_to_clgr_params
# (additive patch; fiducial calls unchanged).
#
# Evaluate at the FIDUCIAL (GR) point: run against the fid_wiggle integrals.
#
# USAGE:
#   julia --project compute_Qbias_derivs.jl \
#       --input=<ClGR_integrals_meta.h5> \
#       --tracer-list=<tracers.txt> --pairs=1-1,2-2,3-3,4-4,5-5 \
#       --cosmo-funcr=<cosmo_funcr.txt> --Omm0=0.3111 --H0=67.66 \
#       --outdir=<out> [--poly-order=5] [--eps=1e-3] [--fNL=0] [--variant=full]
# =============================================================================

include(joinpath(@__DIR__, "calcClGR.jl"))
include(joinpath(@__DIR__, "cosmofns.jl"))
using .CalcClGR
using .cosmofns: cosmofn
using HDF5
using Printf

function parse_args(args)
    o = Dict{Symbol,Any}(
        :input=>nothing, :tracer_list=>nothing, :pairs=>nothing,
        :cosmo_funcr=>nothing, :outdir=>"Qbias_derivs",
        :poly_order=>5, :eps=>1e-3, :fNL=>0.0, :H0=>nothing, :Omm0=>nothing,
        :delta_c=>1.686, :coeff_index=>nothing, :variant=>"full",
    )
    for a in args
        if     startswith(a,"--input=");        o[:input]=String(split(a,"=",limit=2)[2])
        elseif startswith(a,"--tracer-list=");  o[:tracer_list]=String(split(a,"=",limit=2)[2])
        elseif startswith(a,"--pairs=");        o[:pairs]=String(split(a,"=",limit=2)[2])
        elseif startswith(a,"--pairs-file=")
            pf = String(split(a,"=",limit=2)[2])
            isfile(pf) || error("pairs-file not found: $pf")
            o[:pairs] = String(strip(read(pf, String)))
        elseif startswith(a,"--cosmo-funcr=");  o[:cosmo_funcr]=String(split(a,"=",limit=2)[2])
        elseif startswith(a,"--outdir=");       o[:outdir]=String(split(a,"=",limit=2)[2])
        elseif startswith(a,"--poly-order=");   o[:poly_order]=parse(Int,split(a,"=",limit=2)[2])
        elseif startswith(a,"--eps=");          o[:eps]=parse(Float64,split(a,"=",limit=2)[2])
        elseif startswith(a,"--fNL=");          o[:fNL]=parse(Float64,split(a,"=",limit=2)[2])
        elseif startswith(a,"--H0=");           o[:H0]=parse(Float64,split(a,"=",limit=2)[2])
        elseif startswith(a,"--Omm0=");         o[:Omm0]=parse(Float64,split(a,"=",limit=2)[2])
        elseif startswith(a,"--delta-c=");      o[:delta_c]=parse(Float64,split(a,"=",limit=2)[2])
        elseif startswith(a,"--coeff-index=");  o[:coeff_index]=parse(Int,split(a,"=",limit=2)[2])
        elseif startswith(a,"--variant=");      o[:variant]=String(split(a,"=",limit=2)[2])
        end
    end
    for k in (:input,:tracer_list,:pairs,:cosmo_funcr,:H0,:Omm0)
        o[k]===nothing && error("missing required --$(replace(string(k),'_'=>'-'))")
    end
    o[:variant] in ("full","newtonian","kaiser") ||
        error("--variant must be full|newtonian|kaiser (got $(o[:variant]))")
    return o
end

function parse_pairs(s::String)
    out = Tuple{Int,Int}[]
    for tok in split(s, ",")
        ij = split(strip(tok), "-")
        push!(out, (parse(Int,ij[1]), parse(Int,ij[2])))
    end
    return out
end

# tracer list reader (one h5 path per line; blanks and # comments skipped).
function _read_tracer_list(path::String)::Vector{String}
    isfile(path) || error("tracer list file not found: $path")
    paths = String[]
    for ln in eachline(path)
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

function main()
    o = parse_args(ARGS)
    mkpath(o[:outdir])
    N   = o[:poly_order]
    eps = o[:eps]
    pairs = parse_pairs(o[:pairs])
    variant = Symbol(o[:variant])

    println("Loading cosmology: $(o[:cosmo_funcr])"); flush(stdout)
    cf = cosmofn(o[:cosmo_funcr])
    z_of_r    = r -> cf.fzr(r)
    dzdr_of_r = r -> cf.fHr(r)

    println("Loading integrals metadata: $(o[:input])"); flush(stdout)
    meta_rr, meta_ell_values = h5open(o[:input], "r") do f
        (Float64.(read(f, "grid/rr")), Int.(read(f, "grid/ell_values")))
    end
    println("  nr=$(length(meta_rr)), nell=$(length(meta_ell_values)), ell in [$(meta_ell_values[1]),$(meta_ell_values[end])]"); flush(stdout)

    println("Loading tracers: $(o[:tracer_list])"); flush(stdout)
    tracer_paths = _read_tracer_list(o[:tracer_list])
    tracers = [load_tracer_h5(p) for p in tracer_paths]
    n_tracers = length(tracers)

    # 𝒬(z) is SHARED across all samples (same LF + flux limit), so unlike bσ8 we
    # do NOT need a per-sample coefficient.  We still report the sample breakdown
    # for provenance, but a single c_i modulates 𝒬 for EVERY tracer at once.
    function sample_of(path::AbstractString)::Int
        m = match(r"_s(\d+)_", basename(path))
        m === nothing && return 0    # tolerate names without _s<N>_ ; Q is shared anyway
        return parse(Int, m.captures[1])
    end
    tracer_sample = [sample_of(p) for p in tracer_paths]
    samples = sort(unique(tracer_sample))
    nsamp = length(samples)
    counts = [count(==(s), tracer_sample) for s in samples]
    println("  $(n_tracers) tracers in $(nsamp) samples $(samples); " *
            "tracers/sample = $(counts); pairs=$(pairs); variant=$(variant)"); flush(stdout)
    println("  𝒬(z) treated as SHARED across samples -> $(N+1) coefficients (like fσ8)."); flush(stdout)

    contract(; Q_scales=nothing) = begin
        _t0 = time()
        println("      [contract] starting full-pair streaming pass..."); flush(stdout)
        res = compute_Cl_observed_multi(o[:input], tracers, pairs, cf,
            z_of_r, dzdr_of_r, meta_ell_values;
            fNL=o[:fNL], Omm0=o[:Omm0], H0=o[:H0], delta_c=o[:delta_c],
            variant=variant,
            Q_scales=Q_scales, verbose=true)
        println("      [contract] done in $(round(time()-_t0, digits=1)) s"); flush(stdout)
        res
    end

    plus(i)  = (z -> 1.0 + eps * z^i)
    minus(i) = (z -> 1.0 - eps * z^i)

    ells_out = collect(meta_ell_values)

    # Write a derivative cube in the SAME layout as the fiducial Cl_f0.h5:
    #   Cl_all : (nell, n_pairs)         convenience matrix
    #   ell    : (nell,)
    #   pairs/<ti>_<tj>/Cl : (nell,)     per-pair derivative (Fisher reads these)
    function write_deriv(outp::String, dC::AbstractMatrix, attrs::Dict)
        h5open(outp, "w") do f
            f["Cl_all"] = dC                  # keep name Cl_all for loader compatibility
            f["ell"]    = ells_out
            g = create_group(f, "pairs")
            for p in 1:length(pairs)
                ti, tj = pairs[p]
                gp = create_group(g, "$(ti)_$(tj)")
                gp["Cl"] = @view dC[:, p]     # name "Cl" so the fiducial loader finds it
            end
            for (k,v) in attrs
                attributes(f)[k] = v
            end
        end
    end

    # Build the flat coefficient job list in a STABLE order so --coeff-index maps
    # deterministically across array tasks:
    #   index 0 .. N  -> 𝒬 c_i   (shared; one coefficient modulates ALL tracers)
    jobs = Vector{Tuple}()
    for i in 0:N
        push!(jobs, (:Qbias, i))
    end
    njobs = length(jobs)
    println("\nTotal coefficients (jobs) = $(njobs)  [$(N+1) shared 𝒬]"); flush(stdout)

    # Which jobs does THIS process run?  All of them (default) or one (array mode).
    run_indices = if o[:coeff_index] === nothing
        0:(njobs-1)
    else
        ci = o[:coeff_index]
        (0 <= ci < njobs) || error("--coeff-index=$ci out of range 0..$(njobs-1)")
        ci:ci
    end

    function run_Qbias(i)
        @time begin
            # SHARED: every tracer gets the SAME scale function (no per-sample mask).
            Qp = [plus(i)  for _ in 1:n_tracers]
            println("    [c_$(i)] contraction 1/2 (+ε)"); flush(stdout)
            Cl_p = contract(Q_scales=Qp)
            Qm = [minus(i) for _ in 1:n_tracers]
            println("    [c_$(i)] contraction 2/2 (−ε)"); flush(stdout)
            Cl_m = contract(Q_scales=Qm)
            dC = (Cl_p .- Cl_m) ./ (2eps)
            outp = joinpath(o[:outdir], @sprintf("Cl_dQ_c%d.h5", i))
            write_deriv(outp, dC, Dict("coeff"=>String("c_$(i)"),
                "kind"=>"Qbias_shared", "order"=>i, "eps"=>eps,
                "variant"=>String(o[:variant])))
            print("  c_$(i) -> $(basename(outp))  ")
        end; flush(stdout)
    end

    println("\n=== Computing coefficient indices $(first(run_indices))..$(last(run_indices)) ==="); flush(stdout)
    for ci in run_indices
        job = jobs[ci+1]   # jobs is 1-based; coeff-index is 0-based
        run_Qbias(job[2])
    end

    # Provenance meta — write only once (all-in-one mode, or array task 0) to
    # avoid concurrent writers racing on the same file.
    if o[:coeff_index] === nothing || o[:coeff_index] == 0
        h5open(joinpath(o[:outdir], "derivs_meta.h5"), "w") do f
            f["poly_order"]=N; f["eps"]=eps; f["nsamp"]=nsamp
            f["n_Qbias_shared"]=N+1
            f["pairs_i"]=[p[1] for p in pairs]; f["pairs_j"]=[p[2] for p in pairs]
            f["njobs"]=njobs
            attributes(f)["scheme"]="Qbias_shared_polynomial"
            attributes(f)["variant"]=String(o[:variant])
        end
    end
    println("\nDone (indices $(first(run_indices))..$(last(run_indices))) in $(o[:outdir])"); flush(stdout)
end

main()

