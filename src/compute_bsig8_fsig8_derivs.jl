# =============================================================================
# compute_bsig8_fsig8_derivs.jl
#
# d C_l / d(polynomial coefficient) derivative cubes for the Khek et al.
# (arXiv:2212.05760) bsigma8 / fsigma8 nuisance scheme, in the angular-Cl pipeline.
#
# Scheme (their Eq. 27):
#     fsig8(z)     = fsig8_fid(z)     * sum_{i=0}^{N} a_i z^i      (shared)
#     bsig8^(s)(z) = bsig8_fid^(s)(z) * sum_{i=0}^{N} b_i^(s) z^i  (per sample s)
#   fiducial: a_0 = b_0 = 1, higher = 0.
#
# fsig8 ~ f and bsig8 ~ bg (sigma8 constant), so modulating the polynomial ==
# multiplicatively rescaling f(z) / bg(z).  At fiducial:
#   d(fsig8)/da_i      => rescale f  by (1 + eps z^i),  finite-difference
#   d(bsig8^s)/db_i^s  => rescale sample-s bg by (1 + eps z^i)
#
# The w/s/t/l integrals depend only on P(k)+geometry, not bg/f, so they are
# streamed/reused; each derivative is a fast re-contraction via
# compute_Cl_observed_multi(...; f_scale=, bg_scales=).
# Caveat (Option 2): f inside the Step-2 ISW integral is NOT modulated
# (relativistic x growth correction, 2nd-order small).  f modulation DOES flow
# through beta, B, A at contraction.
#
# Evaluate at the FIDUCIAL (GR) point: run against the fid_wiggle integrals.
#
# USAGE:
#   julia --project compute_bsig8_fsig8_derivs.jl \
#       --input=<ClGR_integrals_meta.h5> \
#       --tracer-list=<tracers.txt> --pairs=1-1,2-2,3-3,4-4,5-5 \
#       --cosmo-funcr=<cosmo_funcr.txt> --Omm0=0.3111 --H0=67.66 \
#       --outdir=<out> [--poly-order=5] [--eps=1e-3] [--fNL=0]
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
        :cosmo_funcr=>nothing, :outdir=>"bsig8_fsig8_derivs",
        :poly_order=>5, :eps=>1e-3, :fNL=>0.0, :H0=>nothing, :Omm0=>nothing,
        :delta_c=>1.686, :coeff_index=>nothing,
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
        end
    end
    for k in (:input,:tracer_list,:pairs,:cosmo_funcr,:H0,:Omm0)
        o[k]===nothing && error("missing required --$(replace(string(k),'_'=>'-'))")
    end
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
# Defined locally because it lives in the Step-3 driver, not in the CalcClGR module.
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

    # Map each tracer to its SPHEREx sample by parsing the "_s<N>_" token in the
    # filename (e.g. tracer_cont_s1_a0001.h5 -> sample 1).  The Khek et al.
    # bσ8 coefficients are PER SAMPLE, so a single b_i^(s) scales EVERY tracer
    # belonging to sample s (not one radial sub-slice).
    function sample_of(path::AbstractString)::Int
        m = match(r"_s(\d+)_", basename(path))
        m === nothing && error("cannot parse sample id (_s<N>_) from: $(basename(path))")
        return parse(Int, m.captures[1])
    end
    tracer_sample = [sample_of(p) for p in tracer_paths]
    samples = sort(unique(tracer_sample))
    nsamp = length(samples)
    counts = [count(==(s), tracer_sample) for s in samples]
    println("  $(n_tracers) tracers in $(nsamp) samples $(samples); " *
            "tracers/sample = $(counts); pairs=$(pairs)"); flush(stdout)

    # verbose=true activates the per-ℓ-part progress prints already inside the
    # streaming contraction (slice-read / assemble / contract timings), so we get
    # live progress without editing the module.
    contract(; f_scale=nothing, bg_scales=nothing) = begin
        _t0 = time()
        println("      [contract] starting full-pair streaming pass..."); flush(stdout)
        res = compute_Cl_observed_multi(o[:input], tracers, pairs, cf,
            z_of_r, dzdr_of_r, meta_ell_values;
            fNL=o[:fNL], Omm0=o[:Omm0], H0=o[:H0], delta_c=o[:delta_c],
            variant=:full,
            f_scale=f_scale, bg_scales=bg_scales, verbose=true)
        println("      [contract] done in $(round(time()-_t0, digits=1)) s"); flush(stdout)
        res
    end

    plus(i)  = (z -> 1.0 + eps * z^i)
    minus(i) = (z -> 1.0 - eps * z^i)
    one_fn   = (z -> 1.0)

    ells_out = collect(meta_ell_values)

    # Write a derivative cube in the SAME layout as the fiducial Cl_f0.h5:
    #   Cl_all : (nell, n_pairs)         convenience matrix
    #   ell    : (nell,)
    #   pairs/<ti>_<tj>/dCl : (nell,)    per-pair derivative (Fisher reads these)
    # dC is (nell, n_pairs); column p corresponds to pairs[p]=(ti,tj).
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
    #   index 0..N            -> fσ8 a_i                  (shared)
    #   index N+1 .. end      -> bσ8 b_i^(s), sample-major then order
    jobs = Vector{Tuple}()
    for i in 0:N
        push!(jobs, (:fsig8, i))
    end
    for s in samples
        for i in 0:N
            push!(jobs, (:bsig8, s, i))
        end
    end
    njobs = length(jobs)
    println("\nTotal coefficients (jobs) = $(njobs)  " *
            "[$(N+1) fσ8 + $(nsamp)×$(N+1) bσ8]"); flush(stdout)

    # Which jobs does THIS process run?  All of them (default) or one (array mode).
    run_indices = if o[:coeff_index] === nothing
        0:(njobs-1)
    else
        ci = o[:coeff_index]
        (0 <= ci < njobs) || error("--coeff-index=$ci out of range 0..$(njobs-1)")
        ci:ci
    end

    function run_fsig8(i)
        @time begin
            println("    [a_$(i)] contraction 1/2 (+ε)"); flush(stdout)
            Cl_p = contract(f_scale=plus(i))
            println("    [a_$(i)] contraction 2/2 (−ε)"); flush(stdout)
            Cl_m = contract(f_scale=minus(i))
            dC = (Cl_p .- Cl_m) ./ (2eps)
            outp = joinpath(o[:outdir], @sprintf("Cl_dfsig8_a%d.h5", i))
            write_deriv(outp, dC, Dict("coeff"=>String("a_$(i)"),
                "kind"=>"fsigma8_shared", "order"=>i, "eps"=>eps))
            print("  a_$(i) -> $(basename(outp))  ")
        end; flush(stdout)
    end

    function run_bsig8(s, i)
        @time begin
            bgp = [tracer_sample[k]==s ? plus(i)  : one_fn for k in 1:n_tracers]
            println("    [s$(s) b_$(i)] contraction 1/2 (+ε)"); flush(stdout)
            Cl_p = contract(bg_scales=bgp)
            bgm = [tracer_sample[k]==s ? minus(i) : one_fn for k in 1:n_tracers]
            println("    [s$(s) b_$(i)] contraction 2/2 (−ε)"); flush(stdout)
            Cl_m = contract(bg_scales=bgm)
            dC = (Cl_p .- Cl_m) ./ (2eps)
            outp = joinpath(o[:outdir], @sprintf("Cl_dbsig8_s%d_b%d.h5", s, i))
            write_deriv(outp, dC, Dict("coeff"=>String("b_$(i)^$(s)"),
                "kind"=>"bsigma8_persample", "sample"=>s, "order"=>i, "eps"=>eps))
            print("  s=$(s) b_$(i) -> $(basename(outp))  ")
        end; flush(stdout)
    end

    println("\n=== Computing coefficient indices $(first(run_indices))..$(last(run_indices)) ==="); flush(stdout)
    for ci in run_indices
        job = jobs[ci+1]   # jobs is 1-based; coeff-index is 0-based
        if job[1] === :fsig8
            run_fsig8(job[2])
        else
            run_bsig8(job[2], job[3])
        end
    end

    # Provenance meta — write only once (all-in-one mode, or array task 0) to
    # avoid concurrent writers racing on the same file.
    if o[:coeff_index] === nothing || o[:coeff_index] == 0
        h5open(joinpath(o[:outdir], "derivs_meta.h5"), "w") do f
            f["poly_order"]=N; f["eps"]=eps; f["nsamp"]=nsamp
            f["n_fsig8"]=N+1; f["n_bsig8_per_sample"]=N+1
            f["pairs_i"]=[p[1] for p in pairs]; f["pairs_j"]=[p[2] for p in pairs]
            f["njobs"]=njobs
            attributes(f)["scheme"]="Khek_etal_2212.05760_polynomial"
        end
    end
    println("\nDone (indices $(first(run_indices))..$(last(run_indices))) in $(o[:outdir])"); flush(stdout)
end

main()
