#!/usr/bin/env -S julia --project
# =============================================================================
# generate_spherex_tracers_tiled.jl
#
# Exclusive observed-redshift tiling benchmark for SPHEREx.
#
# Each sample is tiled with contiguous, non-overlapping bins in observed
# redshift.  The bins are uniform in u = ln(1+z), because
#
#     sigma_z(z) = sigma_rel * (1+z)
#
# so a constant width in u corresponds to a constant width in units of the
# local photo-z scatter.  By default there is one bin per local sigma_z:
#
#     Delta ln(1+z) ~= sigma_rel.
#
# The true-redshift kernel for observed bin i is
#
#   phi_i(z_true) proportional to sel_s(z_true)
#       * Integral_{zobs_lo}^{zobs_hi} dz_obs p(z_obs | z_true),
#
# with Gaussian p(z_obs|z_true).  The kernels overlap in TRUE redshift even
# though the observed-z bins are exclusive.
#
# The same script builds the consistent diagonal Poisson noise
#
#   N_ii = 1 / [f_sky * Integral dz_true nbar_s(z_true) P_i(z_true)],
#   N_ij = 0 for i != j,
#
# where nbar_s(z) is read from shot_noise_d{s} = 1/nbar_s(z) when available.
#
# Outputs in TRACER_OUTDIR:
#   tracer_tiled_s{s}_b{idx:04d}.h5
#   tracer_list_tiled.txt
#   pairs_tiled.txt
#   tracer_meta_tiled.h5
#   Nshot_tiled.h5
#   tiled_summary.h5
#
# Important environment variables:
#   SPHEREX_PARAM_H5       parameter file
#   TRACER_OUTDIR          separate tiled output directory
#   TILE_BINS_PER_SIGMA    default 1.0
#   TILE_PAIR_MODE         within_sample (default) or all
#   CONT_NZ_TRUE           default 4097
#   CONT_ZMIN              default 0.05
#   CONT_ZMAX              default 4.6
#   FSKY                   default 1.0
# =============================================================================

using HDF5
using Dierckx
using Printf
using Statistics: median
using SpecialFunctions: erf

const REPO = joinpath(@__DIR__, "..")
const SPHEREX_H5 = get(
    ENV,
    "SPHEREX_PARAM_H5",
    joinpath(REPO, "spherex_params_opt_gaussian.h5"),
)
const OUTDIR = get(
    ENV,
    "TRACER_OUTDIR",
    joinpath(REPO, "examples", "tracers_tiled"),
)

const SURVEY_ZMIN = parse(Float64, get(ENV, "CONT_ZMIN", "0.05"))
const SURVEY_ZMAX = parse(Float64, get(ENV, "CONT_ZMAX", "4.6"))
const BINS_PER_SIGMA = parse(Float64, get(ENV, "TILE_BINS_PER_SIGMA", "1.0"))
const N_Z_TRUE = parse(Int, get(ENV, "CONT_NZ_TRUE", "4097"))
const FSKY = parse(Float64, get(ENV, "FSKY", "1.0"))
const PAIR_MODE = lowercase(strip(get(ENV, "TILE_PAIR_MODE", "within_sample")))

const SQRT2 = sqrt(2.0)

# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------

function trap_weights(x::AbstractVector{<:Real})
    n = length(x)
    n >= 2 || error("trap_weights needs at least two points")

    w = Vector{Float64}(undef, n)
    w[1] = 0.5 * (x[2] - x[1])
    w[end] = 0.5 * (x[end] - x[end - 1])

    @inbounds for k in 2:(n - 1)
        w[k] = 0.5 * (x[k + 1] - x[k - 1])
    end

    return w
end

trapz(x, y) = sum(trap_weights(x) .* y)

function sample_sigmaz_rel(f, sample::Int)::Float64
    if haskey(f, "sigmaz_$sample") && haskey(f, "zmid$sample")
        sigma_z = Float64.(read(f["sigmaz_$sample"]))
        zmid = Float64.(read(f["zmid$sample"]))
        rel = sort(sigma_z ./ (1 .+ zmid))
        return rel[cld(length(rel), 2)]
    end

    return (0.003, 0.01, 0.03, 0.10, 0.20)[sample]
end

# Probability that a galaxy at z_true is observed inside [zlo,zhi].
@inline function observed_bin_probability(
    z_true::Float64,
    zlo::Float64,
    zhi::Float64,
    sigma_rel::Float64,
)::Float64
    sigma_z = sigma_rel * (1.0 + z_true)
    hi = (zhi - z_true) / (SQRT2 * sigma_z)
    lo = (zlo - z_true) / (SQRT2 * sigma_z)
    return clamp(0.5 * (erf(hi) - erf(lo)), 0.0, 1.0)
end

# Exact probability that z_obs lies anywhere inside the survey range.
@inline survey_capture_probability(z_true, sigma_rel) =
    observed_bin_probability(z_true, SURVEY_ZMIN, SURVEY_ZMAX, sigma_rel)

# -----------------------------------------------------------------------------
# Bin choice
# -----------------------------------------------------------------------------

"""
    tiled_edges(zmin,zmax,sigma_rel; bins_per_sigma=1.0)

Return contiguous observed-z bin edges that are uniform in ln(1+z).

For sigma_z(z)=sigma_rel*(1+z), one bin per local photo-z sigma means
Delta ln(1+z) approximately sigma_rel.  `bins_per_sigma=2` gives half-sigma
bins; `bins_per_sigma=0.5` gives two-sigma bins.
"""
function tiled_edges(
    zmin::Float64,
    zmax::Float64,
    sigma_rel::Float64;
    bins_per_sigma::Float64 = 1.0,
)
    bins_per_sigma > 0 || error("TILE_BINS_PER_SIGMA must be positive")

    umin = log1p(zmin)
    umax = log1p(zmax)
    span_u = umax - umin

    nbins = max(2, round(Int, bins_per_sigma * span_u / sigma_rel))
    uedges = collect(range(umin, umax; length = nbins + 1))
    edges = exp.(uedges) .- 1.0

    edges[1] = zmin
    edges[end] = zmax

    return edges
end

# Midpoint in u=ln(1+z), not the arithmetic midpoint in z.
log_midpoint(zlo, zhi) = exp(0.5 * (log1p(zlo) + log1p(zhi))) - 1.0

# -----------------------------------------------------------------------------
# Number-density loader
# -----------------------------------------------------------------------------

"""
    load_nbar_spline(f,sample,ztest,sel_v)

Return nbar_s(z) [gal/sr/unit-z].  Prefer shot_noise_d{s}=1/nbar_s(z).
Fall back to the selection shape normalized to the scalar total density.
"""
function load_nbar_spline(f, sample::Int, ztest, sel_v)
    dense_key = "shot_noise_d$sample"
    scalar_key = "shot_noise_$sample"

    if haskey(f, dense_key)
        inv_nbar = Float64.(read(f[dense_key]))
        length(inv_nbar) == length(ztest) ||
            error("$dense_key length mismatch with ztest")

        nbar = [x > 0 ? 1.0 / x : 0.0 for x in inv_nbar]
        return Spline1D(ztest, nbar, k = 3), dense_key
    end

    if haskey(f, scalar_key)
        total_surface_density = 1.0 / Float64(read(f[scalar_key]))
        norm_sel = trapz(ztest, sel_v)
        norm_sel > 0 || error("selection normalization <=0 for sample $sample")
        nbar = total_surface_density .* (sel_v ./ norm_sel)
        return Spline1D(ztest, nbar, k = 3), "sel_func_$(sample)_com+$scalar_key"
    end

    error(
        "sample $sample lacks both $dense_key and " *
        "(sel_func_$(sample)_com + $scalar_key)",
    )
end

# -----------------------------------------------------------------------------
# Tracer writer
# -----------------------------------------------------------------------------

function write_tiled_tracer(
    sample::Int,
    bin_index::Int,
    zlo::Float64,
    zhi::Float64,
    out_path::String;
    z_true,
    weights,
    bg_spl,
    sel_spl,
    be_spl,
    Q_spl,
    nbar_spl,
    sigma_rel::Float64,
)
    nz = length(z_true)

    probability = Vector{Float64}(undef, nz)
    phi_raw = Vector{Float64}(undef, nz)
    nbar_integrand = Vector{Float64}(undef, nz)

    @inbounds for k in eachindex(z_true)
        zt = z_true[k]
        pbin = observed_bin_probability(zt, zlo, zhi, sigma_rel)
        probability[k] = pbin
        phi_raw[k] = pbin * max(sel_spl(zt), 0.0)
        nbar_integrand[k] = pbin * max(nbar_spl(zt), 0.0)
    end

    phi_norm = sum(weights .* phi_raw)
    phi_norm > 0 || error(
        "nonpositive phi normalization for sample=$sample bin=$bin_index",
    )

    nbar_bin = sum(weights .* nbar_integrand)
    nbar_bin > 0 || error(
        "nonpositive bin surface density for sample=$sample bin=$bin_index",
    )

    phi_out = phi_raw ./ phi_norm
    zmid = log_midpoint(zlo, zhi)

    bg_out = [bg_spl(z) for z in z_true]
    be_out = [be_spl(z) for z in z_true]
    Q_out = [Q_spl(z) for z in z_true]

    h5open(out_path, "w") do fo
        fo["z"] = z_true
        fo["bg"] = bg_out
        fo["be"] = be_out
        fo["Q"] = Q_out
        fo["phi"] = phi_out

        fo["sample"] = sample
        fo["bin_index"] = bin_index
        fo["z_obs"] = zmid
        fo["zmid"] = zmid
        fo["zobs_lo"] = zlo
        fo["zobs_hi"] = zhi
        fo["delta_zobs"] = zhi - zlo
        fo["sigmaz_rel"] = sigma_rel
        fo["phi_raw_norm"] = phi_norm
        fo["nbar_bin"] = nbar_bin
        fo["window_type"] = "exclusive_observed_z_erf_tile"
        fo["is_tiled"] = 1
        fo["is_continuous"] = 0
        fo["bins_per_sigma"] = BINS_PER_SIGMA
    end

    return (
        path = out_path,
        sample = sample,
        bin_index = bin_index,
        z_obs = zmid,
        zlo = zlo,
        zhi = zhi,
        delta_zobs = zhi - zlo,
        sigma_rel = sigma_rel,
        phi_norm = phi_norm,
        nbar_bin = nbar_bin,
    )
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    samples = parse.(Int, split(get(ENV, "CONT_SAMPLES", "1,2,3,4,5"), ","))

    isfile(SPHEREX_H5) || error("parameter file not found: $SPHEREX_H5")
    N_Z_TRUE >= 257 || error("CONT_NZ_TRUE must be at least 257")
    FSKY > 0 || error("FSKY must be positive")
    PAIR_MODE in ("within_sample", "all") ||
        error("TILE_PAIR_MODE must be within_sample or all")

    mkpath(OUTDIR)

    @info "parameter file: $SPHEREX_H5"
    @info "output directory: $OUTDIR"
    @info "TILE_BINS_PER_SIGMA=$BINS_PER_SIGMA"
    @info "CONT_NZ_TRUE=$N_Z_TRUE"
    @info "PAIR_MODE=$PAIR_MODE"

    tracer_meta = NamedTuple[]
    sample_summaries = NamedTuple[]
    nbar_sources = Dict{Int,String}()

    h5open(SPHEREX_H5, "r") do f
        ztest = Float64.(read(f["ztest"]))
        z_trunc = Float64.(read(f["z_trunc"]))

        zmin = max(SURVEY_ZMIN, ztest[1], z_trunc[1])
        zmax = min(SURVEY_ZMAX, ztest[end], z_trunc[end])
        z_true = collect(range(zmin, zmax; length = N_Z_TRUE))
        weights = trap_weights(z_true)

        be_spl = Spline1D(z_trunc, Float64.(read(f["b_e"])), k = 3)
        Q_spl = Spline1D(z_trunc, Float64.(read(f["Q"])), k = 3)

        for sample in samples
            bg_v = Float64.(read(f["b_$sample"]))
            sel_v = Float64.(read(f["sel_func_$(sample)_com"]))

            bg_spl = Spline1D(ztest, bg_v, k = 3)
            sel_spl = Spline1D(ztest, sel_v, k = 3)
            nbar_spl, nbar_source = load_nbar_spline(f, sample, ztest, sel_v)
            nbar_sources[sample] = nbar_source

            sigma_rel = sample_sigmaz_rel(f, sample)
            edges = tiled_edges(
                zmin,
                zmax,
                sigma_rel;
                bins_per_sigma = BINS_PER_SIGMA,
            )
            nbins = length(edges) - 1

            # Closure checks in observed-z probability and galaxy counts.
            p_sum = zeros(Float64, length(z_true))
            nbar_sum = 0.0

            for bin_index in 1:nbins
                zlo = edges[bin_index]
                zhi = edges[bin_index + 1]

                out_path = joinpath(
                    OUTDIR,
                    @sprintf("tracer_tiled_s%d_b%04d.h5", sample, bin_index),
                )

                meta = write_tiled_tracer(
                    sample,
                    bin_index,
                    zlo,
                    zhi,
                    out_path;
                    z_true = z_true,
                    weights = weights,
                    bg_spl = bg_spl,
                    sel_spl = sel_spl,
                    be_spl = be_spl,
                    Q_spl = Q_spl,
                    nbar_spl = nbar_spl,
                    sigma_rel = sigma_rel,
                )

                push!(tracer_meta, meta)
                nbar_sum += meta.nbar_bin

                @inbounds for k in eachindex(z_true)
                    p_sum[k] += observed_bin_probability(
                        z_true[k], zlo, zhi, sigma_rel,
                    )
                end
            end

            p_capture = [survey_capture_probability(z, sigma_rel) for z in z_true]
            partition_error = maximum(abs.(p_sum .- p_capture))

            nbar_capture = sum(
                weights .* [
                    max(nbar_spl(z), 0.0) * survey_capture_probability(z, sigma_rel)
                    for z in z_true
                ],
            )
            count_closure = nbar_sum / nbar_capture - 1.0

            widths_u = diff(log1p.(edges))
            widths_z = diff(edges)

            push!(
                sample_summaries,
                (
                    sample = sample,
                    sigma_rel = sigma_rel,
                    nbins = nbins,
                    median_delta_log1pz = median(widths_u),
                    median_delta_z = median(widths_z),
                    min_delta_z = minimum(widths_z),
                    max_delta_z = maximum(widths_z),
                    partition_error = partition_error,
                    count_closure = count_closure,
                    observed_surface_density = nbar_sum,
                ),
            )

            @printf(
                "S%d: sigma_rel=%.4f  nbins=%d  median Delta ln(1+z)=%.6f (%.3f sigma)  median Delta z=%.5f  closure=%+.3e  partition=%.3e\n",
                sample,
                sigma_rel,
                nbins,
                median(widths_u),
                median(widths_u) / sigma_rel,
                median(widths_z),
                count_closure,
                partition_error,
            )
        end
    end

    ntracer = length(tracer_meta)
    @printf("\nTotal tiled tracers: %d\n", ntracer)

    # -------------------------------------------------------------------------
    # Tracer list
    # -------------------------------------------------------------------------

    list_path = joinpath(OUTDIR, "tracer_list_tiled.txt")
    open(list_path, "w") do io
        println(io, "# exclusive observed-z tiled SPHEREx tracers")
        println(io, "# bins uniform in ln(1+z)")
        println(io, "# TILE_BINS_PER_SIGMA=$BINS_PER_SIGMA")
        println(io, "# total=$ntracer")

        for meta in tracer_meta
            println(io, meta.path)
        end
    end

    # -------------------------------------------------------------------------
    # Pair specification
    # -------------------------------------------------------------------------

    pairs_path = joinpath(OUTDIR, "pairs_tiled.txt")
    npairs = 0

    open(pairs_path, "w") do io
        pieces = String[]

        for i in 1:ntracer, j in i:ntracer
            if PAIR_MODE == "within_sample"
                tracer_meta[i].sample == tracer_meta[j].sample || continue
            end

            push!(pieces, "$i-$j")
            npairs += 1
        end

        print(io, join(pieces, ","))
    end

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    meta_path = joinpath(OUTDIR, "tracer_meta_tiled.h5")
    h5open(meta_path, "w") do fo
        fo["sample"] = [x.sample for x in tracer_meta]
        fo["bin_index"] = [x.bin_index for x in tracer_meta]
        fo["z_obs"] = [x.z_obs for x in tracer_meta]
        fo["zobs_lo"] = [x.zlo for x in tracer_meta]
        fo["zobs_hi"] = [x.zhi for x in tracer_meta]
        fo["delta_zobs"] = [x.delta_zobs for x in tracer_meta]
        fo["sigmaz_rel"] = [x.sigma_rel for x in tracer_meta]
        fo["phi_raw_norm"] = [x.phi_norm for x in tracer_meta]
        fo["nbar_bin"] = [x.nbar_bin for x in tracer_meta]
        fo["tracer_paths"] = [x.path for x in tracer_meta]
        fo["bins_per_sigma"] = BINS_PER_SIGMA
        fo["pair_mode"] = PAIR_MODE
        fo["survey_zmin"] = SURVEY_ZMIN
        fo["survey_zmax"] = SURVEY_ZMAX
    end

    # -------------------------------------------------------------------------
    # Consistent diagonal Poisson noise
    # -------------------------------------------------------------------------

    noise = zeros(Float64, ntracer, ntracer)
    @inbounds for i in 1:ntracer
        noise[i, i] = 1.0 / (FSKY * tracer_meta[i].nbar_bin)
    end

    noise_path = joinpath(OUTDIR, "Nshot_tiled.h5")
    h5open(noise_path, "w") do fo
        fo["N"] = noise
        fo["sample"] = [x.sample for x in tracer_meta]
        fo["z_obs"] = [x.z_obs for x in tracer_meta]
        fo["zobs_lo"] = [x.zlo for x in tracer_meta]
        fo["zobs_hi"] = [x.zhi for x in tracer_meta]
        fo["delta_zobs"] = [x.delta_zobs for x in tracer_meta]
        fo["nbar_bin"] = [x.nbar_bin for x in tracer_meta]
        fo["nbar_eff"] = [x.nbar_bin for x in tracer_meta]
        fo["fsky"] = FSKY

        provenance = create_group(fo, "provenance")
        provenance["noise_type"] = "exclusive observed-z bins; diagonal Poisson"
        provenance["window_type"] = "exclusive_observed_z_erf_tile"
        provenance["bins_per_sigma"] = BINS_PER_SIGMA
        provenance["bin_coordinate"] = "uniform in ln(1+z)"
        provenance["formula"] = "Nii=1/(fsky*int dz_true nbar_s(z_true) P_i(z_true))"
        provenance["params"] = SPHEREX_H5
    end

    # -------------------------------------------------------------------------
    # Summary HDF5
    # -------------------------------------------------------------------------

    summary_path = joinpath(OUTDIR, "tiled_summary.h5")
    h5open(summary_path, "w") do fo
        fo["sample"] = [x.sample for x in sample_summaries]
        fo["sigma_rel"] = [x.sigma_rel for x in sample_summaries]
        fo["nbins"] = [x.nbins for x in sample_summaries]
        fo["median_delta_log1pz"] = [x.median_delta_log1pz for x in sample_summaries]
        fo["median_delta_z"] = [x.median_delta_z for x in sample_summaries]
        fo["min_delta_z"] = [x.min_delta_z for x in sample_summaries]
        fo["max_delta_z"] = [x.max_delta_z for x in sample_summaries]
        fo["partition_error"] = [x.partition_error for x in sample_summaries]
        fo["count_closure"] = [x.count_closure for x in sample_summaries]
        fo["observed_surface_density"] = [x.observed_surface_density for x in sample_summaries]
        fo["bins_per_sigma"] = BINS_PER_SIGMA

        for sample in sort(collect(keys(nbar_sources)))
            fo["nbar_source_$sample"] = nbar_sources[sample]
        end
    end

    println("\nWrote:")
    println("  tracer list : $list_path")
    println("  pair file   : $pairs_path  ($npairs pairs)")
    println("  metadata    : $meta_path")
    println("  shot noise  : $noise_path")
    println("  summary     : $summary_path")

    println("\nRecommended first benchmark:")
    println("  TILE_BINS_PER_SIGMA=1.0  (Delta z_obs approximately sigma_z)")
    println("Then verify convergence with 0.5 and 2.0 without overwriting this directory.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

