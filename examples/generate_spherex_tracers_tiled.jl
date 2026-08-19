#!/usr/bin/env -S julia --project
# =============================================================================
# generate_spherex_tracers_tiled.jl
#
# Physical tomographic SPHEREx benchmark using EXCLUSIVE observed-redshift bins.
#
# For sample s, observed-z bins are contiguous and non-overlapping, with edges
# uniform in u = ln(1+z).  The Gaussian photo-z model is
#
#     sigma_z(z_true) = sigma_rel,s * (1 + z_true),
#
# and the true-z distribution of observed bin i is
#
#     n_i(z_true) = n_s(z_true)
#                   Integral_[zobs_lo,zobs_hi] dz_obs p(z_obs | z_true).
#
# The normalized PowerFull radial window is therefore
#
#     phi_i(z_true) = n_i(z_true) / Integral dz_true n_i(z_true).
#
# Galaxies are assigned to exactly one observed-z bin, so the Poisson noise is
# diagonal within each disjoint catalogue:
#
#     N_ii = 1 / nbar_i,     N_ij = 0  (i != j),
#
# where nbar_i is the surface density per steradian.  f_sky belongs in the
# Fisher prefactor and MUST NOT be inserted into N_ii.
#
# Default binning is conservative: Delta ln(1+z) <= 2 sigma_rel, implemented as
# TILE_BINS_PER_SIGMA=0.5.  Convergence tests should use 1.0 and 2.0 in separate
# output directories.
#
# Outputs in TRACER_OUTDIR:
#   tracer_tiled_s{s}_b{idx:04d}.h5
#   tracer_list_tiled.txt
#   pairs_tiled.txt
#   tracer_meta_tiled.h5
#   Nshot_tiled.h5
#   tiled_summary.h5
#   tiled_config.txt
#
# Important environment variables:
#   SPHEREX_PARAM_H5       parameter file
#   TRACER_OUTDIR          tiled output directory
#   TILE_BINS_PER_SIGMA    default 0.5  (= approximately 2-sigma-wide bins)
#   TILE_PAIR_MODE         within_sample (default) or all
#   TILE_CLEAN_OUTDIR      default true; removes old tiled products in OUTDIR
#   CONT_NZ_TRUE           default 16385
#   CONT_ZMIN              observed-z minimum, default 0.05
#   CONT_ZMAX              observed-z maximum, default 4.6
#   CONT_SAMPLES           default 1,2,3,4,5
# =============================================================================

using HDF5
using Dierckx
using Printf
using Statistics: median
using SpecialFunctions: erf
using Dates

const REPO = normpath(joinpath(@__DIR__, ".."))

const SPHEREX_H5 = get(
    ENV,
    "SPHEREX_PARAM_H5",
    joinpath(REPO, "spherex_params_opt_gaussian.h5"),
)

const OUTDIR = get(
    ENV,
    "TRACER_OUTDIR",
    joinpath(REPO, "examples", "tracers_tiled_2sigma"),
)

const OBS_ZMIN = parse(Float64, get(ENV, "CONT_ZMIN", "0.05"))
const OBS_ZMAX = parse(Float64, get(ENV, "CONT_ZMAX", "4.6"))
const BINS_PER_SIGMA = parse(Float64, get(ENV, "TILE_BINS_PER_SIGMA", "0.5"))
const N_Z_TRUE = parse(Int, get(ENV, "CONT_NZ_TRUE", "16385"))
const PAIR_MODE = lowercase(strip(get(ENV, "TILE_PAIR_MODE", "within_sample")))

const SQRT2 = sqrt(2.0)

function parse_bool_env(name::String, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "true" : "false")))
    raw in ("1", "true", "yes", "y", "on") && return true
    raw in ("0", "false", "no", "n", "off") && return false
    error("$name must be a boolean; got '$raw'")
end

const CLEAN_OUTDIR = parse_bool_env("TILE_CLEAN_OUTDIR", true)

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
    sig_key = "sigmaz_$sample"
    zmid_key = "zmid$sample"

    if haskey(f, sig_key) && haskey(f, zmid_key)
        sigma_z = Float64.(read(f[sig_key]))
        zmid = Float64.(read(f[zmid_key]))
        length(sigma_z) == length(zmid) ||
            error("$sig_key and $zmid_key lengths do not match")

        rel = sort(sigma_z ./ (1 .+ zmid))
        sigma_rel = rel[cld(length(rel), 2)]
        sigma_rel > 0 || error("nonpositive sigma_rel for sample $sample")
        return sigma_rel
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
    sigma_z > 0 || return 0.0

    hi = (zhi - z_true) / (SQRT2 * sigma_z)
    lo = (zlo - z_true) / (SQRT2 * sigma_z)

    return clamp(0.5 * (erf(hi) - erf(lo)), 0.0, 1.0)
end

@inline survey_capture_probability(z_true, sigma_rel) =
    observed_bin_probability(z_true, OBS_ZMIN, OBS_ZMAX, sigma_rel)

# -----------------------------------------------------------------------------
# Bin choice
# -----------------------------------------------------------------------------

"""
    tiled_edges(zmin,zmax,sigma_rel; bins_per_sigma=0.5)

Return contiguous observed-z edges uniform in ln(1+z).

`bins_per_sigma = 0.5` gives approximately two-sigma-wide bins.
`bins_per_sigma = 1.0` gives approximately one-sigma-wide bins.
`bins_per_sigma = 2.0` gives approximately half-sigma-wide bins.

The number of bins is rounded upward so no bin is wider than the requested
Delta ln(1+z) = sigma_rel / bins_per_sigma.
"""
function tiled_edges(
    zmin::Float64,
    zmax::Float64,
    sigma_rel::Float64;
    bins_per_sigma::Float64 = 0.5,
)
    zmax > zmin || error("zmax must exceed zmin")
    sigma_rel > 0 || error("sigma_rel must be positive")
    bins_per_sigma > 0 || error("TILE_BINS_PER_SIGMA must be positive")

    umin = log1p(zmin)
    umax = log1p(zmax)
    span_u = umax - umin

    nbins = max(2, ceil(Int, bins_per_sigma * span_u / sigma_rel))
    uedges = collect(range(umin, umax; length = nbins + 1))
    edges = exp.(uedges) .- 1.0

    edges[1] = zmin
    edges[end] = zmax

    return edges
end

log_midpoint(zlo, zhi) = exp(0.5 * (log1p(zlo) + log1p(zhi))) - 1.0

# -----------------------------------------------------------------------------
# Number-density loader
# -----------------------------------------------------------------------------

"""
    load_nbar_spline(f,sample,ztest,sel_v)

Return nbar_s(z) in galaxies / steradian / unit-z.

Prefer shot_noise_d{s}=1/nbar_s(z).  If unavailable, use the selection shape
normalized to the sample's scalar total surface density.
"""
function load_nbar_spline(f, sample::Int, ztest, sel_v)
    dense_key = "shot_noise_d$sample"
    scalar_key = "shot_noise_$sample"

    if haskey(f, dense_key)
        inv_nbar = Float64.(read(f[dense_key]))
        length(inv_nbar) == length(ztest) ||
            error("$dense_key length mismatch with ztest")

        nbar = [isfinite(x) && x > 0 ? 1.0 / x : 0.0 for x in inv_nbar]
        maximum(nbar) > 0 || error("$dense_key produces zero number density")

        return Spline1D(ztest, nbar, k = 3), dense_key
    end

    if haskey(f, scalar_key)
        shot = Float64(read(f[scalar_key]))
        shot > 0 || error("$scalar_key must be positive")

        total_surface_density = 1.0 / shot
        norm_sel = trapz(ztest, sel_v)
        norm_sel > 0 || error("selection normalization <= 0 for sample $sample")

        nbar = total_surface_density .* (max.(sel_v, 0.0) ./ norm_sel)
        return Spline1D(ztest, nbar, k = 3), "sel_func_$(sample)_com+$scalar_key"
    end

    error(
        "sample $sample lacks both $dense_key and " *
        "(sel_func_$(sample)_com + $scalar_key)",
    )
end

# -----------------------------------------------------------------------------
# Output cleanup
# -----------------------------------------------------------------------------

function clean_old_outputs!(outdir::String)
    isdir(outdir) || return

    fixed_names = Set([
        "tracer_list_tiled.txt",
        "pairs_tiled.txt",
        "tracer_meta_tiled.h5",
        "Nshot_tiled.h5",
        "tiled_summary.h5",
        "tiled_config.txt",
    ])

    for name in readdir(outdir)
        remove_it =
            (startswith(name, "tracer_tiled_s") && endswith(name, ".h5")) ||
            (name in fixed_names)

        remove_it || continue
        path = joinpath(outdir, name)
        @info "removing stale tiled output" path
        rm(path; force = true, recursive = true)
    end
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
    be_spl,
    Q_spl,
    nbar_spl,
    sigma_rel::Float64,
    nbar_source::String,
)
    nz = length(z_true)

    probability = Vector{Float64}(undef, nz)
    phi_raw = Vector{Float64}(undef, nz)

    @inbounds for k in eachindex(z_true)
        zt = z_true[k]
        pbin = observed_bin_probability(zt, zlo, zhi, sigma_rel)
        nbar_true = max(nbar_spl(zt), 0.0)

        probability[k] = pbin
        phi_raw[k] = pbin * nbar_true
    end

    # The physical tomographic window and the bin surface density are the same
    # unnormalized quantity.  This avoids mixing a selection-shape spline into
    # phi while deriving shot noise from an unrelated nbar spline.
    nbar_bin = sum(weights .* phi_raw)
    nbar_bin > 0 || error(
        "nonpositive bin surface density for sample=$sample bin=$bin_index",
    )

    phi_norm = nbar_bin
    phi_out = phi_raw ./ phi_norm
    phi_integral = sum(weights .* phi_out)

    isapprox(phi_integral, 1.0; rtol = 0.0, atol = 2e-10) || error(
        "phi normalization failure for sample=$sample bin=$bin_index: " *
        "integral=$phi_integral",
    )

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
        fo["p_zobs_bin_given_ztrue"] = probability

        fo["sample"] = sample
        fo["bin_index"] = bin_index
        fo["z_obs"] = zmid
        fo["zmid"] = zmid
        fo["zobs_lo"] = zlo
        fo["zobs_hi"] = zhi
        fo["delta_zobs"] = zhi - zlo
        fo["delta_log1pz"] = log1p(zhi) - log1p(zlo)
        fo["sigmaz_rel"] = sigma_rel
        fo["bin_width_sigma"] = (log1p(zhi) - log1p(zlo)) / sigma_rel
        fo["phi_raw_norm"] = phi_norm
        fo["phi_integral"] = phi_integral
        fo["nbar_bin"] = nbar_bin
        fo["shot_noise"] = 1.0 / nbar_bin
        fo["nbar_source"] = nbar_source
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
        delta_log1pz = log1p(zhi) - log1p(zlo),
        sigma_rel = sigma_rel,
        bin_width_sigma = (log1p(zhi) - log1p(zlo)) / sigma_rel,
        phi_norm = phi_norm,
        phi_integral = phi_integral,
        nbar_bin = nbar_bin,
    )
end

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main()
    samples = parse.(Int, strip.(split(get(ENV, "CONT_SAMPLES", "1,2,3,4,5"), ",")))

    isfile(SPHEREX_H5) || error("parameter file not found: $SPHEREX_H5")
    OBS_ZMAX > OBS_ZMIN || error("CONT_ZMAX must exceed CONT_ZMIN")
    N_Z_TRUE >= 4097 || error("CONT_NZ_TRUE must be at least 4097")
    BINS_PER_SIGMA > 0 || error("TILE_BINS_PER_SIGMA must be positive")
    PAIR_MODE in ("within_sample", "all") ||
        error("TILE_PAIR_MODE must be within_sample or all")
    all(s -> 1 <= s <= 5, samples) || error("CONT_SAMPLES entries must lie in 1:5")
    length(unique(samples)) == length(samples) || error("CONT_SAMPLES contains duplicates")

    mkpath(OUTDIR)
    CLEAN_OUTDIR && clean_old_outputs!(OUTDIR)

    @info "parameter file" SPHEREX_H5
    @info "output directory" OUTDIR
    @info "TILE_BINS_PER_SIGMA" BINS_PER_SIGMA
    @info "target bin width in sigma" 1.0 / BINS_PER_SIGMA
    @info "CONT_NZ_TRUE" N_Z_TRUE
    @info "PAIR_MODE" PAIR_MODE

    tracer_meta = NamedTuple[]
    sample_summaries = NamedTuple[]
    nbar_sources = Dict{Int,String}()

    h5open(SPHEREX_H5, "r") do f
        ztest = Float64.(read(f["ztest"]))
        z_trunc = Float64.(read(f["z_trunc"]))

        model_zmin = max(ztest[1], z_trunc[1])
        model_zmax = min(ztest[end], z_trunc[end])
        model_zmax > model_zmin || error("ztest and z_trunc have no common support")

        OBS_ZMIN >= model_zmin || error(
            "observed zmin=$OBS_ZMIN lies below model support zmin=$model_zmin",
        )
        OBS_ZMAX <= model_zmax || error(
            "observed zmax=$OBS_ZMAX lies above model support zmax=$model_zmax",
        )

        # Use the full model-supported true-z interval.  If the parameter file
        # extends beyond the observed survey range, this correctly retains
        # galaxies that scatter into the first/last observed-z bins.
        z_true = collect(range(model_zmin, model_zmax; length = N_Z_TRUE))
        weights = trap_weights(z_true)

        be_spl = Spline1D(z_trunc, Float64.(read(f["b_e"])), k = 3)
        Q_spl = Spline1D(z_trunc, Float64.(read(f["Q"])), k = 3)

        for sample in samples
            bg_v = Float64.(read(f["b_$sample"]))
            sel_v = Float64.(read(f["sel_func_$(sample)_com"]))

            length(bg_v) == length(ztest) || error("b_$sample length mismatch")
            length(sel_v) == length(ztest) || error("sel_func_$(sample)_com length mismatch")

            bg_spl = Spline1D(ztest, bg_v, k = 3)
            nbar_spl, nbar_source = load_nbar_spline(f, sample, ztest, sel_v)
            nbar_sources[sample] = nbar_source

            sigma_rel = sample_sigmaz_rel(f, sample)
            edges = tiled_edges(
                OBS_ZMIN,
                OBS_ZMAX,
                sigma_rel;
                bins_per_sigma = BINS_PER_SIGMA,
            )
            nbins = length(edges) - 1

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
                    be_spl = be_spl,
                    Q_spl = Q_spl,
                    nbar_spl = nbar_spl,
                    sigma_rel = sigma_rel,
                    nbar_source = nbar_source,
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
            nbar_capture > 0 || error("nonpositive captured density for sample $sample")
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
                    bins_per_sigma_measured = sigma_rel / median(widths_u),
                    bin_width_sigma_measured = median(widths_u) / sigma_rel,
                    partition_error = partition_error,
                    count_closure = count_closure,
                    observed_surface_density = nbar_sum,
                ),
            )

            # Printf.@printf in Julia 1.8 requires a literal format string;
            # do not build the format with string concatenation inside the macro.
            @printf("S%d: sigma_rel=%.4f  nbins=%d  median Delta ln(1+z)=%.7f (%.4f sigma wide)  median Delta z=%.6f  count closure=%+.3e  partition=%.3e\n",
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
        println(io, "# target bin width=$(1.0 / BINS_PER_SIGMA) sigma")
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
        first_pair = true

        for i in 1:ntracer, j in i:ntracer
            if PAIR_MODE == "within_sample"
                tracer_meta[i].sample == tracer_meta[j].sample || continue
            end

            first_pair || print(io, ",")
            print(io, "$i-$j")
            first_pair = false
            npairs += 1
        end

        println(io)
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
        fo["delta_log1pz"] = [x.delta_log1pz for x in tracer_meta]
        fo["sigmaz_rel"] = [x.sigma_rel for x in tracer_meta]
        fo["bin_width_sigma"] = [x.bin_width_sigma for x in tracer_meta]
        fo["phi_raw_norm"] = [x.phi_norm for x in tracer_meta]
        fo["phi_integral"] = [x.phi_integral for x in tracer_meta]
        fo["nbar_bin"] = [x.nbar_bin for x in tracer_meta]
        fo["tracer_paths"] = [x.path for x in tracer_meta]
        fo["bins_per_sigma"] = BINS_PER_SIGMA
        fo["pair_mode"] = PAIR_MODE
        fo["observed_zmin"] = OBS_ZMIN
        fo["observed_zmax"] = OBS_ZMAX
        fo["n_z_true"] = N_Z_TRUE
    end

    # -------------------------------------------------------------------------
    # Consistent diagonal Poisson noise
    # -------------------------------------------------------------------------

    noise = zeros(Float64, ntracer, ntracer)
    @inbounds for i in 1:ntracer
        # nbar_bin is already a surface density per steradian.
        # Do NOT multiply or divide by f_sky here.
        noise[i, i] = 1.0 / tracer_meta[i].nbar_bin
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

        provenance = create_group(fo, "provenance")
        provenance["noise_type"] = "exclusive observed-z bins; diagonal Poisson"
        provenance["window_type"] = "exclusive_observed_z_erf_tile"
        provenance["bins_per_sigma"] = BINS_PER_SIGMA
        provenance["target_bin_width_sigma"] = 1.0 / BINS_PER_SIGMA
        provenance["bin_coordinate"] = "uniform in ln(1+z)"
        provenance["formula"] = "Nii=1/(int dz_true nbar_s(z_true) P_i(z_true)); fsky excluded"
        provenance["params"] = SPHEREX_H5
        provenance["pair_mode"] = PAIR_MODE
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
        fo["bins_per_sigma_measured"] = [x.bins_per_sigma_measured for x in sample_summaries]
        fo["bin_width_sigma_measured"] = [x.bin_width_sigma_measured for x in sample_summaries]
        fo["partition_error"] = [x.partition_error for x in sample_summaries]
        fo["count_closure"] = [x.count_closure for x in sample_summaries]
        fo["observed_surface_density"] = [x.observed_surface_density for x in sample_summaries]
        fo["bins_per_sigma"] = BINS_PER_SIGMA
        fo["target_bin_width_sigma"] = 1.0 / BINS_PER_SIGMA

        for sample in sort(collect(keys(nbar_sources)))
            fo["nbar_source_$sample"] = nbar_sources[sample]
        end
    end

    config_path = joinpath(OUTDIR, "tiled_config.txt")
    open(config_path, "w") do io
        println(io, "generated_at=$(Dates.now())")
        println(io, "parameter_file=$SPHEREX_H5")
        println(io, "window_type=exclusive_observed_z_erf_tile")
        println(io, "bin_coordinate=ln(1+z)")
        println(io, "bins_per_sigma=$BINS_PER_SIGMA")
        println(io, "target_bin_width_sigma=$(1.0 / BINS_PER_SIGMA)")
        println(io, "pair_mode=$PAIR_MODE")
        println(io, "observed_zmin=$OBS_ZMIN")
        println(io, "observed_zmax=$OBS_ZMAX")
        println(io, "n_z_true=$N_Z_TRUE")
        println(io, "shot_noise=Nii=1/nbar_bin; no fsky factor")
        println(io, "phi=nbar(z_true)*P(observed bin|z_true), normalized to unity")
    end

    println("\nWrote:")
    println("  tracer list : $list_path")
    println("  pair file   : $pairs_path  ($npairs pairs)")
    println("  metadata    : $meta_path")
    println("  shot noise  : $noise_path")
    println("  summary     : $summary_path")
    println("  config      : $config_path")

    println("\nProduction baseline:")
    println("  TILE_BINS_PER_SIGMA=0.5  (approximately two-sigma-wide bins)")
    println("Convergence runs, in separate directories:")
    println("  TILE_BINS_PER_SIGMA=1.0  (approximately one-sigma-wide bins)")
    println("  TILE_BINS_PER_SIGMA=2.0  (approximately half-sigma-wide bins)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

