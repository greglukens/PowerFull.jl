#!/usr/bin/env -S julia --project
# =============================================================================
# compute_continuous_shotnoise.jl
#
# Build the dense, ℓ-independent shot-noise matrix N_αβ for the continuous
# (photo-z-convolved) tracers produced by generate_spherex_tracers_continuous.jl.
#
#   Usage:
#     julia --project compute_continuous_shotnoise.jl \
#         <spherex_params.h5> <tracer_meta_cont.h5> <Nshot_cont.h5> \
#         [--fsky=1.0] [--nz-true=4096]
#
# --------------------------------------------------------------------------
# WHY IT IS NOT DIAGONAL
# --------------------------------------------------------------------------
# In the tomographic case the bins are (nearly) disjoint, so a galaxy lands in
# exactly one bin and the Poisson noise is N_ij = δ_ij / N̄_i.  In the
# continuous case two observed redshifts z_obs^α and z_obs^β both receive
# galaxies scattered from the SAME true redshift z_true, so their shot noise is
# correlated.  The shared count density between estimators α and β is
#
#     K_αβ = (1/(f_sky N̄_s)) ∫dz_true  p(z_obs^α|z_true) p(z_obs^β|z_true)
#                  (dN_s/dz_true)  /  (norm_α · norm_β),
#
# with norm_α = ∫ p(z_obs^α|z_true) sel_s(z_true) dz_true the same φ
# normalization used when the tracer file was written.  The estimator
# overdensity at α has effective number density n̄_α = K_αα, so the shot-noise
# covariance is
#
#     N_αβ = K_αβ / (K_αα K_ββ),
#
# giving the familiar 1/n̄ on the diagonal and the correlated off-diagonal in
# between.  Cross-sample blocks (different s) have NO shared galaxies, so
# N_αβ = 0 there.  This is the real-space analogue of the off-diagonal SFB
# shot-noise matrix N^obs_{ℓ n n'} of Khek/Grasshorn-Gebhardt/Doré.
#
# In the wide-spacing limit (Δz_obs ≫ σ_z) the off-diagonal terms vanish and
# N_αβ → δ_αβ / N̄_α, recovering the tomographic diagonal form — a built-in
# sanity check.
#
# --------------------------------------------------------------------------
# NUMBER DENSITY n̄_s(z)
# --------------------------------------------------------------------------
# dN_s/dz_true = n̄_{sr,s}(z) [galaxies / steradian / unit z], read DIRECTLY
# from the parameter file.  Preferred key: `shot_noise_d\$s` = 1/n̄_{sr,s}(z)
# on the `ztest` grid (the survey's actual z-dependent inverse density).  If
# only the scalar `shot_noise_\$s` (= 1/N̄_s total) is present, the selection
# shape `sel_func_\$(s)_com` is used for the z-dependence and rescaled to that
# total.  No placeholder fallback — the real density is in the file.
# =============================================================================

using HDF5
using Dierckx
using Printf

# ---- fixed survey z-range (MUST match generate_spherex_tracers_continuous.jl)
# The scatter-kernel integration here reproduces the φ normalization stored in
# the tracer files; if the range differs, norm_α won't match and N_αβ is wrong.
const SURVEY_ZMIN = parse(Float64, get(ENV, "CONT_ZMIN", "0.05"))
const SURVEY_ZMAX = parse(Float64, get(ENV, "CONT_ZMAX", "4.6"))

gauss(x, μ, σ) = exp(-0.5 * ((x - μ) / σ)^2) / (sqrt(2π) * σ)

function trap_weights(x::AbstractVector{<:Real})
    n = length(x); w = Vector{Float64}(undef, n)
    w[1] = 0.5*(x[2]-x[1]); w[end] = 0.5*(x[end]-x[end-1])
    @inbounds for k in 2:n-1; w[k] = 0.5*(x[k+1]-x[k-1]); end
    return w
end
trapz(x, y) = sum(trap_weights(x) .* y)

function parse_args()
    o = Dict{Symbol,Any}(:fsky=>1.0, :nz_true=>4096,
                         :params=>nothing, :meta=>nothing, :out=>nothing)
    pos = String[]
    for a in ARGS
        if startswith(a, "--fsky=");        o[:fsky]   = parse(Float64, split(a,"=")[2])
        elseif startswith(a, "--nz-true="); o[:nz_true]= parse(Int,     split(a,"=")[2])
        elseif startswith(a, "--");         @warn "unknown arg $a"
        else push!(pos, a); end
    end
    length(pos) >= 3 || error("usage: compute_continuous_shotnoise.jl " *
        "<params.h5> <tracer_meta_cont.h5> <Nshot_cont.h5> [--fsky=..] [--nz-true=..]")
    o[:params], o[:meta], o[:out] = pos[1], pos[2], pos[3]
    return o
end

"""
    load_nbar_of_z(params_file, samples) -> Dict{Int,Tuple{Vector,Vector}}

Per-sample number density as a function of TRUE redshift, n̄_{sr,s}(z) in
[gal / sr / unit z], returned as (z_grid, nbar_values) per sample.

Reads the parameter file directly.  Priority of keys:
  1. `shot_noise_d\$s`  = 1 / n̄_{sr,s}(z)  on the `ztest` grid  (z-dependent) ← preferred
  2. `sel_func_\$(s)_com` shape × scalar `shot_noise_\$s` (= 1/N̄_s)  as a fallback
The z-dependent form is exact (it IS the survey's dN/dz per sr); the fallback
assumes dN/dz ∝ selection shape and only fixes the total.
"""
function load_nbar_of_z(params_file::String, samples)
    out = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    h5open(params_file, "r") do f
        ztest = Float64.(read(f["ztest"]))
        for s in samples
            if haskey(f, "shot_noise_d$s")
                inv_nbar = Float64.(read(f["shot_noise_d$s"]))   # = 1 / n̄_sr(z)
                length(inv_nbar) == length(ztest) ||
                    error("shot_noise_d$s length $(length(inv_nbar)) != ztest length $(length(ztest))")
                # invert; guard zeros (no galaxies -> nbar 0, infinite noise)
                nbar = [iv > 0 ? 1.0 / iv : 0.0 for iv in inv_nbar]
                out[s] = (ztest, nbar)
                @info "  sample $s: using z-dependent shot_noise_d$s (exact dN/dz per sr)"
            elseif haskey(f, "sel_func_$(s)_com") && haskey(f, "shot_noise_$s")
                sel = Float64.(read(f["sel_func_$(s)_com"]))
                Nbar_tot = 1.0 / Float64(read(f["shot_noise_$s"]))   # shot_noise_s = 1/N̄_s
                # normalize selection shape to integrate to N̄_tot over ztest
                wt = trap_weights(ztest); tot = sum(wt .* sel)
                nbar = Nbar_tot .* (sel ./ tot)
                out[s] = (ztest, nbar)
                @info "  sample $s: using sel_func shape × scalar shot_noise_$s (fallback)"
            else
                error("sample $s: parameter file lacks both shot_noise_d$s and " *
                      "(sel_func_$(s)_com + shot_noise_$s). Cannot set n̄(z).")
            end
        end
    end
    return out
end

function main()
    o = parse_args()
    @info "params=$(o[:params])  meta=$(o[:meta])  out=$(o[:out])  fsky=$(o[:fsky])"

    # --- load tracer metadata (sample, z_obs, phi_raw_norm) ---
    sample_of, zobs_of, norm_of, paths = h5open(o[:meta], "r") do f
        (Int.(read(f["sample"])), Float64.(read(f["z_obs"])),
         Float64.(read(f["phi_raw_norm"])), read(f["tracer_paths"]))
    end
    nα = length(sample_of)
    samples = sort(unique(sample_of))
    @printf("Loaded %d continuous tracers across samples %s\n", nα, string(samples))

    # --- per-sample σ_z and survey z-range ---
    σrel_of_s = Dict{Int,Float64}()
    zmin_g = Inf; zmax_g = -Inf
    h5open(o[:params], "r") do f
        ztest   = Float64.(read(f["ztest"]))
        z_trunc = Float64.(read(f["z_trunc"]))
        for s in samples
            if haskey(f, "sigmaz_$s") && haskey(f, "zmid$s")
                σz = Float64.(read(f["sigmaz_$s"])); zmid = Float64.(read(f["zmid$s"]))
                rel = sort(σz ./ (1 .+ zmid)); σrel_of_s[s] = rel[cld(length(rel),2)]
            else
                σrel_of_s[s] = (0.003,0.01,0.03,0.1,0.2)[s]
            end
        end
        zmin_g = max(SURVEY_ZMIN, ztest[1], z_trunc[1])
        zmax_g = min(SURVEY_ZMAX, ztest[end], z_trunc[end])
    end

    # --- per-sample n̄(z) and the SELECTION shape the generator used for φ ---
    # n̄_s(z) [gal/sr/unit z] from shot_noise_d$s (the true number density).
    # sel_s(z) = sel_func_$(s)_com — the SAME selection the tracer generator used
    # to build φ_α = p_α·sel/norm_α.  These are NOT assumed equal: the noise uses
    # sel for the weight and n̄ for the Poisson denominator separately.
    nbar_tab = load_nbar_of_z(o[:params], samples)
    nbar_spl = Dict(s => Spline1D(nbar_tab[s][1], nbar_tab[s][2], k=3) for s in samples)

    sel_spl = Dict{Int,Spline1D}()
    h5open(o[:params], "r") do f
        ztest = Float64.(read(f["ztest"]))
        for s in samples
            sel_v = Float64.(read(f["sel_func_$(s)_com"]))
            sel_spl[s] = Spline1D(ztest, sel_v, k=3)
        end
    end

    z_true = collect(range(zmin_g, zmax_g; length=o[:nz_true]))
    wt = trap_weights(z_true)

    nbar_z = Dict{Int,Vector{Float64}}()
    sel_z  = Dict{Int,Vector{Float64}}()
    for s in samples
        nbar_z[s] = [max(nbar_spl[s](z), 0.0) for z in z_true]
        sel_z[s]  = [max(sel_spl[s](z),  0.0) for z in z_true]
    end

    # --- precompute p(z_obs^α | z_true) on the true grid ---
    @info "Precomputing scatter kernels p(z_obs|z_true) for $nα tracers..."
    P = Matrix{Float64}(undef, o[:nz_true], nα)
    @inbounds for α in 1:nα
        s = sample_of[α]; σrel = σrel_of_s[s]; zo = zobs_of[α]
        for k in 1:o[:nz_true]
            zt = z_true[k]; σz = σrel * (1 + zt)
            P[k, α] = gauss(zo, zt, σz)
        end
    end

    # --- N_αβ = (1/f_sky) ∫ φ_α(z) φ_β(z) / n̄(z) dz  (continuous shot noise) ---
    # φ_α(z) = p(z_obs^α|z)·sel_s(z) / norm_α,  norm_α = norm_of[α] = ∫ p_α·sel dz
    # (the EXACT weight the generator stored and the signal projection uses).
    # =>  φ_α φ_β / n̄ = p_α p_β · sel² / (n̄ · norm_α norm_β).
    # The integrand factor sel²/n̄ is computed explicitly (NOT assuming sel ≡ n̄).
    # Broader/deeper samples integrate more galaxies → LOWER noise → higher S/N,
    # matching the SFB result (Khek et al. 2022) where high-σ_z samples dominate
    # the fNL constraint. (This replaces both the earlier K/(KK) form AND the
    # sel≡n̄ shortcut, either of which mis-scaled the per-sample noise.)
    @info "Assembling continuous shot noise N_αβ = (1/f_sky)∫ φ_αφ_β/n̄ dz (block-diagonal)..."
    N = zeros(Float64, nα, nα)
    idx_of_sample = Dict(s => findall(==(s), sample_of) for s in samples)
    fsky = o[:fsky]
    for s in samples
        idxs = idx_of_sample[s]
        ns_blk = length(idxs)
        # integrand factor per z: wt · sel²/n̄   (guard n̄→0)
        nb = nbar_z[s]; sl = sel_z[s]
        fac = similar(nb)
        @inbounds for k in eachindex(nb)
            fac[k] = nb[k] > 0 ? wt[k] * sl[k] * sl[k] / nb[k] : 0.0
        end
        @info "  sample $s: $ns_blk tracers → $(ns_blk*(ns_blk+1)÷2) pairs"
        flush(stderr)
        Threads.@threads for ii in 1:ns_blk
            α = idxs[ii]
            Pα = @view P[:, α]
            @inbounds for jj in ii:ns_blk
                β = idxs[jj]
                Pβ = @view P[:, β]
                acc = 0.0
                @simd for k in 1:o[:nz_true]
                    acc += fac[k] * Pα[k] * Pβ[k]
                end
                val = acc / (fsky * norm_of[α] * norm_of[β])
                N[α, β] = val; N[β, α] = val
            end
        end
    end

    # --- diagnostics ---
    offmax = 0.0; nnmax = 0.0
    @inbounds for s in samples
        idxs = idx_of_sample[s]
        for (ii, α) in enumerate(idxs), (jj, β) in enumerate(idxs)
            ii == jj && continue
            r = abs(N[α,β]) / sqrt(N[α,α]*N[β,β])
            offmax = max(offmax, r)
            if abs(ii-jj) == 1; nnmax = max(nnmax, r); end
        end
    end
    @printf("  max |corr| off-diagonal (within sample): %.3f\n", offmax)
    @printf("  max nearest-neighbor corr             : %.3f\n", nnmax)
    @printf("  (→ 0 means wide spacing/diagonal limit; →1 means heavy coupling)\n")

    # --- write ---
    h5open(o[:out], "w") do f
        f["N"]            = N                    # [nα × nα] shot-noise covariance
        f["sample"]       = sample_of
        f["z_obs"]        = zobs_of
        f["nbar_eff"]     = [N[α,α] > 0 ? 1.0/N[α,α] : 0.0 for α in 1:nα]  # eff n̄_α = 1/N_αα
        f["fsky"]         = fsky
        create_group(f, "provenance")
        f["provenance/params"] = o[:params]
        f["provenance/meta"]   = o[:meta]
        f["provenance/nz_true"]= o[:nz_true]
        for s in samples
            # integrated total N̄_s = ∫ n̄_s(z) dz over the survey range
            f["provenance/Nbar_sr_$s"] = sum(wt .* nbar_z[s])
            f["provenance/sigmaz_rel_$s"] = σrel_of_s[s]
        end
    end
    @printf("\nWrote shot-noise matrix N[%d×%d] to %s\n", nα, nα, o[:out])
    println("Add to the signal as  C̄_ℓ = C_ℓ + N  before inverting in the Fisher step.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
