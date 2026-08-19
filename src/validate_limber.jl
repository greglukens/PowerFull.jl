#!/usr/bin/env -S julia --project
# =============================================================================
#
# >> validate_limber.jl <<
#
# Validate prefix-sum integrals by comparing with Limber approximation.
#
# The base w functions (from TwoFAST) are already tested.
# This script checks the INTEGRALS computed in PowerFull.jl:
#
#   (3/D(r_i)) × ∫₀^{r_i} dlnr₁ rf(r₁) w^{p}_{ℓ,00}(r₁, r_j)
#
# where rf(r) = r × f(r) and the integral is a trapezoidal prefix sum.
#
# At high ℓ, the Limber approximation gives:
#   w^{p}_{ℓ,00}(r₁,r₂) → (ν/r₁)^p P(ν/r₁)/r₁² δ_D(r₁-r₂)
#
# The prefix-sum integral (for i ≥ j) then becomes:
#   (3/D(r_i)) × f(r_j) × (ν/r_j)^p × P(ν/r_j) / r_j²
#
# Note: f(r) is the kernel WITHOUT the extra r factor from rf = r×f.
# The r factor in rf cancels with dlnr = dr/r in the integration measure.
#
# IMPORTANT: At the exact diagonal (i=j), the trapezoidal rule gives the
# delta-function peak half weight (endpoint effect). We evaluate at i = j + offset
# to avoid this numerical artifact.
#
# Supports both single-file and multi-part HDF5 formats.
#
# Usage:
#   julia --project src/validate_limber.jl <integrals.h5>
#   julia --project src/validate_limber.jl <meta.h5>
#
#  March 2026
#  Donghui Jeong
# =============================================================================

using HDF5
using Dierckx
using DelimitedFiles
using Printf

# =============================================================================
# Power spectrum
# =============================================================================

struct PkSpectrum
    pkspl::Spline1D
    kmin::Float64
    kmax::Float64
    nslo::Float64
    nshi::Float64
    kmin_norm::Float64
    kmax_norm::Float64
end

function PkSpectrum(filename::String=joinpath(@__DIR__, "..", "data",
        "astropy_planck_2018_matterpower.dat"))
    data = readdlm(filename, comments=true)
    kk, pk = data[:,1], data[:,2]
    pkspl = Spline1D(kk, pk)

    k0 = kk[1]; P0 = pkspl(k0); Pp0 = derivative(pkspl, k0)
    nslo = k0 * Pp0 / P0
    kmin_norm = P0 / k0^nslo

    k0 = kk[end]; P0 = pkspl(k0); Pp0 = derivative(pkspl, k0)
    nshi = 4 + k0 * Pp0 / P0
    kmax_norm = P0 / (k0^(nshi - 4))

    PkSpectrum(pkspl, kk[1], kk[end], nslo, nshi, kmin_norm, kmax_norm)
end

function (pwr::PkSpectrum)(k::Real)
    if k < pwr.kmin
        return pwr.kmin_norm * k^pwr.nslo
    elseif k > pwr.kmax
        return pwr.kmax_norm * k^(pwr.nshi - 4)
    else
        return pwr.pkspl(k)
    end
end

"""
    transferk(k, pk) -> Float64

Transfer function T(k) = sqrt(P(k) / P_prim(k)) where
P_prim(k) = kmin_norm × k^nslo is the primordial power-law extrapolation
at low k. Matches the definition in run_twofast.jl:transferk so that
tk(k)^n gives the extra kernel weight for the n ≠ 0 base functions
(u for n=-1, v for n=-2).
"""
function transferk(k::Real, pk::PkSpectrum)::Float64
    p0 = pk.kmin_norm * k^pk.nslo
    return sqrt(pk(k) / p0)
end

# =============================================================================
# Cosmological kernel functions (must match build_and_export.jl)
# =============================================================================

include(joinpath(@__DIR__, "cosmofns.jl"))
using .cosmofns

# Kernel functions f(r) — WITHOUT the extra r factor.
# In the prefix sum, rf[k] = r_k * f(r_k) is multiplied by Δlnr = dr/r,
# so the effective integration measure is: rf[k] * Δlnr = r * f(r) * dr/r = f(r) * dr.
# The Limber delta function then picks out f(r_j), not r_j * f(r_j).

"""Kernel f_s(r) = a³ H³ Ωm (f-1) D"""
function f_s(r::Float64, cfns::cosmofn)::Float64
    z, a, H, Om, f, D = cfns(r)
    return a^3 * H^3 * Om * (f - 1) * D
end

"""Kernel f_t(r) = a² H² Ωm D"""
function f_t(r::Float64, cfns::cosmofn)::Float64
    z, a, H, Om, f, D = cfns(r)
    return a^2 * H^2 * Om * D
end

"""Kernel f_l(r) = a² H² Ωm D / r"""
function f_l(r::Float64, cfns::cosmofn)::Float64
    z, a, H, Om, f, D = cfns(r)
    return a^2 * H^2 * Om * D / r
end

# =============================================================================
# Limber prediction for prefix-sum integral
# =============================================================================

"""
    limber_prefix_1D(ell, r_i, r_j, f_fn, cfns, pk, p; n=0) -> Float64

Limber prediction for the stored 1D prefix-sum integral at (r_i, r_j):
    stored[i,j] = (3/D(r_i)) × ∫₀^{r_i} dr₁ f(r₁) b^{p,n}(r₁, r_j)

where b^{p,n} denotes the TwoFAST base function with transfer-function
weight T(k)^n: n=0 → w^p (standard), n=-1 → u^p, n=-2 → v^p.
The TwoFAST integrand is k^p P(k) T(k)^n (see run_twofast.jl:292).

At high ℓ, b^{p,n} → (ν/r₁)^p P(ν/r₁) T(ν/r₁)^n / r₁² δ(r₁-r_j), so:
    → (3/D(r_i)) × f(r_j) × (ν/r_j)^p × P(ν/r_j) × T(ν/r_j)^n / r_j²

Valid for r_i ≥ r_j (i.e. r_j is within the integration range).
"""
function limber_prefix_1D(ell::Int, r_i::Float64, r_j::Float64,
                           f_fn::Function, cfns::cosmofn,
                           pk::PkSpectrum, p::Int; n::Int=0)::Float64
    nu = ell + 0.5
    k = nu / r_j
    D_i = cfns.fDr(r_i)
    f_j = f_fn(r_j, cfns)
    tk_factor = n == 0 ? 1.0 : transferk(k, pk)^n
    return (3.0 / D_i) * f_j * (nu / r_j)^p * pk(k) * tk_factor / r_j^2
end

"""
    limber_prefix_1D_rp(ell, r_i, r_j, f_fn, cfns, pk, p; n=0) -> Float64

Limber prediction for the ;r' prefix-sum integral at (r_i, r_j):
    stored[i,j] = (3/D(r_j)) × ∫₀^{r_j} dr₂ f(r₂) b^{p,n}(r_i, r₂)

At high ℓ, b^{p,n} → (ν/r_i)^p P(ν/r_i) T(ν/r_i)^n / r_i² δ(r_i-r₂), so:
    → (3/D(r_j)) × f(r_i) × (ν/r_i)^p × P(ν/r_i) × T(ν/r_i)^n / r_i²

Valid for r_j ≥ r_i (i.e. r_i is within the integration range).
"""
function limber_prefix_1D_rp(ell::Int, r_i::Float64, r_j::Float64,
                              f_fn::Function, cfns::cosmofn,
                              pk::PkSpectrum, p::Int; n::Int=0)::Float64
    nu = ell + 0.5
    k = nu / r_i
    D_j = cfns.fDr(r_j)
    f_i = f_fn(r_i, cfns)
    tk_factor = n == 0 ? 1.0 : transferk(k, pk)^n
    return (3.0 / D_j) * f_i * (nu / r_i)^p * pk(k) * tk_factor / r_i^2
end

"""
    limber_prefix_1D_deriv(ell, r_i, r_j, f_fn, cfns, pk, p, jp) -> Float64

Limber prediction for the 1D prefix-sum integral with asymmetric base function
w^{p}_{ℓ,0j'}, using the derivative formulation:

    w^{p}_{ℓ,0j'}(r₁, r₂) = ∂_{r₂}^{j'} w^{p-j'}_{ℓ,00}(r₁, r₂)

Applying Limber to w^{p-j'}_{ℓ,00}:

    w^{p}_{ℓ,0j'} → ∂_{r₂}^{j'} [(ν/r₁)^{p-j'} P(ν/r₁)/r₁² δ_D(r₁-r₂)]

The 1D ;r integral reduces to a derivative of the bracket at r=r_j:

    I^{p}_{ℓ,0j';r}(r_i, r_j) = (3/D(r_i)) × d^{j'}/dr^{j'} B(r) |_{r=r_j}

where the bracket is

    B(r) ≡ f(r) × (ν/r)^{p-j'} × P(ν/r) / r²

The derivative is computed via a quintic spline on a dense log-r grid.
Valid for r_j within the integration range [0, r_i].
"""
function limber_prefix_1D_deriv(ell::Int, r_i::Float64, r_j::Float64,
                                 f_fn::Function, cfns::cosmofn,
                                 pk::PkSpectrum, p::Int, jp::Int)::Float64
    nu = ell + 0.5
    p_eff = p - jp

    # Build bracket function B(r) on a dense log-r grid around r_j.
    # Use enough points for a stable quintic spline (k=5) and its derivatives.
    # The r range spans the relevant k = nu/r values.
    r_lo = max(r_j / 3.0, nu / 400.0)   # stay within pk valid range
    r_hi = min(r_j * 3.0, nu / 1e-4)
    n_spl = 256
    ln_r_grid = collect(range(log(r_lo), log(r_hi), length=n_spl))
    r_grid = exp.(ln_r_grid)

    B_grid = similar(r_grid)
    @inbounds for idx in eachindex(r_grid)
        r = r_grid[idx]
        k = nu / r
        if k < 1e-5 || k > 500.0
            B_grid[idx] = 0.0
            continue
        end
        f_val = f_fn(r, cfns)
        B_grid[idx] = f_val * (nu / r)^p_eff * pk(k) / r^2
    end

    # Quintic spline for stable numerical derivatives up to 4th order
    spl = Spline1D(r_grid, B_grid, k=5)

    # Evaluate jp-th derivative at r_j
    local deriv_val::Float64
    if jp == 0
        deriv_val = evaluate(spl, r_j)
    else
        deriv_val = derivative(spl, r_j, nu=jp)
    end

    D_i = cfns.fDr(r_i)
    return (3.0 / D_i) * deriv_val
end

"""
    limber_prefix_1D_deriv_rp(ell, r_i, r_j, f_fn, cfns, pk, p, j) -> Float64

Limber prediction for the 1D ;r' prefix-sum integral with asymmetric base
function w^{p}_{ℓ,j0}, i.e. derivatives on the first argument. By the symmetry
w^{p}_{ℓ,j0}(r, r') = w^{p}_{ℓ,0j}(r', r), the Limber limit is the same
bracket derivative evaluated at r_i (rather than r_j), with prefactor (3/D(r_j)).

Valid for r_i within the integration range [0, r_j].
"""
function limber_prefix_1D_deriv_rp(ell::Int, r_i::Float64, r_j::Float64,
                                    f_fn::Function, cfns::cosmofn,
                                    pk::PkSpectrum, p::Int, j::Int)::Float64
    nu = ell + 0.5
    p_eff = p - j

    r_lo = max(r_i / 3.0, nu / 400.0)
    r_hi = min(r_i * 3.0, nu / 1e-4)
    n_spl = 256
    ln_r_grid = collect(range(log(r_lo), log(r_hi), length=n_spl))
    r_grid = exp.(ln_r_grid)

    B_grid = similar(r_grid)
    @inbounds for idx in eachindex(r_grid)
        r = r_grid[idx]
        k = nu / r
        if k < 1e-5 || k > 500.0
            B_grid[idx] = 0.0
            continue
        end
        f_val = f_fn(r, cfns)
        B_grid[idx] = f_val * (nu / r)^p_eff * pk(k) / r^2
    end

    spl = Spline1D(r_grid, B_grid, k=5)

    local deriv_val::Float64
    if j == 0
        deriv_val = evaluate(spl, r_i)
    else
        deriv_val = derivative(spl, r_i, nu=j)
    end

    D_j = cfns.fDr(r_j)
    return (3.0 / D_j) * deriv_val
end

"""
    limber_prefix_2D(ell, rr, ri_diag, f1_fn, f2_fn, cfns, pk, Δlnr) -> Float64

Limber prediction for the stored 2D prefix-sum integral at diagonal (r, r):
    stored[i,j] = (3/D(r_i))(3/D(r_j)) × ∫∫ dlnr₁ dlnr₂ rf₁(r₁) rf₂(r₂) w(r₁,r₂)

With Limber δ(r₁-r₂), the 2D integral collapses to 1D:
    → (3/D(r))² × ∫₀^r dlnr₁ f₁(r₁) f₂(r₁) (ν/r₁)^{-4} P(ν/r₁) / r₁

(The factor 1/r₁ comes from δ(r₁-r₂) = δ(lnr₁-lnr₂)/r₁ in log coordinates.)
"""
function limber_prefix_2D(ell::Int, rr::Vector{Float64}, ri_diag::Int,
                           f1_fn::Function, f2_fn::Function,
                           cfns::cosmofn, pk::PkSpectrum,
                           Δlnr::Float64)::Float64
    nu = ell + 0.5
    r_diag = rr[ri_diag]
    D_diag = cfns.fDr(r_diag)

    limber_sum = 0.0
    @inbounds for ii in 1:ri_diag
        ri = rr[ii]
        k = nu / ri
        if k < 1e-5 || k > 500.0
            continue
        end
        f1_val = f1_fn(ri, cfns)
        f2_val = f2_fn(ri, cfns)
        limber_sum += f1_val * f2_val * (nu / ri)^(-4) * pk(k) / ri
    end
    return (3.0 / D_diag)^2 * Δlnr * limber_sum
end

# =============================================================================
# Multi-part HDF5 reader
# =============================================================================

struct MultiPartReader
    meta_file::String
    part_files::Vector{String}
    rr::Vector{Float64}
    ell_values::Vector{Int}
    ell_ranges::Matrix{Int}  # [n_parts, 2]
    n_parts::Int
end

function MultiPartReader(input_file::String)
    if !isfile(input_file)
        error("File not found: $input_file")
    end

    # Detect format
    is_split = h5open(input_file, "r") do f
        haskey(f, "metadata") && haskey(f["metadata"], "n_parts")
    end

    if is_split
        # Multi-part: input_file is the meta file
        meta_file = input_file
        basedir = dirname(abspath(input_file))
        rr, ell_values, ell_ranges, part_filenames = h5open(meta_file, "r") do f
            rr = Float64.(read(f["grid/rr"]))
            ell_values = Int.(read(f["grid/ell_values"]))
            ell_ranges = Int.(read(f["metadata/ell_ranges"]))
            part_filenames = String.(read(f["part_files"]))
            (rr, ell_values, ell_ranges, part_filenames)
        end
        part_files = [joinpath(basedir, pf) for pf in part_filenames]
        n_parts = length(part_files)
    else
        # Single file: create a trivial 1-part reader
        rr, ell_values = h5open(input_file, "r") do f
            rr = Float64.(read(f["grid/rr"]))
            ell_values = Int.(read(f["grid/ell_values"]))
            (rr, ell_values)
        end
        n_parts = 1
        part_files = [input_file]
        ell_ranges = reshape([1, length(ell_values)], 1, 2)
        meta_file = input_file
    end

    return MultiPartReader(meta_file, part_files, rr, ell_values, ell_ranges, n_parts)
end

"""Read a specific ell slice from the correct part file. Returns [nr, nr] Matrix."""
function read_ell_slice(reader::MultiPartReader, key::String, group::String,
                         ell_idx::Int)::Union{Nothing, Matrix{Float64}}
    for part_idx in 1:reader.n_parts
        ell_start = reader.ell_ranges[part_idx, 1]
        ell_end = reader.ell_ranges[part_idx, 2]
        if ell_start <= ell_idx <= ell_end
            local_idx = ell_idx - ell_start + 1
            return h5open(reader.part_files[part_idx], "r") do f
                if !haskey(f[group], key)
                    return nothing
                end
                data = read(f["$group/$key"])
                return Matrix{Float64}(data[local_idx, :, :])
            end
        end
    end
    return nothing
end

"""Check if a key exists in the first part file's group."""
function has_key(reader::MultiPartReader, key::String, group::String)::Bool
    return h5open(reader.part_files[1], "r") do f
        haskey(f, group) && haskey(f[group], key)
    end
end

# =============================================================================
# Main
# =============================================================================

# Offset for off-diagonal comparison to avoid trapezoidal endpoint effect.
# At high ℓ, the delta function has width ~1 grid point; offset=4 ensures
# the peak is fully included as an interior point of the trapezoidal sum.
const DIAG_OFFSET = 4

function main(input_file::String;
              matterpower::String=joinpath(@__DIR__, "..", "data",
                  "astropy_planck_2018_matterpower.dat"),
              cosmo_funcr::String=joinpath(@__DIR__, "..", "data", "cosmo_funcr_astropy_planck2018.txt"))
    println("="^70)
    println("Limber Validation of Prefix-Sum Integrals")
    println("="^70)
    println("Input: $input_file")
    println("Matterpower: $matterpower")
    println("Cosmo_funcr: $cosmo_funcr\n")

    pk = PkSpectrum(matterpower)
    cfns_data = cosmofn(cosmo_funcr)
    reader = MultiPartReader(input_file)

    rr = reader.rr
    ell_values = reader.ell_values
    nr = length(rr)
    nell = length(ell_values)
    Δlnr = log(rr[2] / rr[1])

    println("Grid: nr=$nr, r=[$(round(rr[1],digits=1)), $(round(rr[end],digits=1))] Mpc/h")
    println("ell: $nell values, [$(ell_values[1]), $(ell_values[end])]")
    if reader.n_parts > 1
        println("Format: $(reader.n_parts)-part split files")
    end

    # Pick a mid-range r point (avoid boundaries)
    r_mid_idx = nr ÷ 2
    r_mid = rr[r_mid_idx]
    z_mid = cfns_data.fzr(r_mid)
    @printf("\nReference point: r_j = %.1f Mpc/h (z = %.3f), idx = %d\n", r_mid, z_mid, r_mid_idx)
    @printf("Off-diagonal offset: i = j + %d (r_i = %.1f Mpc/h)\n",
            DIAG_OFFSET, rr[r_mid_idx + DIAG_OFFSET])

    # =====================================================================
    # Test 1: 1D prefix-sum integrals vs Limber
    # =====================================================================
    # stored[i,j] = (3/D(r_i)) × Δlnr × Σ_{k=1}^{i} rf[k] w[k,j]  (trapezoidal)
    # Limber(i,j) = (3/D(r_i)) × f(r_j) × (ν/r_j)^p × P(ν/r_j) / r_j²

    # Test cases for ;r: (key, kernel, p, n, j, jp, label)
    # j=jp=0 uses the analytic limber_prefix_1D (supports n ≠ 0);
    # j=0, jp>0 uses the derivative form limber_prefix_1D_deriv.
    test_cases = [
        # p=-2, n=0, j=j'=0
        ("s_m2_0_0_r",     f_s, -2,  0, 0, 0, "s_{-2,00;r}   (density)"),
        ("t_m2_0_0_r",     f_t, -2,  0, 0, 0, "t_{-2,00;r}   (Doppler)"),
        ("tl_m2_0_0_r",     f_l, -2,  0, 0, 0, "l_{-2,00;r}   (lensing)"),
        # p=-4, n=0, j=j'=0
        ("s_m4_0_0_r",     f_s, -4,  0, 0, 0, "s_{-4,00;r}   (density, p=-4)"),
        ("t_m4_0_0_r",     f_t, -4,  0, 0, 0, "t_{-4,00;r}   (Doppler, p=-4)"),
        ("tl_m4_0_0_r",     f_l, -4,  0, 0, 0, "l_{-4,00;r}   (lensing, p=-4)"),
        # p=-4, n=-1, j=j'=0 — u-function base (scrs/scrt/scrl)
        ("scrs_m4_0_0_r",  f_s, -4, -1, 0, 0, "𝔰_{-4,00;r}   (density, n=-1)"),
        ("scrt_m4_0_0_r",  f_t, -4, -1, 0, 0, "𝔱_{-4,00;r}   (Doppler, n=-1)"),
        ("tscrl_m4_0_0_r",  f_l, -4, -1, 0, 0, "𝔩_{-4,00;r}   (lensing, n=-1)"),
        # p=-3, j=0, j'=1 — p_eff = -4, one derivative
        ("s_m3_0_1_r",     f_s, -3,  0, 0, 1, "s_{-3,01;r}   (density, ∂)"),
        ("t_m3_0_1_r",     f_t, -3,  0, 0, 1, "t_{-3,01;r}   (Doppler, ∂)"),
        ("tl_m3_0_1_r",     f_l, -3,  0, 0, 1, "l_{-3,01;r}   (lensing, ∂)"),
        # p=-2, j=0, j'=2 — p_eff = -4, two derivatives
        ("s_m2_0_2_r",     f_s, -2,  0, 0, 2, "s_{-2,02;r}   (density, ∂²)"),
        ("t_m2_0_2_r",     f_t, -2,  0, 0, 2, "t_{-2,02;r}   (Doppler, ∂²)"),
        ("tl_m2_0_2_r",     f_l, -2,  0, 0, 2, "l_{-2,02;r}   (lensing, ∂²)"),
    ]

    for (key, f_fn, p, n_val, j_val, jp_val, label) in test_cases
        if !has_key(reader, key, "integrated")
            println("\n  [SKIP] $key not found")
            continue
        end

        println("\n" * "-"^70)
        @printf("%-30s  at r_j = %.1f Mpc/h (i = j + %d)\n", label, r_mid, DIAG_OFFSET)
        println("-"^70)
        @printf("%6s  %14s  %14s  %10s\n", "ell", "prefix-sum", "Limber", "ratio")
        println("-"^56)

        j = r_mid_idx
        i = j + DIAG_OFFSET

        ell_sample = unique(round.(Int, range(1, nell, length=min(20, nell))))

        for ell_idx in ell_sample
            ell = ell_values[ell_idx]
            nu = ell + 0.5
            k = nu / rr[j]

            if k < 1e-5 || k > 500.0
                continue
            end

            slice = read_ell_slice(reader, key, "integrated", ell_idx)
            if slice === nothing; continue; end
            val_stored = slice[i, j]

            val_limber = if jp_val == 0 && j_val == 0
                limber_prefix_1D(ell, rr[i], rr[j], f_fn, cfns_data, pk, p; n=n_val)
            else
                limber_prefix_1D_deriv(ell, rr[i], rr[j], f_fn, cfns_data, pk, p, jp_val)
            end
            ratio = abs(val_limber) > 1e-30 ? val_stored / val_limber : NaN
            @printf("%6d  %14.6e  %14.6e  %10.4f\n",
                    ell, val_stored, val_limber, ratio)
        end
    end

    # =====================================================================
    # Test 1b: 1D prefix-sum integrals (;r' variant) vs Limber
    # =====================================================================
    # stored[i,j] = (3/D(r_j)) × Δlnr × Σ_{k=1}^{j} rf[k] w[i,k]  (trapezoidal)
    # Limber(i,j) = (3/D(r_j)) × f(r_i) × (ν/r_i)^p × P(ν/r_i) / r_i²
    # NOTE: Need i ≤ j, so we evaluate at i = j - DIAG_OFFSET

    # Test cases for ;r': (key, kernel, p, n, j, jp, label)
    # Stored key *_a_b_rp corresponds to base w^p_{ell, a, b} integrated over r_2.
    # In PowerFull, the asymmetric cases are produced via transpose of the ;r
    # result with j and j' swapped; by the same symmetry the Limber formula is
    # the ;r version with j and j' swapped, evaluated at r_i instead of r_j.
    test_cases_rp = [
        # p=-2, n=0, j=j'=0
        ("s_m2_0_0_rp",     f_s, -2,  0, 0, 0, "s_{-2,00;r'}  (density)"),
        ("t_m2_0_0_rp",     f_t, -2,  0, 0, 0, "t_{-2,00;r'}  (Doppler)"),
        ("tl_m2_0_0_rp",     f_l, -2,  0, 0, 0, "l_{-2,00;r'}  (lensing)"),
        # p=-4, n=0, j=j'=0
        ("s_m4_0_0_rp",     f_s, -4,  0, 0, 0, "s_{-4,00;r'}  (density, p=-4)"),
        ("t_m4_0_0_rp",     f_t, -4,  0, 0, 0, "t_{-4,00;r'}  (Doppler, p=-4)"),
        ("tl_m4_0_0_rp",     f_l, -4,  0, 0, 0, "l_{-4,00;r'}  (lensing, p=-4)"),
        # p=-4, n=-1, j=j'=0 — u-function base
        ("scrs_m4_0_0_rp",  f_s, -4, -1, 0, 0, "𝔰_{-4,00;r'}  (density, n=-1)"),
        ("scrt_m4_0_0_rp",  f_t, -4, -1, 0, 0, "𝔱_{-4,00;r'}  (Doppler, n=-1)"),
        ("tscrl_m4_0_0_rp",  f_l, -4, -1, 0, 0, "𝔩_{-4,00;r'}  (lensing, n=-1)"),
        # p=-3, j=1, j'=0 — stored as transpose of (0,1;r); one derivative on r_i
        ("s_m3_1_0_rp",     f_s, -3,  0, 1, 0, "s_{-3,10;r'}  (density, ∂)"),
        ("t_m3_1_0_rp",     f_t, -3,  0, 1, 0, "t_{-3,10;r'}  (Doppler, ∂)"),
        ("tl_m3_1_0_rp",     f_l, -3,  0, 1, 0, "l_{-3,10;r'}  (lensing, ∂)"),
        # p=-2, j=2, j'=0 — stored as transpose of (0,2;r); two derivatives on r_i
        ("s_m2_2_0_rp",     f_s, -2,  0, 2, 0, "s_{-2,20;r'}  (density, ∂²)"),
        ("t_m2_2_0_rp",     f_t, -2,  0, 2, 0, "t_{-2,20;r'}  (Doppler, ∂²)"),
        ("tl_m2_2_0_rp",     f_l, -2,  0, 2, 0, "l_{-2,20;r'}  (lensing, ∂²)"),
    ]

    for (key, f_fn, p, n_val, j_val, jp_val, label) in test_cases_rp
        if !has_key(reader, key, "integrated")
            println("\n  [SKIP] $key not found")
            continue
        end

        println("\n" * "-"^70)
        @printf("%-30s  at r_j = %.1f Mpc/h (i = j - %d)\n", label, r_mid, DIAG_OFFSET)
        println("-"^70)
        @printf("%6s  %14s  %14s  %10s\n", "ell", "prefix-sum", "Limber", "ratio")
        println("-"^56)

        j = r_mid_idx
        i = j - DIAG_OFFSET   # reversed offset for ;r'

        ell_sample = unique(round.(Int, range(1, nell, length=min(20, nell))))

        for ell_idx in ell_sample
            ell = ell_values[ell_idx]
            nu = ell + 0.5
            k = nu / rr[i]

            if k < 1e-5 || k > 500.0
                continue
            end

            slice = read_ell_slice(reader, key, "integrated", ell_idx)
            if slice === nothing; continue; end
            val_stored = slice[i, j]

            val_limber = if j_val == 0 && jp_val == 0
                limber_prefix_1D_rp(ell, rr[i], rr[j], f_fn, cfns_data, pk, p; n=n_val)
            else
                limber_prefix_1D_deriv_rp(ell, rr[i], rr[j], f_fn, cfns_data, pk, p, j_val)
            end
            ratio = abs(val_limber) > 1e-30 ? val_stored / val_limber : NaN
            @printf("%6d  %14.6e  %14.6e  %10.4f\n",
                    ell, val_stored, val_limber, ratio)
        end
    end

    # =====================================================================
    # Test 2: r dependence at highest ell
    # =====================================================================
    ell_max_idx = nell
    ell_max = ell_values[ell_max_idx]

    key_test = "s_m2_0_0_r"
    if has_key(reader, key_test, "integrated")
        slice = read_ell_slice(reader, key_test, "integrated", ell_max_idx)
        if slice !== nothing
            data_2d = slice

            println("\n" * "="^70)
            @printf("r-dependence at ℓ = %d  (%s, i = j + %d)\n", ell_max, key_test, DIAG_OFFSET)
            println("="^70)
            @printf("%10s  %10s  %14s  %14s  %10s\n",
                    "r_j[Mpc/h]", "k=ν/r_j", "prefix-sum", "Limber", "ratio")
            println("-"^66)

            r_indices = unique(round.(Int, range(2, nr - DIAG_OFFSET - 1, length=min(12, nr-DIAG_OFFSET-2))))
            for j in r_indices
                i = j + DIAG_OFFSET
                nu = ell_max + 0.5
                k = nu / rr[j]
                if k < 1e-5 || k > 500.0
                    continue
                end

                val_stored = data_2d[i, j]
                val_limber = limber_prefix_1D(ell_max, rr[i], rr[j], f_s, cfns_data, pk, -2)
                ratio = abs(val_limber) > 1e-30 ? val_stored / val_limber : NaN
                @printf("%10.1f  %10.4f  %14.6e  %14.6e  %10.4f\n",
                        rr[j], k, val_stored, val_limber, ratio)
            end
        end
    end

    # =====================================================================
    # Test 3: 2D prefix-sum (scrS, scrT, scrL, scrX, scrY, scrZ) vs Limber
    # =====================================================================
    # stored = (3/D)² × Δlnr² × Σ_a Σ_b rf₁[a] rf₂[b] w[a,b]
    # Limber collapses to 1D:
    #   (3/D)² × Δlnr × Σ_a f₁(r_a) f₂(r_a) (ν/r_a)^{-4} P(ν/r_a) / r_a

    lensing_cases = [
        # Diagonal: same kernel × same kernel
        ("scrS_m4_0_0_r_rp", f_s, f_s, "scrS (density×density)"),
        ("scrT_m4_0_0_r_rp", f_t, f_t, "scrT (Doppler×Doppler)"),
        ("tscrL_m4_0_0_r_rp", f_l, f_l, "scrL (lensing×lensing)"),
        # Cross terms: different kernel × different kernel
        ("scrX_m4_0_0_r_rp", f_s, f_t, "scrX (density×Doppler)"),
        ("tscrY_m4_0_0_r_rp", f_s, f_l, "scrY (density×lensing)"),
        ("tscrZ_m4_0_0_r_rp", f_t, f_l, "scrZ (Doppler×lensing)"),
    ]

    for (key, f1_fn, f2_fn, label) in lensing_cases
        if !has_key(reader, key, "integrated")
            println("\n  [SKIP] $key not found")
            continue
        end

        println("\n" * "="^70)
        @printf("2D integral: %-35s\n", label)
        @printf("scrX(r,r) vs Limber: ∫₀ʳ dlnr₁ f₁f₂ (ν/r₁)⁻⁴ P(ν/r₁)/r₁\n")
        println("="^70)

        # ℓ-convergence at r_mid
        @printf("\n  ℓ-convergence at r = %.1f Mpc/h:\n", r_mid)
        @printf("  %6s  %14s  %14s  %10s\n", "ell", "scrX(r,r)", "Limber", "ratio")
        println("  " * "-"^56)

        ell_sample = unique(round.(Int, range(1, nell, length=min(20, nell))))

        for ell_idx in ell_sample
            ell = ell_values[ell_idx]
            slice = read_ell_slice(reader, key, "integrated", ell_idx)
            if slice === nothing; continue; end
            data_2d = slice
            val_stored = data_2d[r_mid_idx, r_mid_idx]

            val_limber = limber_prefix_2D(ell, rr, r_mid_idx, f1_fn, f2_fn,
                                           cfns_data, pk, Δlnr)

            ratio = abs(val_limber) > 1e-30 ? val_stored / val_limber : NaN
            @printf("  %6d  %14.6e  %14.6e  %10.4f\n",
                    ell, val_stored, val_limber, ratio)
        end

        # r-dependence at highest ell
        @printf("\n  r-dependence at ℓ = %d:\n", ell_values[end])
        @printf("  %10s  %14s  %14s  %10s\n", "r [Mpc/h]", "scrX(r,r)", "Limber", "ratio")
        println("  " * "-"^56)

        slice = read_ell_slice(reader, key, "integrated", nell)
        if slice !== nothing
            data_2d = slice

            r_indices = unique(round.(Int, range(3, nr-1, length=min(10, nr-3))))
            for ri_diag in r_indices
                val_stored = data_2d[ri_diag, ri_diag]
                val_limber = limber_prefix_2D(ell_values[end], rr, ri_diag,
                                               f1_fn, f2_fn, cfns_data, pk, Δlnr)

                ratio = abs(val_limber) > 1e-30 ? val_stored / val_limber : NaN
                @printf("  %10.1f  %14.6e  %14.6e  %10.4f\n",
                        rr[ri_diag], val_stored, val_limber, ratio)
            end
        end
    end

    # =====================================================================
    # Summary
    # =====================================================================
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println("Expected: ratio → 1.0 as ℓ → ∞")
    println()
    println("Tested (j=j'=0, analytic Limber formula):")
    println("  1D ;r  : s,t,l × p={-2,-4}       (6 arrays, at i = j + $DIAG_OFFSET)")
    println("  1D ;r' : s,t,l × p={-2,-4}       (6 arrays, at i = j - $DIAG_OFFSET)")
    println("  1D ;r  : scrs,scrt,scrl × p=-4,n=-1 (3 arrays, n=-1 u-base)")
    println("  1D ;r' : scrs,scrt,scrl × p=-4,n=-1 (3 arrays, n=-1 u-base)")
    println("  2D diag: scrS,scrT,scrL           (3 arrays, diagonal r₁=r₂)")
    println("  2D cross: scrX,scrY,scrZ          (3 arrays, diagonal r₁=r₂)")
    println()
    println("All $(24) j=j'=0 arrays stored by PowerFull are now covered.")
    println()
    println("Tested (j≠0, derivative form of Eq. limber_wjj):")
    println("  1D ;r  : s,t,l × (p=-3, j'=1) and (p=-2, j'=2)  (6 arrays)")
    println("  1D ;r' : s,t,l × (p=-3, j=1)  and (p=-2, j=2)   (6 arrays)")
    println()
    println("Expected convergence rates:")
    println("  j=j'=0          : ratio → 1.00 at ℓ=500 (2-5% residual)")
    println("  j'=1 (one deriv): ratio → 0.80-0.85 at ℓ=500 (slow; NLO amplified)")
    println("  j'=2 (two deriv): leading Limber unreliable at ℓ≤500")
    println()
    println("Note: each derivative on r_j amplifies the NLO Limber correction")
    println("(∂_{r_j} of shape function does not share the 1/ν² suppression of")
    println("the leading correction). The stored derivative identity")
    println("  stored w^p_{ℓ,0j'}(r_1,r_j) = ∂^{j'}_{r_j} stored w^{p-j'}_{ℓ,00}(r_1,r_j)")
    println("holds to machine precision in the TwoFAST output (verified at ℓ≤200)")
    println("so the slow convergence is a property of the Limber approximation,")
    println("not of the pipeline.")
    println("="^70)
end

# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    # Split positional from flag args.
    positional = String[]
    matterpower_arg = nothing
    cosmo_funcr_arg = nothing
    for arg in ARGS
        if startswith(arg, "--matterpower=")
            matterpower_arg = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--cosmo-funcr=")
            cosmo_funcr_arg = String(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--")
            @warn "Unknown argument: $arg"
        else
            push!(positional, arg)
        end
    end

    input_file = if length(positional) >= 1
        positional[1]
    elseif isfile("ClGR_integrals.h5")
        "ClGR_integrals.h5"
    elseif isdir("ClGRresult") && isfile("ClGRresult/ClGR_integrals_hires_meta.h5")
        "ClGRresult/ClGR_integrals_hires_meta.h5"
    else
        error("No input file specified and no default found. Usage: julia --project src/validate_limber.jl <file.h5>")
    end

    kwargs = Dict{Symbol,String}()
    matterpower_arg !== nothing && (kwargs[:matterpower]  = matterpower_arg)
    cosmo_funcr_arg !== nothing && (kwargs[:cosmo_funcr]  = cosmo_funcr_arg)
    main(input_file; kwargs...)
end
