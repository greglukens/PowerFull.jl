#!/usr/bin/env -S julia --project
# =============================================================================
# Limber cross-check of observed C_ℓ^{ii}.
#
# Computes, for each tracer:
#   (1) Density-only Limber:
#         C_ℓ^gg(Limber) = ∫ dr / r² · [b_g(r) φ(z(r)) dz/dr · D(r)]² · P(ν/r)
#
#   (2) κ-only Limber (magnification auto):
#         W_κ(r,ℓ) = 2·(1-Q(r)) · (3/2)·Ωm·(H0/c)² · (1+z(r)) · g(r; ℓ)
#         g(r; ℓ) = r · ∫_r^{r_max} dr' φ(z(r')) dz'/dr' · (r' - r)/r'
#         C_ℓ^κκ(Limber) = ∫ dr / r² · W_κ²(r,ℓ) · P(ν/r) / D(r)² × D_eff
#         (here we absorb D into the integrand: W_κ contains (1-Q)·φ·dz/dr
#          factor mimicking the PowerFull "r-multiplication" convention.)
#
#   (3) density+κ cross contribution at Limber (since density and magnification
#       are local-in-r and overlap where φ and (1-Q)φ' coexist) — small in
#       narrow-σ, non-negligible in wide-σ.
#
# Compare ratio PowerFull / (gg_Limber + κκ_Limber + 2·gκ_Limber) at
# ℓ = {50, 100, 200, 500}.
#
# Usage:
#   julia --project examples/compare_cl_limber.jl \
#       production/ClGR_prod_t24_meta.h5 \
#       examples/Cl_spherex_15pairs_prod.h5 \
#       examples/tracer_list_prod.txt
# =============================================================================

using HDF5
using Dierckx
using DelimitedFiles
using Printf

include(joinpath(@__DIR__, "..", "src", "cosmofns.jl"))
using .cosmofns: cosmofn

# ---- Power spectrum with extrapolation (mirrors validate_limber.jl) ---------
struct PkSpectrum
    pkspl::Spline1D
    kmin::Float64; kmax::Float64
    nslo::Float64; nshi::Float64
    kmin_norm::Float64; kmax_norm::Float64
end
function PkSpectrum(path::String)
    data = readdlm(path, comments=true)
    kk, pk = data[:,1], data[:,2]
    pkspl = Spline1D(kk, pk)
    k0 = kk[1];   P0 = pkspl(k0);   Pp0 = derivative(pkspl, k0); nslo = k0 * Pp0 / P0
    kmin_norm = P0 / k0^nslo
    k1 = kk[end]; P1 = pkspl(k1);   Pp1 = derivative(pkspl, k1); nshi = 4 + k1 * Pp1 / P1
    kmax_norm = P1 / (k1^(nshi - 4))
    PkSpectrum(pkspl, kk[1], kk[end], nslo, nshi, kmin_norm, kmax_norm)
end
function (p::PkSpectrum)(k::Real)
    k < p.kmin && return p.kmin_norm * k^p.nslo
    k > p.kmax && return p.kmax_norm * k^(p.nshi - 4)
    return p.pkspl(k)
end

# ---- Tracer loader (mirrors load_tracer_h5) ---------------------------------
struct TracerSpl
    z::Vector{Float64}; phi::Spline1D; bg::Spline1D; Q::Spline1D
    zmin::Float64; zmax::Float64
end
function load_tracer(path::String)
    h5open(path, "r") do f
        z  = read(f["z"])
        bg = read(f["bg"])
        Q  = read(f["Q"])
        ph = read(f["phi"])
        TracerSpl(z, Spline1D(z, ph, k=3), Spline1D(z, bg, k=3), Spline1D(z, Q, k=3),
                  minimum(z), maximum(z))
    end
end

# ---- Build r-grid evaluating φ, bg, Q, D, dzdr, (1-Q), (1+z) as arrays ------
struct TrGrid
    rr::Vector{Float64}; dr::Vector{Float64}
    z::Vector{Float64}; D::Vector{Float64}
    bg::Vector{Float64}; Q::Vector{Float64}; oneMQ::Vector{Float64}
    phi::Vector{Float64}; dzdr::Vector{Float64}
    wW_g::Vector{Float64}   # b_g · φ · dz/dr · D
    wW_mu::Vector{Float64}  # 2·(1-Q) · φ · dz/dr   (magnification "source")
end
function TrGrid(tracer::TracerSpl, rr::Vector{Float64}, cf, Omm0::Float64, H0_kms_Mpc::Float64)
    nr = length(rr)
    dr = Vector{Float64}(undef, nr)
    dr[1]   = 0.5*(rr[2]-rr[1])
    dr[end] = 0.5*(rr[end]-rr[end-1])
    @inbounds for k in 2:nr-1; dr[k] = 0.5*(rr[k+1]-rr[k-1]); end
    z = [cf.fzr(r) for r in rr]
    D = [cf.fDr(r) for r in rr]
    bg = [z[k] ≥ tracer.zmin && z[k] ≤ tracer.zmax ? tracer.bg(z[k]) : 0.0 for k in 1:nr]
    Q  = [z[k] ≥ tracer.zmin && z[k] ≤ tracer.zmax ? tracer.Q(z[k])  : 1.0 for k in 1:nr]
    oneMQ = 1.0 .- Q
    phi = [z[k] ≥ tracer.zmin && z[k] ≤ tracer.zmax ? tracer.phi(z[k]) : 0.0 for k in 1:nr]
    # dz/dr from fHr (conformal ℋ in 1/(Mpc/h))
    dzdr = [cf.fHr(r) for r in rr]
    wW_g  = bg  .* phi .* dzdr .* D
    wW_mu = 2.0 .* oneMQ .* phi .* dzdr
    TrGrid(rr, dr, z, D, bg, Q, oneMQ, phi, dzdr, wW_g, wW_mu)
end

# ---- Density-only Limber C_ℓ -------------------------------------------------
# C_ℓ^gg = ∫ dr / r² · [wW_g(r)]² · P(ν/r)
# (D is already folded into wW_g; the D²·P normalization gives the standard
#  Limber galaxy-clustering expression with P evaluated at k=ν/r, z=z(r) via
#  the growth factor.)
function limber_Cgg(t::TrGrid, pk::PkSpectrum, ell::Int)::Float64
    ν = ell + 0.5
    s = 0.0
    @inbounds for k in eachindex(t.rr)
        r = t.rr[k]
        kval = ν / r
        (kval < pk.kmin*0.1 || kval > pk.kmax*10) && continue
        Pk = pk(kval)
        s += t.wW_g[k]^2 * Pk * t.dr[k] / (r*r)
    end
    return s
end

# ---- κ-only Limber C_ℓ: magnification auto-spectrum contribution ------------
# Build g(r; ℓ) = r · ∫_r^{rmax} dr' [ (r'-r)/r' · φ(z(r')) dz'/dr' ]
# (Only tracer-dependent lensing kernel, independent of ℓ actually.)
# Then C_ℓ^κκ_tracer = ∫ dr / r² · [(3/2) Ωm (H0/c)² · (1+z(r)) · g(r) · 2·(1-Q(r))]² · P(ν/r) · D(r)²
# The extra D(r)² folds cosmological growth into the P(k) at redshift z.
function limber_Ckappa(t::TrGrid, pk::PkSpectrum, ell::Int,
                       Omm0::Float64)::Float64
    ν = ell + 0.5
    nr = length(t.rr)
    # Lensing kernel g(r) = r · ∫_r^rmax dr' (r'-r)/r' · φ(z(r')) dz'/dr'
    # Compute via reverse-cumulative trapezoidal.
    g = zeros(Float64, nr)
    integrand = t.phi .* t.dzdr   # W_source(r') dr'  (already φ·dz/dr)
    @inbounds for i in 1:nr
        r = t.rr[i]
        acc = 0.0
        @inbounds for jp in i+1:nr
            rp = t.rr[jp]
            acc += (rp - r)/rp * integrand[jp] * t.dr[jp]
        end
        g[i] = r * acc
    end
    c_light  = 2.99792458e5
    H0_per_Mpch = 100.0 / c_light        # h/Mpc, since we work in Mpc/h
    prefK = 1.5 * Omm0 * H0_per_Mpch^2
    s = 0.0
    @inbounds for k in eachindex(t.rr)
        r = t.rr[k]
        kval = ν / r
        (kval < pk.kmin*0.1 || kval > pk.kmax*10) && continue
        Pk = pk(kval)
        # W_κ includes the magnification pre-factor 2·(1-Q)
        Wk = prefK * (1.0 + t.z[k]) * g[k] * 2.0 * t.oneMQ[k]
        s += Wk^2 * Pk * t.dr[k] * t.D[k]^2 / (r*r)
    end
    return s
end

# ---- Density × κ cross (integrated Limber) ----------------------------------
# 2 · ∫ dr / r² · wW_g · [prefK · (1+z) · g · 2(1-Q) · D²] · P(ν/r)
function limber_CgKappa(t::TrGrid, pk::PkSpectrum, ell::Int,
                        Omm0::Float64)::Float64
    ν = ell + 0.5
    nr = length(t.rr)
    g = zeros(Float64, nr)
    integrand = t.phi .* t.dzdr
    @inbounds for i in 1:nr
        r = t.rr[i]
        acc = 0.0
        @inbounds for jp in i+1:nr
            rp = t.rr[jp]
            acc += (rp - r)/rp * integrand[jp] * t.dr[jp]
        end
        g[i] = r * acc
    end
    c_light  = 2.99792458e5
    H0_per_Mpch = 100.0 / c_light
    prefK = 1.5 * Omm0 * H0_per_Mpch^2
    s = 0.0
    @inbounds for k in eachindex(t.rr)
        r = t.rr[k]
        kval = ν / r
        (kval < pk.kmin*0.1 || kval > pk.kmax*10) && continue
        Pk = pk(kval)
        Wk = prefK * (1.0 + t.z[k]) * g[k] * 2.0 * t.oneMQ[k]
        s += 2.0 * t.wW_g[k] * Wk * Pk * t.dr[k] * t.D[k] / (r*r)
    end
    return s
end

# ---- Main --------------------------------------------------------------------
function main()
    length(ARGS) ≥ 3 || error("usage: julia examples/compare_cl_limber.jl <meta.h5> <cl_multi.h5> <tracer_list.txt> [--fnl-strip] [--cosmo-funcr=<path>] [--matterpower=<path>]")
    meta_path       = ARGS[1]
    cl_path         = ARGS[2]
    tracer_list     = ARGS[3]
    cosmo_funcr_path = get_arg("--cosmo-funcr=", "cosmo_funcr_astropy_planck2018.txt")
    matterpower_path = get_arg("--matterpower=", "astropy_planck_2018_matterpower.dat")

    println("="^70)
    println("Limber cross-check of PowerFull observed C_ℓ^{ii}")
    println("="^70)

    # Grid + ell values
    rr, ell_values = h5open(meta_path, "r") do f
        (Float64.(read(f, "grid/rr")), Int.(read(f, "grid/ell_values")))
    end
    println("rr: nr=$(length(rr)),  r ∈ [$(round(rr[1],digits=1)), $(round(rr[end],digits=1))]")
    println("ell: $(length(ell_values)) values, [$(ell_values[1]), $(ell_values[end])]")

    # Cosmology + power spectrum
    cf = cosmofn(cosmo_funcr_path)
    pk = PkSpectrum(matterpower_path)
    # Omm0 read from PowerFull defaults; set below. Could parse from Cl provenance.
    Omm0 = 0.30682
    H0   = 67.78
    println("Cosmology: Omm0=$Omm0  H0=$H0  matterpower=$(basename(matterpower_path))")

    # PowerFull Cl (15 pairs)
    ell_pf, Cl_all = h5open(cl_path, "r") do f
        (Int.(read(f, "ell")), Float64.(read(f, "Cl_all")))
    end
    ell_pf == ell_values || error("ell mismatch between meta and Cl file")
    n_pairs = size(Cl_all, 2)
    @assert size(Cl_all, 1) == length(ell_values)

    # Tracer list
    tracer_paths = String[]
    for ln in eachline(tracer_list)
        s = strip(ln); (isempty(s) || startswith(s, "#")) && continue
        push!(tracer_paths, String(s))
    end
    length(tracer_paths) == n_pairs ||
        error("tracer count $(length(tracer_paths)) != n_pairs $n_pairs")

    # Test ells
    test_ells = [50, 100, 200, 500]
    test_ell_idx = [findfirst(==(e), ell_values) for e in test_ells]

    println("\n")
    @printf("%-22s | %4s |  %14s  %14s  %14s  %14s  |  %8s  %8s\n",
            "tracer", "ell", "PF", "Lim_gg", "Lim_κκ", "Lim_gg+κκ+cross",
            "PF/gg", "PF/total")
    println("-"^150)

    for p in 1:n_pairs
        tpath = joinpath(@__DIR__, "..", tracer_paths[p])
        tracer = load_tracer(tpath)
        t = TrGrid(tracer, rr, cf, Omm0, H0)
        for (ie, e) in zip(test_ell_idx, test_ells)
            Cpf    = Cl_all[ie, p]
            Cgg    = limber_Cgg(t, pk, e)
            Ckk    = limber_Ckappa(t, pk, e, Omm0)
            Cgk    = limber_CgKappa(t, pk, e, Omm0)
            Ctot   = Cgg + Ckk + Cgk
            r_gg   = Cpf / Cgg
            r_tot  = Cpf / Ctot
            @printf("%-22s | %4d |  %14.5e  %14.5e  %14.5e  %14.5e  |  %8.4f  %8.4f\n",
                    basename(tpath), e, Cpf, Cgg, Ckk, Ctot, r_gg, r_tot)
        end
        println("-"^150)
    end
end

function get_arg(prefix::String, default::String)::String
    for a in ARGS
        startswith(a, prefix) && return String(split(a, "=", limit=2)[2])
    end
    return default
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
