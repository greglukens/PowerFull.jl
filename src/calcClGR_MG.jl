# calcClGR_MG.jl
# =============================================================================
#
# Standalone module for computing C_ℓ from pre-computed integrals, with
# optional local-limit modified-gravity (μ₀, Σ₀) support.  GR-inert at (0,0).
#
# Requires: JLD2 only
# Does NOT require: TwoFAST.jl, PowerFullTwoFAST.jl, PowerFull.jl
#
# Usage:
#     include("calcClGR.jl")
#     using .CalcClGR
#
#     I = load_integrals("ClGR_integrals.jld2")
#     nr = length(I.rr)
#     params = ClGRParams(
#         D = [D(r) for r in I.rr],
#         aH = [a(r)*H(r) for r in I.rr],   # conformal Hubble ℋ = a·H
#         bg = fill(1.5, nr),
#         ...
#     )
#     Cl_map = compute_Cl_GR(I, params, 100)  # for ℓ=100, returns (nr, nr) array
#
#  December 2025
#  Donghui Jeong
# =============================================================================

module CalcClGR

using JLD2
using HDF5
using Base.Threads
using Interpolations
using Dierckx
using Base.Cartesian: @nexprs

# Modified-gravity helpers (local limit). GR-inert when mu0=Sigma0=0.
include(joinpath(@__DIR__, "mg_cosmo.jl"))
using .MGCosmo: MGModel, build_mg_model

export IntegralCollection, ClGRParams, ClGRParamCache
export load_integrals, load_integrals_hdf5
export compute_Cl_GR, compute_Cl_GR!, compute_Cl_GR_all_ell, compute_Cl_GR_terms
export compute_Cl_GR_batch, compute_Cl_GR_batch!
export show_available_keys
export build_mg_model, MGModel

# =============================================================================
# Data Structures
# =============================================================================

"""
    IntegralCollection

Container for all pre-computed integrals on physical (r₁, r₂) grid, matching paper notation.

# Key format: (type, p, j, jp, sub)
- `type::Symbol`: Base function or integral type (:w, :u, :v, :s, :t, :l, :scrs, :scrS, etc.)
- `p::Int`: Power index (-4, -3, -2, -1, 0)
- `j, jp::Int`: Bessel derivative orders (0, 1, 2)
- `sub::Symbol`: Subscript indicating integration (:r, :rp, :r_rp, :rp_r, :none)

# Naming convention based on n parameter:
- n = 0  → w (base), s/t/l (integrals)
- n = -1 → u (base), scrs/scrt/scrl (integrals)
- n = -2 → v (base)

# Fields
- `data`: Dictionary mapping keys to 3D arrays [n_ell, nr, nr] on physical grid
- `rr`: Radial grid (Mpc/h), same for both axes (r₁ and r₂)
- `ell_values`: Multipole ℓ values
"""
struct IntegralCollection
    data::Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}  # [n_ell, nr, nr]
    rr::Vector{Float64}
    ell_values::Vector{Int}
end

# Some older dump/component code used temporary names like :tl / :tscrY for
# lensing-style integrated kernels. The current build/export files write these
# same arrays under the canonical names :l / :scrY / :scrZ / :scrL / :scrl.
# Keep lookup compatibility here so dump_terms.jl and older scripts do not crash
# on keys such as (:tl, -2, 0, 0, :r).
const _INTEGRAL_TYPE_ALIASES = Dict{Symbol,Symbol}(
    :tl     => :l,
    :tscrY  => :scrY,
    :tscrZ  => :scrZ,
    :tscrL  => :scrL,
    :tscrl  => :scrl,
)

function _canonical_integral_key(I::IntegralCollection, type::Symbol, p::Int, j::Int, jp::Int, sub::Symbol)
    key = (type, p, j, jp, sub)
    haskey(I.data, key) && return key

    if haskey(_INTEGRAL_TYPE_ALIASES, type)
        alias_type = _INTEGRAL_TYPE_ALIASES[type]
        alias_key = (alias_type, p, j, jp, sub)
        haskey(I.data, alias_key) && return alias_key
    end

    return key
end

function Base.getindex(I::IntegralCollection, type::Symbol, p::Int, j::Int, jp::Int, sub::Symbol)
    requested_key = (type, p, j, jp, sub)
    key = _canonical_integral_key(I, type, p, j, jp, sub)
    if !haskey(I.data, key)
        available = sort(collect(keys(I.data)))
        error("Key $requested_key not found. Available keys (first 10): $(available[1:min(10,length(available))])")
    end
    return I.data[key]
end

function Base.haskey(I::IntegralCollection, type::Symbol, p::Int, j::Int, jp::Int, sub::Symbol)
    key = _canonical_integral_key(I, type, p, j, jp, sub)
    return haskey(I.data, key)
end

function Base.keys(I::IntegralCollection)
    return keys(I.data)
end

"""
    ClGRParams

Cosmological parameters for C_ℓ^GR calculation.

Constructor accepts either:
- Arrays: (rr, values) → auto-creates interpolation function
- Function: r -> value → uses directly

# Function fields (internally stored as Functions)
- `D`: Growth factor D(r)
- `aH`: Conformal Hubble ℋ(r) = a(r)·H(r) in units of 1/(Mpc/h)
- `bg`: Galaxy bias b_g(r)
- `β`: RSD parameter β(r) = f(r)/b_g(r)
- `B`: ℬ(r) parameter (velocity contribution)
- `A`: 𝒜(r) parameter (potential contribution)
- `Q`: Magnification bias 𝒬(r)
- `bPhi`: Scale-dependent bias b_Φ(r)

# Scalar fields (constants)
- `f_NL`: Primordial non-Gaussianity parameter (default: 0)
- `Omm0`: Ω_{m,0} at z=0 (default: 0.3)
- `H0`: H₀ in km/s/Mpc (default: 67.0)

# Examples
```julia
# Example 1: Arrays with common rr grid (simplest)
params = ClGRParams(
    rr = rr_grid,
    D = D_values, aH = aH_values, bg = bg_values, β = β_values,
    B = B_values, A = A_values, Q = Q_values, bPhi = bPhi_values
)

# Example 2: Mix of arrays and functions
params = ClGRParams(
    rr = rr_grid,
    D = D_values,           # Array → interpolated
    bg = r -> 1.5,          # Function → direct (constant)
    ...
)

# Example 3: All functions (power user)
params = ClGRParams(
    D = r -> growth_factor(cosmo, r),
    aH = r -> conformal_hubble(cosmo, r),
    ...
)
```
"""
struct ClGRParams
    # Internal: all stored as Functions
    D::Function
    aH::Function   # conformal Hubble ℋ(r) = a(r)·H(r); distinct from H0 below
    bg::Function
    β::Function
    B::Function
    A::Function
    Q::Function
    bPhi::Function

    # Scalars (constants)
    f_NL::Float64
    Omm0::Float64
    H0::Float64    # physical H₀ in km/s/Mpc
end

# Helper: create interpolation function from arrays (log-spaced r)
function _make_interp(rr::Vector{Float64}, values::Vector{Float64})
    log_rr = log10.(rr)
    itp = linear_interpolation(log_rr, values, extrapolation_bc=Flat())
    return r -> itp(log10(r))
end

# Helper: pass-through if already a function
_make_interp(f::Function) = f

# Constructor with flexible inputs
function ClGRParams(;
    rr::Union{Vector{Float64}, Nothing} = nothing,  # common r grid (optional)
    D, aH, bg, β, B, A, Q, bPhi,                     # each can be Array or Function
    f_NL::Float64 = 0.0,
    Omm0::Float64 = 0.3,
    H0::Float64 = 67.0
)
    # Helper to process each parameter
    function _process(param, rr_common)
        if param isa Function
            return param
        elseif param isa Vector{Float64} && rr_common !== nothing
            return _make_interp(rr_common, param)
        elseif param isa Tuple{Vector{Float64}, Vector{Float64}}
            # (rr, values) tuple
            return _make_interp(param[1], param[2])
        else
            error("Parameter must be Function, Vector (with rr provided), or Tuple{Vector,Vector}")
        end
    end

    return ClGRParams(
        _process(D, rr),
        _process(aH, rr),
        _process(bg, rr),
        _process(β, rr),
        _process(B, rr),
        _process(A, rr),
        _process(Q, rr),
        _process(bPhi, rr),
        f_NL, Omm0, H0
    )
end

# =============================================================================
# Parameter Cache for Optimized Computation
# =============================================================================

"""
    ClGRParamCache

Pre-computed parameter cache for optimized C_ℓ^GR computation on physical (r₁, r₂) grid.
Stores all ell-independent quantities to avoid redundant function calls.

# Fields
## Grid info
- `nr::Int`: Grid dimension (same for both r₁ and r₂)
- `rr::Vector{Float64}`: Physical r grid (shared by both axes)

## Parameters at r₁ (vectors of length nr, indexed by i)
- `D1`, `aH1`, `bg1`, `β1`, `B1`, `A1`, `Q1`, `bPhi1`

## Parameters at r₂ (vectors of length nr, indexed by j)
- `D2`, `aH2`, `bg2`, `β2`, `B2`, `A2`, `Q2`, `bPhi2`

## Pre-computed coefficients (nr × nr matrices, indexed by [i,j])
- `prefactor`: D1[i] * D2[j]
- Various coefficient combinations for terms 1-19

## Scalars
- `f_NL`, `Omm0`, `H0`, `fNL_prefactor`
"""
struct ClGRParamCache
    # Grid info
    nr::Int
    rr::Vector{Float64}
    ell_values::Vector{Int}

    # Params at r1 [nr] (indexed by i)
    D1::Vector{Float64}
    aH1::Vector{Float64}
    bg1::Vector{Float64}
    β1::Vector{Float64}
    B1::Vector{Float64}
    A1::Vector{Float64}
    Q1::Vector{Float64}
    bPhi1::Vector{Float64}

    # Params at r2 [nr] (indexed by j — same grid, different role)
    D2::Vector{Float64}
    aH2::Vector{Float64}
    bg2::Vector{Float64}
    β2::Vector{Float64}
    B2::Vector{Float64}
    A2::Vector{Float64}
    Q2::Vector{Float64}
    bPhi2::Vector{Float64}

    # Pre-computed ell-independent coefficients [nr, nr]
    prefactor::Matrix{Float64}  # D1 * D2

    # Term 1 coefficients: bg1*bg2*(w - β1*w - β2*w + β1*β2*w)
    c1_w000::Matrix{Float64}      # prefactor * bg1 * bg2
    c1_w020::Matrix{Float64}      # -prefactor * bg1 * bg2 * β1
    c1_w002::Matrix{Float64}      # -prefactor * bg1 * bg2 * β2
    c1_w022::Matrix{Float64}      # prefactor * bg1 * bg2 * β1 * β2

    # Term 2 coefficients
    c2_wm101::Matrix{Float64}     # prefactor * bg1 * aH2 * B2
    c2_wm121::Matrix{Float64}     # -prefactor * bg1 * aH2 * B2 * β1
    c2_wm110::Matrix{Float64}     # prefactor * bg2 * aH1 * B1
    c2_wm112::Matrix{Float64}     # -prefactor * bg2 * aH1 * B1 * β2

    # Term 3 coefficients
    c3_wm200::Matrix{Float64}     # prefactor * bg1 * aH2^2 * A2
    c3_wm220::Matrix{Float64}     # -prefactor * bg1 * aH2^2 * A2 * β1
    c3_wm310::Matrix{Float64}     # prefactor * aH1 * aH2^2 * A2 * B1

    # Term 4 coefficients
    c4_wm200::Matrix{Float64}     # prefactor * bg2 * aH1^2 * A1
    c4_wm202::Matrix{Float64}     # -prefactor * bg2 * aH1^2 * A1 * β2
    c4_wm301::Matrix{Float64}     # prefactor * aH2 * aH1^2 * A1 * B2

    # Term 5 coefficients (s integrals with _rp subscript)
    c5_sm200rp::Matrix{Float64}   # prefactor * bg1 * (B2/β2)
    c5_sm220rp::Matrix{Float64}   # -prefactor * bg1 * (B2/β2) * β1
    c5_sm310rp::Matrix{Float64}   # prefactor * aH1 * (B2/β2) * B1
    c5_sm400rp::Matrix{Float64}   # prefactor * aH1^2 * (B2/β2) * A1

    # Term 6 coefficients (s integrals with _r subscript)
    c6_sm200r::Matrix{Float64}    # prefactor * bg2 * (B1/β1)
    c6_sm202r::Matrix{Float64}    # -prefactor * bg2 * (B1/β1) * β2
    c6_sm301r::Matrix{Float64}    # prefactor * aH2 * (B1/β1) * B2
    c6_sm400r::Matrix{Float64}    # prefactor * aH2^2 * (B1/β1) * A2

    # Term 7 coefficients (t integrals with _rp)
    c7_tm200rp::Matrix{Float64}   # -2 * prefactor * bg1 * (1-Q2)/r2
    c7_tm220rp::Matrix{Float64}   # 2 * prefactor * bg1 * (1-Q2)/r2 * β1
    c7_tm310rp::Matrix{Float64}   # -2 * prefactor * aH1 * (1-Q2)/r2 * B1
    c7_tm400rp::Matrix{Float64}   # -2 * prefactor * aH1^2 * (1-Q2)/r2 * A1

    # Term 8 coefficients (t integrals with _r)
    c8_tm200r::Matrix{Float64}    # -2 * prefactor * bg2 * (1-Q1)/r1
    c8_tm202r::Matrix{Float64}    # 2 * prefactor * bg2 * (1-Q1)/r1 * β2
    c8_tm301r::Matrix{Float64}    # -2 * prefactor * aH2 * (1-Q1)/r1 * B2
    c8_tm400r::Matrix{Float64}    # -2 * prefactor * aH2^2 * (1-Q1)/r1 * A2

    # Term 9 coefficients (l integrals with _rp)
    c9_lm200rp::Matrix{Float64}   # -2 * prefactor * bg1 * (1-Q2)
    c9_lm220rp::Matrix{Float64}   # 2 * prefactor * bg1 * (1-Q2) * β1
    c9_lm310rp::Matrix{Float64}   # -2 * prefactor * aH1 * (1-Q2) * B1
    c9_lm400rp::Matrix{Float64}   # -2 * prefactor * aH1^2 * (1-Q2) * A1

    # Term 10 coefficients (l integrals with _r)
    c10_lm200r::Matrix{Float64}   # -2 * prefactor * bg2 * (1-Q1)
    c10_lm202r::Matrix{Float64}   # 2 * prefactor * bg2 * (1-Q1) * β2
    c10_lm301r::Matrix{Float64}   # -2 * prefactor * aH2 * (1-Q1) * B2
    c10_lm400r::Matrix{Float64}   # -2 * prefactor * aH2^2 * (1-Q1) * A2

    # Term 11 coefficients (2D integrals scrX, scrY, scrZ)
    c11_scrXrrp::Matrix{Float64}  # -2 * prefactor * (1-Q2) * B1/(f1*r2) = -2 * pf * (1-Q2) * B1/(bg1*β1*r2)
    c11_scrYrrp::Matrix{Float64}  # -2 * prefactor * (1-Q2) * B1/f1     = -2 * pf * (1-Q2) * B1/(bg1*β1)
    c11_scrZrpr::Matrix{Float64}  # 4 * prefactor * (1-Q2) * (1-Q1)/r2

    # Term 12 coefficients
    c12_scrXrpr::Matrix{Float64}  # -2 * prefactor * (1-Q1) * B2/(f2*r1) = -2 * pf * (1-Q1) * B2/(bg2*β2*r1)
    c12_scrYrpr::Matrix{Float64}  # -2 * prefactor * (1-Q1) * B2/f2     = -2 * pf * (1-Q1) * B2/(bg2*β2)
    c12_scrZrrp::Matrix{Float64}  # 4 * prefactor * (1-Q1) * (1-Q2)/r1

    # Term 13 coefficients (u integrals, fNL linear)
    c13_um200::Matrix{Float64}    # prefactor * fNL_pf * bg1 * (bPhi2/D2) * f_NL
    c13_um220::Matrix{Float64}    # -prefactor * fNL_pf * bg1 * (bPhi2/D2) * f_NL * β1
    c13_um310::Matrix{Float64}    # prefactor * fNL_pf * aH1 * (bPhi2/D2) * f_NL * B1
    c13_um400::Matrix{Float64}    # prefactor * fNL_pf * aH1^2 * (bPhi2/D2) * f_NL * A1

    # Term 14 coefficients
    c14_um200::Matrix{Float64}    # prefactor * fNL_pf * bg2 * (bPhi1/D1) * f_NL
    c14_um202::Matrix{Float64}    # -prefactor * fNL_pf * bg2 * (bPhi1/D1) * f_NL * β2
    c14_um301::Matrix{Float64}    # prefactor * fNL_pf * aH2 * (bPhi1/D1) * f_NL * B2
    c14_um400::Matrix{Float64}    # prefactor * fNL_pf * aH2^2 * (bPhi1/D1) * f_NL * A2

    # Term 15 coefficients (scrs, scrt, scrl with _rp)
    c15_scrsm4rp::Matrix{Float64} # prefactor * fNL_pf * (bPhi1/D1) * f_NL * B2/(bg2*β2)
    c15_scrtm4rp::Matrix{Float64} # -2 * prefactor * fNL_pf * (bPhi1/D1) * f_NL * (1-Q2)/r2
    c15_scrlm4rp::Matrix{Float64} # -2 * prefactor * fNL_pf * (bPhi1/D1) * f_NL * (1-Q2)

    # Term 16 coefficients (scrs, scrt, scrl with _r)
    c16_scrsm4r::Matrix{Float64}  # prefactor * fNL_pf * (bPhi2/D2) * f_NL * B1/(bg1*β1)
    c16_scrtm4r::Matrix{Float64}  # -2 * prefactor * fNL_pf * (bPhi2/D2) * f_NL * (1-Q1)/r1
    c16_scrlm4r::Matrix{Float64}  # -2 * prefactor * fNL_pf * (bPhi2/D2) * f_NL * (1-Q1)

    # Term 17 coefficients
    c17_wm211::Matrix{Float64}    # prefactor * aH1 * aH2 * B1 * B2
    c17_wm400::Matrix{Float64}    # prefactor * aH1^2 * aH2^2 * A1 * A2
    c17_scrSrrp::Matrix{Float64}  # prefactor * B1/(bg1*β1) * B2/(bg2*β2)

    # Term 18 coefficients
    c18_scrTrrp::Matrix{Float64}  # 4 * prefactor * (1-Q1)/r1 * (1-Q2)/r2
    c18_scrLrrp::Matrix{Float64}  # 4 * prefactor * (1-Q1) * (1-Q2)

    # Term 19 coefficient
    c19_vm400::Matrix{Float64}    # prefactor * (9/4) * (bPhi1/D1) * (bPhi2/D2) * f_NL^2 * Omm0^2 * (H0/c)^4

    # Scalars
    f_NL::Float64
    Omm0::Float64
    H0::Float64
    fNL_prefactor::Float64

    # Pre-cached integral array references (avoid Dict lookup in hot path)
    W_0_0_0::Array{Float64,3}
    W_0_2_0::Array{Float64,3}
    W_0_0_2::Array{Float64,3}
    W_0_2_2::Array{Float64,3}
    W_m1_0_1::Array{Float64,3}
    W_m1_2_1::Array{Float64,3}
    W_m1_1_0::Array{Float64,3}
    W_m1_1_2::Array{Float64,3}
    W_m2_0_0::Array{Float64,3}
    W_m2_2_0::Array{Float64,3}
    W_m2_0_2::Array{Float64,3}
    W_m2_1_1::Array{Float64,3}
    W_m3_1_0::Array{Float64,3}
    W_m3_0_1::Array{Float64,3}
    W_m4_0_0::Array{Float64,3}

    S_m2_0_0_r::Array{Float64,3}
    S_m2_0_2_r::Array{Float64,3}
    S_m2_0_0_rp::Array{Float64,3}
    S_m2_2_0_rp::Array{Float64,3}
    S_m3_0_1_r::Array{Float64,3}
    S_m3_1_0_rp::Array{Float64,3}
    S_m4_0_0_r::Array{Float64,3}
    S_m4_0_0_rp::Array{Float64,3}

    T_m2_0_0_r::Array{Float64,3}
    T_m2_0_2_r::Array{Float64,3}
    T_m2_0_0_rp::Array{Float64,3}
    T_m2_2_0_rp::Array{Float64,3}
    T_m3_0_1_r::Array{Float64,3}
    T_m3_1_0_rp::Array{Float64,3}
    T_m4_0_0_r::Array{Float64,3}
    T_m4_0_0_rp::Array{Float64,3}

    L_m2_0_0_r::Array{Float64,3}
    L_m2_0_2_r::Array{Float64,3}
    L_m2_0_0_rp::Array{Float64,3}
    L_m2_2_0_rp::Array{Float64,3}
    L_m3_0_1_r::Array{Float64,3}
    L_m3_1_0_rp::Array{Float64,3}
    L_m4_0_0_r::Array{Float64,3}
    L_m4_0_0_rp::Array{Float64,3}

    ScrX_r_rp::Array{Float64,3}
    ScrX_rp_r::Array{Float64,3}
    ScrY_r_rp::Array{Float64,3}
    ScrY_rp_r::Array{Float64,3}
    ScrZ_r_rp::Array{Float64,3}
    ScrZ_rp_r::Array{Float64,3}
    ScrS_r_rp::Array{Float64,3}
    ScrT_r_rp::Array{Float64,3}
    ScrL_r_rp::Array{Float64,3}

    U_m2_0_0::Array{Float64,3}
    U_m2_2_0::Array{Float64,3}
    U_m2_0_2::Array{Float64,3}
    U_m3_1_0::Array{Float64,3}
    U_m3_0_1::Array{Float64,3}
    U_m4_0_0::Array{Float64,3}

    Scrs_m4_r::Array{Float64,3}
    Scrs_m4_rp::Array{Float64,3}
    Scrt_m4_r::Array{Float64,3}
    Scrt_m4_rp::Array{Float64,3}
    Scrl_m4_r::Array{Float64,3}
    Scrl_m4_rp::Array{Float64,3}

    V_m4_0_0::Array{Float64,3}
end

"""
    ClGRParamCache(I, params) -> ClGRParamCache

Single-tracer (auto) convenience that forwards to the two-tracer
constructor with `params` on both axes.  Pre-computes all
ell-independent coefficients for optimized C_ℓ^GR computation on the
physical (r₁, r₂) grid.
"""
function ClGRParamCache(I::IntegralCollection, params::ClGRParams)
    return ClGRParamCache(I, params, params)
end

"""
    ClGRParamCache(I, params_1, params_2)

Cross-tracer cache.  Evaluates D/H/b_g/β/B/A/Q/b_Φ at each r on the
physical grid, using `params_1` for the r₁ axis and `params_2` for the
r₂ axis.  Scalars (f_NL, Omm0, H0) must match — they come from
cosmology, not the tracer.
"""
function ClGRParamCache(I::IntegralCollection,
                        params_1::ClGRParams, params_2::ClGRParams)
    nr = length(I.rr)
    rr = I.rr

    # Scalars must be consistent (these are cosmological, not per-tracer).
    params_1.f_NL == params_2.f_NL || error("ClGRParamCache: params_1.f_NL ($(params_1.f_NL)) != params_2.f_NL ($(params_2.f_NL))")
    params_1.Omm0 == params_2.Omm0 || error("ClGRParamCache: params_1.Omm0 != params_2.Omm0")
    params_1.H0   == params_2.H0   || error("ClGRParamCache: params_1.H0 != params_2.H0")

    f_NL = params_1.f_NL
    Omm0 = params_1.Omm0
    H0   = params_1.H0
    c_light = 2.99792458e5
    fNL_prefactor = 1.5 * Omm0 * (100.0 / c_light)^2  # (h/Mpc)^2 unit

    # Per-axis params [nr] — two independent evaluations.
    D1    = [params_1.D(r)    for r in rr]
    aH1    = [params_1.aH(r)    for r in rr]
    bg1   = [params_1.bg(r)   for r in rr]
    β1    = [params_1.β(r)    for r in rr]
    B1    = [params_1.B(r)    for r in rr]
    A1    = [params_1.A(r)    for r in rr]
    Q1    = [params_1.Q(r)    for r in rr]
    bPhi1 = [params_1.bPhi(r) for r in rr]

    D2    = [params_2.D(r)    for r in rr]
    aH2    = [params_2.aH(r)    for r in rr]
    bg2   = [params_2.bg(r)   for r in rr]
    β2    = [params_2.β(r)    for r in rr]
    B2    = [params_2.B(r)    for r in rr]
    A2    = [params_2.A(r)    for r in rr]
    Q2    = [params_2.Q(r)    for r in rr]
    bPhi2 = [params_2.bPhi(r) for r in rr]

    # Pre-compute prefactor[i,j] = D1[i] * D2[j]
    prefactor = zeros(Float64, nr, nr)
    for j in 1:nr
        for i in 1:nr
            prefactor[i, j] = D1[i] * D2[j]
        end
    end

    # Allocate all coefficient matrices [nr, nr]
    c1_w000 = zeros(Float64, nr, nr)
    c1_w020 = zeros(Float64, nr, nr)
    c1_w002 = zeros(Float64, nr, nr)
    c1_w022 = zeros(Float64, nr, nr)
    c2_wm101 = zeros(Float64, nr, nr)
    c2_wm121 = zeros(Float64, nr, nr)
    c2_wm110 = zeros(Float64, nr, nr)
    c2_wm112 = zeros(Float64, nr, nr)
    c3_wm200 = zeros(Float64, nr, nr)
    c3_wm220 = zeros(Float64, nr, nr)
    c3_wm310 = zeros(Float64, nr, nr)
    c4_wm200 = zeros(Float64, nr, nr)
    c4_wm202 = zeros(Float64, nr, nr)
    c4_wm301 = zeros(Float64, nr, nr)
    c5_sm200rp = zeros(Float64, nr, nr)
    c5_sm220rp = zeros(Float64, nr, nr)
    c5_sm310rp = zeros(Float64, nr, nr)
    c5_sm400rp = zeros(Float64, nr, nr)
    c6_sm200r = zeros(Float64, nr, nr)
    c6_sm202r = zeros(Float64, nr, nr)
    c6_sm301r = zeros(Float64, nr, nr)
    c6_sm400r = zeros(Float64, nr, nr)
    c7_tm200rp = zeros(Float64, nr, nr)
    c7_tm220rp = zeros(Float64, nr, nr)
    c7_tm310rp = zeros(Float64, nr, nr)
    c7_tm400rp = zeros(Float64, nr, nr)
    c8_tm200r = zeros(Float64, nr, nr)
    c8_tm202r = zeros(Float64, nr, nr)
    c8_tm301r = zeros(Float64, nr, nr)
    c8_tm400r = zeros(Float64, nr, nr)
    c9_lm200rp = zeros(Float64, nr, nr)
    c9_lm220rp = zeros(Float64, nr, nr)
    c9_lm310rp = zeros(Float64, nr, nr)
    c9_lm400rp = zeros(Float64, nr, nr)
    c10_lm200r = zeros(Float64, nr, nr)
    c10_lm202r = zeros(Float64, nr, nr)
    c10_lm301r = zeros(Float64, nr, nr)
    c10_lm400r = zeros(Float64, nr, nr)
    c11_scrXrrp = zeros(Float64, nr, nr)
    c11_scrYrrp = zeros(Float64, nr, nr)
    c11_scrZrpr = zeros(Float64, nr, nr)
    c12_scrXrpr = zeros(Float64, nr, nr)
    c12_scrYrpr = zeros(Float64, nr, nr)
    c12_scrZrrp = zeros(Float64, nr, nr)
    c13_um200 = zeros(Float64, nr, nr)
    c13_um220 = zeros(Float64, nr, nr)
    c13_um310 = zeros(Float64, nr, nr)
    c13_um400 = zeros(Float64, nr, nr)
    c14_um200 = zeros(Float64, nr, nr)
    c14_um202 = zeros(Float64, nr, nr)
    c14_um301 = zeros(Float64, nr, nr)
    c14_um400 = zeros(Float64, nr, nr)
    c15_scrsm4rp = zeros(Float64, nr, nr)
    c15_scrtm4rp = zeros(Float64, nr, nr)
    c15_scrlm4rp = zeros(Float64, nr, nr)
    c16_scrsm4r = zeros(Float64, nr, nr)
    c16_scrtm4r = zeros(Float64, nr, nr)
    c16_scrlm4r = zeros(Float64, nr, nr)
    c17_wm211 = zeros(Float64, nr, nr)
    c17_wm400 = zeros(Float64, nr, nr)
    c17_scrSrrp = zeros(Float64, nr, nr)
    c18_scrTrrp = zeros(Float64, nr, nr)
    c18_scrLrrp = zeros(Float64, nr, nr)
    c19_vm400 = zeros(Float64, nr, nr)

    # Compute all coefficients on physical (i, j) grid
    @inbounds @threads for j in 1:nr
        for i in 1:nr
            r1 = rr[i]
            r2 = rr[j]
            pf = prefactor[i, j]

            # Local param values
            _D1, _D2 = D1[i], D2[j]
            _aH1, _aH2 = aH1[i], aH2[j]
            _bg1, _bg2 = bg1[i], bg2[j]
            _β1, _β2 = β1[i], β2[j]
            _B1, _B2 = B1[i], B2[j]
            _A1, _A2 = A1[i], A2[j]
            _Q1, _Q2 = Q1[i], Q2[j]
            _bPhi1, _bPhi2 = bPhi1[i], bPhi2[j]

            # Common sub-expressions
            bg1_bg2 = _bg1 * _bg2
            oneMinusQ1 = 1.0 - _Q1
            oneMinusQ2 = 1.0 - _Q2
            B1_over_β1 = _B1 / _β1
            B2_over_β2 = _B2 / _β2
            aH1_B1 = _aH1 * _B1
            aH2_B2 = _aH2 * _B2
            aH1sq_A1 = _aH1^2 * _A1
            aH2sq_A2 = _aH2^2 * _A2
            bPhi1_over_D1 = _bPhi1 / _D1
            bPhi2_over_D2 = _bPhi2 / _D2

            # Term 1: bg1*bg2*(w - β1*w - β2*w + β1*β2*w)
            c1_w000[i, j] = pf * bg1_bg2
            c1_w020[i, j] = -pf * bg1_bg2 * _β1
            c1_w002[i, j] = -pf * bg1_bg2 * _β2
            c1_w022[i, j] = pf * bg1_bg2 * _β1 * _β2

            # Term 2
            c2_wm101[i, j] = pf * _bg1 * aH2_B2
            c2_wm121[i, j] = -pf * _bg1 * aH2_B2 * _β1
            c2_wm110[i, j] = pf * _bg2 * aH1_B1
            c2_wm112[i, j] = -pf * _bg2 * aH1_B1 * _β2

            # Term 3
            c3_wm200[i, j] = pf * _bg1 * aH2sq_A2
            c3_wm220[i, j] = -pf * _bg1 * aH2sq_A2 * _β1
            c3_wm310[i, j] = pf * _aH1 * aH2sq_A2 * _B1

            # Term 4
            c4_wm200[i, j] = pf * _bg2 * aH1sq_A1
            c4_wm202[i, j] = -pf * _bg2 * aH1sq_A1 * _β2
            c4_wm301[i, j] = pf * _aH2 * aH1sq_A1 * _B2

            # Term 5
            term5_base = pf * _bg1 * B2_over_β2
            c5_sm200rp[i, j] = term5_base
            c5_sm220rp[i, j] = -term5_base * _β1
            c5_sm310rp[i, j] = pf * _aH1 * B2_over_β2 * _B1
            c5_sm400rp[i, j] = pf * aH1sq_A1 * B2_over_β2

            # Term 6
            term6_base = pf * _bg2 * B1_over_β1
            c6_sm200r[i, j] = term6_base
            c6_sm202r[i, j] = -term6_base * _β2
            c6_sm301r[i, j] = pf * _aH2 * B1_over_β1 * _B2
            c6_sm400r[i, j] = pf * aH2sq_A2 * B1_over_β1

            # Term 7
            term7_base = -2.0 * pf * _bg1 * oneMinusQ2 / r2
            c7_tm200rp[i, j] = term7_base
            c7_tm220rp[i, j] = -term7_base * _β1
            c7_tm310rp[i, j] = -2.0 * pf * _aH1 * oneMinusQ2 / r2 * _B1
            c7_tm400rp[i, j] = -2.0 * pf * aH1sq_A1 * oneMinusQ2 / r2

            # Term 8
            term8_base = -2.0 * pf * _bg2 * oneMinusQ1 / r1
            c8_tm200r[i, j] = term8_base
            c8_tm202r[i, j] = -term8_base * _β2
            c8_tm301r[i, j] = -2.0 * pf * _aH2 * oneMinusQ1 / r1 * _B2
            c8_tm400r[i, j] = -2.0 * pf * aH2sq_A2 * oneMinusQ1 / r1

            # Term 9
            term9_base = -2.0 * pf * _bg1 * oneMinusQ2
            c9_lm200rp[i, j] = term9_base
            c9_lm220rp[i, j] = -term9_base * _β1
            c9_lm310rp[i, j] = -2.0 * pf * _aH1 * oneMinusQ2 * _B1
            c9_lm400rp[i, j] = -2.0 * pf * aH1sq_A1 * oneMinusQ2

            # Term 10
            term10_base = -2.0 * pf * _bg2 * oneMinusQ1
            c10_lm200r[i, j] = term10_base
            c10_lm202r[i, j] = -term10_base * _β2
            c10_lm301r[i, j] = -2.0 * pf * _aH2 * oneMinusQ1 * _B2
            c10_lm400r[i, j] = -2.0 * pf * aH2sq_A2 * oneMinusQ1

            # Term 11
            # NOTE: per Eq. (4.24), X and Y prefactor on the i-side is B1/f1.
            # B1_over_β1 = B1/β1 = B1·bg1/f1, so we divide by _bg1 to get B1/f1.
            # Previous code had _bg1 * B1_over_β1 = bg1²·B1/f1 (wrong by bg1²).
            c11_scrXrrp[i, j] = -2.0 * pf * oneMinusQ2 * B1_over_β1 / _bg1 / r2
            c11_scrYrrp[i, j] = -2.0 * pf * oneMinusQ2 * B1_over_β1 / _bg1
            c11_scrZrpr[i, j] = 4.0 * pf * oneMinusQ2 * oneMinusQ1 / r2

            # Term 12
            # See Term-11 note: j-side B/f prefactor needs /_bg2 division.
            c12_scrXrpr[i, j] = -2.0 * pf * oneMinusQ1 * B2_over_β2 / _bg2 / r1
            c12_scrYrpr[i, j] = -2.0 * pf * oneMinusQ1 * B2_over_β2 / _bg2
            c12_scrZrrp[i, j] = 4.0 * pf * oneMinusQ1 * oneMinusQ2 / r1

            # Terms 13-16: f_NL terms
            fNL_coeff = fNL_prefactor * f_NL

            # Term 13
            term13_base = pf * fNL_coeff * _bg1 * bPhi2_over_D2
            c13_um200[i, j] = term13_base
            c13_um220[i, j] = -term13_base * _β1
            c13_um310[i, j] = pf * fNL_coeff * _aH1 * bPhi2_over_D2 * _B1
            c13_um400[i, j] = pf * fNL_coeff * aH1sq_A1 * bPhi2_over_D2

            # Term 14
            term14_base = pf * fNL_coeff * _bg2 * bPhi1_over_D1
            c14_um200[i, j] = term14_base
            c14_um202[i, j] = -term14_base * _β2
            c14_um301[i, j] = pf * fNL_coeff * _aH2 * bPhi1_over_D1 * _B2
            c14_um400[i, j] = pf * fNL_coeff * aH2sq_A2 * bPhi1_over_D1

            # Term 15
            c15_scrsm4rp[i, j] = pf * fNL_coeff * bPhi1_over_D1 * _B2 / (_bg2 * _β2)
            c15_scrtm4rp[i, j] = -2.0 * pf * fNL_coeff * bPhi1_over_D1 * oneMinusQ2 / r2
            c15_scrlm4rp[i, j] = -2.0 * pf * fNL_coeff * bPhi1_over_D1 * oneMinusQ2

            # Term 16
            c16_scrsm4r[i, j] = pf * fNL_coeff * bPhi2_over_D2 * _B1 / (_bg1 * _β1)
            c16_scrtm4r[i, j] = -2.0 * pf * fNL_coeff * bPhi2_over_D2 * oneMinusQ1 / r1
            c16_scrlm4r[i, j] = -2.0 * pf * fNL_coeff * bPhi2_over_D2 * oneMinusQ1

            # Term 17
            c17_wm211[i, j] = pf * aH1_B1 * aH2_B2
            c17_wm400[i, j] = pf * aH1sq_A1 * aH2sq_A2
            c17_scrSrrp[i, j] = pf * B1_over_β1 / _bg1 * B2_over_β2 / _bg2

            # Term 18
            c18_scrTrrp[i, j] = 4.0 * pf * oneMinusQ1 / r1 * oneMinusQ2 / r2
            c18_scrLrrp[i, j] = 4.0 * pf * oneMinusQ1 * oneMinusQ2

            # Term 19
            fNL_sq_coeff = (9.0/4.0) * Omm0^2 * (100.0/c_light)^4 * f_NL^2
            c19_vm400[i, j] = pf * fNL_sq_coeff * bPhi1_over_D1 * bPhi2_over_D2
        end
    end

    # Extract integral arrays from I (avoid Dict lookup in hot path)
    W_0_0_0 = I[:w, 0, 0, 0, :none]
    W_0_2_0 = I[:w, 0, 2, 0, :none]
    W_0_0_2 = I[:w, 0, 0, 2, :none]
    W_0_2_2 = I[:w, 0, 2, 2, :none]
    W_m1_0_1 = I[:w, -1, 0, 1, :none]
    W_m1_2_1 = I[:w, -1, 2, 1, :none]
    W_m1_1_0 = I[:w, -1, 1, 0, :none]
    W_m1_1_2 = I[:w, -1, 1, 2, :none]
    W_m2_0_0 = I[:w, -2, 0, 0, :none]
    W_m2_2_0 = I[:w, -2, 2, 0, :none]
    W_m2_0_2 = I[:w, -2, 0, 2, :none]
    W_m2_1_1 = I[:w, -2, 1, 1, :none]
    W_m3_1_0 = I[:w, -3, 1, 0, :none]
    W_m3_0_1 = I[:w, -3, 0, 1, :none]
    W_m4_0_0 = I[:w, -4, 0, 0, :none]

    S_m2_0_0_r = I[:s, -2, 0, 0, :r]
    S_m2_0_2_r = I[:s, -2, 0, 2, :r]
    S_m2_0_0_rp = I[:s, -2, 0, 0, :rp]
    S_m2_2_0_rp = I[:s, -2, 2, 0, :rp]
    S_m3_0_1_r = I[:s, -3, 0, 1, :r]
    S_m3_1_0_rp = I[:s, -3, 1, 0, :rp]
    S_m4_0_0_r = I[:s, -4, 0, 0, :r]
    S_m4_0_0_rp = I[:s, -4, 0, 0, :rp]

    T_m2_0_0_r = I[:t, -2, 0, 0, :r]
    T_m2_0_2_r = I[:t, -2, 0, 2, :r]
    T_m2_0_0_rp = I[:t, -2, 0, 0, :rp]
    T_m2_2_0_rp = I[:t, -2, 2, 0, :rp]
    T_m3_0_1_r = I[:t, -3, 0, 1, :r]
    T_m3_1_0_rp = I[:t, -3, 1, 0, :rp]
    T_m4_0_0_r = I[:t, -4, 0, 0, :r]
    T_m4_0_0_rp = I[:t, -4, 0, 0, :rp]

    L_m2_0_0_r = I[:tl, -2, 0, 0, :r]
    L_m2_0_2_r = I[:tl, -2, 0, 2, :r]
    L_m2_0_0_rp = I[:tl, -2, 0, 0, :rp]
    L_m2_2_0_rp = I[:tl, -2, 2, 0, :rp]
    L_m3_0_1_r = I[:tl, -3, 0, 1, :r]
    L_m3_1_0_rp = I[:tl, -3, 1, 0, :rp]
    L_m4_0_0_r = I[:tl, -4, 0, 0, :r]
    L_m4_0_0_rp = I[:tl, -4, 0, 0, :rp]

    ScrX_r_rp = I[:scrX, -4, 0, 0, :r_rp]
    ScrX_rp_r = I[:scrX, -4, 0, 0, :rp_r]
    ScrY_r_rp = I[:tscrY, -4, 0, 0, :r_rp]
    ScrY_rp_r = I[:tscrY, -4, 0, 0, :rp_r]
    ScrZ_r_rp = I[:tscrZ, -4, 0, 0, :r_rp]
    ScrZ_rp_r = I[:tscrZ, -4, 0, 0, :rp_r]
    ScrS_r_rp = I[:scrS, -4, 0, 0, :r_rp]
    ScrT_r_rp = I[:scrT, -4, 0, 0, :r_rp]
    ScrL_r_rp = I[:tscrL, -4, 0, 0, :r_rp]

    U_m2_0_0 = I[:u, -2, 0, 0, :none]
    U_m2_2_0 = I[:u, -2, 2, 0, :none]
    U_m2_0_2 = I[:u, -2, 0, 2, :none]
    U_m3_1_0 = I[:u, -3, 1, 0, :none]
    U_m3_0_1 = I[:u, -3, 0, 1, :none]
    U_m4_0_0 = I[:u, -4, 0, 0, :none]

    Scrs_m4_r = I[:scrs, -4, 0, 0, :r]
    Scrs_m4_rp = I[:scrs, -4, 0, 0, :rp]
    Scrt_m4_r = I[:scrt, -4, 0, 0, :r]
    Scrt_m4_rp = I[:scrt, -4, 0, 0, :rp]
    Scrl_m4_r = I[:tscrl, -4, 0, 0, :r]
    Scrl_m4_rp = I[:tscrl, -4, 0, 0, :rp]

    V_m4_0_0 = I[:v, -4, 0, 0, :none]

    ell_values = I.ell_values

    return ClGRParamCache(
        nr, rr, ell_values,
        D1, aH1, bg1, β1, B1, A1, Q1, bPhi1,
        D2, aH2, bg2, β2, B2, A2, Q2, bPhi2,
        prefactor,
        c1_w000, c1_w020, c1_w002, c1_w022,
        c2_wm101, c2_wm121, c2_wm110, c2_wm112,
        c3_wm200, c3_wm220, c3_wm310,
        c4_wm200, c4_wm202, c4_wm301,
        c5_sm200rp, c5_sm220rp, c5_sm310rp, c5_sm400rp,
        c6_sm200r, c6_sm202r, c6_sm301r, c6_sm400r,
        c7_tm200rp, c7_tm220rp, c7_tm310rp, c7_tm400rp,
        c8_tm200r, c8_tm202r, c8_tm301r, c8_tm400r,
        c9_lm200rp, c9_lm220rp, c9_lm310rp, c9_lm400rp,
        c10_lm200r, c10_lm202r, c10_lm301r, c10_lm400r,
        c11_scrXrrp, c11_scrYrrp, c11_scrZrpr,
        c12_scrXrpr, c12_scrYrpr, c12_scrZrrp,
        c13_um200, c13_um220, c13_um310, c13_um400,
        c14_um200, c14_um202, c14_um301, c14_um400,
        c15_scrsm4rp, c15_scrtm4rp, c15_scrlm4rp,
        c16_scrsm4r, c16_scrtm4r, c16_scrlm4r,
        c17_wm211, c17_wm400, c17_scrSrrp,
        c18_scrTrrp, c18_scrLrrp,
        c19_vm400,
        f_NL, Omm0, H0, fNL_prefactor,
        # Integral arrays
        W_0_0_0, W_0_2_0, W_0_0_2, W_0_2_2,
        W_m1_0_1, W_m1_2_1, W_m1_1_0, W_m1_1_2,
        W_m2_0_0, W_m2_2_0, W_m2_0_2, W_m2_1_1,
        W_m3_1_0, W_m3_0_1, W_m4_0_0,
        S_m2_0_0_r, S_m2_0_2_r, S_m2_0_0_rp, S_m2_2_0_rp,
        S_m3_0_1_r, S_m3_1_0_rp, S_m4_0_0_r, S_m4_0_0_rp,
        T_m2_0_0_r, T_m2_0_2_r, T_m2_0_0_rp, T_m2_2_0_rp,
        T_m3_0_1_r, T_m3_1_0_rp, T_m4_0_0_r, T_m4_0_0_rp,
        L_m2_0_0_r, L_m2_0_2_r, L_m2_0_0_rp, L_m2_2_0_rp,
        L_m3_0_1_r, L_m3_1_0_rp, L_m4_0_0_r, L_m4_0_0_rp,
        ScrX_r_rp, ScrX_rp_r, ScrY_r_rp, ScrY_rp_r,
        ScrZ_r_rp, ScrZ_rp_r, ScrS_r_rp, ScrT_r_rp, ScrL_r_rp,
        U_m2_0_0, U_m2_2_0, U_m2_0_2, U_m3_1_0, U_m3_0_1, U_m4_0_0,
        Scrs_m4_r, Scrs_m4_rp, Scrt_m4_r, Scrt_m4_rp, Scrl_m4_r, Scrl_m4_rp,
        V_m4_0_0
    )
end

# =============================================================================
# Loading Functions
# =============================================================================

# Helper to decode p values ("m" prefix → negative)
function _decode_p(p_str::String)
    if startswith(p_str, "m")
        return -parse(Int, p_str[2:end])
    else
        return parse(Int, p_str)
    end
end

# Helper to parse key from JLD2
function _parse_key(key::String, is_base::Bool)
    parts = split(key, "_")
    type = Symbol(parts[1])
    p = _decode_p(String(parts[2]))
    j = parse(Int, parts[3])
    jp = parse(Int, parts[4])

    if is_base
        sub = :none
    else
        sub_str = join(parts[5:end], "_")
        sub = Symbol(sub_str)
    end

    return (type, p, j, jp, sub)
end

"""
    load_integrals(filename::String) -> IntegralCollection

Load pre-computed integrals from JLD2 file.
"""
function load_integrals(filename::String)
    data = Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}()

    local rr, ell_values

    jldopen(filename, "r") do f
        # Load grid information
        rr = Float64.(f["grid/rr"])
        ell_values = f["grid/ell_values"]

        # Load base TwoFAST results (w, u, v)
        # Files store [nr, nr, n_ell]; convert to [n_ell, nr, nr] for cache-optimal access
        if haskey(f, "base")
            for key in keys(f["base"])
                type, p, j, jp, sub = _parse_key(key, true)
                arr = Float64.(f["base/$key"])
                data[(type, p, j, jp, sub)] = ndims(arr) == 3 ? permutedims(arr, (3, 1, 2)) : arr
            end
        end

        # Load integrated quantities
        if haskey(f, "integrated")
            for key in keys(f["integrated"])
                type, p, j, jp, sub = _parse_key(key, false)
                arr = Float64.(f["integrated/$key"])
                data[(type, p, j, jp, sub)] = ndims(arr) == 3 ? permutedims(arr, (3, 1, 2)) : arr
            end
        end
    end

    # Get actual sizes from loaded data
    nr = length(rr)
    n_ell = length(ell_values)

    println("Loaded $(length(data)) arrays from $filename")
    println("  - r grid: $nr points, range [$(round(rr[1], digits=1)), $(round(rr[end], digits=1))] Mpc/h")
    println("  - ℓ values: $n_ell, range [$(ell_values[1]), $(ell_values[end])]")

    return IntegralCollection(data, rr, ell_values)
end

"""
    load_integrals_hdf5(filename::String) -> IntegralCollection

Load pre-computed integrals from HDF5 file.

Supports two formats:
1. Single file (format_version 1.0): All data in one .h5 file
2. Split files (format_version 2.0): Meta file + part files
   - Pass the `*_meta.h5` file as filename
   - Part files must be in the same directory
"""
function load_integrals_hdf5(filename::String)
    # Check if this is a split format (meta file)
    is_split = false
    if endswith(filename, "_meta.h5")
        is_split = true
    else
        # Check format_version inside the file
        h5open(filename, "r") do f
            if haskey(f, "metadata/format_version")
                ver = read(f, "metadata/format_version")
                is_split = (ver == "2.0")
            end
        end
    end

    if is_split
        return _load_integrals_hdf5_split(filename)
    else
        return _load_integrals_hdf5_single(filename)
    end
end

"""
Load from single HDF5 file (format version 1.0)
"""
function _load_integrals_hdf5_single(filename::String)
    data = Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}()

    local rr, ell_values

    h5open(filename, "r") do f
        # Load grid information
        rr = Float64.(read(f, "grid/rr"))
        ell_values = read(f, "grid/ell_values")

        # Load base TwoFAST results (w, u, v)
        # Files store [nr, nr, n_ell]; convert to [n_ell, nr, nr] for cache-optimal access
        if haskey(f, "base")
            base_grp = f["base"]
            for key in keys(base_grp)
                type, p, j, jp, sub = _parse_key(key, true)
                arr = Float64.(read(base_grp, key))
                data[(type, p, j, jp, sub)] = ndims(arr) == 3 ? permutedims(arr, (3, 1, 2)) : arr
            end
        end

        # Load integrated quantities
        if haskey(f, "integrated")
            int_grp = f["integrated"]
            for key in keys(int_grp)
                type, p, j, jp, sub = _parse_key(key, false)
                arr = Float64.(read(int_grp, key))
                data[(type, p, j, jp, sub)] = ndims(arr) == 3 ? permutedims(arr, (3, 1, 2)) : arr
            end
        end
    end

    # Get actual sizes from loaded data
    nr = length(rr)
    n_ell = length(ell_values)

    println("Loaded $(length(data)) arrays from $filename (HDF5)")
    println("  - r grid: $nr points, range [$(round(rr[1], digits=1)), $(round(rr[end], digits=1))] Mpc/h")
    println("  - ℓ values: $n_ell, range [$(ell_values[1]), $(ell_values[end])]")

    return IntegralCollection(data, rr, ell_values)
end

"""
Load from split HDF5 files (format version 2.0)
"""
function _load_integrals_hdf5_split(meta_filename::String)
    # Get directory of meta file
    meta_dir = dirname(meta_filename)
    if isempty(meta_dir)
        meta_dir = "."
    end

    local rr, ell_values, part_files, ell_ranges

    # Load meta file
    h5open(meta_filename, "r") do f
        rr = Float64.(read(f, "grid/rr"))
        ell_values = read(f, "grid/ell_values")
        part_files = read(f, "part_files")
        ell_ranges = read(f, "metadata/ell_ranges")
    end

    n_ell = length(ell_values)
    nr = length(rr)
    n_parts = length(part_files)

    println("Loading split HDF5 ($n_parts parts)...")

    # Initialize data dictionary - we'll build full arrays by concatenating parts
    data = Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}()

    # Load each part file and concatenate
    for (part_idx, part_file) in enumerate(part_files)
        part_path = joinpath(meta_dir, part_file)
        ell_start = ell_ranges[part_idx, 1]
        ell_end = ell_ranges[part_idx, 2]

        println("  Loading $part_file (ell $ell_start:$ell_end)...")

        h5open(part_path, "r") do f
            # Load base TwoFAST results
            # Split files store [n_ell_part, nr, nr] — already in target layout
            if haskey(f, "base")
                base_grp = f["base"]
                for key in keys(base_grp)
                    tuple_key = _parse_key(key, true)
                    arr_part = Float64.(read(base_grp, key))

                    if !haskey(data, tuple_key)
                        # First part: allocate full array in [n_ell, nr, nr] layout
                        data[tuple_key] = zeros(Float64, n_ell, nr, nr)
                    end
                    # Split files already in [n_ell_part, nr, nr] — direct copy
                    data[tuple_key][ell_start:ell_end, :, :] = arr_part
                end
            end

            # Load integrated quantities
            if haskey(f, "integrated")
                int_grp = f["integrated"]
                for key in keys(int_grp)
                    tuple_key = _parse_key(key, false)
                    arr_part = Float64.(read(int_grp, key))

                    if !haskey(data, tuple_key)
                        data[tuple_key] = zeros(Float64, n_ell, nr, nr)
                    end
                    # Split files already in [n_ell_part, nr, nr] — direct copy
                    data[tuple_key][ell_start:ell_end, :, :] = arr_part
                end
            end
        end
    end

    println("Loaded $(length(data)) arrays from $n_parts part files")
    println("  - r grid: $nr points, range [$(round(rr[1], digits=1)), $(round(rr[end], digits=1))] Mpc/h")
    println("  - ℓ values: $n_ell, range [$(ell_values[1]), $(ell_values[end])]")

    return IntegralCollection(data, rr, ell_values)
end

"""
    show_available_keys(I::IntegralCollection)

Print all available keys in the IntegralCollection.
"""
function show_available_keys(I::IntegralCollection)
    sorted_keys = sort(collect(keys(I.data)))
    println("Available integrals ($(length(sorted_keys)) total):")
    for k in sorted_keys
        println("  I[$(k[1]), $(k[2]), $(k[3]), $(k[4]), $(k[5])]")
    end
end

# =============================================================================
# Helper Functions
# =============================================================================

"""
    _find_nearest_idx(rr::Vector{Float64}, r::Float64) -> Union{Int, Nothing}

Find the nearest index in rr for value r.
Returns nothing if r is outside the grid range.
"""
function _find_nearest_idx(rr::Vector{Float64}, r::Float64)
    r < rr[1] && return nothing
    r > rr[end] && return nothing

    # For log-uniform grid, use log interpolation
    log_r = log10(r)
    log_rr = log10.(rr)
    idx = searchsortedfirst(log_rr, log_r)

    if idx == 1
        return 1
    elseif idx > length(rr)
        return length(rr)
    else
        return abs(log_rr[idx] - log_r) < abs(log_rr[idx-1] - log_r) ? idx : idx-1
    end
end

"""
    _find_ell_idx(ell_values::Vector{Int}, ell::Int) -> Int

Find the index of ell in ell_values. Error if not found.
"""
function _find_ell_idx(ell_values::Vector{Int}, ell::Int)
    ell_idx = findfirst(==(ell), ell_values)
    if isnothing(ell_idx)
        error("ℓ = $ell not found in ell_values. Available range: [$(ell_values[1]), $(ell_values[end])]")
    end
    return ell_idx
end

# =============================================================================
# Optimized Computation Functions (using ClGRParamCache)
# =============================================================================

"""
    compute_Cl_GR!(result::Matrix{Float64}, I::IntegralCollection,
                   cache::ClGRParamCache, ell::Int)

In-place computation of C_ℓ^GR for a single ℓ value using pre-computed cache.
Writes result to pre-allocated `result` matrix on physical (r₁, r₂) grid.

# Arguments
- `result`: Pre-allocated (nr, nr) matrix to store output
- `I`: IntegralCollection with pre-computed integrals
- `cache`: ClGRParamCache with pre-computed coefficients
- `ell`: The multipole ℓ value
"""
function compute_Cl_GR!(result::Matrix{Float64}, I::IntegralCollection,
                        cache::ClGRParamCache, ell::Int)
    ell_idx = _find_ell_idx(I.ell_values, ell)
    nr = cache.nr

    # Pre-extract integral slices for this ell (contiguous copy from [nell, nr, nr])
    w_0_0_0 = I[:w, 0, 0, 0, :none][ell_idx, :, :]
    w_0_2_0 = I[:w, 0, 2, 0, :none][ell_idx, :, :]
    w_0_0_2 = I[:w, 0, 0, 2, :none][ell_idx, :, :]
    w_0_2_2 = I[:w, 0, 2, 2, :none][ell_idx, :, :]
    w_m1_0_1 = I[:w, -1, 0, 1, :none][ell_idx, :, :]
    w_m1_2_1 = I[:w, -1, 2, 1, :none][ell_idx, :, :]
    w_m1_1_0 = I[:w, -1, 1, 0, :none][ell_idx, :, :]
    w_m1_1_2 = I[:w, -1, 1, 2, :none][ell_idx, :, :]
    w_m2_0_0 = I[:w, -2, 0, 0, :none][ell_idx, :, :]
    w_m2_2_0 = I[:w, -2, 2, 0, :none][ell_idx, :, :]
    w_m2_0_2 = I[:w, -2, 0, 2, :none][ell_idx, :, :]
    w_m2_1_1 = I[:w, -2, 1, 1, :none][ell_idx, :, :]
    w_m3_1_0 = I[:w, -3, 1, 0, :none][ell_idx, :, :]
    w_m3_0_1 = I[:w, -3, 0, 1, :none][ell_idx, :, :]
    w_m4_0_0 = I[:w, -4, 0, 0, :none][ell_idx, :, :]

    s_m2_0_0_r = I[:s, -2, 0, 0, :r][ell_idx, :, :]
    s_m2_0_2_r = I[:s, -2, 0, 2, :r][ell_idx, :, :]
    s_m2_0_0_rp = I[:s, -2, 0, 0, :rp][ell_idx, :, :]
    s_m2_2_0_rp = I[:s, -2, 2, 0, :rp][ell_idx, :, :]
    s_m3_0_1_r = I[:s, -3, 0, 1, :r][ell_idx, :, :]
    s_m3_1_0_rp = I[:s, -3, 1, 0, :rp][ell_idx, :, :]
    s_m4_0_0_r = I[:s, -4, 0, 0, :r][ell_idx, :, :]
    s_m4_0_0_rp = I[:s, -4, 0, 0, :rp][ell_idx, :, :]

    t_m2_0_0_r = I[:t, -2, 0, 0, :r][ell_idx, :, :]
    t_m2_0_2_r = I[:t, -2, 0, 2, :r][ell_idx, :, :]
    t_m2_0_0_rp = I[:t, -2, 0, 0, :rp][ell_idx, :, :]
    t_m2_2_0_rp = I[:t, -2, 2, 0, :rp][ell_idx, :, :]
    t_m3_0_1_r = I[:t, -3, 0, 1, :r][ell_idx, :, :]
    t_m3_1_0_rp = I[:t, -3, 1, 0, :rp][ell_idx, :, :]
    t_m4_0_0_r = I[:t, -4, 0, 0, :r][ell_idx, :, :]
    t_m4_0_0_rp = I[:t, -4, 0, 0, :rp][ell_idx, :, :]

    l_m2_0_0_r = I[:tl, -2, 0, 0, :r][ell_idx, :, :]
    l_m2_0_2_r = I[:tl, -2, 0, 2, :r][ell_idx, :, :]
    l_m2_0_0_rp = I[:tl, -2, 0, 0, :rp][ell_idx, :, :]
    l_m2_2_0_rp = I[:tl, -2, 2, 0, :rp][ell_idx, :, :]
    l_m3_0_1_r = I[:tl, -3, 0, 1, :r][ell_idx, :, :]
    l_m3_1_0_rp = I[:tl, -3, 1, 0, :rp][ell_idx, :, :]
    l_m4_0_0_r = I[:tl, -4, 0, 0, :r][ell_idx, :, :]
    l_m4_0_0_rp = I[:tl, -4, 0, 0, :rp][ell_idx, :, :]

    scrX_r_rp = I[:scrX, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrX_rp_r = I[:scrX, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrY_r_rp = I[:tscrY, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrY_rp_r = I[:tscrY, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrZ_r_rp = I[:tscrZ, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrZ_rp_r = I[:tscrZ, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrS_r_rp = I[:scrS, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrT_r_rp = I[:scrT, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrL_r_rp = I[:tscrL, -4, 0, 0, :r_rp][ell_idx, :, :]

    u_m2_0_0 = I[:u, -2, 0, 0, :none][ell_idx, :, :]
    u_m2_2_0 = I[:u, -2, 2, 0, :none][ell_idx, :, :]
    u_m2_0_2 = I[:u, -2, 0, 2, :none][ell_idx, :, :]
    u_m3_1_0 = I[:u, -3, 1, 0, :none][ell_idx, :, :]
    u_m3_0_1 = I[:u, -3, 0, 1, :none][ell_idx, :, :]
    u_m4_0_0 = I[:u, -4, 0, 0, :none][ell_idx, :, :]

    scrs_m4_r = I[:scrs, -4, 0, 0, :r][ell_idx, :, :]
    scrs_m4_rp = I[:scrs, -4, 0, 0, :rp][ell_idx, :, :]
    scrt_m4_r = I[:scrt, -4, 0, 0, :r][ell_idx, :, :]
    scrt_m4_rp = I[:scrt, -4, 0, 0, :rp][ell_idx, :, :]
    scrl_m4_r = I[:tscrl, -4, 0, 0, :r][ell_idx, :, :]
    scrl_m4_rp = I[:tscrl, -4, 0, 0, :rp][ell_idx, :, :]

    v_m4_0_0 = I[:v, -4, 0, 0, :none][ell_idx, :, :]

    # Loop using cached coefficients on physical (i, j) grid
    @threads for j in 1:nr
        @inbounds for i in 1:nr
            # Sum terms using pre-computed coefficients
            val = cache.c1_w000[i, j] * w_0_0_0[i, j] +
                  cache.c1_w020[i, j] * w_0_2_0[i, j] +
                  cache.c1_w002[i, j] * w_0_0_2[i, j] +
                  cache.c1_w022[i, j] * w_0_2_2[i, j]

            val += cache.c2_wm101[i, j] * w_m1_0_1[i, j] +
                   cache.c2_wm121[i, j] * w_m1_2_1[i, j] +
                   cache.c2_wm110[i, j] * w_m1_1_0[i, j] +
                   cache.c2_wm112[i, j] * w_m1_1_2[i, j]

            val += (cache.c3_wm200[i, j] + cache.c4_wm200[i, j]) * w_m2_0_0[i, j] +
                   cache.c3_wm220[i, j] * w_m2_2_0[i, j] +
                   cache.c4_wm202[i, j] * w_m2_0_2[i, j] +
                   cache.c3_wm310[i, j] * w_m3_1_0[i, j] +
                   cache.c4_wm301[i, j] * w_m3_0_1[i, j]

            val += cache.c5_sm200rp[i, j] * s_m2_0_0_rp[i, j] +
                   cache.c5_sm220rp[i, j] * s_m2_2_0_rp[i, j] +
                   cache.c5_sm310rp[i, j] * s_m3_1_0_rp[i, j] +
                   cache.c5_sm400rp[i, j] * s_m4_0_0_rp[i, j]

            val += cache.c6_sm200r[i, j] * s_m2_0_0_r[i, j] +
                   cache.c6_sm202r[i, j] * s_m2_0_2_r[i, j] +
                   cache.c6_sm301r[i, j] * s_m3_0_1_r[i, j] +
                   cache.c6_sm400r[i, j] * s_m4_0_0_r[i, j]

            val += cache.c7_tm200rp[i, j] * t_m2_0_0_rp[i, j] +
                   cache.c7_tm220rp[i, j] * t_m2_2_0_rp[i, j] +
                   cache.c7_tm310rp[i, j] * t_m3_1_0_rp[i, j] +
                   cache.c7_tm400rp[i, j] * t_m4_0_0_rp[i, j]

            val += cache.c8_tm200r[i, j] * t_m2_0_0_r[i, j] +
                   cache.c8_tm202r[i, j] * t_m2_0_2_r[i, j] +
                   cache.c8_tm301r[i, j] * t_m3_0_1_r[i, j] +
                   cache.c8_tm400r[i, j] * t_m4_0_0_r[i, j]

            val += cache.c9_lm200rp[i, j] * l_m2_0_0_rp[i, j] +
                   cache.c9_lm220rp[i, j] * l_m2_2_0_rp[i, j] +
                   cache.c9_lm310rp[i, j] * l_m3_1_0_rp[i, j] +
                   cache.c9_lm400rp[i, j] * l_m4_0_0_rp[i, j]

            val += cache.c10_lm200r[i, j] * l_m2_0_0_r[i, j] +
                   cache.c10_lm202r[i, j] * l_m2_0_2_r[i, j] +
                   cache.c10_lm301r[i, j] * l_m3_0_1_r[i, j] +
                   cache.c10_lm400r[i, j] * l_m4_0_0_r[i, j]

            val += cache.c11_scrXrrp[i, j] * scrX_r_rp[i, j] +
                   cache.c11_scrYrrp[i, j] * scrY_r_rp[i, j] +
                   cache.c11_scrZrpr[i, j] * scrZ_rp_r[i, j]

            val += cache.c12_scrXrpr[i, j] * scrX_rp_r[i, j] +
                   cache.c12_scrYrpr[i, j] * scrY_rp_r[i, j] +
                   cache.c12_scrZrrp[i, j] * scrZ_r_rp[i, j]

            val += (cache.c13_um200[i, j] + cache.c14_um200[i, j]) * u_m2_0_0[i, j] +
                   cache.c13_um220[i, j] * u_m2_2_0[i, j] +
                   cache.c14_um202[i, j] * u_m2_0_2[i, j] +
                   cache.c13_um310[i, j] * u_m3_1_0[i, j] +
                   cache.c14_um301[i, j] * u_m3_0_1[i, j] +
                   (cache.c13_um400[i, j] + cache.c14_um400[i, j]) * u_m4_0_0[i, j]

            val += cache.c15_scrsm4rp[i, j] * scrs_m4_rp[i, j] +
                   cache.c15_scrtm4rp[i, j] * scrt_m4_rp[i, j] +
                   cache.c15_scrlm4rp[i, j] * scrl_m4_rp[i, j]

            val += cache.c16_scrsm4r[i, j] * scrs_m4_r[i, j] +
                   cache.c16_scrtm4r[i, j] * scrt_m4_r[i, j] +
                   cache.c16_scrlm4r[i, j] * scrl_m4_r[i, j]

            val += cache.c17_wm211[i, j] * w_m2_1_1[i, j] +
                   cache.c17_wm400[i, j] * w_m4_0_0[i, j] +
                   cache.c17_scrSrrp[i, j] * scrS_r_rp[i, j]

            val += cache.c18_scrTrrp[i, j] * scrT_r_rp[i, j] +
                   cache.c18_scrLrrp[i, j] * scrL_r_rp[i, j]

            val += cache.c19_vm400[i, j] * v_m4_0_0[i, j]

            result[i, j] = val
        end
    end

    return result
end

"""
    compute_Cl_GR_batch!(results::Array{Float64,3}, cache::ClGRParamCache, ells::Vector{Int})

Batch computation of C_ℓ^GR for multiple ℓ values on physical (r₁, r₂) grid.
Computes coefficients once per (i, j), then loops over ells efficiently.
Uses cached integral array references for type-stable, allocation-free inner loop.

# Arguments
- `results`: Pre-allocated (n_ells, nr, nr) array for output
- `cache`: ClGRParamCache (contains all integral arrays and coefficients)
- `ells`: Vector of ℓ values to compute
"""
function compute_Cl_GR_batch!(results::Array{Float64,3}, cache::ClGRParamCache, ells::Vector{Int})
    n_ells = length(ells)
    nr = cache.nr

    # Pre-compute ell indices using cached ell_values
    ell_indices = [_find_ell_idx(cache.ell_values, ell) for ell in ells]

    # Extract all arrays from cache BEFORE @threads to avoid closure boxing
    # Coefficient matrices
    c1_w000 = cache.c1_w000
    c1_w020 = cache.c1_w020
    c1_w002 = cache.c1_w002
    c1_w022 = cache.c1_w022
    c2_wm101 = cache.c2_wm101
    c2_wm121 = cache.c2_wm121
    c2_wm110 = cache.c2_wm110
    c2_wm112 = cache.c2_wm112
    c3_wm200 = cache.c3_wm200
    c3_wm220 = cache.c3_wm220
    c3_wm310 = cache.c3_wm310
    c4_wm200 = cache.c4_wm200
    c4_wm202 = cache.c4_wm202
    c4_wm301 = cache.c4_wm301
    c5_sm200rp = cache.c5_sm200rp
    c5_sm220rp = cache.c5_sm220rp
    c5_sm310rp = cache.c5_sm310rp
    c5_sm400rp = cache.c5_sm400rp
    c6_sm200r = cache.c6_sm200r
    c6_sm202r = cache.c6_sm202r
    c6_sm301r = cache.c6_sm301r
    c6_sm400r = cache.c6_sm400r
    c7_tm200rp = cache.c7_tm200rp
    c7_tm220rp = cache.c7_tm220rp
    c7_tm310rp = cache.c7_tm310rp
    c7_tm400rp = cache.c7_tm400rp
    c8_tm200r = cache.c8_tm200r
    c8_tm202r = cache.c8_tm202r
    c8_tm301r = cache.c8_tm301r
    c8_tm400r = cache.c8_tm400r
    c9_lm200rp = cache.c9_lm200rp
    c9_lm220rp = cache.c9_lm220rp
    c9_lm310rp = cache.c9_lm310rp
    c9_lm400rp = cache.c9_lm400rp
    c10_lm200r = cache.c10_lm200r
    c10_lm202r = cache.c10_lm202r
    c10_lm301r = cache.c10_lm301r
    c10_lm400r = cache.c10_lm400r
    c11_scrXrrp = cache.c11_scrXrrp
    c11_scrYrrp = cache.c11_scrYrrp
    c11_scrZrpr = cache.c11_scrZrpr
    c12_scrXrpr = cache.c12_scrXrpr
    c12_scrYrpr = cache.c12_scrYrpr
    c12_scrZrrp = cache.c12_scrZrrp
    c13_um200 = cache.c13_um200
    c13_um220 = cache.c13_um220
    c13_um310 = cache.c13_um310
    c13_um400 = cache.c13_um400
    c14_um200 = cache.c14_um200
    c14_um202 = cache.c14_um202
    c14_um301 = cache.c14_um301
    c14_um400 = cache.c14_um400
    c15_scrsm4rp = cache.c15_scrsm4rp
    c15_scrtm4rp = cache.c15_scrtm4rp
    c15_scrlm4rp = cache.c15_scrlm4rp
    c16_scrsm4r = cache.c16_scrsm4r
    c16_scrtm4r = cache.c16_scrtm4r
    c16_scrlm4r = cache.c16_scrlm4r
    c17_wm211 = cache.c17_wm211
    c17_wm400 = cache.c17_wm400
    c17_scrSrrp = cache.c17_scrSrrp
    c18_scrTrrp = cache.c18_scrTrrp
    c18_scrLrrp = cache.c18_scrLrrp
    c19_vm400 = cache.c19_vm400

    # Integral arrays
    W_0_0_0 = cache.W_0_0_0
    W_0_2_0 = cache.W_0_2_0
    W_0_0_2 = cache.W_0_0_2
    W_0_2_2 = cache.W_0_2_2
    W_m1_0_1 = cache.W_m1_0_1
    W_m1_2_1 = cache.W_m1_2_1
    W_m1_1_0 = cache.W_m1_1_0
    W_m1_1_2 = cache.W_m1_1_2
    W_m2_0_0 = cache.W_m2_0_0
    W_m2_2_0 = cache.W_m2_2_0
    W_m2_0_2 = cache.W_m2_0_2
    W_m2_1_1 = cache.W_m2_1_1
    W_m3_1_0 = cache.W_m3_1_0
    W_m3_0_1 = cache.W_m3_0_1
    W_m4_0_0 = cache.W_m4_0_0
    S_m2_0_0_r = cache.S_m2_0_0_r
    S_m2_0_2_r = cache.S_m2_0_2_r
    S_m2_0_0_rp = cache.S_m2_0_0_rp
    S_m2_2_0_rp = cache.S_m2_2_0_rp
    S_m3_0_1_r = cache.S_m3_0_1_r
    S_m3_1_0_rp = cache.S_m3_1_0_rp
    S_m4_0_0_r = cache.S_m4_0_0_r
    S_m4_0_0_rp = cache.S_m4_0_0_rp
    T_m2_0_0_r = cache.T_m2_0_0_r
    T_m2_0_2_r = cache.T_m2_0_2_r
    T_m2_0_0_rp = cache.T_m2_0_0_rp
    T_m2_2_0_rp = cache.T_m2_2_0_rp
    T_m3_0_1_r = cache.T_m3_0_1_r
    T_m3_1_0_rp = cache.T_m3_1_0_rp
    T_m4_0_0_r = cache.T_m4_0_0_r
    T_m4_0_0_rp = cache.T_m4_0_0_rp
    L_m2_0_0_r = cache.L_m2_0_0_r
    L_m2_0_2_r = cache.L_m2_0_2_r
    L_m2_0_0_rp = cache.L_m2_0_0_rp
    L_m2_2_0_rp = cache.L_m2_2_0_rp
    L_m3_0_1_r = cache.L_m3_0_1_r
    L_m3_1_0_rp = cache.L_m3_1_0_rp
    L_m4_0_0_r = cache.L_m4_0_0_r
    L_m4_0_0_rp = cache.L_m4_0_0_rp
    ScrX_r_rp = cache.ScrX_r_rp
    ScrX_rp_r = cache.ScrX_rp_r
    ScrY_r_rp = cache.ScrY_r_rp
    ScrY_rp_r = cache.ScrY_rp_r
    ScrZ_r_rp = cache.ScrZ_r_rp
    ScrZ_rp_r = cache.ScrZ_rp_r
    ScrS_r_rp = cache.ScrS_r_rp
    ScrT_r_rp = cache.ScrT_r_rp
    ScrL_r_rp = cache.ScrL_r_rp
    U_m2_0_0 = cache.U_m2_0_0
    U_m2_2_0 = cache.U_m2_2_0
    U_m2_0_2 = cache.U_m2_0_2
    U_m3_1_0 = cache.U_m3_1_0
    U_m3_0_1 = cache.U_m3_0_1
    U_m4_0_0 = cache.U_m4_0_0
    Scrs_m4_r = cache.Scrs_m4_r
    Scrs_m4_rp = cache.Scrs_m4_rp
    Scrt_m4_r = cache.Scrt_m4_r
    Scrt_m4_rp = cache.Scrt_m4_rp
    Scrl_m4_r = cache.Scrl_m4_r
    Scrl_m4_rp = cache.Scrl_m4_rp
    V_m4_0_0 = cache.V_m4_0_0

    # Main loop: parallelize over j (outer loop) on physical grid
    @threads for j in 1:nr
        @inbounds for i in 1:nr
            # Load cached coefficients once per (i, j)
            c1_000 = c1_w000[i, j]
            c1_020 = c1_w020[i, j]
            c1_002 = c1_w002[i, j]
            c1_022 = c1_w022[i, j]
            c2_m101 = c2_wm101[i, j]
            c2_m121 = c2_wm121[i, j]
            c2_m110 = c2_wm110[i, j]
            c2_m112 = c2_wm112[i, j]
            c34_m200 = c3_wm200[i, j] + c4_wm200[i, j]
            c3_m220 = c3_wm220[i, j]
            c4_m202 = c4_wm202[i, j]
            c3_m310 = c3_wm310[i, j]
            c4_m301 = c4_wm301[i, j]
            c5_200rp = c5_sm200rp[i, j]
            c5_220rp = c5_sm220rp[i, j]
            c5_310rp = c5_sm310rp[i, j]
            c5_400rp = c5_sm400rp[i, j]
            c6_200r = c6_sm200r[i, j]
            c6_202r = c6_sm202r[i, j]
            c6_301r = c6_sm301r[i, j]
            c6_400r = c6_sm400r[i, j]
            c7_200rp = c7_tm200rp[i, j]
            c7_220rp = c7_tm220rp[i, j]
            c7_310rp = c7_tm310rp[i, j]
            c7_400rp = c7_tm400rp[i, j]
            c8_200r = c8_tm200r[i, j]
            c8_202r = c8_tm202r[i, j]
            c8_301r = c8_tm301r[i, j]
            c8_400r = c8_tm400r[i, j]
            c9_200rp = c9_lm200rp[i, j]
            c9_220rp = c9_lm220rp[i, j]
            c9_310rp = c9_lm310rp[i, j]
            c9_400rp = c9_lm400rp[i, j]
            c10_200r = c10_lm200r[i, j]
            c10_202r = c10_lm202r[i, j]
            c10_301r = c10_lm301r[i, j]
            c10_400r = c10_lm400r[i, j]
            c11_Xrrp = c11_scrXrrp[i, j]
            c11_Yrrp = c11_scrYrrp[i, j]
            c11_Zrpr = c11_scrZrpr[i, j]
            c12_Xrpr = c12_scrXrpr[i, j]
            c12_Yrpr = c12_scrYrpr[i, j]
            c12_Zrrp = c12_scrZrrp[i, j]
            c1314_m200 = c13_um200[i, j] + c14_um200[i, j]
            c13_m220 = c13_um220[i, j]
            c14_m202 = c14_um202[i, j]
            c13_m310 = c13_um310[i, j]
            c14_m301 = c14_um301[i, j]
            c1314_m400 = c13_um400[i, j] + c14_um400[i, j]
            c15_srp = c15_scrsm4rp[i, j]
            c15_trp = c15_scrtm4rp[i, j]
            c15_lrp = c15_scrlm4rp[i, j]
            c16_sr = c16_scrsm4r[i, j]
            c16_tr = c16_scrtm4r[i, j]
            c16_lr = c16_scrlm4r[i, j]
            c17_m211 = c17_wm211[i, j]
            c17_m400 = c17_wm400[i, j]
            c17_Srrp = c17_scrSrrp[i, j]
            c18_Trrp = c18_scrTrrp[i, j]
            c18_Lrrp = c18_scrLrrp[i, j]
            c19_v400 = c19_vm400[i, j]

            # Inner loop over ells - only integral lookups vary
            # NOTE: Expression split into 5 groups to avoid LLVM allocation bug
            # with large expressions (>30 terms causes heap allocations)
            for i_ell in 1:n_ells
                ell_idx = ell_indices[i_ell]

                # Group 1: Terms 1-4 (W integrals)
                @inbounds val1 = c1_000 * W_0_0_0[ell_idx, i, j] +
                      c1_020 * W_0_2_0[ell_idx, i, j] +
                      c1_002 * W_0_0_2[ell_idx, i, j] +
                      c1_022 * W_0_2_2[ell_idx, i, j] +
                      c2_m101 * W_m1_0_1[ell_idx, i, j] +
                      c2_m121 * W_m1_2_1[ell_idx, i, j] +
                      c2_m110 * W_m1_1_0[ell_idx, i, j] +
                      c2_m112 * W_m1_1_2[ell_idx, i, j] +
                      c34_m200 * W_m2_0_0[ell_idx, i, j] +
                      c3_m220 * W_m2_2_0[ell_idx, i, j] +
                      c4_m202 * W_m2_0_2[ell_idx, i, j] +
                      c3_m310 * W_m3_1_0[ell_idx, i, j] +
                      c4_m301 * W_m3_0_1[ell_idx, i, j]

                # Group 2: Terms 5-7 (S and T integrals part 1)
                @inbounds val2 = c5_200rp * S_m2_0_0_rp[ell_idx, i, j] +
                      c5_220rp * S_m2_2_0_rp[ell_idx, i, j] +
                      c5_310rp * S_m3_1_0_rp[ell_idx, i, j] +
                      c5_400rp * S_m4_0_0_rp[ell_idx, i, j] +
                      c6_200r * S_m2_0_0_r[ell_idx, i, j] +
                      c6_202r * S_m2_0_2_r[ell_idx, i, j] +
                      c6_301r * S_m3_0_1_r[ell_idx, i, j] +
                      c6_400r * S_m4_0_0_r[ell_idx, i, j] +
                      c7_200rp * T_m2_0_0_rp[ell_idx, i, j] +
                      c7_220rp * T_m2_2_0_rp[ell_idx, i, j] +
                      c7_310rp * T_m3_1_0_rp[ell_idx, i, j] +
                      c7_400rp * T_m4_0_0_rp[ell_idx, i, j]

                # Group 3: Terms 8-10 (T and L integrals)
                @inbounds val3 = c8_200r * T_m2_0_0_r[ell_idx, i, j] +
                      c8_202r * T_m2_0_2_r[ell_idx, i, j] +
                      c8_301r * T_m3_0_1_r[ell_idx, i, j] +
                      c8_400r * T_m4_0_0_r[ell_idx, i, j] +
                      c9_200rp * L_m2_0_0_rp[ell_idx, i, j] +
                      c9_220rp * L_m2_2_0_rp[ell_idx, i, j] +
                      c9_310rp * L_m3_1_0_rp[ell_idx, i, j] +
                      c9_400rp * L_m4_0_0_rp[ell_idx, i, j] +
                      c10_200r * L_m2_0_0_r[ell_idx, i, j] +
                      c10_202r * L_m2_0_2_r[ell_idx, i, j] +
                      c10_301r * L_m3_0_1_r[ell_idx, i, j] +
                      c10_400r * L_m4_0_0_r[ell_idx, i, j]

                # Group 4: Terms 11-14 (Scr and U integrals)
                @inbounds val4 = c11_Xrrp * ScrX_r_rp[ell_idx, i, j] +
                      c11_Yrrp * ScrY_r_rp[ell_idx, i, j] +
                      c11_Zrpr * ScrZ_rp_r[ell_idx, i, j] +
                      c12_Xrpr * ScrX_rp_r[ell_idx, i, j] +
                      c12_Yrpr * ScrY_rp_r[ell_idx, i, j] +
                      c12_Zrrp * ScrZ_r_rp[ell_idx, i, j] +
                      c1314_m200 * U_m2_0_0[ell_idx, i, j] +
                      c13_m220 * U_m2_2_0[ell_idx, i, j] +
                      c14_m202 * U_m2_0_2[ell_idx, i, j] +
                      c13_m310 * U_m3_1_0[ell_idx, i, j] +
                      c14_m301 * U_m3_0_1[ell_idx, i, j] +
                      c1314_m400 * U_m4_0_0[ell_idx, i, j]

                # Group 5: Terms 15-19 (remaining terms)
                @inbounds val5 = c15_srp * Scrs_m4_rp[ell_idx, i, j] +
                      c15_trp * Scrt_m4_rp[ell_idx, i, j] +
                      c15_lrp * Scrl_m4_rp[ell_idx, i, j] +
                      c16_sr * Scrs_m4_r[ell_idx, i, j] +
                      c16_tr * Scrt_m4_r[ell_idx, i, j] +
                      c16_lr * Scrl_m4_r[ell_idx, i, j] +
                      c17_m211 * W_m2_1_1[ell_idx, i, j] +
                      c17_m400 * W_m4_0_0[ell_idx, i, j] +
                      c17_Srrp * ScrS_r_rp[ell_idx, i, j] +
                      c18_Trrp * ScrT_r_rp[ell_idx, i, j] +
                      c18_Lrrp * ScrL_r_rp[ell_idx, i, j] +
                      c19_v400 * V_m4_0_0[ell_idx, i, j]

                results[i_ell, i, j] = val1 + val2 + val3 + val4 + val5
            end
        end
    end

    return results
end

"""
    compute_Cl_GR_batch(I::IntegralCollection, params::ClGRParams,
                        ells::Vector{Int}) -> Array{Float64,3}

Convenience wrapper for batch computation on physical (r₁, r₂) grid.
Allocates cache and result array, then calls in-place version.

# Arguments
- `I`: IntegralCollection
- `params`: ClGRParams
- `ells`: Vector of ℓ values (or pass I.ell_values for all)

# Returns
- Array{Float64,3} of shape (n_ells, nr, nr)
"""
function compute_Cl_GR_batch(I::IntegralCollection, params::ClGRParams,
                             ells::Vector{Int})
    cache = ClGRParamCache(I, params)
    results = zeros(Float64, length(ells), cache.nr, cache.nr)
    compute_Cl_GR_batch!(results, cache, ells)
    return results
end

# =============================================================================
# Streaming variant: load one integral array at a time from split HDF5
# =============================================================================

"""
    _build_array_coeff_pairs(cache) -> Vector{Tuple{Matrix{Float64},Tuple,Bool}}

Enumerate all (effective_coeff, array_key, is_base) triples that the
19-term sum reduces to.  Two pairs of terms share an integral array
(W_m2_0_0 via c3/c4, U_m2_0_0 and U_m4_0_0 via c13/c14), so their
coefficient matrices are pre-summed here to halve I/O for those arrays.
"""
function _build_array_coeff_pairs(cache::ClGRParamCache)
    pairs = Tuple{Matrix{Float64}, NTuple{5,Any}, Bool}[]

    # W base (is_base=true, sub=:none)
    push!(pairs, (cache.c1_w000, (:w, 0, 0, 0, :none), true))
    push!(pairs, (cache.c1_w020, (:w, 0, 2, 0, :none), true))
    push!(pairs, (cache.c1_w002, (:w, 0, 0, 2, :none), true))
    push!(pairs, (cache.c1_w022, (:w, 0, 2, 2, :none), true))
    push!(pairs, (cache.c2_wm101, (:w, -1, 0, 1, :none), true))
    push!(pairs, (cache.c2_wm121, (:w, -1, 2, 1, :none), true))
    push!(pairs, (cache.c2_wm110, (:w, -1, 1, 0, :none), true))
    push!(pairs, (cache.c2_wm112, (:w, -1, 1, 2, :none), true))
    push!(pairs, (cache.c3_wm200 .+ cache.c4_wm200, (:w, -2, 0, 0, :none), true))
    push!(pairs, (cache.c3_wm220,                 (:w, -2, 2, 0, :none), true))
    push!(pairs, (cache.c4_wm202,                 (:w, -2, 0, 2, :none), true))
    push!(pairs, (cache.c3_wm310,                 (:w, -3, 1, 0, :none), true))
    push!(pairs, (cache.c4_wm301,                 (:w, -3, 0, 1, :none), true))
    push!(pairs, (cache.c17_wm211,                (:w, -2, 1, 1, :none), true))
    push!(pairs, (cache.c17_wm400,                (:w, -4, 0, 0, :none), true))

    # U base (fNL linear)
    push!(pairs, (cache.c13_um200 .+ cache.c14_um200, (:u, -2, 0, 0, :none), true))
    push!(pairs, (cache.c13_um220,                    (:u, -2, 2, 0, :none), true))
    push!(pairs, (cache.c14_um202,                    (:u, -2, 0, 2, :none), true))
    push!(pairs, (cache.c13_um310,                    (:u, -3, 1, 0, :none), true))
    push!(pairs, (cache.c14_um301,                    (:u, -3, 0, 1, :none), true))
    push!(pairs, (cache.c13_um400 .+ cache.c14_um400, (:u, -4, 0, 0, :none), true))

    # V base (fNL squared)
    push!(pairs, (cache.c19_vm400, (:v, -4, 0, 0, :none), true))

    # s integrated (:r, :rp)
    push!(pairs, (cache.c6_sm200r,  (:s, -2, 0, 0, :r),  false))
    push!(pairs, (cache.c6_sm202r,  (:s, -2, 0, 2, :r),  false))
    push!(pairs, (cache.c6_sm301r,  (:s, -3, 0, 1, :r),  false))
    push!(pairs, (cache.c6_sm400r,  (:s, -4, 0, 0, :r),  false))
    push!(pairs, (cache.c5_sm200rp, (:s, -2, 0, 0, :rp), false))
    push!(pairs, (cache.c5_sm220rp, (:s, -2, 2, 0, :rp), false))
    push!(pairs, (cache.c5_sm310rp, (:s, -3, 1, 0, :rp), false))
    push!(pairs, (cache.c5_sm400rp, (:s, -4, 0, 0, :rp), false))

    # t integrated
    push!(pairs, (cache.c8_tm200r,  (:t, -2, 0, 0, :r),  false))
    push!(pairs, (cache.c8_tm202r,  (:t, -2, 0, 2, :r),  false))
    push!(pairs, (cache.c8_tm301r,  (:t, -3, 0, 1, :r),  false))
    push!(pairs, (cache.c8_tm400r,  (:t, -4, 0, 0, :r),  false))
    push!(pairs, (cache.c7_tm200rp, (:t, -2, 0, 0, :rp), false))
    push!(pairs, (cache.c7_tm220rp, (:t, -2, 2, 0, :rp), false))
    push!(pairs, (cache.c7_tm310rp, (:t, -3, 1, 0, :rp), false))
    push!(pairs, (cache.c7_tm400rp, (:t, -4, 0, 0, :rp), false))

    # l/tl integrated lensing blocks.  Current Step-2 files store canonical :l; old tilde :tl files are still supported.
    push!(pairs, (cache.c10_lm200r,  (:tl, -2, 0, 0, :r),  false))
    push!(pairs, (cache.c10_lm202r,  (:tl, -2, 0, 2, :r),  false))
    push!(pairs, (cache.c10_lm301r,  (:tl, -3, 0, 1, :r),  false))
    push!(pairs, (cache.c10_lm400r,  (:tl, -4, 0, 0, :r),  false))
    push!(pairs, (cache.c9_lm200rp,  (:tl, -2, 0, 0, :rp), false))
    push!(pairs, (cache.c9_lm220rp,  (:tl, -2, 2, 0, :rp), false))
    push!(pairs, (cache.c9_lm310rp,  (:tl, -3, 1, 0, :rp), false))
    push!(pairs, (cache.c9_lm400rp,  (:tl, -4, 0, 0, :rp), false))

    # scrX plus Y/Z lensing 2-point blocks.  Current Step-2 files store canonical :scrY/:scrZ; old tilde files are still supported.
    push!(pairs, (cache.c11_scrXrrp, (:scrX, -4, 0, 0, :r_rp), false))
    push!(pairs, (cache.c12_scrXrpr, (:scrX, -4, 0, 0, :rp_r), false))
    push!(pairs, (cache.c11_scrYrrp, (:tscrY, -4, 0, 0, :r_rp), false))
    push!(pairs, (cache.c12_scrYrpr, (:tscrY, -4, 0, 0, :rp_r), false))
    push!(pairs, (cache.c12_scrZrrp, (:tscrZ, -4, 0, 0, :r_rp), false))
    push!(pairs, (cache.c11_scrZrpr, (:tscrZ, -4, 0, 0, :rp_r), false))

    # scrS, scrT, and L lensing-lensing block.  Current Step-2 files store canonical :scrL; old tilde :tscrL is still supported.
    push!(pairs, (cache.c17_scrSrrp, (:scrS, -4, 0, 0, :r_rp), false))
    push!(pairs, (cache.c18_scrTrrp, (:scrT, -4, 0, 0, :r_rp), false))
    push!(pairs, (cache.c18_scrLrrp, (:tscrL, -4, 0, 0, :r_rp), false))

    # scrl/tscrl integrated PNG-lensing blocks.  Current Step-2 files store canonical :scrl; old tilde :tscrl is still supported.
    push!(pairs, (cache.c16_scrsm4r,  (:scrs,  -4, 0, 0, :r),  false))
    push!(pairs, (cache.c15_scrsm4rp, (:scrs,  -4, 0, 0, :rp), false))
    push!(pairs, (cache.c16_scrtm4r,  (:scrt,  -4, 0, 0, :r),  false))
    push!(pairs, (cache.c15_scrtm4rp, (:scrt,  -4, 0, 0, :rp), false))
    push!(pairs, (cache.c16_scrlm4r,  (:tscrl, -4, 0, 0, :r),  false))
    push!(pairs, (cache.c15_scrlm4rp, (:tscrl, -4, 0, 0, :rp), false))

    return pairs
end

"""
    _filter_variant!(entries, variant::Symbol) -> entries

In-place filter of the 64-entry sep list for the variants that are a
strict subset of the full 19-term coefficient structure:
- `:full`     — keep everything.
- `:gaussian` — keep everything (fNL terms zero out via `fNL=0`).
- `:kaiser`   — keep entries 1–4 (Term 1, `bg²·w_{0,jj'}` with β RSD).

`:newtonian` is NOT a subset of the 19-term sep list — its coefficient
structure has extra `β·α/r` and `α₁α₂/(r₁r₂)` factors on the 9 paper
bases — so it's built separately by
`_build_newtonian_coeff_pairs_sep` and does not pass through this
filter.  The Newtonian bases reuse 9 of the 61 unique slices: the 4
Term-1 `w^0_{ℓ,jj'}` bases plus `w^{-1}_{ℓ,10}`, `w^{-1}_{ℓ,01}`,
`w^{-1}_{ℓ,12}`, `w^{-1}_{ℓ,21}`, and `w^{-2}_{ℓ,11}`.
"""
function _filter_variant!(entries::Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}},
                           variant::Symbol)
    if variant === :full || variant === :gaussian
        return entries
    elseif variant === :kaiser
        return resize!(entries, 4)
    elseif variant === :fi
        return entries[45:58]       # terms 13--16, coefficient is proportional to f_NL
    elseif variant === :fi_kaiser
        # Kaiser f_NL derivative ONLY:
        # PNG × {galaxy, Kaiser RSD}.  Do NOT include A/B GR terms,
        # since those depend on Q and are not part of Kaiser.
        return entries[[45, 46, 49, 50]]
    elseif variant === :fi_newtonian
        error(":fi_newtonian must be built via _build_fi_newtonian_coeff_pairs_sep, not filtered")
    elseif variant === :ff
        return entries[64:64]       # term 19, coefficient is proportional to f_NL^2
    elseif variant === :newtonian
        error(":newtonian must be built via _build_newtonian_coeff_pairs_sep, not filtered")
    else
        error("unknown variant $variant (expected :full, :gaussian, :kaiser, :newtonian, :fi, :fi_kaiser, :fi_newtonian, :ff)")
    end
end

"""
    _compute_alpha_newtonian(be_of_z, cfns, rr) -> Vector{Float64}

Evaluate on the r-grid

    α(r)  ≡ α_1(r) + α_2(r)
          = d ln[r² a³ n̄_g] / d ln r + d ln(fD) / d ln r
          = 2 - aH(r)·r·b_e(r) - aH(r)·r·(f(r) + d ln f(r)/d ln a)
          = 2 - aH(r)·r·(b_e(r) + f(r) + d ln f / d ln a)

used in the paper's Newtonian C_ℓ expression.  `be_of_z(z)` is the
tracer-specific evolution bias (same `Tracer.be` field); `cfns` is a
`cosmofns.cosmofn` instance (provides `fDr, ffr, far, fHr, fzr`).

`d ln f / d ln a` is obtained as a cubic spline derivative of
`ln f(r)` vs `ln a(r)` on the supplied r-grid.
"""
function _compute_alpha_newtonian(be_of_z::Function, cfns,
                                   rr::Vector{Float64})::Vector{Float64}
    n = length(rr)
    f  = [cfns.ffr(r) for r in rr]
    a  = [cfns.far(r) for r in rr]
    aH = [cfns.far(r) * cfns.fHr(r) for r in rr]
    z  = [cfns.fzr(r) for r in rr]
    be = [be_of_z(zi) for zi in z]

    # dln f / dln a = d(log f) / d(log a) via a spline derivative.
    # ln a is monotonic in r (a decreases as r increases in flat ΛCDM),
    # so sort by ln a ascending before fitting.
    ln_a_vec = log.(a)
    ln_f_vec = log.(f)
    perm = sortperm(ln_a_vec)
    spl  = Spline1D(ln_a_vec[perm], ln_f_vec[perm], k=3)
    dlnf_dlna = [derivative(spl, la) for la in ln_a_vec]

    α = similar(rr)
    @inbounds for i in 1:n
        α[i] = 2.0 - aH[i] * rr[i] * (be[i] + f[i] + dlnf_dlna[i])
    end
    return α
end

"""
    _build_newtonian_coeff_pairs_sep(cache, α1, α2) -> (entries, unique_keys)

Separable form of the paper's Newtonian C_ℓ(r₁,r₂):

    C_ℓ^N(r₁,r₂) = b_{g,1} b_{g,2} D₁ D₂ [
        w^0_{ℓ,00}
      − β₁·w^0_{ℓ,20} − β₂·w^0_{ℓ,02} + β₁β₂·w^0_{ℓ,22}
      − β₁·α₁/r₁·w^{-1}_{ℓ,10} − β₂·α₂/r₂·w^{-1}_{ℓ,01}
      + β₁β₂·α₁/r₁·w^{-1}_{ℓ,12} + β₁β₂·α₂/r₂·w^{-1}_{ℓ,21}
      + β₁β₂·α₁α₂/(r₁r₂)·w^{-2}_{ℓ,11} ]

where β_a = f(r_a)/b_g(r_a) and α_a = α(r_a).  Every coefficient is a
rank-1 outer product `u[i] * v[j]`, so each entry is expressed as
`(u::Vector{Float64}, v::Vector{Float64}, slice_idx::Int)` matching
the kernel's expectation.  Slice indices reuse the unique-keys list
from `_build_array_coeff_pairs_sep`; only 9 of the 61 unique slices
are referenced.
"""
function _build_newtonian_coeff_pairs_sep(cache::ClGRParamCache,
                                           α1::Vector{Float64},
                                           α2::Vector{Float64})
    rr  = cache.rr
    D1, D2 = cache.D1, cache.D2
    bg1, bg2 = cache.bg1, cache.bg2
    β1, β2 = cache.β1, cache.β2

    D1_bg1 = D1 .* bg1
    D1_f1  = D1 .* (β1 .* bg1)              # D1 · f(r1)
    D1_f1_α1_r1 = D1_f1 .* α1 ./ rr         # D1 · f(r1) · α(r1) / r1

    D2_bg2 = D2 .* bg2
    D2_f2  = D2 .* (β2 .* bg2)              # D2 · f(r2)
    D2_f2_α2_r2 = D2_f2 .* α2 ./ rr         # D2 · f(r2) · α(r2) / r2

    # Slice indices in the unique-keys list from _build_array_coeff_pairs_sep:
    #   1: (w,0,0,0)   2: (w,0,2,0)   3: (w,0,0,2)   4: (w,0,2,2)
    #   5: (w,-1,0,1)  6: (w,-1,2,1)  7: (w,-1,1,0)  8: (w,-1,1,2)
    #  14: (w,-2,1,1)
    entries = Tuple{Vector{Float64}, Vector{Float64}, Int}[]
    push!(entries, ( D1_bg1,          D2_bg2,         1 ))  # +w^0_{l,00}
    push!(entries, (-D1_f1,           D2_bg2,         2 ))  # -β1 w^0_{l,20}
    push!(entries, ( D1_bg1,         -D2_f2,          3 ))  # -β2 w^0_{l,02}
    push!(entries, ( D1_f1,           D2_f2,          4 ))  # +β1β2 w^0_{l,22}
    push!(entries, (-D1_f1_α1_r1,     D2_bg2,         7 ))  # -β1·α1/r1 w^{-1}_{l,10}
    push!(entries, ( D1_bg1,         -D2_f2_α2_r2,    5 ))  # -β2·α2/r2 w^{-1}_{l,01}
    push!(entries, ( D1_f1_α1_r1,     D2_f2,          8 ))  # +β1β2·α1/r1 w^{-1}_{l,12}
    push!(entries, ( D1_f1,           D2_f2_α2_r2,    6 ))  # +β1β2·α2/r2 w^{-1}_{l,21}
    push!(entries, ( D1_f1_α1_r1,     D2_f2_α2_r2,   14 ))  # +β1β2·α1α2/(r1r2) w^{-2}_{l,11}

    # Reuse the full keys list (indexing stays consistent with the 64-entry
    # builder); the Newtonian kernel only references 9 of 61 slices but the
    # streaming reader still loads all 61 unique entries per part — the
    # unused ones are wasted I/O.  This matches the filter-based kaiser
    # variant behavior; a newtonian-only kernel that trims the slice list
    # could be added later if the wasted I/O becomes a bottleneck.
    _, unique_keys = _build_array_coeff_pairs_sep(cache; variant=:full)
    return entries, unique_keys
end

"""
    _build_fi_newtonian_coeff_pairs_sep(cache, α1, α2) -> (entries, unique_keys)

Same PNG-bias cross terms as `:fi_kaiser`, but with the Newtonian
velocity/selection pieces included:

    fg + gf + fr + rf + fv_n + vf_n

where

    fv_n ∝ -f(r₂) α(r₂)/r₂ · u^{-3}_{ℓ,01}
    vf_n ∝ -f(r₁) α(r₁)/r₁ · u^{-3}_{ℓ,10}

This matches the component-analysis definition of `fi_n`.
"""
function _build_fi_newtonian_coeff_pairs_sep(cache::ClGRParamCache,
                                              α1::Vector{Float64},
                                              α2::Vector{Float64})
    D1, bg1, β1, bPhi1 = cache.D1, cache.bg1, cache.β1, cache.bPhi1
    D2, bg2, β2, bPhi2 = cache.D2, cache.bg2, cache.β2, cache.bPhi2
    rr = cache.rr
    Omm0, f_NL = cache.Omm0, cache.f_NL
    c_light = 2.99792458e5
    fNL_prefactor = 1.5 * Omm0 * (100.0 / c_light)^2
    fNL_coeff     = fNL_prefactor * f_NL

    D1_bg1    = D1 .* bg1
    D1_f1     = D1_bg1 .* β1
    D1_f1_α_r = D1_f1 .* α1 ./ rr

    D2_bg2    = D2 .* bg2
    D2_f2     = D2_bg2 .* β2
    D2_f2_α_r = D2_f2 .* α2 ./ rr

    entries = Tuple{Vector{Float64}, Vector{Float64}, Int}[]

    # fg, rf: fNL bias on r₂ crossed with density/Kaiser/Newtonian velocity on r₁
    push!(entries, ( fNL_coeff .* D1_bg1,     bPhi2, 16))  # fg: + b₁ · fNL₂ · u^{-2}_{00}
    push!(entries, (-fNL_coeff .* D1_f1,      bPhi2, 17))  # rf: - f₁ · fNL₂ · u^{-2}_{20}
    push!(entries, (-fNL_coeff .* D1_f1_α_r,  bPhi2, 19))  # vf_n: - f₁ α₁/r₁ · fNL₂ · u^{-3}_{10}

    # gf, fr: fNL bias on r₁ crossed with density/Kaiser/Newtonian velocity on r₂
    push!(entries, ( fNL_coeff .* bPhi1, D2_bg2,    16))  # gf: + fNL₁ · b₂ · u^{-2}_{00}
    push!(entries, ( fNL_coeff .* bPhi1, -D2_f2,    18))  # fr: - fNL₁ · f₂ · u^{-2}_{02}
    push!(entries, ( fNL_coeff .* bPhi1, -D2_f2_α_r,20))  # fv_n: - fNL₁ · f₂ α₂/r₂ · u^{-3}_{01}

    _, unique_keys = _build_array_coeff_pairs_sep(cache; variant=:full)
    return entries, unique_keys
end

"""
    _build_array_coeff_pairs_sep(cache; variant=:full) -> (entries, unique_keys)

Separable form of `_build_array_coeff_pairs`: every 19-term coefficient
matrix `c_k[i,j]` is a rank-1 outer product `u_k[i] * v_k[j]`, because
each term is `D1[i]·D2[j]·F1(i)·F2(j)·const`.

Three of the originally-merged pairs (`c3+c4` for W_{-2,0,0}, `c13+c14`
for U_{-2,0,0} and U_{-4,0,0}) are rank-2 and so are split into two
rank-1 entries apiece.  The slice is still read once (shared between the
two entries).

`variant` trims the entry list to one of the physical subsets used by
the hierarchy plot (`:full`, `:gaussian`, `:kaiser`, `:newtonian`; see
`_filter_variant!`).  `:full` and `:gaussian` keep all 64 entries; the
fNL contribution is controlled separately by `fNL` in `ClGRParams`.

Returns:
- `entries`: Vector of `(u::Vector{Float64}, v::Vector{Float64}, slice_idx::Int)`,
  with `slice_idx` indexing into `unique_keys`.
- `unique_keys`: Vector of `(arr_key::NTuple{5,Any}, is_base::Bool)` —
  one entry per unique HDF5 slice to load per part.
"""
function _build_array_coeff_pairs_sep(cache::ClGRParamCache; variant::Symbol=:full)
    D1, aH1, bg1, β1, B1, A1, Q1, bPhi1 =
        cache.D1, cache.aH1, cache.bg1, cache.β1, cache.B1, cache.A1, cache.Q1, cache.bPhi1
    D2, aH2, bg2, β2, B2, A2, Q2, bPhi2 =
        cache.D2, cache.aH2, cache.bg2, cache.β2, cache.B2, cache.A2, cache.Q2, cache.bPhi2
    rr = cache.rr
    Omm0, H0, f_NL = cache.Omm0, cache.H0, cache.f_NL
    c_light = 2.99792458e5
    fNL_prefactor = 1.5 * Omm0 * (100.0 / c_light)^2
    fNL_coeff     = fNL_prefactor * f_NL
    fNL_sq_coeff  = (9.0/4.0) * Omm0^2 * (100.0/c_light)^4 * f_NL^2

    # Pre-compute reusable i-side and j-side 1D vectors.
    D1_bg1        = D1 .* bg1
    D1_bg1_β1     = D1_bg1 .* β1
    D1_aH1_B1     = D1 .* aH1 .* B1
    D1_aH1sq_A1   = D1 .* aH1.^2 .* A1
    D1_B1_over_β1 = D1 .* (B1 ./ β1)
    D1_oneMQ1     = D1 .* (1.0 .- Q1)
    D1_oneMQ1_r1  = D1_oneMQ1 ./ rr
    D1_bg1_B1_β1  = D1_bg1 .* (B1 ./ β1)
    D1_B1_bg1β1   = D1 .* (B1 ./ (bg1 .* β1))

    D2_bg2        = D2 .* bg2
    D2_bg2_β2     = D2_bg2 .* β2
    D2_aH2_B2     = D2 .* aH2 .* B2
    D2_aH2sq_A2   = D2 .* aH2.^2 .* A2
    D2_B2_over_β2 = D2 .* (B2 ./ β2)
    D2_oneMQ2     = D2 .* (1.0 .- Q2)
    D2_oneMQ2_r2  = D2_oneMQ2 ./ rr
    D2_bg2_B2_β2  = D2_bg2 .* (B2 ./ β2)
    D2_B2_bg2β2   = D2 .* (B2 ./ (bg2 .* β2))

    # Slice keys (must match on-disk naming — same as _build_array_coeff_pairs)
    keys_list = Tuple{NTuple{5,Any}, Bool}[
        ((:w, 0, 0, 0, :none), true),        # 1  W_0_0_0
        ((:w, 0, 2, 0, :none), true),        # 2
        ((:w, 0, 0, 2, :none), true),        # 3
        ((:w, 0, 2, 2, :none), true),        # 4
        ((:w, -1, 0, 1, :none), true),       # 5
        ((:w, -1, 2, 1, :none), true),       # 6
        ((:w, -1, 1, 0, :none), true),       # 7
        ((:w, -1, 1, 2, :none), true),       # 8
        ((:w, -2, 0, 0, :none), true),       # 9  W_m2_0_0 (c3+c4 split)
        ((:w, -2, 2, 0, :none), true),       # 10
        ((:w, -2, 0, 2, :none), true),       # 11
        ((:w, -3, 1, 0, :none), true),       # 12
        ((:w, -3, 0, 1, :none), true),       # 13
        ((:w, -2, 1, 1, :none), true),       # 14
        ((:w, -4, 0, 0, :none), true),       # 15
        ((:u, -2, 0, 0, :none), true),       # 16 U_m2_0_0 (c13+c14 split)
        ((:u, -2, 2, 0, :none), true),       # 17
        ((:u, -2, 0, 2, :none), true),       # 18
        ((:u, -3, 1, 0, :none), true),       # 19
        ((:u, -3, 0, 1, :none), true),       # 20
        ((:u, -4, 0, 0, :none), true),       # 21 U_m4_0_0 (c13+c14 split)
        ((:v, -4, 0, 0, :none), true),       # 22
        ((:s, -2, 0, 0, :r), false),         # 23
        ((:s, -2, 0, 2, :r), false),         # 24
        ((:s, -3, 0, 1, :r), false),         # 25
        ((:s, -4, 0, 0, :r), false),         # 26
        ((:s, -2, 0, 0, :rp), false),        # 27
        ((:s, -2, 2, 0, :rp), false),        # 28
        ((:s, -3, 1, 0, :rp), false),        # 29
        ((:s, -4, 0, 0, :rp), false),        # 30
        ((:t, -2, 0, 0, :r), false),         # 31
        ((:t, -2, 0, 2, :r), false),         # 32
        ((:t, -3, 0, 1, :r), false),         # 33
        ((:t, -4, 0, 0, :r), false),         # 34
        ((:t, -2, 0, 0, :rp), false),        # 35
        ((:t, -2, 2, 0, :rp), false),        # 36
        ((:t, -3, 1, 0, :rp), false),        # 37
        ((:t, -4, 0, 0, :rp), false),        # 38
        ((:tl, -2, 0, 0, :r), false),        # 39
        ((:tl, -2, 0, 2, :r), false),        # 40
        ((:tl, -3, 0, 1, :r), false),        # 41
        ((:tl, -4, 0, 0, :r), false),        # 42
        ((:tl, -2, 0, 0, :rp), false),       # 43
        ((:tl, -2, 2, 0, :rp), false),       # 44
        ((:tl, -3, 1, 0, :rp), false),       # 45
        ((:tl, -4, 0, 0, :rp), false),       # 46
        ((:scrX,  -4, 0, 0, :r_rp), false),  # 47
        ((:scrX,  -4, 0, 0, :rp_r), false),  # 48
        ((:tscrY, -4, 0, 0, :r_rp), false),  # 49
        ((:tscrY, -4, 0, 0, :rp_r), false),  # 50
        ((:tscrZ, -4, 0, 0, :r_rp), false),  # 51
        ((:tscrZ, -4, 0, 0, :rp_r), false),  # 52
        ((:scrS,  -4, 0, 0, :r_rp), false),  # 53
        ((:scrT,  -4, 0, 0, :r_rp), false),  # 54
        ((:tscrL, -4, 0, 0, :r_rp), false),  # 55
        ((:scrs,  -4, 0, 0, :r), false),     # 56
        ((:scrs,  -4, 0, 0, :rp), false),    # 57
        ((:scrt,  -4, 0, 0, :r), false),     # 58
        ((:scrt,  -4, 0, 0, :rp), false),    # 59
        ((:tscrl, -4, 0, 0, :r), false),     # 60
        ((:tscrl, -4, 0, 0, :rp), false),    # 61
    ]

    entries = Tuple{Vector{Float64}, Vector{Float64}, Int}[]

    # Term 1 → W_0_0_0 (1), W_0_2_0 (2), W_0_0_2 (3), W_0_2_2 (4)
    push!(entries, (D1_bg1,     D2_bg2,      1))
    push!(entries, (-D1_bg1_β1, D2_bg2,      2))
    push!(entries, (D1_bg1,    -D2_bg2_β2,   3))
    push!(entries, (D1_bg1_β1,  D2_bg2_β2,   4))

    # Term 2 → W_m1_0_1 (5), W_m1_2_1 (6), W_m1_1_0 (7), W_m1_1_2 (8)
    push!(entries, (D1_bg1,     D2_aH2_B2,   5))
    push!(entries, (-D1_bg1_β1, D2_aH2_B2,   6))
    push!(entries, (D1_aH1_B1,  D2_bg2,      7))
    push!(entries, (D1_aH1_B1, -D2_bg2_β2,   8))

    # Term 3 (bg1 × aH2^2 A2) — shares slice 9 W_m2_0_0 with term 4
    push!(entries, (D1_bg1,     D2_aH2sq_A2, 9))
    push!(entries, (-D1_bg1_β1, D2_aH2sq_A2, 10))   # W_m2_2_0
    push!(entries, (D1_aH1_B1,  D2_aH2sq_A2, 12))   # W_m3_1_0

    # Term 4 (bg2 × aH1^2 A1) — slice 9 again, 11 W_m2_0_2, 13 W_m3_0_1
    push!(entries, (D1_aH1sq_A1, D2_bg2,     9))
    push!(entries, (D1_aH1sq_A1, -D2_bg2_β2, 11))
    push!(entries, (D1_aH1sq_A1, D2_aH2_B2,  13))

    # Term 5 → S_m2_0_0_rp (27), S_m2_2_0_rp (28), S_m3_1_0_rp (29), S_m4_0_0_rp (30)
    push!(entries, (D1_bg1,      D2_B2_over_β2, 27))
    push!(entries, (-D1_bg1_β1,  D2_B2_over_β2, 28))
    push!(entries, (D1_aH1_B1,   D2_B2_over_β2, 29))
    push!(entries, (D1_aH1sq_A1, D2_B2_over_β2, 30))

    # Term 6 → S_m2_0_0_r (23), S_m2_0_2_r (24), S_m3_0_1_r (25), S_m4_0_0_r (26)
    push!(entries, (D1_B1_over_β1, D2_bg2,      23))
    push!(entries, (D1_B1_over_β1, -D2_bg2_β2,  24))
    push!(entries, (D1_B1_over_β1, D2_aH2_B2,   25))
    push!(entries, (D1_B1_over_β1, D2_aH2sq_A2, 26))

    # Term 7 → T_*_rp (35,36,37,38)
    push!(entries, (-2.0 .* D1_bg1,      D2_oneMQ2_r2, 35))
    push!(entries, ( 2.0 .* D1_bg1_β1,   D2_oneMQ2_r2, 36))
    push!(entries, (-2.0 .* D1_aH1_B1,   D2_oneMQ2_r2, 37))
    push!(entries, (-2.0 .* D1_aH1sq_A1, D2_oneMQ2_r2, 38))

    # Term 8 → T_*_r (31,32,33,34)
    push!(entries, (-2.0 .* D1_oneMQ1_r1, D2_bg2,      31))
    push!(entries, ( 2.0 .* D1_oneMQ1_r1, D2_bg2_β2,   32))
    push!(entries, (-2.0 .* D1_oneMQ1_r1, D2_aH2_B2,   33))
    push!(entries, (-2.0 .* D1_oneMQ1_r1, D2_aH2sq_A2, 34))

    # Term 9 → L_*_rp (43..46)
    push!(entries, (-2.0 .* D1_bg1,      D2_oneMQ2, 43))
    push!(entries, ( 2.0 .* D1_bg1_β1,   D2_oneMQ2, 44))
    push!(entries, (-2.0 .* D1_aH1_B1,   D2_oneMQ2, 45))
    push!(entries, (-2.0 .* D1_aH1sq_A1, D2_oneMQ2, 46))

    # Term 10 → L_*_r (39..42)
    push!(entries, (-2.0 .* D1_oneMQ1, D2_bg2,      39))
    push!(entries, ( 2.0 .* D1_oneMQ1, D2_bg2_β2,   40))
    push!(entries, (-2.0 .* D1_oneMQ1, D2_aH2_B2,   41))
    push!(entries, (-2.0 .* D1_oneMQ1, D2_aH2sq_A2, 42))

    # Term 11 → ScrX_r_rp (47), tscrY_r_rp (49), tscrZ_rp_r (52)
    # NOTE: per Eq. (4.24), the X and Y prefactor on the i-side is B1/f1, NOT
    # bg1·B1/β1 (which equals bg1²·B1/f1).  Use D1_B1_bg1β1 = D1·B1/(bg1·β1) =
    # D1·B1/f1, matching the same B/f convention used in Term 17 (slice 53).
    push!(entries, (-2.0 .* D1_B1_bg1β1, D2_oneMQ2_r2, 47))
    push!(entries, (-2.0 .* D1_B1_bg1β1, D2_oneMQ2,    49))
    push!(entries, ( 4.0 .* D1_oneMQ1,    D2_oneMQ2_r2, 52))

    # Term 12 → ScrX_rp_r (48), tscrY_rp_r (50), tscrZ_r_rp (51)
    # See Term-11 note: j-side B/f prefactor is D2_B2_bg2β2 = D2·B2/f2.
    push!(entries, (-2.0 .* D1_oneMQ1_r1, D2_B2_bg2β2, 48))
    push!(entries, (-2.0 .* D1_oneMQ1,    D2_B2_bg2β2, 50))
    push!(entries, ( 4.0 .* D1_oneMQ1_r1, D2_oneMQ2,    51))

    # Term 13 (fNL, bg1·bPhi2/D2) → U_m2_0_0 (16), U_m2_2_0 (17), U_m3_1_0 (19), U_m4_0_0 (21)
    #   c13 = pf·fNL_coeff·bg1·bPhi2/D2 = D1·bg1·fNL_coeff · bPhi2
    push!(entries, ( fNL_coeff .* D1_bg1,      bPhi2, 16))
    push!(entries, (-fNL_coeff .* D1_bg1_β1,   bPhi2, 17))
    push!(entries, ( fNL_coeff .* D1_aH1_B1,   bPhi2, 19))
    push!(entries, ( fNL_coeff .* D1_aH1sq_A1, bPhi2, 21))

    # Term 14 (fNL, bg2·bPhi1/D1) → U_m2_0_0 (16), U_m2_0_2 (18), U_m3_0_1 (20), U_m4_0_0 (21)
    #   c14 = pf·fNL_coeff·bg2·bPhi1/D1 = bPhi1·fNL_coeff · D2·bg2
    push!(entries, ( fNL_coeff .* bPhi1,  D2_bg2,      16))
    push!(entries, ( fNL_coeff .* bPhi1, -D2_bg2_β2,   18))
    push!(entries, ( fNL_coeff .* bPhi1,  D2_aH2_B2,   20))
    push!(entries, ( fNL_coeff .* bPhi1,  D2_aH2sq_A2, 21))

    # Term 15 (fNL, bPhi1/D1 side) → scrs_rp (57), scrt_rp (59), tscrl_rp (61)
    #   c = pf·fNL_coeff·bPhi1/D1·[j_stuff] = fNL_coeff·bPhi1 · D2·[j_stuff]
    push!(entries, ( fNL_coeff .* bPhi1,        D2_B2_bg2β2,  57))
    push!(entries, (-2.0 .* fNL_coeff .* bPhi1, D2_oneMQ2_r2, 59))
    push!(entries, (-2.0 .* fNL_coeff .* bPhi1, D2_oneMQ2,    61))

    # Term 16 (fNL, bPhi2/D2 side) → scrs_r (56), scrt_r (58), tscrl_r (60)
    push!(entries, ( fNL_coeff .* D1_B1_bg1β1,   bPhi2, 56))
    push!(entries, (-2.0 .* fNL_coeff .* D1_oneMQ1_r1, bPhi2, 58))
    push!(entries, (-2.0 .* fNL_coeff .* D1_oneMQ1,    bPhi2, 60))

    # Term 17 → W_m2_1_1 (14), W_m4_0_0 (15), ScrS_r_rp (53)
    push!(entries, (D1_aH1_B1,      D2_aH2_B2,      14))
    push!(entries, (D1_aH1sq_A1,    D2_aH2sq_A2,    15))
    push!(entries, (D1_B1_bg1β1,    D2_B2_bg2β2,    53))

    # Term 18 → ScrT_r_rp (54), tscrL_r_rp (55)
    push!(entries, ( 4.0 .* D1_oneMQ1_r1, D2_oneMQ2_r2, 54))
    push!(entries, ( 4.0 .* D1_oneMQ1,    D2_oneMQ2,    55))

    # Term 19 → V_m4_0_0 (22)
    #   c19 = pf·fNL_sq_coeff·bPhi1/D1·bPhi2/D2 = fNL_sq_coeff·bPhi1 · bPhi2
    push!(entries, ( fNL_sq_coeff .* bPhi1, bPhi2, 22))

    # IMPORTANT: for :fi, :fi_kaiser, and :ff the filter returns a NEW
    # vector slice.  The old code called this function but ignored the
    # returned vector, so those variants silently kept the full 64-entry
    # list.  With fNL=0 this made Cl_fi/Cl_ff/Cl_fi_kaiser identical to
    # Cl_f0.
    entries = _filter_variant!(entries, variant)
    unique_keys = keys_list
    return entries, unique_keys
end

"""
    _load_one_array_split(meta_path, part_files, ell_ranges, n_ell, nr,
                           arr_key, is_base) -> Array{Float32,3}

Load a single (type,p,j,jp,sub) array across all part files, concatenating
along the ell axis.  Returns [n_ell, nr, nr] Float32 (kept from disk
layout to halve memory relative to Float64).
"""
function _load_one_array_split(meta_dir::String, part_files::Vector{String},
                                ell_ranges::Matrix{Int},
                                n_ell::Int, nr::Int,
                                arr_key::NTuple{5,Any}, is_base::Bool)
    type, p, j_, jp_, sub = arr_key
    key_str = "$(type)_$(_encode_p(Int(p)))_$(Int(j_))_$(Int(jp_))"
    if sub != :none
        key_str = key_str * "_$(sub)"
    end
    group_path = (is_base ? "base/" : "integrated/") * key_str

    arr = Array{Float32,3}(undef, n_ell, nr, nr)
    for (part_idx, part_file) in enumerate(part_files)
        part_path = joinpath(meta_dir, part_file)
        ell_lo = Int(ell_ranges[part_idx, 1])
        ell_hi = Int(ell_ranges[part_idx, 2])
        h5open(part_path, "r") do f
            haskey(f, group_path) || error("Missing HDF5 key '$group_path' in $part_path")
            raw = read(f[group_path])            # [n_ell_part, nr, nr] Float32 on disk
            @inbounds arr[ell_lo:ell_hi, :, :] = raw
        end
    end
    return arr
end

# Minimal _encode_p local to CalcClGR (PowerFull has its own; duplicate to
# avoid module cycle).
_encode_p(p::Int) = p < 0 ? "m$(abs(p))" : string(p)

"""
    compute_Cl_GR_streaming!(result, cache, meta_path, ells)

Memory-lean C_ℓ^GR assembly.  Loads integral arrays one at a time from
the split HDF5 (meta + part files), accumulating the corresponding
coefficient × array contribution into `result`.  Peak memory is one
integral array (≈2.3 GB Float32 at Nr=4096) plus the `result` buffer
(≈5 GB Float64), instead of ~260 GB when all arrays are held in memory.

The result and the dense in-memory `compute_Cl_GR_batch!` agree to
Float32 round-off (relative ≈ 1e-6).
"""
function compute_Cl_GR_streaming!(result::Array{Float64,3},
                                   cache::ClGRParamCache,
                                   meta_path::String,
                                   ells::Vector{Int};
                                   verbose::Bool=false)
    fill!(result, 0.0)
    nr = cache.nr

    ells == cache.ell_values ||
        error("compute_Cl_GR_streaming! requires ells == cache.ell_values (a full, in-order sweep).  For ell-subset requests, use the dense compute_Cl_GR_batch.")

    # Parse meta file
    meta_dir = dirname(meta_path)
    isempty(meta_dir) && (meta_dir = ".")
    part_files_raw, ell_ranges, rr_meta, ell_values_meta = h5open(meta_path, "r") do f
        pf = read(f, "part_files")
        er = Int.(read(f, "metadata/ell_ranges"))
        rr0 = Float64.(read(f, "grid/rr"))
        ev0 = Int.(read(f, "grid/ell_values"))
        (pf, er, rr0, ev0)
    end
    part_files = String.(part_files_raw)
    n_parts = length(part_files)

    pairs = _build_array_coeff_pairs(cache)
    n_pairs = length(pairs)

    # Split pair list into parallel coefficient- and key- arrays for
    # type-stable access inside the hot kernel.
    coeffs  = Matrix{Float64}[p[1] for p in pairs]
    arr_keys = NTuple{5,Any}[p[2]  for p in pairs]
    is_bases = Bool[p[3]            for p in pairs]

    # Prefetched slices across parts via a Channel on the :interactive pool.
    # Consumer accumulates each part's 61-slice bundle using a dense-style
    # kernel that writes each result element exactly once.
    for p_idx in 1:n_parts
        part_path = joinpath(meta_dir, part_files[p_idx])
        ell_lo = Int(ell_ranges[p_idx, 1])
        ell_hi = Int(ell_ranges[p_idx, 2])
        nell_part = ell_hi - ell_lo + 1
        verbose && @info "[part $p_idx/$n_parts] ell $ell_lo:$ell_hi  ($n_pairs arrays)"

        # Load all 61 slices for this part.  Peak extra memory: 61 * nr^2 *
        # nell_part * 4 B ≈ 7 GB at Nr=4096 / nell_part=25.  Loaded in
        # parallel via :interactive thread while the previous part's
        # compute may still be finishing is a future refinement; here we
        # block on load before compute for correctness and simplicity.
        slices = Vector{Array{Float32,3}}(undef, n_pairs)
        h5open(part_path, "r") do f
            for k in 1:n_pairs
                slices[k] = _read_slice(f, arr_keys[k], is_bases[k], ell_values_meta[ell_lo:ell_hi], rr_meta)
            end
        end

        # Dense-style accumulate: one write per (ell, i, j) output.
        # Pack into fixed-length NTuples so the inner k-loop is unrolled
        # at compile time, eliminating the vector indirection that was
        # allocating ~4 G times per full run on the previous `Vector`
        # version.
        coeffs_tup = NTuple{n_pairs, Matrix{Float64}}(coeffs)
        slices_tup = NTuple{n_pairs, Array{Float32,3}}(slices)
        _accumulate_part_dense!(result, coeffs_tup, slices_tup,
                                 ell_lo, nell_part, nr)

        # Release slices before loading the next part.
        for k in 1:n_pairs
            slices[k] = Array{Float32,3}(undef, 0, 0, 0)
        end
    end
    return result
end

"""
    _accumulate_part_dense!(result, coeffs::NTuple{N,Matrix{Float64}},
                             slices::NTuple{N,Array{Float32,3}},
                             ell_lo, nell_part, nr) where {N}

Dense-style inner kernel.  For each output element (ell, i, j), sum
all N = 61 `coeffs[k][i,j] * Float64(slices[k][e, i, j])` contributions
into a scalar accumulator, then write result once.  NTuple argument
types make Julia's compiler unroll the inner k-loop and emit direct
loads instead of Vector-indirection: this is the change that collapses
the ~4 G spurious allocations/run observed on the `Vector`-backed
version.
"""
function _accumulate_part_dense!(result::Array{Float64,3},
                                  coeffs::NTuple{61, Matrix{Float64}},
                                  slices::NTuple{61, Array{Float32,3}},
                                  ell_lo::Int, nell_part::Int,
                                  nr::Int)
    ell_off = ell_lo - 1
    @threads for j in 1:nr
        val_e = Vector{Float64}(undef, nell_part)
        @inbounds for i in 1:nr
            @simd for e in 1:nell_part
                val_e[e] = 0.0
            end
            # @nexprs literally copies the body 61 times with k = 1..61,
            # which turns `coeffs[k]` into a compile-time tuple index so
            # the per-k read has no Vector-indirection.  This is what
            # named local variables do in the dense compute_Cl_GR_batch!
            # kernel; we just pull the arrays from an NTuple instead of
            # the ClGRParamCache fields.
            @nexprs 61 k -> begin
                let c = coeffs[k][i, j], slice_k = slices[k]
                    @simd for e in 1:nell_part
                        val_e[e] = muladd(c, Float64(slice_k[e, i, j]), val_e[e])
                    end
                end
            end
            @simd for e in 1:nell_part
                result[ell_off + e, i, j] += val_e[e]
            end
        end
    end
    return nothing
end


"""
    _hdf5_key_string(type, p, j, jp, sub, is_base) -> String

Return the on-disk HDF5 dataset name for one PowerFull integral key.
"""
function _hdf5_key_string(type::Symbol, p::Int, j_::Int, jp_::Int, sub::Symbol, is_base::Bool)::String
    key_str = "$(type)_$(_encode_p(p))_$(j_)_$(jp_)"
    if !is_base && sub != :none
        key_str *= "_$(sub)"
    end
    return key_str
end

"""
    _candidate_disk_types(type) -> Vector{Symbol}

Return candidate on-disk type names for a requested integral type.  This keeps
backward compatibility with older Step-2 files that wrote bare/temporary tilde
lensing blocks (`tl`, `tscrY`, `tscrZ`, `tscrL`, `tscrl`) while preferring the
new canonical paper-observable blocks (`l`, `scrY`, `scrZ`, `scrL`, `scrl`) when
those are present.
"""
function _candidate_disk_types(type::Symbol)::Vector{Symbol}
    cand_types = Symbol[type]

    # If the request is an old temporary/tilde name, prefer the new canonical
    # on-disk dataset first.  This is the important case for current Step-2
    # outputs: the coefficient list still requests :tl/:tscrY/... for legacy
    # compatibility, but build_and_export.jl now writes :l/:scrY/... after the
    # paper-lensing conversion.
    if haskey(_INTEGRAL_TYPE_ALIASES, type)
        pushfirst!(cand_types, _INTEGRAL_TYPE_ALIASES[type])
    end

    # If the request is a canonical name, allow fallback to older tilde files.
    for (old_type, canonical_type) in _INTEGRAL_TYPE_ALIASES
        if canonical_type == type
            push!(cand_types, old_type)
        end
    end

    return unique(cand_types)
end

"""
    _read_raw_slice_with_type(f, arr_key, is_base) -> (Array{Float32,3}, Symbol, String)

Read one on-disk slice without applying any lensing-geometry conversion, and
also return the actual on-disk type that matched.  The matched type matters for
lensing: canonical `l/scrY/scrZ/scrL/scrl` datasets are already converted by
new Step-2 output, while old `tl/tscrY/tscrZ/tscrL/tscrl` datasets are bare and
must be converted exactly once in `_read_slice`.
"""
function _read_raw_slice_with_type(f::HDF5.File, arr_key::NTuple{5,Any}, is_base::Bool)
    type, p, j_, jp_, sub = arr_key

    grp = is_base ? "base" : "integrated"
    if !haskey(f, grp)
        error("HDF5 group '$grp' not found in part file. Top-level keys: $(collect(keys(f)))")
    end
    g = f[grp]

    tried = String[]
    for t in _candidate_disk_types(type)
        key_str = _hdf5_key_string(t, Int(p), Int(j_), Int(jp_), sub, is_base)
        push!(tried, "$grp/$key_str")
        if haskey(g, key_str)
            return Float32.(read(g[key_str])), t, "$grp/$key_str"
        end
    end

    available = collect(keys(g))
    preview = available[1:min(length(available), 30)]
    error("HDF5 key not found in part file. Tried: $(tried). Available $grp keys (first $(length(preview))): $(preview)")
end

"""
    _read_raw_slice(f::HDF5.File, arr_key, is_base) -> Array{Float32,3}

Read one on-disk slice without applying any lensing-geometry conversion.
"""
function _read_raw_slice(f::HDF5.File, arr_key::NTuple{5,Any}, is_base::Bool)::Array{Float32,3}
    arr, _, _ = _read_raw_slice_with_type(f, arr_key, is_base)
    return arr
end


@inline function _apply_lensing_ellfac!(arr::Array{Float32,3}, ell_values_part::Vector{Int}, power::Int)
    nell = size(arr, 1)
    @inbounds for e in 1:nell
        fac = Float32((0.5 * ell_values_part[e] * (ell_values_part[e] + 1))^power)
        @simd for idx in 1:(size(arr,2)*size(arr,3))
            arr[e + (idx-1)*nell] *= fac
        end
    end
    return arr
end

@inline function _subtract_col_scaled!(out::Array{Float32,3}, subtrahend::Array{Float32,3}, rr::Vector{Float64})
    nell, nr, _ = size(out)
    @inbounds for j in 1:nr
        invr = Float32(1.0 / rr[j])
        for i in 1:nr
            @simd for e in 1:nell
                out[e,i,j] -= invr * subtrahend[e,i,j]
            end
        end
    end
    return out
end

@inline function _subtract_row_scaled!(out::Array{Float32,3}, subtrahend::Array{Float32,3}, rr::Vector{Float64})
    nell, nr, _ = size(out)
    @inbounds for j in 1:nr
        for i in 1:nr
            invr = Float32(1.0 / rr[i])
            @simd for e in 1:nell
                out[e,i,j] -= invr * subtrahend[e,i,j]
            end
        end
    end
    return out
end

@inline function _add_T_over_r1r2!(out::Array{Float32,3}, T::Array{Float32,3}, rr::Vector{Float64})
    nell, nr, _ = size(out)
    @inbounds for j in 1:nr
        invr2 = Float32(1.0 / rr[j])
        for i in 1:nr
            inv = Float32((1.0 / rr[i])) * invr2
            @simd for e in 1:nell
                out[e,i,j] += inv * T[e,i,j]
            end
        end
    end
    return out
end


"""
    _read_slice(f::HDF5.File, arr_key, is_base, ell_values_part, rr) -> Array{Float32,3}

Read one slice from a split Step-2 part file.

Current `build_and_export.jl` writes the lensing kernels under canonical
paper-observable names (`l`, `scrl`, `scrY`, `scrZ`, `scrL`).  Those datasets
already include the geometry subtraction and the ell(ell+1)/2 factors, so Step 3
must read them as-is.

For backward compatibility, if an older part file only contains the temporary
bare/tilde names (`tl`, `tscrl`, `tscrY`, `tscrZ`, `tscrL`), this function still
performs the old on-load conversion exactly once.
"""
function _read_slice(f::HDF5.File, arr_key::NTuple{5,Any}, is_base::Bool,
                     ell_values_part::Vector{Int}, rr::Vector{Float64})::Array{Float32,3}
    type, p, j_, jp_, sub = arr_key
    raw, disk_type, _disk_path = _read_raw_slice_with_type(f, arr_key, is_base)

    is_base && return raw

    # New Step-2 files already contain paper-observable lensing kernels under
    # canonical names.  Do NOT apply the lensing geometry/ell-factor again.
    if disk_type in (:l, :scrl, :scrY, :scrZ, :scrL)
        return raw
    end

    # Older Step-2 files contain bare/temporary tilde lensing kernels.  Convert
    # those on load, exactly once.

    # 1D w-weighted lensing: l = E_l * (tl - t/r_source)
    if disk_type == :tl
        t = _read_raw_slice(f, (:t, p, j_, jp_, sub), false)
        if sub == :r
            _subtract_row_scaled!(raw, t, rr)
        elseif sub == :rp
            _subtract_col_scaled!(raw, t, rr)
        else
            error("Unexpected 1D lensing subscript for $arr_key")
        end
        return _apply_lensing_ellfac!(raw, ell_values_part, 1)
    end

    # 1D u-weighted lensing: scrl = E_l * (tscrl - scrt/r_source)
    if disk_type == :tscrl
        t = _read_raw_slice(f, (:scrt, p, j_, jp_, sub), false)
        if sub == :r
            _subtract_row_scaled!(raw, t, rr)
        elseif sub == :rp
            _subtract_col_scaled!(raw, t, rr)
        else
            error("Unexpected 1D PNG-lensing subscript for $arr_key")
        end
        return _apply_lensing_ellfac!(raw, ell_values_part, 1)
    end

    # 2D ISW×lensing: Y = E_l * (tildeY - X/r_lensed_source)
    if disk_type == :tscrY
        X = _read_raw_slice(f, (:scrX, p, j_, jp_, sub), false)
        if sub == :r_rp          # ISW on r1, lensing source on r2
            _subtract_col_scaled!(raw, X, rr)
        elseif sub == :rp_r      # ISW on r2, lensing source on r1
            _subtract_row_scaled!(raw, X, rr)
        else
            error("Unexpected Y subscript for $arr_key")
        end
        return _apply_lensing_ellfac!(raw, ell_values_part, 1)
    end

    # 2D TD×lensing: Z = E_l * (tildeZ - T/r_lensed_source)
    if disk_type == :tscrZ
        T = _read_raw_slice(f, (:scrT, p, j_, jp_, :r_rp), false)
        if sub == :r_rp          # TD on r1, lensing source on r2
            _subtract_col_scaled!(raw, T, rr)
        elseif sub == :rp_r      # lensing source on r1, TD on r2
            _subtract_row_scaled!(raw, T, rr)
        else
            error("Unexpected Z subscript for $arr_key")
        end
        return _apply_lensing_ellfac!(raw, ell_values_part, 1)
    end

    # 2D lensing×lensing:
    # L = E_l^2 * (tildeL - tildeZ_{lens r1,TD r2}/r1 - tildeZ_{TD r1,lens r2}/r2 + T/(r1*r2))
    if disk_type == :tscrL
        Z_r_rp = _read_raw_slice(f, (:tscrZ, p, j_, jp_, :r_rp), false)
        Z_rp_r = _read_raw_slice(f, (:tscrZ, p, j_, jp_, :rp_r), false)
        T      = _read_raw_slice(f, (:scrT,  p, j_, jp_, :r_rp), false)
        _subtract_row_scaled!(raw, Z_rp_r, rr)
        _subtract_col_scaled!(raw, Z_r_rp, rr)
        _add_T_over_r1r2!(raw, T, rr)
        return _apply_lensing_ellfac!(raw, ell_values_part, 2)
    end

    return raw
end
"""
    _accumulate_slice!(result, coeff, slice, ell_lo, nell_part, nr) -> nothing

Type-stable inner kernel for per-part accumulation.  Adds
`coeff[i,j] · Float64(slice[e_local, i, j])` into
`result[ell_lo + e_local - 1, i, j]`, threaded over `e_local`.  Keeping
this in a separate concrete-typed method prevents closure-capture
boxing inside `@threads`.
"""
@inline function _accumulate_slice!(result::Array{Float64,3},
                                     coeff::Matrix{Float64},
                                     slice::Array{Float32,3},
                                     ell_lo::Int, nell_part::Int, nr::Int)
    # Both `slice` (nell_part, nr, nr) and `result` (n_ells, nr, nr) are
    # column-major, so their stride-1 axis is the leading ell dimension.
    # Threading over `j` with `e_local` innermost keeps every inner read
    # and write contiguous; the alternative (e_local outer, i innermost)
    # strides both arrays by `nell_part` and `n_ells` respectively, which
    # on Nr=4096 measured ~5× slower due to cache thrashing.
    ell_off = ell_lo - 1
    @threads for j in 1:nr
        @inbounds for i in 1:nr
            c = coeff[i, j]
            @simd for e_local in 1:nell_part
                result[ell_off + e_local, i, j] += c * Float64(slice[e_local, i, j])
            end
        end
    end
    return nothing
end

"""
    compute_Cl_GR_batch_streaming(meta_path, params_1, params_2, ells;
                                   rr_override=nothing, ell_values_override=nothing)

Streaming version of `compute_Cl_GR_batch(I, params_1, params_2, ells)`.
Reads only enough metadata (rr grid + ell list) from `meta_path` to
build a `ClGRParamCache` for the cross-tracer coefficients, then
assembles C_ℓ^GR via `compute_Cl_GR_streaming!`.  Never loads the full
IntegralCollection.
"""
function compute_Cl_GR_batch_streaming(meta_path::String,
                                       params_1::ClGRParams,
                                       params_2::ClGRParams,
                                       ells::Vector{Int};
                                       verbose::Bool=false)
    # Read just grid info to build a minimal IntegralCollection shell.
    rr, ell_values = h5open(meta_path, "r") do f
        (Float64.(read(f, "grid/rr")), Int.(read(f, "grid/ell_values")))
    end

    # ClGRParamCache's constructor extracts every integral array reference
    # from I by tuple key.  For streaming we do not need the real data,
    # but the constructor still has to find keys by lookup — populate a
    # shell IntegralCollection with all 61 keys mapped to a single shared
    # 0×0×0 dummy array.  Streaming compute reads from HDF5, never from
    # these references.
    dummy = zeros(Float64, 0, 0, 0)
    needed_keys = (
        # W base
        (:w, 0, 0, 0, :none),  (:w, 0, 2, 0, :none),  (:w, 0, 0, 2, :none),  (:w, 0, 2, 2, :none),
        (:w, -1, 0, 1, :none), (:w, -1, 2, 1, :none), (:w, -1, 1, 0, :none), (:w, -1, 1, 2, :none),
        (:w, -2, 0, 0, :none), (:w, -2, 2, 0, :none), (:w, -2, 0, 2, :none), (:w, -2, 1, 1, :none),
        (:w, -3, 1, 0, :none), (:w, -3, 0, 1, :none), (:w, -4, 0, 0, :none),
        # U base
        (:u, -2, 0, 0, :none), (:u, -2, 2, 0, :none), (:u, -2, 0, 2, :none),
        (:u, -3, 1, 0, :none), (:u, -3, 0, 1, :none), (:u, -4, 0, 0, :none),
        # V base
        (:v, -4, 0, 0, :none),
        # s integrated
        (:s, -2, 0, 0, :r),  (:s, -2, 0, 2, :r),  (:s, -3, 0, 1, :r),  (:s, -4, 0, 0, :r),
        (:s, -2, 0, 0, :rp), (:s, -2, 2, 0, :rp), (:s, -3, 1, 0, :rp), (:s, -4, 0, 0, :rp),
        # t integrated
        (:t, -2, 0, 0, :r),  (:t, -2, 0, 2, :r),  (:t, -3, 0, 1, :r),  (:t, -4, 0, 0, :r),
        (:t, -2, 0, 0, :rp), (:t, -2, 2, 0, :rp), (:t, -3, 1, 0, :rp), (:t, -4, 0, 0, :rp),
        # tl integrated (lensing-kernel bare)
        (:tl, -2, 0, 0, :r),  (:tl, -2, 0, 2, :r),  (:tl, -3, 0, 1, :r),  (:tl, -4, 0, 0, :r),
        (:tl, -2, 0, 0, :rp), (:tl, -2, 2, 0, :rp), (:tl, -3, 1, 0, :rp), (:tl, -4, 0, 0, :rp),
        # 2D bare blocks
        (:scrX, -4, 0, 0, :r_rp), (:scrX, -4, 0, 0, :rp_r),
        (:tscrY, -4, 0, 0, :r_rp), (:tscrY, -4, 0, 0, :rp_r),
        (:tscrZ, -4, 0, 0, :r_rp), (:tscrZ, -4, 0, 0, :rp_r),
        (:scrS, -4, 0, 0, :r_rp), (:scrT, -4, 0, 0, :r_rp), (:tscrL, -4, 0, 0, :r_rp),
        # fNL 1D
        (:scrs,  -4, 0, 0, :r),  (:scrs,  -4, 0, 0, :rp),
        (:scrt,  -4, 0, 0, :r),  (:scrt,  -4, 0, 0, :rp),
        (:tscrl, -4, 0, 0, :r),  (:tscrl, -4, 0, 0, :rp),
    )
    data_shell = Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}()
    for k in needed_keys
        data_shell[k] = dummy
    end
    I_shell = IntegralCollection(data_shell, rr, ell_values)
    cache = ClGRParamCache(I_shell, params_1, params_2)

    result = zeros(Float64, length(ells), cache.nr, cache.nr)
    compute_Cl_GR_streaming!(result, cache, meta_path, ells; verbose=verbose)
    return result
end

export compute_Cl_GR_streaming!, compute_Cl_GR_batch_streaming

# =============================================================================
# Multi-pair streaming: compute C_ℓ^{ij} for many (tracer_a, tracer_b) pairs
# in a single part-I/O pass.  The only per-pair state is the rank-1 (a, b)
# decomposition of each of the 19-term coefficients pre-fused with the
# tracer-specific radial weights; slices are read once and reused across
# all pairs.
# =============================================================================

"""
    _fused_entries_for_pair(cache, u1, u2) -> Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}

Given a pair's coefficient cache and the tracer-specific quadrature
weights `u1[i] = φ₁(z(r_i))·(dz/dr)(r_i)·dr_i` and `u2[j]` (analogous),
return the 64 fused sep entries `(a_k, b_k, slice_idx)` with
`a_k = u1 .* u_k`, `b_k = u2 .* v_k`.  The radial-selection integral is
then literally `Cl_obs[ell, pair] = Σ_k Σ_ij a_k[i]·b_k[j]·slice_k[e,i,j]`.
"""
function _fused_entries_for_pair(cache::ClGRParamCache,
                                  u1::Vector{Float64}, u2::Vector{Float64})
    entries, _ = _build_array_coeff_pairs_sep(cache)
    fused = Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}(undef, length(entries))
    for k in eachindex(entries)
        a_k, b_k, sidx = entries[k]
        fused[k] = (u1 .* a_k, u2 .* b_k, sidx)
    end
    return fused
end

"""
    compute_Cl_obs_streaming_multi!(Cl_obs, fused_per_pair, unique_keys,
                                     meta_path, ells; verbose=false) -> Cl_obs

Multi-pair streaming C_ℓ^{ij} assembly.  For each part:

1. Read every unique slice referenced by any pair (typically 61 arrays,
   ≈7 GB Float32 at Nr=4096, nell_part=25) ONCE.
2. For each pair p, run the fused bilinear contraction
      Cl_obs[ell, p] += Σ_k Σ_ij a_k_p[i]·b_k_p[j]·slice_k[e_local, i, j]
   over the 64 sep entries.

`Cl_obs` is `Matrix{Float64}(n_ell, n_pairs)` and is filled, not
accumulated, on entry (call `fill!(Cl_obs, 0.0)` internally).
"""
function compute_Cl_obs_streaming_multi!(Cl_obs::Matrix{Float64},
                                          fused_per_pair::Vector{Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}},
                                          unique_keys::Vector{Tuple{NTuple{5,Any}, Bool}},
                                          meta_path::String,
                                          ells::Vector{Int};
                                          verbose::Bool=false)
    fill!(Cl_obs, 0.0)
    n_pairs = length(fused_per_pair)
    n_keys  = length(unique_keys)
    n_ell_total = length(ells)
    size(Cl_obs, 1) == n_ell_total ||
        error("Cl_obs first dimension $(size(Cl_obs,1)) != length(ells) $(n_ell_total)")
    size(Cl_obs, 2) == n_pairs ||
        error("Cl_obs second dimension $(size(Cl_obs,2)) != n_pairs $(n_pairs)")

    meta_dir = dirname(meta_path)
    isempty(meta_dir) && (meta_dir = ".")
    part_files_raw, ell_ranges, rr_meta, ell_values_meta = h5open(meta_path, "r") do f
        pf = read(f, "part_files")
        er = Int.(read(f, "metadata/ell_ranges"))
        rr0 = Float64.(read(f, "grid/rr"))
        ev0 = Int.(read(f, "grid/ell_values"))
        (pf, er, rr0, ev0)
    end
    part_files = String.(part_files_raw)
    n_parts = length(part_files)

    nr = length(fused_per_pair[1][1][1])  # a-vector length

    for p_idx in 1:n_parts
        part_path = joinpath(meta_dir, part_files[p_idx])
        ell_lo = Int(ell_ranges[p_idx, 1])
        ell_hi = Int(ell_ranges[p_idx, 2])
        nell_part = ell_hi - ell_lo + 1
        verbose && @info "[part $p_idx/$n_parts] ell $ell_lo:$ell_hi  ($n_pairs pairs, $n_keys slices)"

        # Load all unique slices for this part.
        slices = Vector{Array{Float32,3}}(undef, n_keys)
        t_read = @elapsed begin
            h5open(part_path, "r") do f
                for k in 1:n_keys
                    arr_key, is_base = unique_keys[k]
                    slices[k] = _read_slice(f, arr_key, is_base, ell_values_meta[ell_lo:ell_hi], rr_meta)
                end
            end
        end
        verbose && @info "  slice read: $(round(t_read, digits=1)) s  ($(round(n_keys*nell_part*nr*nr*4/1e9, digits=2)) GB)"

        t_acc = @elapsed _accumulate_part_multipair!(Cl_obs, fused_per_pair, slices,
                                                      ell_lo, nell_part, nr)
        verbose && @info "  accumulate: $(round(t_acc, digits=1)) s"

        for k in 1:n_keys
            slices[k] = Array{Float32,3}(undef, 0, 0, 0)
        end
    end
    return Cl_obs
end

"""
    _accumulate_part_multipair!(Cl_obs, fused_per_pair, slices,
                                 ell_lo, nell_part, nr)

Hot kernel: for each pair `p`, for each fused sep entry `(a, b, sidx)`,
accumulate `Cl_obs[ell_off + e_local, p] += Σ_ij a[i]·b[j]·slices[sidx][e_local, i, j]`
via the two-mat-vec decomposition

    tmp[e, i] = Σ_j slice[e, i, j] · b[j]   (fixed e, i innermost stride-1 on e)
    Cl_obs[ell_off + e, p] += Σ_i a[i] · tmp[e, i]

Threaded over pairs.  Each thread allocates its own `tmp` (nell_part × nr
Float64 ≈ 800 KB at Nr=4096) once per pair iteration.
"""
function _accumulate_part_multipair!(Cl_obs::Matrix{Float64},
                                      fused_per_pair::Vector{Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}},
                                      slices::Vector{Array{Float32,3}},
                                      ell_lo::Int, nell_part::Int, nr::Int)
    ell_off = ell_lo - 1
    n_pairs = length(fused_per_pair)
    @threads for p in 1:n_pairs
        entries = fused_per_pair[p]
        tmp = Matrix{Float64}(undef, nell_part, nr)
        for (a, b, sidx) in entries
            slice = slices[sidx]
            fill!(tmp, 0.0)
            # tmp[e, i] = Σ_j slice[e, i, j] * b[j]
            @inbounds for j in 1:nr
                bj = b[j]
                for i in 1:nr
                    @simd for e in 1:nell_part
                        tmp[e, i] = muladd(Float64(slice[e, i, j]), bj, tmp[e, i])
                    end
                end
            end
            # Cl_obs[ell_off + e, p] += Σ_i a[i] * tmp[e, i]
            @inbounds for i in 1:nr
                ai = a[i]
                @simd for e in 1:nell_part
                    Cl_obs[ell_off + e, p] = muladd(ai, tmp[e, i], Cl_obs[ell_off + e, p])
                end
            end
        end
    end
    return nothing
end

export compute_Cl_obs_streaming_multi!, _fused_entries_for_pair

# =============================================================================
# Sample-grouped variant: when many tracer pairs share the same bias pair
# (e.g. within-sample tomography), assemble C_ℓ^GR(r₁, r₂) once per unique
# sample-pair-type and reuse it for every pair of that type.  Reduces the
# 64-sep-entry contraction from O(n_pairs) to O(n_types), typically 15
# types vs 41,041 pairs for a 5-sample × 286-bin SphereX production.
# =============================================================================

"""
    compute_Cl_obs_streaming_grouped!(
        Cl_obs, pair_type, type_sep_entries, pair_weights, unique_keys,
        meta_path, ells; verbose=false) -> Cl_obs

Per-part workflow (memory budget ≈ 7 GB slices + 2 GB type matrices at
Nr=1155 / nell_part≤25):

1. Read every unique slice referenced by any sep entry.
2. For each unique sample-pair-type `t`, assemble the Float32 matrix
   `M_t[e, i, j] = Σ_k u_{t,k}[i] · v_{t,k}[j] · slices[sidx_{t,k}][e, i, j]`
   by looping over the type's ~64 rank-1 sep entries.
3. For each pair `p`, fetch `M_{pair_type[p]}` and contract with the
   pair's radial weights `(u1, u2)`:
   `Cl_obs[ell_off + e, p] += u1^T · M_t[e, :, :] · u2`.

`pair_type[p]` is a 1-based index into `type_sep_entries`.  Weights are
`(u1::Vector{Float64}(nr), u2::Vector{Float64}(nr))` holding
`φ(z(r))·(dz/dr)·dr` for the two tracers of pair `p`.  The Cl_obs
array is FILLED, not accumulated (reset to zero on entry).
"""
function compute_Cl_obs_streaming_grouped!(
        Cl_obs::Matrix{Float64},
        pair_type::Vector{Int},
        type_sep_entries::Vector{Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}},
        pair_weights::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
        unique_keys::Vector{Tuple{NTuple{5,Any}, Bool}},
        meta_path::String,
        ells::Vector{Int};
        verbose::Bool=false)

    fill!(Cl_obs, 0.0)
    n_pairs  = length(pair_weights)
    n_types  = length(type_sep_entries)
    n_keys   = length(unique_keys)
    length(pair_type) == n_pairs ||
        error("pair_type length $(length(pair_type)) != n_pairs $n_pairs")
    size(Cl_obs, 1) == length(ells) ||
        error("Cl_obs first dim $(size(Cl_obs,1)) != length(ells) $(length(ells))")
    size(Cl_obs, 2) == n_pairs ||
        error("Cl_obs second dim $(size(Cl_obs,2)) != n_pairs $n_pairs")

    meta_dir = dirname(meta_path); isempty(meta_dir) && (meta_dir = ".")
    part_files_raw, ell_ranges, rr_meta, ell_values_meta = h5open(meta_path, "r") do f
        (read(f, "part_files"), Int.(read(f, "metadata/ell_ranges")), Float64.(read(f, "grid/rr")), Int.(read(f, "grid/ell_values")))
    end
    part_files = String.(part_files_raw)
    n_parts    = length(part_files)

    nr = length(pair_weights[1][1])

    for p_idx in 1:n_parts
        part_path = joinpath(meta_dir, part_files[p_idx])
        ell_lo    = Int(ell_ranges[p_idx, 1])
        ell_hi    = Int(ell_ranges[p_idx, 2])
        nell_part = ell_hi - ell_lo + 1
        verbose && @info "[part $p_idx/$n_parts] ell $ell_lo:$ell_hi  ($n_types types, $n_pairs pairs, $n_keys slices)"

        # 1. Load all unique slices for this part.
        slices = Vector{Array{Float32,3}}(undef, n_keys)
        t_read = @elapsed begin
            h5open(part_path, "r") do f
                for k in 1:n_keys
                    arr_key, is_base = unique_keys[k]
                    slices[k] = _read_slice(f, arr_key, is_base, ell_values_meta[ell_lo:ell_hi], rr_meta)
                end
            end
        end
        verbose && @info "  slice read: $(round(t_read, digits=1)) s"

        # 2. Assemble one Float64 C_ℓ^GR matrix per type.  Using Float64
        # (vs Float32) keeps the accumulation precision on par with the
        # per-pair kernel, which converts each Float32 slice element to
        # Float64 at access and accumulates in Float64.  Memory cost per
        # part: n_types × nell_part × nr² × 8 B (≈ 4 GB for 15 types at
        # Nr=1155, nell_part=25 — within the 40 GB budget).
        type_mats = Vector{Array{Float64,3}}(undef, n_types)
        t_asm = @elapsed begin
            @threads for t in 1:n_types
                M = zeros(Float64, nell_part, nr, nr)
                entries = type_sep_entries[t]
                for (u, v, sidx) in entries
                    slice = slices[sidx]
                    @inbounds for j in 1:nr
                        vj = v[j]
                        for i in 1:nr
                            uv = u[i] * vj
                            @simd for e in 1:nell_part
                                M[e, i, j] = muladd(uv, Float64(slice[e, i, j]), M[e, i, j])
                            end
                        end
                    end
                end
                type_mats[t] = M
            end
        end
        verbose && @info "  type-matrix assemble: $(round(t_asm, digits=1)) s"

        # 3. Per-pair bilinear contraction with the pair's type matrix.
        ell_off = ell_lo - 1
        t_con = @elapsed _contract_pairs_grouped!(Cl_obs, pair_type, pair_weights,
                                                   type_mats, ell_off, nell_part, nr)
        verbose && @info "  pair contract: $(round(t_con, digits=1)) s"

        # Release per-part arrays before next iter.
        for k in 1:n_keys
            slices[k] = Array{Float32,3}(undef, 0, 0, 0)
        end
        for t in 1:n_types
            type_mats[t] = Array{Float64,3}(undef, 0, 0, 0)
        end
    end
    return Cl_obs
end

"""
    _contract_pairs_grouped!(Cl_obs, pair_type, pair_weights, type_mats,
                              ell_off, nell_part, nr)

Threaded over pairs.  Each pair computes `u1' * M_t[e, :, :] * u2` via a
matvec-then-dot pattern: `tmp[i] = Σ_j M_t[e, i, j] · u2[j]`, then
`Σ_i u1[i] · tmp[i]`.  `tmp` is a per-thread `Vector{Float64}(nr)` reused
across `e` values; allocating once per pair iteration amortizes across
nell_part.
"""
function _contract_pairs_grouped!(Cl_obs::Matrix{Float64},
                                   pair_type::Vector{Int},
                                   pair_weights::Vector{Tuple{Vector{Float64}, Vector{Float64}}},
                                   type_mats::Vector{Array{Float64,3}},
                                   ell_off::Int, nell_part::Int, nr::Int)
    # M (Float64) has layout [nell_part, nr, nr] column-major → M[e,i,j]
    # stride 1 in e.  Accumulate tmp[e, i] = Σ_j M[e, i, j]·u2[j] in the
    # (j, i, e) nesting, mirroring the per-pair kernel so every inner
    # access is stride-1.
    n_pairs = length(pair_type)
    @threads for p in 1:n_pairs
        t = pair_type[p]
        (u1, u2) = pair_weights[p]
        M = type_mats[t]
        tmp = Matrix{Float64}(undef, nell_part, nr)
        fill!(tmp, 0.0)
        @inbounds for j in 1:nr
            vj = u2[j]
            for i in 1:nr
                @simd for e in 1:nell_part
                    tmp[e, i] = muladd(M[e, i, j], vj, tmp[e, i])
                end
            end
        end
        @inbounds for i in 1:nr
            ui = u1[i]
            @simd for e in 1:nell_part
                Cl_obs[ell_off + e, p] = muladd(ui, tmp[e, i], Cl_obs[ell_off + e, p])
            end
        end
    end
    return nothing
end

export compute_Cl_obs_streaming_grouped!

"""
    compute_Cl_GR_batch(I, params_1, params_2, ells) -> Array{Float64,3}

Cross-tracer variant: `params_1` for the r₁ axis, `params_2` for r₂.
Returns `results[ell_idx, i, j]` = C_ℓ^GR(r₁=I.rr[i], r₂=I.rr[j]).
"""
function compute_Cl_GR_batch(I::IntegralCollection,
                             params_1::ClGRParams, params_2::ClGRParams,
                             ells::Vector{Int})
    cache = ClGRParamCache(I, params_1, params_2)
    results = zeros(Float64, length(ells), cache.nr, cache.nr)
    compute_Cl_GR_batch!(results, cache, ells)
    return results
end

# =============================================================================
# Main Computation: Eq. (C.14) - Original Interface (backward compatible)
# =============================================================================

"""
    compute_Cl_GR(I::IntegralCollection, params::ClGRParams, ell::Int) -> Array{Float64,2}

Compute C_ℓ^GR(r₁, r₂) from Eq. (C.14) for a given ℓ value.
Returns a 2D array of size (nr, nr) on the physical (r₁, r₂) grid.

Note: For computing multiple ℓ values, use `compute_Cl_GR_batch` instead,
which creates a cache once and reuses it for better performance.

# Arguments
- `I::IntegralCollection`: Collection of all integrals on physical grid
- `params::ClGRParams`: Physical parameters (functions of r)
- `ell::Int`: The multipole ℓ value (not index!)

# Returns
- `Array{Float64,2}`: C_ℓ^GR values of size (nr, nr)
  - result[i, j] corresponds to r₁ = I.rr[i], r₂ = I.rr[j]
"""
function compute_Cl_GR(I::IntegralCollection, params::ClGRParams, ell::Int)
    # Find ell index
    ell_idx = _find_ell_idx(I.ell_values, ell)

    # Get actual sizes from data
    nr = length(I.rr)
    result = zeros(Float64, nr, nr)

    # Extract scalar parameters
    f_NL = params.f_NL
    Omm0 = params.Omm0
    H0 = params.H0
    fNL_prefactor = 1.5 * Omm0 * (100.0 / 2.99792458e5)^2

    # Pre-extract integral slices for this ell (copy to contiguous [nr, nr])
    w_0_0_0 = I[:w, 0, 0, 0, :none][ell_idx, :, :]
    w_0_2_0 = I[:w, 0, 2, 0, :none][ell_idx, :, :]
    w_0_0_2 = I[:w, 0, 0, 2, :none][ell_idx, :, :]
    w_0_2_2 = I[:w, 0, 2, 2, :none][ell_idx, :, :]
    w_m1_0_1 = I[:w, -1, 0, 1, :none][ell_idx, :, :]
    w_m1_2_1 = I[:w, -1, 2, 1, :none][ell_idx, :, :]
    w_m1_1_0 = I[:w, -1, 1, 0, :none][ell_idx, :, :]
    w_m1_1_2 = I[:w, -1, 1, 2, :none][ell_idx, :, :]
    w_m2_0_0 = I[:w, -2, 0, 0, :none][ell_idx, :, :]
    w_m2_2_0 = I[:w, -2, 2, 0, :none][ell_idx, :, :]
    w_m2_0_2 = I[:w, -2, 0, 2, :none][ell_idx, :, :]
    w_m2_1_1 = I[:w, -2, 1, 1, :none][ell_idx, :, :]
    w_m3_1_0 = I[:w, -3, 1, 0, :none][ell_idx, :, :]
    w_m3_0_1 = I[:w, -3, 0, 1, :none][ell_idx, :, :]
    w_m4_0_0 = I[:w, -4, 0, 0, :none][ell_idx, :, :]

    s_m2_0_0_r = I[:s, -2, 0, 0, :r][ell_idx, :, :]
    s_m2_0_2_r = I[:s, -2, 0, 2, :r][ell_idx, :, :]
    s_m2_0_0_rp = I[:s, -2, 0, 0, :rp][ell_idx, :, :]
    s_m2_2_0_rp = I[:s, -2, 2, 0, :rp][ell_idx, :, :]
    s_m3_0_1_r = I[:s, -3, 0, 1, :r][ell_idx, :, :]
    s_m3_1_0_rp = I[:s, -3, 1, 0, :rp][ell_idx, :, :]
    s_m4_0_0_r = I[:s, -4, 0, 0, :r][ell_idx, :, :]
    s_m4_0_0_rp = I[:s, -4, 0, 0, :rp][ell_idx, :, :]

    t_m2_0_0_r = I[:t, -2, 0, 0, :r][ell_idx, :, :]
    t_m2_0_2_r = I[:t, -2, 0, 2, :r][ell_idx, :, :]
    t_m2_0_0_rp = I[:t, -2, 0, 0, :rp][ell_idx, :, :]
    t_m2_2_0_rp = I[:t, -2, 2, 0, :rp][ell_idx, :, :]
    t_m3_0_1_r = I[:t, -3, 0, 1, :r][ell_idx, :, :]
    t_m3_1_0_rp = I[:t, -3, 1, 0, :rp][ell_idx, :, :]
    t_m4_0_0_r = I[:t, -4, 0, 0, :r][ell_idx, :, :]
    t_m4_0_0_rp = I[:t, -4, 0, 0, :rp][ell_idx, :, :]

    l_m2_0_0_r = I[:tl, -2, 0, 0, :r][ell_idx, :, :]
    l_m2_0_2_r = I[:tl, -2, 0, 2, :r][ell_idx, :, :]
    l_m2_0_0_rp = I[:tl, -2, 0, 0, :rp][ell_idx, :, :]
    l_m2_2_0_rp = I[:tl, -2, 2, 0, :rp][ell_idx, :, :]
    l_m3_0_1_r = I[:tl, -3, 0, 1, :r][ell_idx, :, :]
    l_m3_1_0_rp = I[:tl, -3, 1, 0, :rp][ell_idx, :, :]
    l_m4_0_0_r = I[:tl, -4, 0, 0, :r][ell_idx, :, :]
    l_m4_0_0_rp = I[:tl, -4, 0, 0, :rp][ell_idx, :, :]

    scrX_r_rp = I[:scrX, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrX_rp_r = I[:scrX, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrY_r_rp = I[:tscrY, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrY_rp_r = I[:tscrY, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrZ_r_rp = I[:tscrZ, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrZ_rp_r = I[:tscrZ, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrS_r_rp = I[:scrS, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrT_r_rp = I[:scrT, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrL_r_rp = I[:tscrL, -4, 0, 0, :r_rp][ell_idx, :, :]

    u_m2_0_0 = I[:u, -2, 0, 0, :none][ell_idx, :, :]
    u_m2_2_0 = I[:u, -2, 2, 0, :none][ell_idx, :, :]
    u_m2_0_2 = I[:u, -2, 0, 2, :none][ell_idx, :, :]
    u_m3_1_0 = I[:u, -3, 1, 0, :none][ell_idx, :, :]
    u_m3_0_1 = I[:u, -3, 0, 1, :none][ell_idx, :, :]
    u_m4_0_0 = I[:u, -4, 0, 0, :none][ell_idx, :, :]

    scrs_m4_r = I[:scrs, -4, 0, 0, :r][ell_idx, :, :]
    scrs_m4_rp = I[:scrs, -4, 0, 0, :rp][ell_idx, :, :]
    scrt_m4_r = I[:scrt, -4, 0, 0, :r][ell_idx, :, :]
    scrt_m4_rp = I[:scrt, -4, 0, 0, :rp][ell_idx, :, :]
    scrl_m4_r = I[:tscrl, -4, 0, 0, :r][ell_idx, :, :]
    scrl_m4_rp = I[:tscrl, -4, 0, 0, :rp][ell_idx, :, :]

    v_m4_0_0 = I[:v, -4, 0, 0, :none][ell_idx, :, :]

    # Loop over physical (r₁, r₂) grid - parallelized over j (r₂ axis)
    @threads for j in 1:nr
        @inbounds for i in 1:nr
            r1 = I.rr[i]
            r2 = I.rr[j]

            # Get parameters at r1 and r2 (function calls)
            D1, D2 = params.D(r1), params.D(r2)
            aH1, aH2 = params.aH(r1), params.aH(r2)
            bg1, bg2 = params.bg(r1), params.bg(r2)
            β1, β2 = params.β(r1), params.β(r2)
            B1, B2 = params.B(r1), params.B(r2)
            A1, A2 = params.A(r1), params.A(r2)
            Q1, Q2 = params.Q(r1), params.Q(r2)
            bPhi1, bPhi2 = params.bPhi(r1), params.bPhi(r2)

            prefactor = D1 * D2

            # Line 1
            term1 = bg1 * bg2 * (
                w_0_0_0[i, j] - β1 * w_0_2_0[i, j]
                - β2 * w_0_0_2[i, j] + β1 * β2 * w_0_2_2[i, j]
            )

            # Line 2
            term2 = (
                bg1 * aH2 * B2 * (w_m1_0_1[i, j] - β1 * w_m1_2_1[i, j])
                + bg2 * aH1 * B1 * (w_m1_1_0[i, j] - β2 * w_m1_1_2[i, j])
            )

            # Line 3
            term3 = bg1 * aH2^2 * A2 * (
                w_m2_0_0[i, j] - β1 * w_m2_2_0[i, j]
                + (aH1 / bg1) * B1 * w_m3_1_0[i, j]
            )

            # Line 4
            term4 = bg2 * aH1^2 * A1 * (
                w_m2_0_0[i, j] - β2 * w_m2_0_2[i, j]
                + (aH2 / bg2) * B2 * w_m3_0_1[i, j]
            )

            # Line 5
            term5 = bg1 * (B2 / β2) * (
                s_m2_0_0_rp[i, j] - β1 * s_m2_2_0_rp[i, j]
                + (aH1 / bg1) * B1 * s_m3_1_0_rp[i, j]
                + (aH1^2 / bg1) * A1 * s_m4_0_0_rp[i, j]
            )

            # Line 6
            term6 = bg2 * (B1 / β1) * (
                s_m2_0_0_r[i, j] - β2 * s_m2_0_2_r[i, j]
                + (aH2 / bg2) * B2 * s_m3_0_1_r[i, j]
                + (aH2^2 / bg2) * A2 * s_m4_0_0_r[i, j]
            )

            # Line 7
            term7 = -2 * bg1 * ((1 - Q2) / r2) * (
                t_m2_0_0_rp[i, j] - β1 * t_m2_2_0_rp[i, j]
                + (aH1 / bg1) * B1 * t_m3_1_0_rp[i, j]
                + (aH1^2 / bg1) * A1 * t_m4_0_0_rp[i, j]
            )

            # Line 8
            term8 = -2 * bg2 * ((1 - Q1) / r1) * (
                t_m2_0_0_r[i, j] - β2 * t_m2_0_2_r[i, j]
                + (aH2 / bg2) * B2 * t_m3_0_1_r[i, j]
                + (aH2^2 / bg2) * A2 * t_m4_0_0_r[i, j]
            )

            # Line 9
            term9 = -2 * bg1 * (1 - Q2) * (
                l_m2_0_0_rp[i, j] - β1 * l_m2_2_0_rp[i, j]
                + (aH1 / bg1) * B1 * l_m3_1_0_rp[i, j]
                + (aH1^2 / bg1) * A1 * l_m4_0_0_rp[i, j]
            )

            # Line 10
            term10 = -2 * bg2 * (1 - Q1) * (
                l_m2_0_0_r[i, j] - β2 * l_m2_0_2_r[i, j]
                + (aH2 / bg2) * B2 * l_m3_0_1_r[i, j]
                + (aH2^2 / bg2) * A2 * l_m4_0_0_r[i, j]
            )

            # Line 11
            term11 = -2 * bg1 * (1 - Q2) * (
                (B1 / (β1 * r2)) * scrX_r_rp[i, j]
                + (B1 / β1) * scrY_r_rp[i, j]
                - 2 * ((1 - Q1) / (bg1 * r2)) * scrZ_rp_r[i, j]
            )

            # Line 12
            term12 = -2 * bg2 * (1 - Q1) * (
                (B2 / (β2 * r1)) * scrX_rp_r[i, j]
                + (B2 / β2) * scrY_rp_r[i, j]
                - 2 * ((1 - Q2) / (bg2 * r1)) * scrZ_r_rp[i, j]
            )

            # Line 13
            term13 = fNL_prefactor * bg1 * (bPhi2 / D2) * f_NL * (
                u_m2_0_0[i, j] - β1 * u_m2_2_0[i, j]
                + (aH1 / bg1) * B1 * u_m3_1_0[i, j]
                + (aH1^2 / bg1) * A1 * u_m4_0_0[i, j]
            )

            # Line 14
            term14 = fNL_prefactor * bg2 * (bPhi1 / D1) * f_NL * (
                u_m2_0_0[i, j] - β2 * u_m2_0_2[i, j]
                + (aH2 / bg2) * B2 * u_m3_0_1[i, j]
                + (aH2^2 / bg2) * A2 * u_m4_0_0[i, j]
            )

            # Line 15
            term15 = fNL_prefactor * (bPhi1 / D1) * f_NL * (
                (B2 / (bg2 * β2)) * scrs_m4_rp[i, j]
                - 2 * (1 - Q2) * ((1 / r2) * scrt_m4_rp[i, j] + scrl_m4_rp[i, j])
            )

            # Line 16
            term16 = fNL_prefactor * (bPhi2 / D2) * f_NL * (
                (B1 / (bg1 * β1)) * scrs_m4_r[i, j]
                - 2 * (1 - Q1) * ((1 / r1) * scrt_m4_r[i, j] + scrl_m4_r[i, j])
            )

            # Line 17
            term17 = (
                aH1 * aH2 * B1 * B2 * w_m2_1_1[i, j]
                + aH1^2 * aH2^2 * A1 * A2 * w_m4_0_0[i, j]
                + (B1 / (bg1 * β1)) * (B2 / (bg2 * β2)) * scrS_r_rp[i, j]
            )

            # Line 18
            term18 = (
                4 * ((1 - Q1) / r1) * ((1 - Q2) / r2) * scrT_r_rp[i, j]
                + 4 * (1 - Q1) * (1 - Q2) * scrL_r_rp[i, j]
            )

            # Line 19
            term19 = (9/4) * (bPhi1 / D1) * (bPhi2 / D2) * f_NL^2 * Omm0^2 * (100.0/2.99792458e5)^4 * v_m4_0_0[i, j]

            # Sum all terms
            result[i, j] = prefactor * (
                term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 +
                term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19
            )
        end
    end

    return result
end

"""
    compute_Cl_GR_all_ell(I::IntegralCollection, params::ClGRParams) -> Array{Float64,3}

Compute C_ℓ^GR for all ℓ values in the collection on physical (r₁, r₂) grid.
Uses optimized batch computation with pre-computed coefficient cache.

# Returns
- `Array{Float64,3}`: C_ℓ^GR values of size (n_ell, nr, nr)
"""
function compute_Cl_GR_all_ell(I::IntegralCollection, params::ClGRParams)
    # Use the optimized batch function
    return compute_Cl_GR_batch(I, params, I.ell_values)
end

"""
    compute_Cl_GR_terms(I::IntegralCollection, params::ClGRParams, ell::Int) -> NamedTuple

Compute individual terms of C_ℓ^GR for debugging/analysis on physical (r₁, r₂) grid.
Returns NamedTuple with 2D arrays (nr, nr) for each of the 19 terms.

# Returns
NamedTuple with fields:
- `term1` through `term19`: Individual contributions (each is nr × nr array)
- `total`: Sum of all terms (= C_ℓ^GR)
- `density_rsd`: Lines 1-4 (standard density and RSD terms)
- `doppler_1D`: Lines 5-10 (1D integral Doppler terms)
- `lensing_2D`: Lines 11-12 (2D integral lensing terms)
- `fNL_linear`: Lines 13-16 (linear f_NL terms)
- `fNL_cross`: Lines 17-18 (cross terms)
- `fNL_squared`: Line 19 (f_NL² term)
"""
function compute_Cl_GR_terms(I::IntegralCollection, params::ClGRParams, ell::Int)
    ell_idx = _find_ell_idx(I.ell_values, ell)
    nr = length(I.rr)

    # Initialize arrays for each term
    term1 = zeros(Float64, nr, nr)
    term2 = zeros(Float64, nr, nr)
    term3 = zeros(Float64, nr, nr)
    term4 = zeros(Float64, nr, nr)
    term5 = zeros(Float64, nr, nr)
    term6 = zeros(Float64, nr, nr)
    term7 = zeros(Float64, nr, nr)
    term8 = zeros(Float64, nr, nr)
    term9 = zeros(Float64, nr, nr)
    term10 = zeros(Float64, nr, nr)
    term11 = zeros(Float64, nr, nr)
    term12 = zeros(Float64, nr, nr)
    term13 = zeros(Float64, nr, nr)
    term14 = zeros(Float64, nr, nr)
    term15 = zeros(Float64, nr, nr)
    term16 = zeros(Float64, nr, nr)
    term17 = zeros(Float64, nr, nr)
    term18 = zeros(Float64, nr, nr)
    term19 = zeros(Float64, nr, nr)

    # Extract scalar parameters
    f_NL = params.f_NL
    Omm0 = params.Omm0
    H0 = params.H0
    fNL_prefactor = 1.5 * Omm0 * (100.0 / 2.99792458e5)^2

    # Pre-extract integral slices for this ell (copy to contiguous [nr, nr])
    w_0_0_0 = I[:w, 0, 0, 0, :none][ell_idx, :, :]
    w_0_2_0 = I[:w, 0, 2, 0, :none][ell_idx, :, :]
    w_0_0_2 = I[:w, 0, 0, 2, :none][ell_idx, :, :]
    w_0_2_2 = I[:w, 0, 2, 2, :none][ell_idx, :, :]
    w_m1_0_1 = I[:w, -1, 0, 1, :none][ell_idx, :, :]
    w_m1_2_1 = I[:w, -1, 2, 1, :none][ell_idx, :, :]
    w_m1_1_0 = I[:w, -1, 1, 0, :none][ell_idx, :, :]
    w_m1_1_2 = I[:w, -1, 1, 2, :none][ell_idx, :, :]
    w_m2_0_0 = I[:w, -2, 0, 0, :none][ell_idx, :, :]
    w_m2_2_0 = I[:w, -2, 2, 0, :none][ell_idx, :, :]
    w_m2_0_2 = I[:w, -2, 0, 2, :none][ell_idx, :, :]
    w_m2_1_1 = I[:w, -2, 1, 1, :none][ell_idx, :, :]
    w_m3_1_0 = I[:w, -3, 1, 0, :none][ell_idx, :, :]
    w_m3_0_1 = I[:w, -3, 0, 1, :none][ell_idx, :, :]
    w_m4_0_0 = I[:w, -4, 0, 0, :none][ell_idx, :, :]

    s_m2_0_0_r = I[:s, -2, 0, 0, :r][ell_idx, :, :]
    s_m2_0_2_r = I[:s, -2, 0, 2, :r][ell_idx, :, :]
    s_m2_0_0_rp = I[:s, -2, 0, 0, :rp][ell_idx, :, :]
    s_m2_2_0_rp = I[:s, -2, 2, 0, :rp][ell_idx, :, :]
    s_m3_0_1_r = I[:s, -3, 0, 1, :r][ell_idx, :, :]
    s_m3_1_0_rp = I[:s, -3, 1, 0, :rp][ell_idx, :, :]
    s_m4_0_0_r = I[:s, -4, 0, 0, :r][ell_idx, :, :]
    s_m4_0_0_rp = I[:s, -4, 0, 0, :rp][ell_idx, :, :]

    t_m2_0_0_r = I[:t, -2, 0, 0, :r][ell_idx, :, :]
    t_m2_0_2_r = I[:t, -2, 0, 2, :r][ell_idx, :, :]
    t_m2_0_0_rp = I[:t, -2, 0, 0, :rp][ell_idx, :, :]
    t_m2_2_0_rp = I[:t, -2, 2, 0, :rp][ell_idx, :, :]
    t_m3_0_1_r = I[:t, -3, 0, 1, :r][ell_idx, :, :]
    t_m3_1_0_rp = I[:t, -3, 1, 0, :rp][ell_idx, :, :]
    t_m4_0_0_r = I[:t, -4, 0, 0, :r][ell_idx, :, :]
    t_m4_0_0_rp = I[:t, -4, 0, 0, :rp][ell_idx, :, :]

    l_m2_0_0_r = I[:tl, -2, 0, 0, :r][ell_idx, :, :]
    l_m2_0_2_r = I[:tl, -2, 0, 2, :r][ell_idx, :, :]
    l_m2_0_0_rp = I[:tl, -2, 0, 0, :rp][ell_idx, :, :]
    l_m2_2_0_rp = I[:tl, -2, 2, 0, :rp][ell_idx, :, :]
    l_m3_0_1_r = I[:tl, -3, 0, 1, :r][ell_idx, :, :]
    l_m3_1_0_rp = I[:tl, -3, 1, 0, :rp][ell_idx, :, :]
    l_m4_0_0_r = I[:tl, -4, 0, 0, :r][ell_idx, :, :]
    l_m4_0_0_rp = I[:tl, -4, 0, 0, :rp][ell_idx, :, :]

    scrX_r_rp = I[:scrX, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrX_rp_r = I[:scrX, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrY_r_rp = I[:tscrY, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrY_rp_r = I[:tscrY, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrZ_r_rp = I[:tscrZ, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrZ_rp_r = I[:tscrZ, -4, 0, 0, :rp_r][ell_idx, :, :]
    scrS_r_rp = I[:scrS, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrT_r_rp = I[:scrT, -4, 0, 0, :r_rp][ell_idx, :, :]
    scrL_r_rp = I[:tscrL, -4, 0, 0, :r_rp][ell_idx, :, :]

    u_m2_0_0 = I[:u, -2, 0, 0, :none][ell_idx, :, :]
    u_m2_2_0 = I[:u, -2, 2, 0, :none][ell_idx, :, :]
    u_m2_0_2 = I[:u, -2, 0, 2, :none][ell_idx, :, :]
    u_m3_1_0 = I[:u, -3, 1, 0, :none][ell_idx, :, :]
    u_m3_0_1 = I[:u, -3, 0, 1, :none][ell_idx, :, :]
    u_m4_0_0 = I[:u, -4, 0, 0, :none][ell_idx, :, :]

    scrs_m4_r = I[:scrs, -4, 0, 0, :r][ell_idx, :, :]
    scrs_m4_rp = I[:scrs, -4, 0, 0, :rp][ell_idx, :, :]
    scrt_m4_r = I[:scrt, -4, 0, 0, :r][ell_idx, :, :]
    scrt_m4_rp = I[:scrt, -4, 0, 0, :rp][ell_idx, :, :]
    scrl_m4_r = I[:tscrl, -4, 0, 0, :r][ell_idx, :, :]
    scrl_m4_rp = I[:tscrl, -4, 0, 0, :rp][ell_idx, :, :]

    v_m4_0_0 = I[:v, -4, 0, 0, :none][ell_idx, :, :]

    # Loop over physical (r₁, r₂) grid - parallelized over j (r₂ axis)
    @threads for j in 1:nr
        @inbounds for i in 1:nr
            r1 = I.rr[i]
            r2 = I.rr[j]

            # Get parameters at r1 and r2 (function calls)
            D1, D2 = params.D(r1), params.D(r2)
            aH1, aH2 = params.aH(r1), params.aH(r2)
            bg1, bg2 = params.bg(r1), params.bg(r2)
            β1, β2 = params.β(r1), params.β(r2)
            B1, B2 = params.B(r1), params.B(r2)
            A1, A2 = params.A(r1), params.A(r2)
            Q1, Q2 = params.Q(r1), params.Q(r2)
            bPhi1, bPhi2 = params.bPhi(r1), params.bPhi(r2)

            prefactor = D1 * D2

            # Line 1: b_{g,1} b_{g,2} (w⁰_{00} - β₁ w⁰_{20} - β₂ w⁰_{02} + β₁β₂ w⁰_{22})
            term1[i, j] = prefactor * bg1 * bg2 * (
                w_0_0_0[i, j] - β1 * w_0_2_0[i, j]
                - β2 * w_0_0_2[i, j] + β1 * β2 * w_0_2_2[i, j]
            )

            # Line 2
            term2[i, j] = prefactor * (
                bg1 * aH2 * B2 * (w_m1_0_1[i, j] - β1 * w_m1_2_1[i, j])
                + bg2 * aH1 * B1 * (w_m1_1_0[i, j] - β2 * w_m1_1_2[i, j])
            )

            # Line 3
            term3[i, j] = prefactor * bg1 * aH2^2 * A2 * (
                w_m2_0_0[i, j] - β1 * w_m2_2_0[i, j]
                + (aH1 / bg1) * B1 * w_m3_1_0[i, j]
            )

            # Line 4
            term4[i, j] = prefactor * bg2 * aH1^2 * A1 * (
                w_m2_0_0[i, j] - β2 * w_m2_0_2[i, j]
                + (aH2 / bg2) * B2 * w_m3_0_1[i, j]
            )

            # Line 5
            term5[i, j] = prefactor * bg1 * (B2 / β2) * (
                s_m2_0_0_rp[i, j] - β1 * s_m2_2_0_rp[i, j]
                + (aH1 / bg1) * B1 * s_m3_1_0_rp[i, j]
                + (aH1^2 / bg1) * A1 * s_m4_0_0_rp[i, j]
            )

            # Line 6
            term6[i, j] = prefactor * bg2 * (B1 / β1) * (
                s_m2_0_0_r[i, j] - β2 * s_m2_0_2_r[i, j]
                + (aH2 / bg2) * B2 * s_m3_0_1_r[i, j]
                + (aH2^2 / bg2) * A2 * s_m4_0_0_r[i, j]
            )

            # Line 7
            term7[i, j] = prefactor * (-2) * bg1 * ((1 - Q2) / r2) * (
                t_m2_0_0_rp[i, j] - β1 * t_m2_2_0_rp[i, j]
                + (aH1 / bg1) * B1 * t_m3_1_0_rp[i, j]
                + (aH1^2 / bg1) * A1 * t_m4_0_0_rp[i, j]
            )

            # Line 8
            term8[i, j] = prefactor * (-2) * bg2 * ((1 - Q1) / r1) * (
                t_m2_0_0_r[i, j] - β2 * t_m2_0_2_r[i, j]
                + (aH2 / bg2) * B2 * t_m3_0_1_r[i, j]
                + (aH2^2 / bg2) * A2 * t_m4_0_0_r[i, j]
            )

            # Line 9
            term9[i, j] = prefactor * (-2) * bg1 * (1 - Q2) * (
                l_m2_0_0_rp[i, j] - β1 * l_m2_2_0_rp[i, j]
                + (aH1 / bg1) * B1 * l_m3_1_0_rp[i, j]
                + (aH1^2 / bg1) * A1 * l_m4_0_0_rp[i, j]
            )

            # Line 10
            term10[i, j] = prefactor * (-2) * bg2 * (1 - Q1) * (
                l_m2_0_0_r[i, j] - β2 * l_m2_0_2_r[i, j]
                + (aH2 / bg2) * B2 * l_m3_0_1_r[i, j]
                + (aH2^2 / bg2) * A2 * l_m4_0_0_r[i, j]
            )

            # Line 11
            term11[i, j] = prefactor * (-2) * bg1 * (1 - Q2) * (
                (B1 / (β1 * r2)) * scrX_r_rp[i, j]
                + (B1 / β1) * scrY_r_rp[i, j]
                - 2 * ((1 - Q1) / (bg1 * r2)) * scrZ_rp_r[i, j]
            )

            # Line 12
            term12[i, j] = prefactor * (-2) * bg2 * (1 - Q1) * (
                (B2 / (β2 * r1)) * scrX_rp_r[i, j]
                + (B2 / β2) * scrY_rp_r[i, j]
                - 2 * ((1 - Q2) / (bg2 * r1)) * scrZ_r_rp[i, j]
            )

            # Line 13
            term13[i, j] = prefactor * fNL_prefactor * bg1 * (bPhi2 / D2) * f_NL * (
                u_m2_0_0[i, j] - β1 * u_m2_2_0[i, j]
                + (aH1 / bg1) * B1 * u_m3_1_0[i, j]
                + (aH1^2 / bg1) * A1 * u_m4_0_0[i, j]
            )

            # Line 14
            term14[i, j] = prefactor * fNL_prefactor * bg2 * (bPhi1 / D1) * f_NL * (
                u_m2_0_0[i, j] - β2 * u_m2_0_2[i, j]
                + (aH2 / bg2) * B2 * u_m3_0_1[i, j]
                + (aH2^2 / bg2) * A2 * u_m4_0_0[i, j]
            )

            # Line 15
            term15[i, j] = prefactor * fNL_prefactor * (bPhi1 / D1) * f_NL * (
                (B2 / (bg2 * β2)) * scrs_m4_rp[i, j]
                - 2 * (1 - Q2) * ((1 / r2) * scrt_m4_rp[i, j] + scrl_m4_rp[i, j])
            )

            # Line 16
            term16[i, j] = prefactor * fNL_prefactor * (bPhi2 / D2) * f_NL * (
                (B1 / (bg1 * β1)) * scrs_m4_r[i, j]
                - 2 * (1 - Q1) * ((1 / r1) * scrt_m4_r[i, j] + scrl_m4_r[i, j])
            )

            # Line 17
            term17[i, j] = prefactor * (
                aH1 * aH2 * B1 * B2 * w_m2_1_1[i, j]
                + aH1^2 * aH2^2 * A1 * A2 * w_m4_0_0[i, j]
                + (B1 / (bg1 * β1)) * (B2 / (bg2 * β2)) * scrS_r_rp[i, j]
            )

            # Line 18
            term18[i, j] = prefactor * (
                4 * ((1 - Q1) / r1) * ((1 - Q2) / r2) * scrT_r_rp[i, j]
                + 4 * (1 - Q1) * (1 - Q2) * scrL_r_rp[i, j]
            )

            # Line 19
            term19[i, j] = prefactor * (9/4) * (bPhi1 / D1) * (bPhi2 / D2) * f_NL^2 * Omm0^2 * (100.0/2.99792458e5)^4 * v_m4_0_0[i, j]
        end
    end

    # Compute grouped terms
    total = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 +
            term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19
    density_rsd = term1 + term2 + term3 + term4
    doppler_1D = term5 + term6 + term7 + term8 + term9 + term10
    lensing_2D = term11 + term12
    fNL_linear = term13 + term14 + term15 + term16
    fNL_cross = term17 + term18
    fNL_squared = term19

    return (
        term1 = term1, term2 = term2, term3 = term3, term4 = term4,
        term5 = term5, term6 = term6, term7 = term7, term8 = term8,
        term9 = term9, term10 = term10, term11 = term11, term12 = term12,
        term13 = term13, term14 = term14, term15 = term15, term16 = term16,
        term17 = term17, term18 = term18, term19 = term19,
        total = total,
        density_rsd = density_rsd,
        doppler_1D = doppler_1D,
        lensing_2D = lensing_2D,
        fNL_linear = fNL_linear,
        fNL_cross = fNL_cross,
        fNL_squared = fNL_squared
    )
end

# =============================================================================
# compute_Cl_observed — z-space integral with user-supplied selection φ(z)
# =============================================================================

"""
    compute_Cl_observed(I, params_1, params_2, phi_1_of_z, phi_2_of_z,
                        z_of_r, dzdr_of_r, ells) -> Vector{Float64}

Apply the radial selection integral

    C_ℓ^{ij} = ∫dz₁ φᵢ(z₁) ∫dz₂ φⱼ(z₂) · C_ℓ^GR(r(z₁), r(z₂))

to the cross-tracer C_ℓ^GR(r₁, r₂) matrix computed from `params_1` and
`params_2`, using the r-grid of `I` as the integration nodes.  The
caller supplies:

- `phi_1_of_z, phi_2_of_z`: selection functions of z, each normalized so
  that ∫ φ(z) dz = 1 over the z-range of `I.rr`.
- `z_of_r, dzdr_of_r`: functions of r that return z(r) and dz/dr(r),
  typically built from a `cosmofns.cosmofn` instance.

The z-integral is done on the r-grid via r-space trapezoidal rule, with
the Jacobian W(r) = φ(z(r)) · dz/dr(r).  `∫W(r)dr = ∫φ(z)dz = 1`.

Returns a vector of observed `C_ℓ^{ij}` indexed by `ells`.  `ells` must
be a subset of `I.ell_values`.
"""
function compute_Cl_observed(I_or_path::Union{IntegralCollection, String},
                              params_1::ClGRParams, params_2::ClGRParams,
                              phi_1_of_z::Function, phi_2_of_z::Function,
                              z_of_r::Function, dzdr_of_r::Function,
                              ells::Vector{Int};
                              verbose::Bool=false)::Vector{Float64}

    streaming = isa(I_or_path, String)
    rr = streaming ?
        h5open(I_or_path, "r") do f; Float64.(read(f, "grid/rr")); end :
        I_or_path.rr
    nr = length(rr)

    # Per-axis r-space weights: W_a(r) = φ_a(z(r)) · dz/dr(r)
    W1 = Vector{Float64}(undef, nr)
    W2 = Vector{Float64}(undef, nr)
    @inbounds for k in 1:nr
        r = rr[k]
        z = z_of_r(r)
        dzdr = dzdr_of_r(r)
        W1[k] = phi_1_of_z(z) * dzdr
        W2[k] = phi_2_of_z(z) * dzdr
    end

    # Trapezoidal weights on the (possibly non-uniform) r grid.
    dr = Vector{Float64}(undef, nr)
    dr[1]    = 0.5 * (rr[2] - rr[1])
    dr[end]  = 0.5 * (rr[end] - rr[end-1])
    @inbounds for k in 2:nr-1
        dr[k] = 0.5 * (rr[k+1] - rr[k-1])
    end

    # Combined per-axis quadrature weight.
    u1 = W1 .* dr
    u2 = W2 .* dr

    # Sanity check + quadrature renormalization: ∫W(r)dr should equal 1 since φ
    # is z-normalized.  Coarse log-grid quadrature of a narrow low-z window can
    # overshoot, so enforce the known unit-area normalization on the Cl grid.
    n1 = sum(u1); n2 = sum(u2)
    (n1 > 0 && isfinite(n1)) || error("φ_1 selection norm non-positive/non-finite (n1=$n1).")
    (n2 > 0 && isfinite(n2)) || error("φ_2 selection norm non-positive/non-finite (n2=$n2).")
    if abs(n1 - 1) > 0.05
        @info "∫φ_1(z(r))(dz/dr)dr = $(round(n1, digits=4)); renormalizing to unit area (quadrature correction)."
    end
    if abs(n2 - 1) > 0.05
        @info "∫φ_2(z(r))(dz/dr)dr = $(round(n2, digits=4)); renormalizing to unit area (quadrature correction)."
    end
    u1 ./= n1
    u2 ./= n2

    # Per-ℓ 2D integral of C_ℓ^GR(r_a, r_b) weighted by u1[a] · u2[b].
    Cl_rr = streaming ?
        compute_Cl_GR_batch_streaming(I_or_path, params_1, params_2, ells; verbose=verbose) :
        compute_Cl_GR_batch(I_or_path, params_1, params_2, ells)   # [nell, nr, nr]
    nell = length(ells)
    Cl_obs = Vector{Float64}(undef, nell)
    @inbounds for e in 1:nell
        acc = 0.0
        @inbounds for j in 1:nr
            w2 = u2[j]
            @inbounds for i in 1:nr
                acc += u1[i] * w2 * Cl_rr[e, i, j]
            end
        end
        Cl_obs[e] = acc
    end
    return Cl_obs
end

export compute_Cl_observed

# =============================================================================
# Tracer: per-sample z-space bias + selection, and its bridge to ClGRParams.
# =============================================================================

"""
    Tracer

Per-sample galaxy properties and radial selection as functions of z.

- `bg(z)`    galaxy bias
- `be(z)`   evolution bias
- `Q(z)`    magnification bias
- `phi(z)`  radial selection, normalized ∫φ(z)dz = 1
- `bPhi(z)` scale-dependent bias; if `nothing`, the bridge uses 2·δ_c·(b_g−1)
- `zmin, zmax` spline validity range (informational)
"""
struct Tracer
    bg::Function
    be::Function
    Q::Function
    phi::Function
    bPhi::Union{Nothing,Function}
    zmin::Float64
    zmax::Float64
    # Sample id (integer label shared across all z-bins of the same SphereX
    # sample).  When two tracers carry the same `sample`, their bias
    # functions b_g, b_e, Q, b_Φ agree — only φ(z) differs — so the
    # sample-grouped Cl kernel can reuse one C_ℓ^GR(r₁, r₂) matrix for
    # every pair sharing the `(sample_i, sample_j)` type.  Defaults to
    # `nothing` when the tracer h5 lacks `/sample`, in which case the
    # per-pair kernel is used.
    sample::Union{Nothing,Int}
end

"""
    load_tracer_h5(path) -> Tracer

HDF5 keys: `/z`, `/bg`, `/be`, `/Q`, `/phi` (all Vector{Float64} on shared
grid `/z`), and optional `/bPhi`.  `/phi` must already satisfy ∫φ(z)dz = 1.
Splines are cubic (k=3).
"""
function load_tracer_h5(path::String)::Tracer
    isfile(path) || error("tracer h5 file not found: $path")
    h5open(path, "r") do f
        for k in ("z", "bg", "be", "Q", "phi")
            haskey(f, k) || error("tracer h5 missing required key '/$k': $path")
        end
        z    = read(f["z"])
        bg_v = read(f["bg"])
        be_v = read(f["be"])
        Q_v  = read(f["Q"])
        phi_v= read(f["phi"])
        n = length(z)
        for (k, v) in ((:bg, bg_v), (:be, be_v), (:Q, Q_v), (:phi, phi_v))
            length(v) == n || error("tracer h5 '/$k' length $(length(v)) != /z length $n in $path")
        end

        bg_spl  = Spline1D(z, bg_v,  k=3)
        be_spl  = Spline1D(z, be_v,  k=3)
        Q_spl   = Spline1D(z, Q_v,   k=3)
        phi_spl = Spline1D(z, phi_v, k=3)

        bPhi_fn = if haskey(f, "bPhi")
            bPhi_v = read(f["bPhi"])
            length(bPhi_v) == n || error("tracer h5 '/bPhi' length $(length(bPhi_v)) != /z length $n in $path")
            bPhi_spl = Spline1D(z, bPhi_v, k=3)
            zz::Real -> bPhi_spl(zz)
        else
            nothing
        end

        sample_id = haskey(f, "sample") ? Int(read(f["sample"])) : nothing

        return Tracer(
            (zz::Real) -> bg_spl(zz),
            (zz::Real) -> be_spl(zz),
            (zz::Real) -> Q_spl(zz),
            (zz::Real) -> phi_spl(zz),
            bPhi_fn,
            minimum(z), maximum(z),
            sample_id,
        )
    end
end

"""
    _make_dlnHdlna(cfns; n=400) -> Function

Build a model-independent estimator of d ln H / d ln a as a function of
r, splining ln H against ln a over the cosmology's own r-grid.

This replaces the ΛCDM-only identity  d ln H / d ln a = -(3/2) Ω_m(a),
which holds only when Ḣ is fixed by matter + a cosmological constant.
For a general expansion history H(a) (e.g. w ≠ -1 dark energy, modified
gravity, curvature) the friction term must come from the actual H(a),
not from Ω_m.  We obtain it numerically from `cfns.fHr` / `cfns.far`
using the same cubic-spline-derivative idiom as `_compute_alpha_newtonian`.
The returned closure maps r -> d ln H / d ln a evaluated at a(r).
"""
function _make_dlnHdlna(cfns; rmin::Float64 = 1.0, rmax::Float64 = 1.0e4,
                         n::Int = 400)
    rs = exp10.(range(log10(rmin), log10(rmax), length=n))
    a  = [cfns.far(r) for r in rs]
    H  = [cfns.fHr(r) for r in rs]

    # Build strictly-increasing (ln a, ln H) knots. Spline1D REQUIRES strictly
    # increasing x; the cosmology table saturates a→const at low z and may clamp
    # at its high-z edge, producing duplicate/flat ln a that otherwise make the
    # spline return NaN. Sort by ln a, then keep only strictly-increasing knots
    # (skip non-positive a/H, which would give NaN under log).
    pts = Tuple{Float64,Float64}[]
    for i in eachindex(a)
        (a[i] > 0 && H[i] > 0) || continue
        push!(pts, (log(a[i]), log(H[i])))
    end
    sort!(pts, by = first)
    ln_a = Float64[]; ln_H = Float64[]
    for (la, lh) in pts
        if isempty(ln_a) || la > last(ln_a) + 1e-12   # strictly increasing
            push!(ln_a, la); push!(ln_H, lh)
        end
    end
    length(ln_a) >= 4 || error("_make_dlnHdlna: only $(length(ln_a)) usable (ln a, ln H) knots; cosmology table too coarse or degenerate over r∈[$rmin,$rmax].")

    spl = Spline1D(ln_a, ln_H, k=3)
    # Clamp the evaluation point into the fitted ln a range so points just past
    # the table edge return the boundary derivative instead of NaN.
    la_lo, la_hi = first(ln_a), last(ln_a)
    return function (r)
        la = log(cfns.far(r))
        la = clamp(la, la_lo, la_hi)
        return derivative(spl, la)
    end
end

"""
    tracer_to_clgr_params(tracer, cfns; fNL, Omm0, H0, delta_c=1.686) -> ClGRParams

Compose a z-space `Tracer` with r-space cosmology (via a `cosmofn`
instance) to produce a `ClGRParams` struct the 19-term assembly
consumes.  `delta_c` is only used when `tracer.bPhi === nothing`, in
which case b_Φ(r) defaults to 2·δ_c·(b_g(r)−1).

The velocity coefficient 𝒞(r) uses d ln H / d ln a obtained directly
from the supplied expansion history (see `_make_dlnHdlna`) rather than
the ΛCDM-only substitution -(3/2)Ω_m, so the resulting ℬ and 𝒜 are
valid for a general H(a).
"""
function tracer_to_clgr_params(tracer::Tracer, cfns;
                                fNL::Float64, Omm0::Float64, H0::Float64,
                                delta_c::Float64 = 1.686,
                                mg::Union{Nothing,MGModel} = nothing,
                                f_scale::Union{Nothing,Function} = nothing,
                                bg_scale::Union{Nothing,Function} = nothing)::ClGRParams
    mg_active = (mg !== nothing) && (!mg.is_gr)

    # Growth D and rate f: modified D₀,f₀ when MG active, else table GR D,f.
    D_fn = mg_active ? (r -> mg.D0_r(r)) : (r -> cfns.fDr(r))
    f_base = mg_active ? (r -> mg.f0_r(r)) : (r -> cfns.ffr(r))

    Om_fn  = r -> cfns.fOmr(r)
    a_fn   = r -> cfns.far(r)
    Hph_fn = r -> cfns.fHr(r)
    z_fn   = r -> cfns.fzr(r)
    aH_fn   = r -> a_fn(r) * Hph_fn(r)         # conformal ℋ(r) = a·H

    # Optional multiplicative modulation of fσ8 and bσ8 (Khek et al. 2212.05760
    # polynomial scheme).  f_scale(z), bg_scale(z) default to 1 (no change), so
    # the fiducial call is byte-identical to before.  fσ8 ∝ f (σ8 a constant),
    # so rescaling fσ8 ≡ rescaling f; same for bσ8 ∝ bg.  NOTE: f modulation here
    # flows into β, B, and A (contraction-time f-dependence).  The f-dependence
    # buried in the Step-2 ISW integral is NOT modulated (relativistic × growth
    # correction, second-order small — see notes).
    f_fn = if f_scale === nothing
        f_base
    else
        r -> f_base(r) * f_scale(z_fn(r))
    end

    bg_fn = if bg_scale === nothing
        r -> tracer.bg(z_fn(r))
    else
        r -> tracer.bg(z_fn(r)) * bg_scale(z_fn(r))
    end

    be_fn = r -> tracer.be(z_fn(r))
    Q_fn  = r -> tracer.Q(z_fn(r))
    β_fn  = r -> f_fn(r) / bg_fn(r)            # β = f₀/b_g  (MG-aware via f_fn)

    # Model-independent d ln H / d ln a from the supplied H(a).
    dlnHdlna_fn = _make_dlnHdlna(cfns)

    function C_fn(r)
        Q = Q_fn(r); aHr = aH_fn(r) * r
        dlnH_dlna = dlnHdlna_fn(r)           # general H(a); ΛCDM gives -(3/2)Ω_m
        return -dlnH_dlna - (2.0 / aHr) * (1 - Q) - 2 * Q
    end

    # ℬ̄ = b_e + 𝒞 - 1;  ℬ = f₀·ℬ̄  (the velocity-coefficient field B uses f₀).
    Bbar_fn = r -> (be_fn(r) + C_fn(r) - 1)
    B_fn    = r -> f_fn(r) * Bbar_fn(r)

    # 𝒜 field.  In the LOCAL MG limit this is the full 𝒜̃₀ (Eq. A.0), which
    # reduces EXACTLY to the GR 𝒜 (the verbatim Eq.-101 form below) when
    # δμ₀=δΣ₀=δμ₀'=δΣ₀'=0 and f₀=f.  (Verified: 𝒜_code|_GR ≡ 𝒜̃₀|_GR.)
    function A_fn(r)
        Om = Om_fn(r); f0 = f_fn(r); Q = Q_fn(r)
        be = be_fn(r); Bbar = Bbar_fn(r)
        if !mg_active
            # GR: verbatim Eq.-101 (unfactored, no ΛCDM assumption outside 𝒞).
            C = C_fn(r)
            return 1.5 * Om * (be * (1 - 2f0 / (3Om)) + 1 + 2f0 / Om + C - f0 - 2Q)
        end
        # MG local limit: full 𝒜̃₀ (Eq. A.0).  Primed quantities are wrt
        # conformal time η; the formula needs them divided by ℋ, i.e.
        #   δX₀'(η)/ℋ = d(δX₀)/dlna  — exactly what dX0_prime_r/aH gives.
        aH    = aH_fn(r)
        dmu0  = mg.dmu0_r(r)
        dSig0 = mg.dSig0_r(r)
        dmu0p_over_H  = mg.dmu0_prime_r(r)  / aH    # δμ₀'(η)/ℋ
        dSig0p_over_H = mg.dSig0_prime_r(r) / aH    # δΣ₀'(η)/ℋ
        return 1.5 * Om * (
              (Bbar + 2 - 2Q) * (1 + dmu0)
            - f0
            - (2f0 / (3Om)) * (be - 3)
            - (2 * dSig0p_over_H - dmu0p_over_H)
            - 2 * f0 * dSig0
            + f0 * dmu0^2
            - 2 * f0 * dmu0 * dSig0
        )
    end

    # b_Φ / D : the fNL scale-dependent-bias amplitude uses the MODIFIED D₀
    # (it appears as b_Φ/D₀ in Eq. Fell_0_local). bPhi itself is z-only.
    bPhi_fn = if tracer.bPhi === nothing
        r -> 2 * delta_c * (bg_fn(r) - 1)
    else
        r -> tracer.bPhi(z_fn(r))
    end

    return ClGRParams(
        D = D_fn, aH = aH_fn, bg = bg_fn, β = β_fn,
        B = B_fn, A = A_fn, Q = Q_fn, bPhi = bPhi_fn,
        f_NL = fNL, Omm0 = Omm0, H0 = H0,
    )
end

export Tracer, load_tracer_h5, tracer_to_clgr_params

# =============================================================================
# Multi-pair observed-Cl driver: shared part-I/O across all pairs.
# =============================================================================

"""
    compute_Cl_observed_multi(meta_path, tracers, pair_indices,
                               cfns, z_of_r, dzdr_of_r, ells;
                               fNL=0.0, Omm0=0.30682, H0=67.78, delta_c=1.686,
                               verbose=false) -> Matrix{Float64}

Multi-pair analogue of `compute_Cl_observed`.  Computes C_ℓ^{ij} for
every requested pair of tracers in a single streaming pass over the
integrals HDF5, sharing the slice reads across all pairs.

Arguments:
- `meta_path` — split-h5 meta file (integrals).
- `tracers::Vector{Tracer}` — loaded `Tracer` objects (length n_tracers).
- `pair_indices::Vector{Tuple{Int,Int}}` — 1-based (i, j) indices into
  `tracers`; may include both auto (i=j) and cross pairs.
- `cfns` — `cosmofns.cosmofn` instance used to build per-tracer ClGRParams.
- `z_of_r, dzdr_of_r` — functions used for the r-space radial-selection weights.
- `ells` — ell list (must equal the integrals' full ell list, as for
  the single-pair streaming path).

Returns `Cl_obs[ell_idx, pair_idx]`.
"""
function compute_Cl_observed_multi(meta_path::String,
                                    tracers::Vector{Tracer},
                                    pair_indices::Vector{Tuple{Int,Int}},
                                    cfns,
                                    z_of_r::Function, dzdr_of_r::Function,
                                    ells::Vector{Int};
                                    fNL::Float64=0.0,
                                    Omm0::Float64=0.30682,
                                    H0::Float64=67.78,
                                    delta_c::Float64=1.686,
                                    variant::Symbol=:full,
                                    mu0::Float64=0.0,
                                    Sigma0::Float64=0.0,
                                    f_scale::Union{Nothing,Function}=nothing,
                                    bg_scales::Union{Nothing,Vector}=nothing,
                                    verbose::Bool=false)::Matrix{Float64}

    rr, ell_values = h5open(meta_path, "r") do f
        (Float64.(read(f, "grid/rr")), Int.(read(f, "grid/ell_values")))
    end
    ells == ell_values ||
        error("compute_Cl_observed_multi requires ells == integrals' full ell_values (streaming path).")
    nr = length(rr)
    n_pairs = length(pair_indices)
    n_tracers = length(tracers)

    # Per-tracer ClGRParams (build once, reuse across pairs that reference it).
    # Component variants are meant to write the COEFFICIENTS:
    #   Cl_fi          = coefficient of f_NL
    #   Cl_fi_kaiser   = Kaiser subset of coefficient of f_NL
    #   Cl_fi_newtonian= Newtonian subset of coefficient of f_NL
    #   Cl_ff          = coefficient of f_NL^2
    # Therefore evaluate those component kernels with f_NL=1 regardless of
    # the CLI default.  Full/Gaussian/Newtonian/Kaiser still use the user fNL.
    component_variant = variant in (:fi, :fi_kaiser, :fi_newtonian, :ff)
    fNL_eff = component_variant ? 1.0 : fNL
    if verbose && component_variant && fNL != 1.0
        println("  [component mode] variant=$(variant): using fNL_eff=1.0 to write coefficient file (input fNL=$(fNL))")
        flush(stdout)
    end

    params = Vector{ClGRParams}(undef, n_tracers)
    mg = ((mu0 != 0.0) || (Sigma0 != 0.0)) ? build_mg_model(cfns; mu0=mu0, Sigma0=Sigma0) : nothing
    if verbose && mg !== nothing
        println("  [MG] local-limit modified gravity: μ₀=$mu0, Σ₀=$Sigma0 (D₀,f₀ from modified growth ODE; 𝒜→𝒜̃₀)")
        flush(stdout)
    end
    for k in 1:n_tracers
        bgsc = (bg_scales === nothing) ? nothing : bg_scales[k]
        params[k] = tracer_to_clgr_params(tracers[k], cfns;
            fNL=fNL_eff, Omm0=Omm0, H0=H0, delta_c=delta_c, mg=mg,
            f_scale=f_scale, bg_scale=bgsc)
    end

    # Shared shell IntegralCollection so the cache constructor can find keys.
    dummy = zeros(Float64, 0, 0, 0)
    needed_keys = (
        (:w, 0, 0, 0, :none),  (:w, 0, 2, 0, :none),  (:w, 0, 0, 2, :none),  (:w, 0, 2, 2, :none),
        (:w, -1, 0, 1, :none), (:w, -1, 2, 1, :none), (:w, -1, 1, 0, :none), (:w, -1, 1, 2, :none),
        (:w, -2, 0, 0, :none), (:w, -2, 2, 0, :none), (:w, -2, 0, 2, :none), (:w, -2, 1, 1, :none),
        (:w, -3, 1, 0, :none), (:w, -3, 0, 1, :none), (:w, -4, 0, 0, :none),
        (:u, -2, 0, 0, :none), (:u, -2, 2, 0, :none), (:u, -2, 0, 2, :none),
        (:u, -3, 1, 0, :none), (:u, -3, 0, 1, :none), (:u, -4, 0, 0, :none),
        (:v, -4, 0, 0, :none),
        (:s, -2, 0, 0, :r),  (:s, -2, 0, 2, :r),  (:s, -3, 0, 1, :r),  (:s, -4, 0, 0, :r),
        (:s, -2, 0, 0, :rp), (:s, -2, 2, 0, :rp), (:s, -3, 1, 0, :rp), (:s, -4, 0, 0, :rp),
        (:t, -2, 0, 0, :r),  (:t, -2, 0, 2, :r),  (:t, -3, 0, 1, :r),  (:t, -4, 0, 0, :r),
        (:t, -2, 0, 0, :rp), (:t, -2, 2, 0, :rp), (:t, -3, 1, 0, :rp), (:t, -4, 0, 0, :rp),
        (:tl, -2, 0, 0, :r),  (:tl, -2, 0, 2, :r),  (:tl, -3, 0, 1, :r),  (:tl, -4, 0, 0, :r),
        (:tl, -2, 0, 0, :rp), (:tl, -2, 2, 0, :rp), (:tl, -3, 1, 0, :rp), (:tl, -4, 0, 0, :rp),
        (:scrX, -4, 0, 0, :r_rp), (:scrX, -4, 0, 0, :rp_r),
        (:tscrY, -4, 0, 0, :r_rp), (:tscrY, -4, 0, 0, :rp_r),
        (:tscrZ, -4, 0, 0, :r_rp), (:tscrZ, -4, 0, 0, :rp_r),
        (:scrS, -4, 0, 0, :r_rp), (:scrT, -4, 0, 0, :r_rp), (:tscrL, -4, 0, 0, :r_rp),
        (:scrs,  -4, 0, 0, :r),  (:scrs,  -4, 0, 0, :rp),
        (:scrt,  -4, 0, 0, :r),  (:scrt,  -4, 0, 0, :rp),
        (:tscrl, -4, 0, 0, :r),  (:tscrl, -4, 0, 0, :rp),
    )
    data_shell = Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}()
    for k in needed_keys
        data_shell[k] = dummy
    end
    I_shell = IntegralCollection(data_shell, rr, ell_values)

    # Trapezoidal weights on the r-grid (shared across pairs).
    dr = Vector{Float64}(undef, nr)
    dr[1]   = 0.5 * (rr[2] - rr[1])
    dr[end] = 0.5 * (rr[end] - rr[end-1])
    @inbounds for k in 2:nr-1
        dr[k] = 0.5 * (rr[k+1] - rr[k-1])
    end

    # Per-tracer u_t[i] = φ_t(z(r_i)) · dz/dr(r_i) · dr_i
    u_per_tracer = Vector{Vector{Float64}}(undef, n_tracers)
    for t in 1:n_tracers
        u = Vector{Float64}(undef, nr)
        phi_t = tracers[t].phi
        @inbounds for k in 1:nr
            r = rr[k]
            z = z_of_r(r)
            u[k] = phi_t(z) * dzdr_of_r(r) * dr[k]
        end
        n = sum(u)
        if n <= 0 || !isfinite(n)
            error("tracer[$t]: non-positive/non-finite selection norm (n=$n); check φ and z(r) range.")
        end
        if abs(n - 1) > 0.05
            @info "tracer[$t]: discrete ∫φ(z(r))(dz/dr)dr = $(round(n, digits=4)); " *
                  "renormalizing radial weight to unit area (coarse-grid quadrature correction; " *
                  "φ is analytically unit-normalized)."
        end
        u ./= n        # enforce ∫φ=1 on the Cl integration grid (selection is unit-area by definition)
        u_per_tracer[t] = u
    end

    # For the :newtonian variant, pre-compute α(r) per tracer.  α depends
    # on tracer.be (via α₁) and on cosmology (via α₂) — for SphereX's
    # common b_e across samples α is the same across tracers, but we
    # compute per-tracer to be safe.
    α_per_tracer = variant in (:newtonian, :fi_newtonian) ?
        [_compute_alpha_newtonian(tracers[t].be, cfns, rr) for t in 1:n_tracers] :
        Vector{Vector{Float64}}()

    # Detect sample-grouped path: every tracer has a non-nothing `sample`
    # field, and pairs can be partitioned by (min(sa, sb), max(sa, sb)).
    # If at least one tracer is missing `sample`, fall back to the
    # per-pair kernel.
    use_grouped = all(t -> t.sample !== nothing, tracers)

    if use_grouped
        # Map each pair to a 1-based sample-pair-type index.  Types are
        # enumerated as the sorted unique (sample_a, sample_b) tuples,
        # with sample_a ≤ sample_b, because ClGRParamCache(params_a, params_b)
        # and (params_b, params_a) give the C_ℓ^GR matrix transposed —
        # equivalent under (i, j) swap — so we only need one matrix per
        # unordered sample pair.
        pair_sample = Vector{Tuple{Int,Int}}(undef, n_pairs)
        for p in 1:n_pairs
            (ti, tj) = pair_indices[p]
            sa = tracers[ti].sample::Int
            sb = tracers[tj].sample::Int
            pair_sample[p] = (min(sa, sb), max(sa, sb))
        end
        type_keys = sort(unique(pair_sample))
        n_types = length(type_keys)
        type_idx_of = Dict(k => i for (i, k) in enumerate(type_keys))

        # For each type, find one representative pair to build the sep
        # entries from (all pairs of the same type share biases, so any
        # representative gives the same (u, v, sidx) list).
        type_rep_params = Vector{Tuple{ClGRParams, ClGRParams, Bool}}(undef, n_types)
        for p in 1:n_pairs
            (ti, tj) = pair_indices[p]
            (sa, sb) = pair_sample[p]
            t_i = type_idx_of[(sa, sb)]
            if !isassigned(type_rep_params, t_i)
                swapped = sa == tracers[ti].sample::Int ? false : true
                # Preserve the same (sample_a, sample_b) ordering as type_keys
                # for consistency across pairs of the same type.
                if sa == tracers[ti].sample::Int
                    type_rep_params[t_i] = (params[ti], params[tj], false)
                else
                    type_rep_params[t_i] = (params[tj], params[ti], true)
                end
            end
        end

        # Per-type representative tracer indices to pull α from (for :newtonian).
        # Tuple{Int,Int} is isbits, so `isassigned(v, i)` on an `undef`-
        # initialized Vector is always true and can't be used as an
        # "already-set" sentinel — track first-seen with a BitVector
        # instead, and take the first pair of each type as the rep.
        type_rep_tracers = Vector{Tuple{Int,Int}}(undef, n_types)
        rep_seen = falses(n_types)
        for p in 1:n_pairs
            (ti, tj) = pair_indices[p]
            (sa, sb) = pair_sample[p]
            t_i = type_idx_of[(sa, sb)]
            if !rep_seen[t_i]
                if sa == tracers[ti].sample::Int
                    type_rep_tracers[t_i] = (ti, tj)
                else
                    type_rep_tracers[t_i] = (tj, ti)
                end
                rep_seen[t_i] = true
            end
        end

        type_sep_entries = Vector{Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}}(undef, n_types)
        unique_keys_out = Vector{Tuple{NTuple{5,Any}, Bool}}()
        for t_i in 1:n_types
            p1, p2, _ = type_rep_params[t_i]
            cache_t = ClGRParamCache(I_shell, p1, p2)
            if variant === :newtonian
                (rt1, rt2) = type_rep_tracers[t_i]
                entries, keys_t = _build_newtonian_coeff_pairs_sep(
                    cache_t, α_per_tracer[rt1], α_per_tracer[rt2])
            elseif variant === :fi_newtonian
                (rt1, rt2) = type_rep_tracers[t_i]
                entries, keys_t = _build_fi_newtonian_coeff_pairs_sep(
                    cache_t, α_per_tracer[rt1], α_per_tracer[rt2])
            else
                entries, keys_t = _build_array_coeff_pairs_sep(cache_t; variant=variant)
            end
            type_sep_entries[t_i] = entries
            if isempty(unique_keys_out)
                unique_keys_out = keys_t
            end
        end

        # Pair type (Int index) + radial weights (u1, u2) in the same
        # axis order as the type's (params_a, params_b): if the type's
        # representative was built with (params_ti, params_tj) then pair
        # weight is (u_ti, u_tj); if the representative was swapped then
        # pair weight must swap too.
        pair_type = Vector{Int}(undef, n_pairs)
        pair_weights = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, n_pairs)
        for p in 1:n_pairs
            (ti, tj) = pair_indices[p]
            (sa, sb) = pair_sample[p]
            t_i = type_idx_of[(sa, sb)]
            pair_type[p] = t_i
            if sa == tracers[ti].sample::Int
                pair_weights[p] = (u_per_tracer[ti], u_per_tracer[tj])
            else
                pair_weights[p] = (u_per_tracer[tj], u_per_tracer[ti])
            end
        end

        verbose && @info "Grouped kernel: $n_pairs pairs → $n_types sample-pair types"
        Cl_obs = zeros(Float64, length(ells), n_pairs)
        compute_Cl_obs_streaming_grouped!(Cl_obs, pair_type, type_sep_entries,
                                           pair_weights, unique_keys_out,
                                           meta_path, ells; verbose=verbose)
        return Cl_obs
    end

    # Fall back: per-pair kernel (no sample info, or mixed).
    fused_per_pair = Vector{Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}}(undef, n_pairs)
    unique_keys_out = Vector{Tuple{NTuple{5,Any}, Bool}}()
    for p in 1:n_pairs
        (ti, tj) = pair_indices[p]
        (1 ≤ ti ≤ n_tracers) && (1 ≤ tj ≤ n_tracers) ||
            error("pair $p: (ti=$ti, tj=$tj) out of range 1:$n_tracers")
        cache_p = ClGRParamCache(I_shell, params[ti], params[tj])
        if variant === :newtonian
            entries, keys_p = _build_newtonian_coeff_pairs_sep(
                cache_p, α_per_tracer[ti], α_per_tracer[tj])
        elseif variant === :fi_newtonian
            entries, keys_p = _build_fi_newtonian_coeff_pairs_sep(
                cache_p, α_per_tracer[ti], α_per_tracer[tj])
        else
            entries, keys_p = _build_array_coeff_pairs_sep(cache_p; variant=variant)
        end
        if isempty(unique_keys_out)
            unique_keys_out = keys_p
        end
        # Pre-fuse radial weights into the sep entries.
        u1 = u_per_tracer[ti]; u2 = u_per_tracer[tj]
        fused = Vector{Tuple{Vector{Float64}, Vector{Float64}, Int}}(undef, length(entries))
        for k in eachindex(entries)
            a_k, b_k, sidx = entries[k]
            fused[k] = (u1 .* a_k, u2 .* b_k, sidx)
        end
        fused_per_pair[p] = fused
    end

    Cl_obs = zeros(Float64, length(ells), n_pairs)
    compute_Cl_obs_streaming_multi!(Cl_obs, fused_per_pair, unique_keys_out,
                                     meta_path, ells; verbose=verbose)
    return Cl_obs
end

export compute_Cl_observed_multi

end # module CalcClGR
