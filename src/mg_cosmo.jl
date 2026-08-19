# mg_cosmo.jl
# =============================================================================
#
# >> mg_cosmo.jl <<
#
# Shared modified-gravity (MG) helpers for the PowerFull pipeline, in the
# "local modification" / no-scale-dependence regime (x^2 << 1, keeping only
# the k^0 pieces of mu, Sigma, eta).  Implements:
#
#   * delta_mu_0(z)  = (mu_0   / Omega_Lambda) * Omega_de(z)          [Eq. 3.39 left]
#   * delta_Sig_0(z) = (Sigma_0/ Omega_Lambda) * Omega_de(z)          [Eq. 3.40 left]
#   * delta_Sig_0'(z) wrt CONFORMAL TIME (d/dη), i.e. ℋ·d/dlna, to match the
#     ISW integrand's  δΣ₀'(r'')/ℋ(r'')  combination in Eqs. (4.x) I_isw_00.
#   * Modified linear growth D_0(z), f_0(z) from the redshift-only growth ODE
#         D₀'' + ℋ D₀' - (3/2) ℋ² Ω_m [1 + δμ₀] D₀ = 0          [Eq. 3.34]
#     solved as a function of ln a (numerically stable), then re-expressed on
#     the cosmology's own r-grid via splines, returning callables D_0(r), f_0(r).
#
# GR limit: mu_0 = 0, Sigma_0 = 0  ⇒  δμ₀ = δΣ₀ = 0, D_0 = D_GR, f_0 = f_GR.
#
# Fiducial assumptions (match fid_wiggle = flat ΛCDM):
#   * Flat universe, radiation negligible at the redshifts of interest, so
#         Ω_de(z) = 1 - Ω_m(z)        (from the table's Ω_m column),
#         Ω_Λ      = Ω_de(z=0) = 1 - Ω_{m,0}.
#   * The expansion history H(a) is taken from the cosmology table (cfns.fHr),
#     so the Hubble-friction term in the growth ODE is fully general; only the
#     Poisson source carries the [1+δμ₀] modification.
#
# This module touches NO GR code paths when mu_0 = Sigma_0 = 0.
#
#  2026
# =============================================================================

module MGCosmo

using Dierckx

export MGModel, build_mg_model
export delta_mu0_z, delta_Sig0_z, dSig0_dconftime_r

# Speed of light [km/s] — matches PowerFull convention (fHr stores H/c in h/Mpc).
const C_LIGHT = 2.99792458e5

"""
    MGModel

Container for the MG modification functions on the cosmology's r-grid.
All callables take comoving distance r [Mpc/h] (so they compose directly
with the rest of the pipeline, which is r-indexed).

Fields (all `Function` of r unless noted):
- `mu0`, `Sigma0` :: Float64   — the two MG amplitudes (scalars)
- `OmLambda`      :: Float64   — Ω_Λ = 1 - Ω_{m,0}
- `dmu0_r`        — δμ₀(r)
- `dSig0_r`       — δΣ₀(r)
- `dSig0_prime_r` — δΣ₀'(r) wrt CONFORMAL TIME (d/dη = ℋ d/dlna)
- `D0_r`          — modified growth D₀(r) (normalized like the table's D)
- `f0_r`          — modified growth rate f₀(r) = -(1+z) dD₀/dz = dlnD₀/dlna
- `is_gr`         :: Bool      — true when mu0 == 0 && Sigma0 == 0
"""
struct MGModel
    mu0::Float64
    Sigma0::Float64
    OmLambda::Float64
    dmu0_r::Function
    dSig0_r::Function
    dSig0_prime_r::Function
    dmu0_prime_r::Function
    D0_r::Function
    f0_r::Function
    is_gr::Bool
end

# -----------------------------------------------------------------------------
# δμ₀(z), δΣ₀(z) as pure-z helpers (used in tests / standalone evaluation).
# -----------------------------------------------------------------------------
"δμ₀ given μ₀, Ω_de(z), Ω_Λ."
@inline delta_mu0_z(mu0::Float64, Ode::Float64, OmLambda::Float64) = (mu0 / OmLambda) * Ode

"δΣ₀ given Σ₀, Ω_de(z), Ω_Λ."
@inline delta_Sig0_z(Sigma0::Float64, Ode::Float64, OmLambda::Float64) = (Sigma0 / OmLambda) * Ode

# -----------------------------------------------------------------------------
# Modified growth solver (in ln a), returning splines D0(lna), f0(lna).
# -----------------------------------------------------------------------------
"""
    _solve_growth_lna(cfns, mu0, OmLambda; a_init, n_steps) -> (lna, D0, f0)

Integrate the redshift-only modified growth ODE in x = ln a:

    D'' + (2 + dlnH/dlna) D' - (3/2) Ω_m(a) [1 + δμ₀(a)] D = 0          (')=d/dx

This is Eq. (3.34) rewritten in ln a (the ℋ-friction term becomes
(2 + dlnH/dlna)·D' after converting d/dη → ℋ d/dx and dividing by ℋ²).
Both Ω_m(a) and dlnH/dlna come from the supplied expansion history H(a),
so the only MG ingredient is the Poisson factor [1+δμ₀].

Growing-mode initial conditions at high redshift (matter domination):
D ∝ a, so D(a_i)=a_i, D'(a_i)=a_i in ln a (since dD/dx = a dD/da = a).

Returns (lna grid, D0 on grid, f0=dlnD0/dx on grid).  D0 is left UNnormalized
here; the caller rescales to match the table's growth normalization.
"""
function _solve_growth_lna(cfns, mu0::Float64, OmLambda::Float64;
                           a_init::Float64 = 1.0e-3, n_steps::Int = 40_000)
    # Build Ω_m(a) and H(a) interpolants in ln a from the table (r-indexed).
    # Sample the table densely over its valid r-range, convert to (a, Ω_m, H).
    rmin, rmax = 1.0, 1.0e4
    rs = exp10.(range(log10(rmin), log10(rmax), length = 2_000))
    pts = Tuple{Float64,Float64,Float64}[]   # (ln a, Ω_m, H)
    for r in rs
        a = cfns.far(r); H = cfns.fHr(r); Om = cfns.fOmr(r)
        (a > 0 && H > 0) || continue
        push!(pts, (log(a), Om, H))
    end
    sort!(pts, by = first)
    # strictly-increasing ln a knots
    la = Float64[]; Omv = Float64[]; Hv = Float64[]
    for (l, o, h) in pts
        if isempty(la) || l > last(la) + 1e-12
            push!(la, l); push!(Omv, o); push!(Hv, h)
        end
    end
    length(la) >= 4 || error("MGCosmo: too few (ln a) knots from cosmology table.")

    Om_spl   = Spline1D(la, Omv, k = 3)
    lnH_spl  = Spline1D(la, log.(Hv), k = 3)
    la_lo, la_hi = first(la), last(la)

    Om_at(x)        = Om_spl(clamp(x, la_lo, la_hi))
    dlnHdx_at(x)    = derivative(lnH_spl, clamp(x, la_lo, la_hi))
    dmu0_at(x)      = (mu0 / OmLambda) * (1.0 - Om_at(x))   # Ω_de = 1 - Ω_m (flat)

    # RK4 over x = ln a from ln(a_init) to 0 (a=1).
    x0 = log(a_init); x1 = 0.0
    xs = collect(range(x0, x1, length = n_steps))
    h  = xs[2] - xs[1]
    D  = Vector{Float64}(undef, n_steps)
    Dp = Vector{Float64}(undef, n_steps)
    D[1]  = a_init          # D ∝ a in matter domination
    Dp[1] = a_init          # dD/dx = a (since D=a ⇒ dD/dx = dD/dlna = a)

    f_ode(x, D_, Dp_) = begin
        Dpp = -(2.0 + dlnHdx_at(x)) * Dp_ + 1.5 * Om_at(x) * (1.0 + dmu0_at(x)) * D_
        (Dp_, Dpp)
    end

    @inbounds for i in 1:(n_steps - 1)
        x = xs[i]
        k1D, k1Dp = f_ode(x,           D[i],               Dp[i])
        k2D, k2Dp = f_ode(x + h/2,     D[i] + h/2*k1D,     Dp[i] + h/2*k1Dp)
        k3D, k3Dp = f_ode(x + h/2,     D[i] + h/2*k2D,     Dp[i] + h/2*k2Dp)
        k4D, k4Dp = f_ode(x + h,       D[i] + h*k3D,       Dp[i] + h*k3Dp)
        D[i+1]  = D[i]  + h/6*(k1D  + 2k2D  + 2k3D  + k4D)
        Dp[i+1] = Dp[i] + h/6*(k1Dp + 2k2Dp + 2k3Dp + k4Dp)
    end

    f0 = Dp ./ D   # f = dlnD/dlna
    return xs, D, f0
end

"""
    build_mg_model(cfns; mu0, Sigma0) -> MGModel

Construct the MG model for the given (mu0, Sigma0) on top of the GR cosmology
`cfns` (a `cosmofns.cosmofn`).  Solves the modified growth ODE and builds the
r-indexed δμ₀, δΣ₀, δΣ₀'(d/dη), D₀, f₀ callables.

The modified D₀ is normalized to agree with the table's GR growth D at the
LOWEST redshift sampled (deep matter domination), where MG is negligible —
this fixes the same normalization convention the rest of the pipeline uses for
the GR D (the power-spectrum normalization is carried separately in P(k), so
only the SHAPE/relative D(z) matters for the kernels).
"""
function build_mg_model(cfns; mu0::Float64, Sigma0::Float64)::MGModel
    OmLambda = 1.0 - cfns.fOmr(_r_at_amax(cfns))   # Ω_Λ = 1 - Ω_{m,0}

    is_gr = (mu0 == 0.0 && Sigma0 == 0.0)

    # δμ₀(r), δΣ₀(r): Ω_de(z) = 1 - Ω_m(z) (flat, radiation neglected).
    dmu0_r  = r -> (mu0    / OmLambda) * (1.0 - cfns.fOmr(r))
    dSig0_r = r -> (Sigma0 / OmLambda) * (1.0 - cfns.fOmr(r))

    # δΣ₀'(r) wrt CONFORMAL TIME η:  δΣ₀'(η) = ℋ · d(δΣ₀)/dlna.
    # δΣ₀(a) = (Σ₀/Ω_Λ)(1 - Ω_m(a))  ⇒  d/dlna = -(Σ₀/Ω_Λ) dΩ_m/dlna.
    # Build dΩ_m/dlna from a spline of Ω_m vs ln a, then multiply by ℋ = aH.
    dSig0_prime_r = _make_dSig0_prime(cfns, Sigma0, OmLambda)
    # δμ₀'(η) = ℋ·d(δμ₀)/dlna, with δμ₀(a) = (μ₀/Ω_Λ)(1-Ω_m(a)).
    dmu0_prime_r  = _make_dSig0_prime(cfns, mu0, OmLambda)

    # Modified growth on ln a, then map to r via a(r) → ln a.
    lna, D0_lna, f0_lna = _solve_growth_lna(cfns, mu0, OmLambda)
    D0_spl = Spline1D(lna, D0_lna, k = 3)
    f0_spl = Spline1D(lna, f0_lna, k = 3)
    la_lo, la_hi = first(lna), last(lna)

    # Normalize D₀ to the table's GR D at the earliest (highest-z) sample,
    # where MG ≈ 0.  This keeps D₀(z) on the same footing as cfns.fDr(z).
    r_norm = _r_at_amin(cfns)
    a_norm = cfns.far(r_norm)
    x_norm = clamp(log(a_norm), la_lo, la_hi)
    D0_at_norm = D0_spl(x_norm)
    D_GR_at_norm = cfns.fDr(r_norm)
    norm = (D0_at_norm != 0.0) ? (D_GR_at_norm / D0_at_norm) : 1.0

    D0_r = r -> norm * D0_spl(clamp(log(cfns.far(r)), la_lo, la_hi))
    f0_r = r -> f0_spl(clamp(log(cfns.far(r)), la_lo, la_hi))

    # In the GR limit, fall back to the table's own D and f exactly (avoids any
    # tiny ODE-vs-table mismatch propagating into "GR" reference spectra).
    if is_gr
        D0_r = r -> cfns.fDr(r)
        f0_r = r -> cfns.ffr(r)
    end

    return MGModel(mu0, Sigma0, OmLambda,
                   dmu0_r, dSig0_r, dSig0_prime_r, dmu0_prime_r, D0_r, f0_r, is_gr)
end

# δΣ₀'(η) closure: ℋ · d(δΣ₀)/dlna,  with d(δΣ₀)/dlna = -(Σ₀/Ω_Λ) dΩ_m/dlna.
function _make_dSig0_prime(cfns, Sigma0::Float64, OmLambda::Float64)
    rmin, rmax = 1.0, 1.0e4
    rs = exp10.(range(log10(rmin), log10(rmax), length = 2_000))
    pts = Tuple{Float64,Float64}[]
    for r in rs
        a = cfns.far(r); Om = cfns.fOmr(r)
        a > 0 || continue
        push!(pts, (log(a), Om))
    end
    sort!(pts, by = first)
    la = Float64[]; Omv = Float64[]
    for (l, o) in pts
        if isempty(la) || l > last(la) + 1e-12
            push!(la, l); push!(Omv, o)
        end
    end
    Om_spl = Spline1D(la, Omv, k = 3)
    la_lo, la_hi = first(la), last(la)
    return function (r)
        x   = clamp(log(cfns.far(r)), la_lo, la_hi)
        dOm = derivative(Om_spl, x)                 # dΩ_m/dlna
        aH  = cfns.far(r) * cfns.fHr(r)             # conformal ℋ = aH  [h/Mpc]
        return aH * (-(Sigma0 / OmLambda) * dOm)    # δΣ₀'(η)
    end
end

# Helpers to find the r at the extreme scale factors covered by the table.
function _r_at_amax(cfns)   # a → 1 (z → 0): smallest r in the usable range
    rs = exp10.(range(log10(1.0), log10(1.0e4), length = 4_000))
    best_r = rs[1]; best_a = cfns.far(rs[1])
    for r in rs
        a = cfns.far(r)
        if a > best_a && a <= 1.0 + 1e-9
            best_a = a; best_r = r
        end
    end
    return best_r
end
function _r_at_amin(cfns)   # smallest a (highest z) in the usable range
    rs = exp10.(range(log10(1.0), log10(1.0e4), length = 4_000))
    best_r = rs[end]; best_a = cfns.far(rs[end])
    for r in rs
        a = cfns.far(r)
        if a > 0 && a < best_a
            best_a = a; best_r = r
        end
    end
    return best_r
end

end # module MGCosmo
