# build_and_export_mg_kernels.jl
# =============================================================================
#
# MG override for Step 2 cosmology kernels.  `include()` this in the MG build
# driver AFTER `cosmofns` and `MGCosmo` are in scope.  Provides MG versions of
# the line-of-sight integrand kernels fs/ft/fl1 and the array builder
# `compute_kernel_arrays`, which differ from GR only by the factors
#
#     ISW  (s):  a³H³Ω · { [f₀-1][1+δΣ₀] + δΣ₀'/ℋ } · D₀      (Eq. I_isw_00)
#     TD   (t):  a²H²Ω · [1+δΣ₀] · D₀                          (Eq. I_td_00)
#     lens (l):  t-kernel / r                                  (Eq. I_kappa_00)
#     and the s/t/l prefactor 3/D(r) uses the MODIFIED D₀(r).
#
# When the MGModel is GR (mu0=Sigma0=0), δΣ₀=δΣ₀'=0 and D₀=D_GR, f₀=f_GR, so
# these reduce EXACTLY to the GR fs/ft/fl1 and D_r — bit-identical to the
# original build (modulo the GR-limit fast-path in build_mg_model that returns
# the table D/f directly).
#
# NOTE on the conformal Hubble convention.  The GR kernels use
#     fs = a³H³Ω(f-1)D,    ft = a²H²ΩD,
# i.e. ℋ = aH appears as (aH)³ and (aH)² respectively (H here = cfns.fHr in
# h/Mpc, already H/c).  The paper's I_isw_00 carries ℋ³Ω and the ISW combination
#     {[f₀-1][1+δΣ₀] + δΣ₀'/ℋ}.   Since δΣ₀'(η) = ℋ·d(δΣ₀)/dlna (see MGCosmo),
# the term δΣ₀'/ℋ is dimensionless and ℋ-free — exactly d(δΣ₀)/dlna — so we
# implement it that way to avoid a spurious factor of ℋ.
# =============================================================================

using .MGCosmo: MGModel

# ---- MG integrand kernels (r-indexed), given a prebuilt MGModel `mg` ----------

@inline function fs_mg(r::Real, cfns, mg::MGModel)::Float64
    a  = cfns.far(r); H = cfns.fHr(r); Om = cfns.fOmr(r)
    f0 = mg.f0_r(r);  D0 = mg.D0_r(r)
    dSig0       = mg.dSig0_r(r)
    dSig0dlna   = mg.dSig0_prime_r(r) / (a * H)   # δΣ₀'(η)/ℋ = d(δΣ₀)/dlna
    isw_factor  = (f0 - 1.0) * (1.0 + dSig0) + dSig0dlna
    return a^3 * H^3 * Om * isw_factor * D0
end

@inline function ft_mg(r::Real, cfns, mg::MGModel)::Float64
    a  = cfns.far(r); H = cfns.fHr(r); Om = cfns.fOmr(r)
    D0 = mg.D0_r(r); dSig0 = mg.dSig0_r(r)
    return a^2 * H^2 * Om * (1.0 + dSig0) * D0
end

@inline fl1_mg(r::Real, cfns, mg::MGModel)::Float64 = ft_mg(r, cfns, mg) / r

"""
    compute_kernel_arrays_mg(rr, cfns, mg) -> (rf_s, rf_t, rf_l, D_r)

MG analogue of `compute_kernel_arrays`.  Returns r·f_kernel arrays and the
MODIFIED growth D₀(r) used to form inv3D = 3/D₀ in the s/t/l prefactor.
"""
function compute_kernel_arrays_mg(rr::Vector{Float64}, cfns, mg::MGModel)
    nr = length(rr)
    rf_s = Vector{Float64}(undef, nr)
    rf_t = Vector{Float64}(undef, nr)
    rf_l = Vector{Float64}(undef, nr)
    D_r  = Vector{Float64}(undef, nr)
    @inbounds for k in 1:nr
        r = rr[k]
        rf_s[k] = r * fs_mg(r, cfns, mg)
        rf_t[k] = r * ft_mg(r, cfns, mg)
        rf_l[k] = r * fl1_mg(r, cfns, mg)
        D_r[k]  = mg.D0_r(r)
    end
    return rf_s, rf_t, rf_l, D_r
end
