# LucasPatch.jl — direct (non-FFTLog) computation of base functions in
# regions where TwoFASTpp's 9-term reconstruction loses precision to
# catastrophic cancellation. Currently provides the (p=0, j=2, jp=2)
# slice at ell=2 — the worst hot-spot at production q values.
#
# Algorithm: integrate
#   w^0_{2,22}(r1, r2) = (2/π) ∫ k² P(k) j_2''(kr1) j_2''(kr2) dk
# directly via QuadGK using closed-form j_2'' (no spherical Bessel calls).
# Slower than the Lucas/quadosc method (~2 sec/point) but self-contained
# inside TwoFASTpp and bypasses the 9-term cancellation entirely.
#
# For faster computation (~30× speedup) use the standalone
# `scripts/lucas_w22_ell2_patch.slurm` which calls Henry Gebhardt's
# Quad_jar_jbt (Lucas 1995 quadosc) external module.

module LucasPatch

using QuadGK
using Distributed
using Printf

export lucas_patch_w_0_22_ell2, j2pp

# Closed-form j_2''(z) = (1/z - 17/z³ + 36/z⁵) sin(z) + (5/z² - 36/z⁴) cos(z)
# Series expansion for small z to avoid 1/z^n cancellation.
@inline function j2pp(z::Float64)::Float64
    az = abs(z)
    if az < 0.1
        z2 = z*z
        return 2/15 - (2/35)*z2 + (1/252)*z2*z2 - (1/8910)*z2*z2*z2
    else
        s, c = sincos(z)
        z2 = z*z
        z3 = z*z2
        z4 = z2*z2
        z5 = z*z4
        return (1/z - 17/z3 + 36/z5)*s + (5/z2 - 36/z4)*c
    end
end

const TWO_OVER_PI = 2.0 / pi

"""
    lucas_patch_w_0_22_ell2(rr, pk_func; kmin=1e-5, kmax=1e3, rtol=1e-7)

Compute w^0_{ell=2, j=2, jp=2}(r1, r2) on the symmetric (rr × rr) grid
via direct QuadGK integration of `(2/π) k² P(k) j_2''(kr1) j_2''(kr2) dk`.

Bypasses the 9-term FFTLog reconstruction and its cancellation problem
at low ell with j=jp=2.

Arguments:
- `rr`: r-grid (Vector{Float64})
- `pk_func`: callable `k -> P(k)` (any user-supplied power spectrum)
- `kmin`, `kmax`: integration range (default `[1e-5, 1e3]` h/Mpc)
- `rtol`: QuadGK relative tolerance (default 1e-7)

Returns: Matrix{Float64} of shape `(length(rr), length(rr))`.

Parallelized via Distributed if `nworkers() > 1`.  The module and its
dependencies must be loaded on every worker before calling, otherwise
`pmap` will hit `UndefVarError` on the remotes:

    using Distributed
    addprocs(24)
    @everywhere using QuadGK
    @everywhere include(joinpath(@__DIR__, "LucasPatch.jl"))  # adjust path
    @everywhere using .LucasPatch

Cost: ~2 sec/point on single core. For 1155×1155/2 ≈ 670K unique points
and 24 workers, total ~9 hours. For production use, prefer the standalone
script `scripts/lucas_w22_ell2_patch.slurm` which uses Lucas's quadosc
algorithm and is ~30× faster.
"""
function lucas_patch_w_0_22_ell2(rr::Vector{Float64}, pk_func;
                                  kmin::Float64=1e-5, kmax::Float64=1e3,
                                  rtol::Float64=1e-7)
    Nr = length(rr)
    @printf("[LucasPatch] Computing w_0_2_2 ell=2 on %d×%d grid (%d unique points)\n",
            Nr, Nr, Nr*(Nr+1)÷2)

    function w22_at(r1::Float64, r2::Float64)::Float64
        f_lnk(lnk) = begin
            k = exp(lnk)
            TWO_OVER_PI * k^3 * pk_func(k) * j2pp(k*r1) * j2pp(k*r2)
        end
        val, _ = QuadGK.quadgk(f_lnk, log(kmin), log(kmax);
                                rtol=rtol, atol=1e-30, order=15)
        return val
    end

    ij_list = [(i, j) for i in 1:Nr, j in 1:Nr if i <= j]
    t0 = time()
    if nworkers() > 1
        results = pmap(ij_list; batch_size=200) do ij
            i, j = ij
            (i, j, w22_at(rr[i], rr[j]))
        end
    else
        results = [(ij[1], ij[2], w22_at(rr[ij[1]], rr[ij[2]])) for ij in ij_list]
    end
    @printf("[LucasPatch] %.1f min total\n", (time() - t0)/60)

    M = zeros(Float64, Nr, Nr)
    for (i, j, v) in results
        M[i, j] = v
        M[j, i] = v
    end
    return M
end

end # module LucasPatch
