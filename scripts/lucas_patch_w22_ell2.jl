#!/usr/bin/env julia
# ==============================================================================
# Lucas BLAS patch for w^p=0_{ell=2, j=2, jp=2}(r1, r2)
#
# Computes
#   w_0_22(r1, r2) = (2/π) ∫ k² P(k) j_2''(k r1) j_2''(k r2) dk
# directly via Lucas, vectorized as matrix product M = A · Aᵀ where
#   A[i, n] = √(weight_n · K_n) · j_2''(k_n · r_i),  K_n = (2/π) k_n² P(k_n)
# A is (Nr × Nk); single BLAS dgemm gives the (Nr × Nr) symmetric slice.
#
# Output: HDF5 file with the patched (Nr × Nr) matrix and metadata.
# This script does NOT modify the build; that is a separate, gated step.
# ==============================================================================

using HDF5
using SpecialFunctions
using LinearAlgebra
using Dierckx
using Printf
using Statistics

const TWO_OVER_PI = 2.0 / pi

# ---- argparse (positional) ---------------------------------------------------
function usage()
    println("usage: julia lucas_patch_w22_ell2.jl <build_meta.h5> <matterpower.dat> [Nk]")
    println("  build_meta.h5   : <prefix>_meta.h5 (e.g. ClGR_d0p0015_meta.h5)")
    println("  matterpower.dat : two-column k[h/Mpc]  P(k)[(Mpc/h)^3]")
    println("  Nk              : number of log-spaced k-points (default 8192)")
    exit(1)
end
length(ARGS) < 2 && usage()
const meta_path = ARGS[1]
const pk_path   = ARGS[2]
const Nk        = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 8192

# ---- 1. j_l''(x) for ell = 2 -------------------------------------------------
# j_l''(x) = ((l+1)(l+2)/x² - 1) j_l(x) - (2/x) j_{l-1}(x)
# For ell = 2: j_2''(x) = (12/x² - 1) j_2(x) - (2/x) j_1(x).
# Small-x: j_2(x) ≈ x²/15 (1 - x²/14), j_1(x) ≈ x/3 (1 - x²/10);
#   limit j_2''(x) → 2/15 - 3x²/(7·15) for x → 0.
@inline function j2pp(x::Float64)::Float64
    if x < 1e-3
        # Taylor through O(x²) — well within Float64 precision for x < 1e-3.
        return (2.0/15.0) * (1.0 - 3.0 * x^2 / 14.0)
    end
    j2 = sphericalbesselj(2, x)
    j1 = sphericalbesselj(1, x)
    return (12.0 / x^2 - 1.0) * j2 - (2.0 / x) * j1
end

# ---- 2. Read PowerFull meta + grid -------------------------------------------
hm = h5open(meta_path, "r")
const rr        = read(hm["grid/rr"])::Vector{Float64}
const ell_vals  = read(hm["grid/ell_values"])::Vector{Int}
const ell_ranges = read(hm["metadata/ell_ranges"])::Matrix{Int}
const part_files = read(hm["part_files"])
const Nr = length(rr)
close(hm)

@assert ell_vals[1] == 2 "first ell expected to be 2 (got $(ell_vals[1]))"
const ell_idx_2_global = 1                         # index in flat ell list
const part_idx_for_ell2 = 1                        # part containing ell=2
const ell_idx_2_local = 1                          # slice within that part
@printf("Nr=%d   r-grid range = [%.3f, %.3f] Mpc/h   dlnr=%.6e\n",
        Nr, rr[1], rr[end], log(rr[2] / rr[1]))

# ---- 3. P(k) spline (no extrapolation outside file range) --------------------
data = open(pk_path) do io
    lines = filter(l -> !startswith(strip(l), '#') && !isempty(strip(l)), readlines(io))
    [parse.(Float64, split(l)) for l in lines]
end
kk_pk = [d[1] for d in data]; pk_pk = [d[2] for d in data]
const pk_spl = Spline1D(log.(kk_pk), log.(pk_pk); k=3, bc="error")
@inline P_of_k(k::Float64) = exp(evaluate(pk_spl, log(k)))

# ---- 4. k-grid: log-spaced, kmin/kmax matching TwoFAST defaults --------------
const kmin = 1.0e-5
const kmax = 1.0e3
const lnk_grid = collect(range(log(kmin), log(kmax); length=Nk))
const k_grid   = exp.(lnk_grid)
const dlnk     = lnk_grid[2] - lnk_grid[1]
# Composite trapezoidal weights for ∫ f dk = ∫ f·k d(ln k)
const w_quad = fill(dlnk, Nk); w_quad[1] *= 0.5; w_quad[end] *= 0.5

# K(k) = (2/π) k² P(k); positive everywhere (P(k) ≥ 0).
const K_arr = TWO_OVER_PI .* (k_grid .^ 2) .* P_of_k.(k_grid)
const sq_factor = sqrt.(w_quad .* k_grid .* K_arr)   # √(weight · K)
@printf("Nk=%d   k ∈ [%.2e, %.2e]  dlnk=%.4e\n", Nk, kmin, kmax, dlnk)

# ---- 5. Build A[i, n] = √(w_n · k_n · K_n) · j_2''(k_n · r_i) ----------------
println("Building A matrix ($Nr × $Nk) ...")
const A = Matrix{Float64}(undef, Nr, Nk)
@time begin
    Threads.@threads for n in 1:Nk
        kn = k_grid[n]
        sf = sq_factor[n]
        @inbounds for i in 1:Nr
            A[i, n] = sf * j2pp(kn * rr[i])
        end
    end
end

# ---- 6. M = A · Aᵀ (single BLAS dgemm) ---------------------------------------
println("Computing M = A · Aᵀ ($Nr × $Nr) via BLAS ...")
M = Matrix{Float64}(undef, Nr, Nr)
@time mul!(M, A, A')
# Symmetrize against round-off
@inbounds for j in 1:Nr, i in 1:j-1
    avg = 0.5 * (M[i, j] + M[j, i])
    M[i, j] = avg; M[j, i] = avg
end

# ---- 7. Self-convergence: recompute at 2× Nk and report relative shift -------
# Lucas direct has no cancellation (positive integrand structure); it converges
# monotonically with Nk. Use this as internal check instead of slow BigFloat.
println("\nSelf-convergence check (recompute with 2×Nk) ...")
const Nk2 = 2 * Nk
const lnk2 = collect(range(log(kmin), log(kmax); length=Nk2))
const k2 = exp.(lnk2)
const dlnk2 = lnk2[2] - lnk2[1]
const w2 = fill(dlnk2, Nk2); w2[1] *= 0.5; w2[end] *= 0.5
const K2 = TWO_OVER_PI .* (k2 .^ 2) .* P_of_k.(k2)
const sf2 = sqrt.(w2 .* k2 .* K2)
A2 = Matrix{Float64}(undef, Nr, Nk2)
@time begin
    Threads.@threads for n in 1:Nk2
        kn = k2[n]; sf = sf2[n]
        @inbounds for i in 1:Nr
            A2[i, n] = sf * j2pp(kn * rr[i])
        end
    end
end
M2 = Matrix{Float64}(undef, Nr, Nr)
@time mul!(M2, A2, A2')
@inbounds for j in 1:Nr, i in 1:j-1
    avg = 0.5 * (M2[i, j] + M2[j, i]); M2[i, j] = avg; M2[j, i] = avg
end

println("\nSpot-check at known (r1, r2) points:")
for (r1, r2, label) in [(3602.80, 2053.53, "R57 (cross)"),
                          (3602.80, 3234.19, "R90 (cross)"),
                          (1000.0, 1000.0,  "auto-pair r=1000"),
                          (3602.80, 3602.80,  "auto-pair r1=r1")]
    i = argmin(abs.(rr .- r1)); j = argmin(abs.(rr .- r2))
    v1 = M[i, j]; v2 = M2[i, j]
    rel = (v2 - v1) / abs(v2)
    @printf("  %-22s  rr[%d]=%.3f rr[%d]=%.3f   M(Nk=%d)=%+.6e   M(Nk=%d)=%+.6e   rel shift=%+.2e\n",
            label, i, rr[i], j, rr[j], Nk, v1, Nk2, v2, rel)
end

# ---- 8. Compare to PowerFull's existing w_0_2_2 ell=2 slice ------------------
build_dir = dirname(meta_path)
prefix = replace(basename(meta_path), "_meta.h5" => "")
part1 = joinpath(build_dir, "$(prefix)_part_$(lpad(part_idx_for_ell2, 3, '0')).h5")
println("\nReading PowerFull's w_0_2_2 ell=2 slice from: $part1")
hp = h5open(part1, "r")
pf_slice = read(hp["base/w_0_2_2"])[ell_idx_2_local, :, :]   # 1155 × 1155 Float32
close(hp)
pf_slice_f64 = Float64.(pf_slice)

diff = M .- pf_slice_f64
absM = abs.(M)
denom = max.(absM, eps(Float64))
rel = abs.(diff) ./ denom
println("\nDiff stats (Lucas BLAS − PowerFull on ell=2 slice):")
@printf("  L2 norm of diff / L2 norm of M     = %.3e\n",
        sqrt(sum(diff.^2)) / sqrt(sum(absM.^2)))
@printf("  max |diff|                          = %.3e (PowerFull |val| %.3e at that point)\n",
        maximum(abs, diff), absM[argmax(abs.(diff))])
@printf("  median rel|diff| (where |M|>1e-15) = %.3e\n",
        median(rel[absM .> 1e-15]))
@printf("  mean rel|diff| (where |M|>1e-15)   = %.3e\n",
        mean(rel[absM .> 1e-15]))
@printf("  90th pctl rel|diff|                = %.3e\n",
        quantile(rel[absM .> 1e-15], 0.9))

# ---- 9. Save patched matrix --------------------------------------------------
out_path = joinpath(build_dir, "$(prefix)_lucas_w22_ell2.h5")
println("\nWriting patched slice to: $out_path")
h5open(out_path, "w") do h
    write(h, "w_0_2_2_ell2", Float32.(M))   # match storage dtype
    write(h, "rr", rr)
    write(h, "ell", 2)
    write(h, "Nk", Nk)
    write(h, "kmin", kmin)
    write(h, "kmax", kmax)
    write(h, "source_meta", meta_path)
    write(h, "source_pk", pk_path)
end
println("Done.")
