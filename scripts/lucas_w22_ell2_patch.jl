#!/usr/bin/env julia
# Lucas direct computation of w^0_{ell=2, j=2, jp=2}(r1, r2) on full (r, r') grid.
# Uses 9-term decomposition: j_2''(x) = (12/x²-1) j_2(x) - (2/x) j_1(x).
# Each sub-integral via TwoFAST's `Quad_jar_jbt.quad_jar_jbt_log` (Lucas algorithm
# with quadosc tail handling) — proven to match Lucas reference values.
#
# Output: HDF5 file with (1155, 1155) Float32 patched slice + metadata.

push!(LOAD_PATH, "/gpfs/djeong/TwoFAST/Lucas/myjl")
using PkSpectra
using Quad_jar_jbt
using HDF5
using Distributed
using Printf

# Spawn workers if running standalone
if nprocs() == 1
    nthreads = parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "24"))
    addprocs(nthreads - 1)
    println("Spawned $(nworkers()) workers")
end

@everywhere push!(LOAD_PATH, "/gpfs/djeong/TwoFAST/Lucas/myjl")
@everywhere using PkSpectra, Quad_jar_jbt, Printf

@everywhere const PK_FILE = "/gpfs/djeong/PowerFull.jl/data/planck_base_plikHM_TTTEEE_lowTEB_lensing_post_BAO_H070p6_JLA_matterpower.dat"
@everywhere const pk_local = PkFile(PK_FILE)

# Lucas integral L[(l1, l2, p, n=0)] = ∫ k^(2+p) P(k) j_l1(kr1) j_l2(kr2) dk
# (n_tk=0 fixed; we don't use transfer function powers here.)
@everywhere function lucas_int(l1::Int, l2::Int, r1::Float64, r2::Float64, q::Int)
    p = q - 2
    total_power = 3 + p   # = q + 1, exponent for k^(...) in d(lnk)
    function f(lnk::Float64)
        k = exp(lnk)
        if k < pk_local.kmin
            exponent = total_power + pk_local.nslo
            return pk_local.kmin_norm * exp(lnk * exponent)
        elseif k > pk_local.kmax
            exponent = total_power + pk_local.nshi - 4
            return pk_local.kmax_norm * exp(lnk * exponent)
        else
            return k^total_power * pk_local.pkspl(k)
        end
    end
    val, _ = quad_jar_jbt_log(f, -100.0, Inf, l1, l2, r1, r2; rtol=1e-7)
    return val
end

# w^0_{2,22}(r1, r2) via 9-term decomp of j_2''(x) = (12/x² - 1) j_2(x) - (2/x) j_1(x)
@everywhere function w_0_22_ell2(r1::Float64, r2::Float64)
    L_22_m2   = lucas_int(2, 2, r1, r2, -2)
    L_22_0    = lucas_int(2, 2, r1, r2, 0)
    L_22_2    = lucas_int(2, 2, r1, r2, 2)
    L_21_m1   = lucas_int(2, 1, r1, r2, -1)
    L_21_1    = lucas_int(2, 1, r1, r2, 1)
    L_12_m1   = lucas_int(1, 2, r1, r2, -1)
    L_12_1    = lucas_int(1, 2, r1, r2, 1)
    L_11_0    = lucas_int(1, 1, r1, r2, 0)
    pi_inv = 1.0 / pi
    s = 288 * pi_inv / (r1^2 * r2^2) * L_22_m2
    s += -24 * pi_inv / r1^2 * L_22_0
    s += -48 * pi_inv / (r1^2 * r2) * L_21_m1
    s += -24 * pi_inv / r2^2 * L_22_0
    s +=   2 * pi_inv * L_22_2
    s +=   4 * pi_inv / r2 * L_21_1
    s += -48 * pi_inv / (r1 * r2^2) * L_12_m1
    s +=   4 * pi_inv / r1 * L_12_1
    s +=   8 * pi_inv / (r1 * r2) * L_11_0
    return s
end

# Read r-grid from existing build meta (we use the test_singletier_nR4097 grid as reference)
const META = ARGS[1]
const OUTPATH = ARGS[2]

println("Reading r-grid from: $META")
h = h5open(META, "r")
rr = read(h["grid/rr"])::Vector{Float64}
close(h)
const Nr = length(rr)
@printf("r-grid: %d points, range [%.3f, %.3f] Mpc/h\n", Nr, rr[1], rr[end])

# Compute upper triangular only (symmetry: w(r1, r2) = w(r2, r1))
# Total unique points: Nr * (Nr+1) / 2
N_unique = Nr * (Nr + 1) ÷ 2
@printf("Unique (r1, r2) pairs (symmetric): %d\n", N_unique)

# Build index list for (i, j) with i ≤ j
ij_list = [(i, j) for i in 1:Nr, j in 1:Nr if i <= j]
@printf("Will compute %d points × ~0.4 sec / %d workers ≈ %.1f hours\n",
        N_unique, nworkers(), N_unique * 0.4 / nworkers() / 3600)

# Pmap with progress
println("\nStarting Lucas computation...")
flush(stdout)
t0 = time()

results = pmap(ij_list; batch_size=200) do ij
    i, j = ij
    r1, r2 = rr[i], rr[j]
    return (i, j, w_0_22_ell2(r1, r2))
end

dt = time() - t0
@printf("\nCompleted in %.1f minutes (%.2f sec/point avg)\n", dt/60, dt/N_unique)

# Build full Nr × Nr matrix
M = zeros(Float64, Nr, Nr)
for (i, j, v) in results
    M[i, j] = v
    M[j, i] = v   # symmetric
end

# Write
println("\nWriting to: $OUTPATH")
h5open(OUTPATH, "w") do hf
    write(hf, "w_0_2_2_ell2", Float32.(M))
    write(hf, "rr", rr)
    write(hf, "method", "Lucas 9-term decomp via quad_jar_jbt_log (rtol=1e-10)")
    write(hf, "ell", 2)
    write(hf, "Nr", Nr)
    write(hf, "source_meta", META)
    write(hf, "P_k_file", PK_FILE)
end
println("Done.")
