#!/usr/bin/env julia
# Read base w_0_2_2 at (r1, r2) from each dlnR cache, compare to Lucas direct.
using PowerFull, JLD2, Dierckx, Printf

const DATADIR = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump"
const Nr, nR = 4096, 2049

# Two test (r1, r2):
const CASES = [
    ("R57", 3602.80, 2053.53,  -4.5534e-8),  # Lucas direct ell=2 w_0_22
    ("R90", 3602.80, 3234.19,  -8.3695e-7),
]

# T0 set + Mid set
const SET_T0  = [(0.004,  2, 30, "T0"), (0.005, 2, 30, "T0"), (0.006, 2, 30, "T0")]
const SET_MID = [(0.0015, 2, 210, "Mid"), (0.002, 2, 210, "Mid"), (0.0025, 2, 210, "Mid")]

for (tag, r1, r2, lucas) in CASES
    println("\n=== $tag  r1=$r1  r2=$r2  Lucas-direct=$lucas ===")
    println("  dlnR     R-grid full     Rcut [Rmin, Rmax]    R-interp value      |val-Lucas|/|Lucas|")

    for (dlnR, ellmin, ellmax, label) in vcat(SET_T0, SET_MID)
        # Force reload (LOOKUP_TABLE caches across calls).
        PowerFull.clear_cache!()
        try
            data = PowerFull.wpljjprime(0, 2, 2, 0, 2;
                Nr=Nr, nR=nR, dlnR=dlnR,
                ellmin=ellmin, ellmax=ellmax, datadir=DATADIR)
            # data has wrRl (Nr, nR) and rr, RR
            rr = data.rr
            RR = data.RR
            wrRl = data.wrRl  # already at ell=2 (single ell load)

            # Find rr[i] closest to r1
            ir1 = argmin(abs.(rr .- r1))
            ir2 = argmin(abs.(rr .- r2))
            R_target = r2 / r1
            rr_close1, rr_close2 = rr[ir1], rr[ir2]

            # Interpolate w_integrand[ir1, :] at R = r2/rr[ir1] via R-spline
            ln_RR = log.(RR)
            ln_R_target = log(R_target)
            spl = Spline1D(ln_RR, wrRl[ir1, :]; k=3, bc="zero")
            w_val = spl(ln_R_target)

            ratio = (abs(w_val) - abs(lucas)) / abs(lucas)
            R_min, R_max = exp(-(nR-1)/2 * dlnR), exp((nR-1)/2 * dlnR)
            @printf("  %-7.4f  [%.4g, %.4g]   no-cut              %+.4e        %+.2f%%\n",
                    dlnR, R_min, R_max, w_val, ratio*100)
        catch e
            println("  $dlnR  failed: $e")
        end
    end
end
