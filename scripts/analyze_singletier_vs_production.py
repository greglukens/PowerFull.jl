#!/usr/bin/env python3
"""
Compare Option A (nR=4097 single-tier) vs current production-equivalent (T0+Mid stitched).

Targets:
1. Off-diagonal pair Cl kink reduction (where fNL signal lives)
2. Per-array L2 ratio at boundary ell (was kink at ell=20→21)
3. Overall convergence improvement

Inputs:
- /gpfs/djeong/PowerFull.jl/production/results/test_singletier_nR4097/cl/Cl_nR4097.h5
- /gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl/Cl_d0p0015.h5  (Mid finest reference)
- /gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl/Cl_d0p005_full.h5 (T0 full)
"""
import os, sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CL_4097 = "/gpfs/djeong/PowerFull.jl/production/results/test_singletier_nR4097/cl/Cl_nR4097.h5"
CL_MID  = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl/Cl_d0p0015.h5"
CL_T0   = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl/Cl_d0p005_full.h5"

OUTDIR = "/gpfs/djeong/PowerFull.jl/production/results/test_singletier_nR4097/figs"
os.makedirs(OUTDIR, exist_ok=True)

# Off-diagonal (fNL-relevant) pairs + dominant for sanity
PAIRS = [
    ("86_88", 0.7,  "off-diag"),
    ("103_106", 0.31, "off-diag"),
    ("277_280", 0.25, "off-diag"),
    ("136_139", 0.5, "near-zero-cross"),
    ("4_5", 0.93, "near-auto"),
    ("4_4", 1.00, "auto"),
]


def load(path, pair):
    if not os.path.exists(path):
        return None, None
    with h5py.File(path, "r") as f:
        ell = f[f"pairs/{pair}/ell"][:]
        cl = f[f"pairs/{pair}/Cl"][:]
    return ell, cl


def main():
    if not os.path.exists(CL_4097):
        sys.exit(f"Cl_nR4097.h5 not found yet — wait for step3 to finish.\n  Path: {CL_4097}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for idx, (pair, R, kind) in enumerate(PAIRS):
        ax = axes[idx]
        ell_4097, cl_4097 = load(CL_4097, pair)
        ell_mid,  cl_mid  = load(CL_MID,  pair)
        ell_t0,   cl_t0   = load(CL_T0,   pair)

        # Reference = nR4097 (most converged)
        if ell_4097 is None: continue

        # Production-equivalent stitched: T0 ell≤20, Mid ell≥21
        ell_prod = np.concatenate([ell_t0[ell_t0 <= 20], ell_mid[ell_mid >= 21]]) if ell_t0 is not None else ell_mid
        cl_prod = np.concatenate([cl_t0[ell_t0 <= 20], cl_mid[ell_mid >= 21]]) if ell_t0 is not None else cl_mid

        # rel diff vs nR4097
        rel_prod = np.full(len(ell_4097), np.nan)
        rel_mid = np.full(len(ell_4097), np.nan)
        for i, e in enumerate(ell_4097):
            j = np.where(ell_prod == e)[0]
            if len(j) > 0 and abs(cl_4097[i]) > 1e-25:
                rel_prod[i] = (cl_prod[j[0]] - cl_4097[i]) / cl_4097[i]
            j = np.where(ell_mid == e)[0]
            if len(j) > 0 and abs(cl_4097[i]) > 1e-25:
                rel_mid[i] = (cl_mid[j[0]] - cl_4097[i]) / cl_4097[i]

        ax.semilogx(ell_4097, rel_prod, '-', color='darkred', label='Production stitch (T0+Mid) − ref', lw=1.2)
        ax.semilogx(ell_4097, rel_mid,  '--', color='darkblue', label='Mid d0p0015 − ref', lw=1.0, alpha=0.7)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axvline(20, color='gray', ls=':', alpha=0.5, label='_nolegend_')
        ax.set_title(f"{pair}  R={R}  ({kind})", fontsize=11)
        ax.set_xlabel("ell")
        ax.set_ylabel(r"rel diff vs nR4097")
        ax.set_ylim(-0.05, 0.05)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle("Option A (nR=4097 single-tier) reference: Cl rel diff for production builds",
                 fontsize=13)
    plt.tight_layout()
    out = os.path.join(OUTDIR, "singletier_vs_production_kink.pdf")
    plt.savefig(out, dpi=120)
    print(f"saved {out}")

    # Print summary table at ell=20 boundary
    print("\n=== Cl kink at ell=20 (after Option A patch) ===")
    print(f"{'pair':10s} {'R':6s}  {'4097':>14s}  {'prod':>14s}  {'rel diff':>12s}")
    for (pair, R, kind) in PAIRS:
        ell_a, cl_a = load(CL_4097, pair)
        ell_t0, cl_t0 = load(CL_T0, pair)
        if ell_a is None or ell_t0 is None: continue
        ia = np.where(ell_a == 20)[0]
        it = np.where(ell_t0 == 20)[0]
        if len(ia) == 0 or len(it) == 0: continue
        v_a = cl_a[ia[0]]; v_t = cl_t0[it[0]]
        rel = (v_t - v_a)/v_a if abs(v_a) > 1e-25 else np.nan
        print(f"{pair:10s} {R:.2f}  {v_a:+.4e}  {v_t:+.4e}  {rel:+.2e}")


if __name__ == "__main__":
    main()
