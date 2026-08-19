#!/usr/bin/env python3
"""dlnR test: 10-build comparison plot focused on ell~20 boundary.

Loads the 10 Cl_*.h5 outputs from production/results/test_dlnR_jump/cl/
and produces a multi-panel comparison plot showing how Cl varies with
dlnR and R-cutoff at low ell.

Reference build = d0p0015 (Mid, finest dlnR, full R range) — assumed
most accurate.  All others plotted as ratio relative to this.

Output: examples/figs/dlnRtest_comparison.pdf
"""
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

CLDIR = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl"
OUT   = "/gpfs/djeong/PowerFull.jl/examples/figs"
os.makedirs(OUT, exist_ok=True)

# Build name → (label, color, linestyle, group, dlnR)
BUILDS = {
    "d0p004":        dict(label=r"$d\ln R=0.004$ (T0)",       c="darkred",       ls="-",   grp="T0",  dlnR=0.004,  rcut=False),
    "d0p005_full":   dict(label=r"$d\ln R=0.005$ full",       c="darkblue",      ls="-",   grp="T0",  dlnR=0.005,  rcut=False),
    "d0p005_cut":    dict(label=r"$d\ln R=0.005$ cut",        c="darkblue",      ls="--",  grp="T0",  dlnR=0.005,  rcut=True),
    "d0p006_full":   dict(label=r"$d\ln R=0.006$ full",       c="darkgreen",     ls="-",   grp="T0",  dlnR=0.006,  rcut=False),
    "d0p006_cut":    dict(label=r"$d\ln R=0.006$ cut",        c="darkgreen",     ls="--",  grp="T0",  dlnR=0.006,  rcut=True),
    "d0p0015":       dict(label=r"$d\ln R=0.0015$ (Mid)",     c="darkmagenta",   ls="-",   grp="Mid", dlnR=0.0015, rcut=False),
    "d0p002_full":   dict(label=r"$d\ln R=0.002$ full",       c="darkorange",    ls="-",   grp="Mid", dlnR=0.002,  rcut=False),
    "d0p002_cut":    dict(label=r"$d\ln R=0.002$ cut",        c="darkorange",    ls="--",  grp="Mid", dlnR=0.002,  rcut=True),
    "d0p0025_full":  dict(label=r"$d\ln R=0.0025$ full",      c="darkgoldenrod", ls="-",   grp="Mid", dlnR=0.0025, rcut=False),
    "d0p0025_cut":   dict(label=r"$d\ln R=0.0025$ cut",       c="darkgoldenrod", ls="--",  grp="Mid", dlnR=0.0025, rcut=True),
}

REFERENCE = "d0p0015"  # finest Mid dlnR, full R-range — most accurate.

# Representative pairs to plot.
# s4: tracers 257-275 (19 z-bins).  s5: tracers 276-286 (11 z-bins).
# Production ell~20 kink originally seen in s4/s5 — focus there.
PAIRS = ["260_260", "260_263", "270_273", "278_278", "280_283", "283_286"]


def load_pair(name, pair):
    path = os.path.join(CLDIR, f"Cl_{name}.h5")
    with h5py.File(path, "r") as f:
        ell = f[f"pairs/{pair}/ell"][:]
        cl  = f[f"pairs/{pair}/Cl"][:]
    return ell, cl


def plot_pair(ax_top, ax_bot, pair_name):
    # Reference for ratio
    ell_ref, cl_ref = load_pair(REFERENCE, pair_name)
    ax_top.set_title(rf"pair {pair_name}", fontsize=10)

    for name, meta in BUILDS.items():
        try:
            ell, cl = load_pair(name, pair_name)
        except KeyError:
            continue
        # Top panel: |Cl|
        ax_top.plot(ell, np.abs(cl), color=meta["c"], ls=meta["ls"],
                    lw=1.4 if name == REFERENCE else 1.0,
                    alpha=1.0 if name == REFERENCE else 0.85,
                    label=meta["label"] if pair_name == PAIRS[0] else None)
        # Bottom panel: ratio to ref (where ell overlaps)
        common = np.intersect1d(ell, ell_ref)
        ix = np.searchsorted(ell, common)
        ix_ref = np.searchsorted(ell_ref, common)
        ratio = cl[ix] / cl_ref[ix_ref] - 1.0
        ax_bot.plot(common, ratio, color=meta["c"], ls=meta["ls"],
                    lw=1.0, alpha=0.85)

    ax_top.set_yscale("log")
    ax_top.set_xlim(2, 50)
    ax_top.set_xticklabels([])
    ax_top.grid(alpha=0.3)
    ax_top.set_ylabel(r"$|C_\ell|$")
    ax_bot.axhline(0, color="k", lw=0.5)
    ax_bot.axvline(20.5, color="0.5", lw=0.5, ls=":")
    ax_bot.set_xlim(2, 50)
    ax_bot.set_ylim(-0.30, 0.30)
    ax_bot.set_yticks([-0.25, -0.10, 0, 0.10, 0.25])
    ax_bot.set_yticklabels([r"$-25\%$", r"$-10\%$", "0", r"$+10\%$", r"$+25\%$"])
    ax_bot.set_xlabel(r"$\ell$")
    ax_bot.set_ylabel(rf"$C_\ell/C_\ell^{{\rm ref}}-1$")
    ax_bot.grid(alpha=0.3)


def main():
    # 2x2 panels (4 pairs) with bigger axes + legend in dedicated bottom strip.
    pairs_4 = ["260_260", "270_273", "278_278", "283_286"]
    fig = plt.figure(figsize=(13, 11))
    # Outer grid: 2x2 panels + legend strip at bottom.
    gs_outer = fig.add_gridspec(3, 2, height_ratios=[3.5, 3.5, 1.0],
                                hspace=0.30, wspace=0.22)

    for k, pair_name in enumerate(pairs_4):
        r, c = divmod(k, 2)
        gs_inner = gs_outer[r, c].subgridspec(2, 1, hspace=0.05, height_ratios=[2.0, 1])
        ax_top = fig.add_subplot(gs_inner[0])
        ax_bot = fig.add_subplot(gs_inner[1], sharex=ax_top)
        plot_pair(ax_top, ax_bot, pair_name)

    # Dedicated legend axes spanning bottom row, both columns.
    ax_leg = fig.add_subplot(gs_outer[2, :])
    ax_leg.axis("off")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center", ncol=2, fontsize=11,
                  frameon=True, framealpha=0.95, edgecolor="0.7",
                  borderpad=0.6, labelspacing=0.7, columnspacing=2.0,
                  handlelength=2.5, handletextpad=0.6)

    fig.suptitle(r"dlnR test: bin-to-bin $C_\ell$ vs reference $d\ln R = 0.0015$ (Mid finest, full $R$)  —  variant=full, $f_{\rm NL}=5$",
                 y=0.995, fontsize=13)
    out = os.path.join(OUT, "dlnRtest_comparison.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
