#!/usr/bin/env python3
"""
Per-sample hierarchy plots.  Six Cl curves per panel:
  realspace (density only, no RSD)
  Kaiser    (density + β RSD)
  Newtonian (Kaiser + β·α/r corrections)
  Gaussian  (all 19 GR terms, fNL=0)
  Full      (all 19 GR terms, fNL=5; red dashed)
  fNL only  = Full − Gaussian                 (magenta dotted)

Reads:
  examples/Cl_hierarchy_{realspace,kaiser,newtonian,gaussian,full}.h5

Writes (five PDFs):
  examples/figs/Cl_hierarchy_s{1..5}.pdf

Layout: for every sample, each page fixes one reference bin i and
shows 4 panels for (i, i+Δ), Δ ∈ {0,1,2,3}, trimming panels whose j
index exceeds the sample's last bin.
"""

import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX   = os.path.join(REPO, "examples")
FIGS = os.path.join(EX, "figs")
os.makedirs(FIGS, exist_ok=True)

FNL  = 5.0

# 1-based bin ranges in the 286-tracer list.
SAMPLE_BINS = {1: (1, 102), 2: (103, 204), 3: (205, 256),
               4: (257, 275), 5: (276, 286)}

# Curves and their styling.  Keep in render order (bottom to top).
CURVES = [
    ("realspace",  "realspace",              "0.35",        "-",  1.0),
    ("kaiser",     "Kaiser",                 "darkblue",    "-",  1.1),
    ("newtonian",  "Newtonian",              "darkcyan",    "-",  1.1),
    ("gaussian",   "Gaussian (full GR)",     "darkgreen",   "-",  1.1),
    ("full",       r"Full (GR + $f_{\mathrm{NL}}=%g$)" % FNL, "darkred", "--", 1.3),
    ("fnl_only",   r"$f_{\mathrm{NL}}$ only", "darkmagenta", ":",  1.3),
]


def pair_index_map():
    """Rebuild the sample-major 1-based pair-index map used by
    generate_per_sample_pairs.jl.  Returns
        idx[(sample, i_within_sample, j_within_sample)] = 1-based pair idx
    with i<=j, j-i <= 3, and both inside the same sample."""
    idx = {}
    p = 1
    for s in (1, 2, 3, 4, 5):
        lo, hi = SAMPLE_BINS[s]
        nbin = hi - lo + 1
        for i in range(nbin):
            for d in range(4):
                j = i + d
                if j >= nbin:
                    break
                idx[(s, i, j)] = p
                p += 1
    return idx


def load_variants():
    """Returns (ell, {curve_key: Cl_all[n_pair, n_ell]}).
    h5py reads the Julia column-major [n_ell, n_pair] dataset as
    [n_pair, n_ell], so first index = pair."""
    ell = None
    data = {}
    for key in ("realspace", "kaiser", "newtonian", "gaussian", "full"):
        path = os.path.join(EX, f"Cl_hierarchy_{key}.h5")
        with h5py.File(path, "r") as f:
            e = f["ell"][:].astype(int)
            cl = f["Cl_all"][:]
        if ell is None:
            ell = e
        else:
            assert np.array_equal(ell, e), f"ell mismatch in {key}"
        data[key] = cl
    data["fnl_only"] = data["full"] - data["gaussian"]
    return ell, data


def draw_panel(ax, ell, data, pair_idx, title, show_legend):
    p = pair_idx - 1
    for key, label, color, ls, lw in CURVES:
        y = np.abs(data[key][p, :])
        mask = y > 0
        ax.loglog(ell[mask], y[mask],
                  color=color, linestyle=ls, linewidth=lw, label=label)
    ax.set_xlim(ell[0], ell[-1])
    ax.set_xlabel(r"$\ell$", fontsize=12)
    ax.set_ylabel(r"$|C_\ell^{ij}|$", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.4)
    if show_legend:
        ax.legend(fontsize=9, loc="best", framealpha=0.85)


def plot_sample(pdf, sample, ell, data, pair_idx_map):
    lo, hi = SAMPLE_BINS[sample]
    nbin = hi - lo + 1
    for i in range(nbin):
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
        axes = axes.ravel()
        for k in range(4):
            j = i + k
            if j >= nbin:
                axes[k].set_axis_off()
                continue
            pair_idx = pair_idx_map[(sample, i, j)]
            title = r"sample %d: bin %d-%d ($\Delta=%d$)" % (sample, i+1, j+1, k)
            show_legend = (i == 0 and k == 0)
            draw_panel(axes[k], ell, data, pair_idx, title, show_legend)
        fig.suptitle(f"Sample {sample}, reference bin {i+1} / {nbin}", fontsize=13)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    ell, data = load_variants()
    pim = pair_index_map()
    for sample in (1, 2, 3, 4, 5):
        out = os.path.join(FIGS, f"Cl_hierarchy_s{sample}.pdf")
        with PdfPages(out) as pdf:
            plot_sample(pdf, sample, ell, data, pim)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
