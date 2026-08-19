#!/usr/bin/env python3
"""
Per-ell Frobenius L2 ratio:  ||I_full - I_cut||_F / ||I_full||_F
for each base / 1D / 2D array, comparing full vs cut at same dlnR.
Paper-quality: 5 highlighted arrays in DB/DR/DG/DP/DO palette with LaTeX
labels; remaining 56 arrays as gray envelope.
"""
import os, sys, glob, json
import numpy as np
import h5py
import matplotlib.pyplot as plt

BUILDS_DIR = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/builds"
OUTDIR = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl/figs"
CACHE = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl/l2_cache.json"

# (title, prefix_full, prefix_cut, output slug)
PAIRS = [
    (r"$\mathrm{d}\ln R = 0.002$, cut $R \in [0.215,\,4.654]$",
     "ClGR_d0p002_full",  "ClGR_d0p002_cut",  "d0p002"),
    (r"$\mathrm{d}\ln R = 0.0025$, cut $R \in [0.215,\,4.654]$",
     "ClGR_d0p0025_full", "ClGR_d0p0025_cut", "d0p0025"),
    (r"$\mathrm{d}\ln R = 0.005$, cut $R \in [0.0167,\,60.05]$",
     "ClGR_d0p005_full",  "ClGR_d0p005_cut",  "d0p005"),
    (r"$\mathrm{d}\ln R = 0.006$, cut $R \in [0.0167,\,60.05]$",
     "ClGR_d0p006_full",  "ClGR_d0p006_cut",  "d0p006"),
]

# Palette (Material 900 + extensions)
DB = "#1a237e"  # indigo 900
DR = "#b71c1c"  # red 900
DG = "#2e7d32"  # green 900
DP = "#4a148c"  # purple 900
DO = "#e65100"  # orange 900
GRAY = "#bdbdbd"

# Five representative arrays highlighted in every panel.
HIGHLIGHT = [
    # (group, name, latex_label, color, linestyle)
    ("integrated", "s_m2_0_0_r",       r"$s^{(-2)}_{\ell,00;r}$",            DB, "-"),
    ("integrated", "s_m3_0_1_r",       r"$s^{(-3)}_{\ell,01;r}$",            DG, "-"),
    ("integrated", "t_m2_0_2_r",       r"$t^{(-2)}_{\ell,02;r}$",            DR, "-"),
    ("integrated", "scrS_m4_0_0_r_rp", r"$\mathcal{S}^{(-4)}_{\ell,00}$",    DP, "-"),
    ("base",       "w_0_2_2",          r"$w^{(0)}_{\ell,22}$",               DO, "-"),
]


def list_part_files(prefix):
    metas = sorted(glob.glob(f"{BUILDS_DIR}/{prefix}_part_*.h5"))
    if not metas:
        sys.exit(f"no part files for {prefix}")
    return metas


def load_meta(prefix):
    with h5py.File(f"{BUILDS_DIR}/{prefix}_meta.h5", "r") as h:
        ell = h["grid/ell_values"][:]
        rr = h["grid/rr"][:]
        ell_ranges = h["metadata/ell_ranges"][:].T
    return ell, rr, ell_ranges


def list_array_names(prefix):
    parts = list_part_files(prefix)
    names = {"base": set(), "integrated": set()}
    with h5py.File(parts[0], "r") as h:
        for grp in ("base", "integrated"):
            if grp in h:
                names[grp].update(h[grp].keys())
    return names


def _cache_load():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def _cache_save(cache):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f)


def compute_or_load(prefix_a, prefix_b):
    cache = _cache_load()
    key = f"{prefix_a}__{prefix_b}"
    if key in cache:
        d = cache[key]
        ell = np.array(d["ell"], dtype=np.int64)
        ratios = {tuple(k.split("|", 1)): np.asarray(v, dtype=np.float64)
                  for k, v in d["ratios"].items()}
        print(f"  cache hit: {key}")
        return ell, ratios
    print(f"  computing L2 (will cache): {key}")
    ell, ratios = compute_l2_per_ell(prefix_a, prefix_b)
    cache[key] = {
        "ell": ell.tolist(),
        "ratios": {f"{g}|{n}": np.where(np.isnan(r), None, r).tolist()
                   for (g, n), r in ratios.items()},
    }
    _cache_save(cache)
    return ell, ratios


def compute_l2_per_ell(prefix_full, prefix_cut):
    ell_full, rr_full, eranges_full = load_meta(prefix_full)
    ell_cut, rr_cut, _              = load_meta(prefix_cut)
    assert np.array_equal(ell_full, ell_cut)
    assert np.allclose(rr_full, rr_cut)
    Nell = len(ell_full)

    parts_full = list_part_files(prefix_full)
    parts_cut  = list_part_files(prefix_cut)
    assert len(parts_full) == len(parts_cut)

    names = list_array_names(prefix_full)
    out = {}
    for grp in ("base", "integrated"):
        for nm in sorted(names[grp]):
            out[(grp, nm)] = np.full(Nell, np.nan)

    for ip, (pf, pc) in enumerate(zip(parts_full, parts_cut)):
        ell_lo, ell_hi = eranges_full[ip]
        sl = slice(ell_lo - 1, ell_hi)
        with h5py.File(pf, "r") as hf, h5py.File(pc, "r") as hc:
            for grp in ("base", "integrated"):
                if grp not in hf or grp not in hc:
                    continue
                for nm in sorted(set(hf[grp].keys()) & set(hc[grp].keys())):
                    A = np.asarray(hf[grp][nm][:], dtype=np.float64)
                    B = np.asarray(hc[grp][nm][:], dtype=np.float64)
                    if A.shape != B.shape or A.ndim != 3:
                        continue
                    if A.shape[0] != (sl.stop - sl.start):
                        A = np.moveaxis(A, -1, 0)
                        B = np.moveaxis(B, -1, 0)
                    diff = A - B
                    norm_full = np.linalg.norm(A.reshape(A.shape[0], -1), axis=1)
                    norm_diff = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
                    ratio = np.where(norm_full > 1e-30, norm_diff / norm_full, np.nan)
                    out[(grp, nm)][sl] = ratio
    return ell_full, out


def plot_one(title, ell, ratios, outpath):
    print(f"--- {title} ---")
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    # Gray envelope: every array except the highlighted ones.
    hl_keys = {(g, n) for g, n, *_ in HIGHLIGHT}
    for key, r in ratios.items():
        if key in hl_keys:
            continue
        ax.semilogy(ell, np.maximum(r, 1e-15), color=GRAY, lw=0.5, alpha=0.35,
                    zorder=1)

    # Highlighted arrays on top with palette colors and LaTeX legend.
    for grp, nm, lab, color, ls in HIGHLIGHT:
        r = ratios.get((grp, nm))
        if r is None:
            continue
        ax.semilogy(ell, np.maximum(r, 1e-15), color=color, lw=1.6, ls=ls,
                    label=lab, zorder=3)

    # Reference levels
    ax.axhline(1e-3, color="0.55", ls="--", lw=0.7, alpha=0.7, zorder=2)
    ax.axhline(1e-4, color="0.55", ls=":",  lw=0.7, alpha=0.7, zorder=2)

    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$\epsilon_\ell = \|I^{\mathrm{full}} - I^{\mathrm{cut}}\|_F\,/\,\|I^{\mathrm{full}}\|_F$",
                  fontsize=11)
    ax.set_xlim(ell.min(), ell.max())
    ax.set_ylim(1e-15, 1.0)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True, framealpha=0.92,
              labelspacing=0.3, handlelength=1.5, borderpad=0.4)

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    print(f"  saved {outpath}")

    for el in (2, 10, 20):
        if el not in ell:
            continue
        ie = list(ell).index(el)
        items = [(k, ratios[k][ie]) for k in ratios if not np.isnan(ratios[k][ie])]
        items.sort(key=lambda x: -x[1])
        print(f"  ell={el}: top-5 worst arrays")
        for k, r in items[:5]:
            print(f"    {k[0]}/{k[1]:30s}  L2 ratio = {r:.3e}")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    for title, pf, pc, slug in PAIRS:
        ell, ratios = compute_or_load(pf, pc)
        outpath = f"{OUTDIR}/L2_full_vs_cut_{slug}.pdf"
        plot_one(title, ell, ratios, outpath)


if __name__ == "__main__":
    main()
