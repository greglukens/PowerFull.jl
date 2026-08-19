#!/usr/bin/env python3
"""
Per-ell L2 ratio for dlnR convergence (cut vs cut at SAME R-cut, varying dlnR).
Within tier — pure dlnR effect with R-range controlled.
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

# (title, prefix_a, prefix_b, output slug)
PAIRS = [
    (r"$R \in [0.215,\,4.654]$: $\mathrm{d}\ln R = 0.002$ vs $0.0025$",
     "ClGR_d0p002_cut", "ClGR_d0p0025_cut",
     "d0p002_vs_d0p0025"),
    (r"$R \in [0.0167,\,60.05]$: $\mathrm{d}\ln R = 0.005$ vs $0.006$",
     "ClGR_d0p005_cut", "ClGR_d0p006_cut",
     "d0p005_vs_d0p006"),
]

# Palette (Material 900 + extensions)
DB = "#1a237e"
DR = "#b71c1c"
DG = "#2e7d32"
DP = "#4a148c"
DO = "#e65100"
GRAY = "#bdbdbd"

HIGHLIGHT = [
    ("integrated", "s_m2_0_0_r",       r"$s^{(-2)}_{\ell,00;r}$",            DB, "-"),
    ("integrated", "s_m3_0_1_r",       r"$s^{(-3)}_{\ell,01;r}$",            DG, "-"),
    ("integrated", "t_m2_0_2_r",       r"$t^{(-2)}_{\ell,02;r}$",            DR, "-"),
    ("integrated", "scrS_m4_0_0_r_rp", r"$\mathcal{S}^{(-4)}_{\ell,00}$",    DP, "-"),
    ("base",       "w_0_2_2",          r"$w^{(0)}_{\ell,22}$",               DO, "-"),
]


def list_part_files(prefix):
    return sorted(glob.glob(f"{BUILDS_DIR}/{prefix}_part_*.h5"))


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


def compute_l2_per_ell(prefix_a, prefix_b):
    ell_a, rr_a, eranges_a = load_meta(prefix_a)
    ell_b, rr_b, _         = load_meta(prefix_b)
    assert np.array_equal(ell_a, ell_b)
    assert np.allclose(rr_a, rr_b)
    Nell = len(ell_a)

    parts_a = list_part_files(prefix_a)
    parts_b = list_part_files(prefix_b)
    assert len(parts_a) == len(parts_b)

    names = list_array_names(prefix_a)
    out = {}
    for grp in ("base", "integrated"):
        for nm in sorted(names[grp]):
            out[(grp, nm)] = np.full(Nell, np.nan)

    for ip, (pa, pb) in enumerate(zip(parts_a, parts_b)):
        ell_lo, ell_hi = eranges_a[ip]
        sl = slice(ell_lo - 1, ell_hi)
        with h5py.File(pa, "r") as ha, h5py.File(pb, "r") as hb:
            for grp in ("base", "integrated"):
                if grp not in ha or grp not in hb:
                    continue
                for nm in sorted(set(ha[grp].keys()) & set(hb[grp].keys())):
                    A = np.asarray(ha[grp][nm][:], dtype=np.float64)
                    B = np.asarray(hb[grp][nm][:], dtype=np.float64)
                    if A.shape != B.shape or A.ndim != 3:
                        continue
                    if A.shape[0] != (sl.stop - sl.start):
                        A = np.moveaxis(A, -1, 0)
                        B = np.moveaxis(B, -1, 0)
                    diff = A - B
                    norm_a = np.linalg.norm(A.reshape(A.shape[0], -1), axis=1)
                    norm_d = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
                    ratio = np.where(norm_a > 1e-30, norm_d / norm_a, np.nan)
                    out[(grp, nm)][sl] = ratio
    return ell_a, out


def plot_one(title, ell, ratios, outpath):
    print(f"--- {title} ---")
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    hl_keys = {(g, n) for g, n, *_ in HIGHLIGHT}
    for key, r in ratios.items():
        if key in hl_keys:
            continue
        ax.semilogy(ell, np.maximum(r, 1e-15), color=GRAY, lw=0.5, alpha=0.35,
                    zorder=1)

    for grp, nm, lab, color, ls in HIGHLIGHT:
        r = ratios.get((grp, nm))
        if r is None:
            continue
        ax.semilogy(ell, np.maximum(r, 1e-15), color=color, lw=1.6, ls=ls,
                    label=lab, zorder=3)

    ax.axhline(1e-3, color="0.55", ls="--", lw=0.7, alpha=0.7, zorder=2)
    ax.axhline(1e-4, color="0.55", ls=":",  lw=0.7, alpha=0.7, zorder=2)

    ax.set_xlabel(r"Multipole $\ell$")
    ax.set_ylabel(r"$\epsilon_\ell = \|I^A - I^B\|_F\,/\,\|I^A\|_F$",
                  fontsize=11)
    ax.set_xlim(ell.min(), ell.max())
    ax.set_ylim(1e-12, 1.0)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    ax.legend(loc="upper left", fontsize=9.0, ncol=2, frameon=True,
              framealpha=0.92, labelspacing=0.3, columnspacing=1.0,
              handlelength=1.4, borderpad=0.4)

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
    for title, pa, pb, slug in PAIRS:
        ell, ratios = compute_or_load(pa, pb)
        outpath = f"{OUTDIR}/L2_dlnR_{slug}.pdf"
        plot_one(title, ell, ratios, outpath)


if __name__ == "__main__":
    main()
