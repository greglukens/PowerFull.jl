#!/usr/bin/env python3
"""
R-tail validation: compare the low-ell baseline build (nR=4097, dlnR=0.002,
R in [0.0167, 60.05]) against a wider build (nR=5201, dlnR=0.002,
R in [0.0055, 181.27]).  L2 ratio measures the contribution of the
R-tail outside [0.0167, 60.05].

Paper-quality: 5 highlighted arrays in DB/DR/DG/DP/DO palette with LaTeX
labels; remaining 56 arrays as gray envelope.  Cached.
"""
import os, sys, glob, json
import numpy as np
import h5py
import matplotlib.pyplot as plt

# nR=4097 (production baseline) and nR=5201 (R-tail probe) live next to each other.
META_4097 = "/gpfs/djeong/PowerFull.jl/production/results/test_singletier_nR4097/builds/ClGR_nR4097_meta.h5"
META_5201 = "/gpfs/djeong/PowerFull.jl/production/results/test_singletier_nR5201/builds/ClGR_nR5201_meta.h5"
OUTDIR = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl/figs"
CACHE = "/gpfs/djeong/PowerFull.jl/production/results/test_dlnR_jump/cl/l2_cache.json"

PAIRS = [
    (r"$\mathrm{d}\ln R = 0.002$: $n_R = 5201$ vs $4097$",
     META_5201, META_4097,
     "Rtail_nR5201_vs_nR4097"),
]

# Palette
DB = "#1a237e"; DR = "#b71c1c"; DG = "#2e7d32"; DP = "#4a148c"; DO = "#e65100"
GRAY = "#bdbdbd"

HIGHLIGHT = [
    ("integrated", "s_m2_0_0_r",       r"$s^{(-2)}_{\ell,00;r}$",            DB, "-"),
    ("integrated", "s_m3_0_1_r",       r"$s^{(-3)}_{\ell,01;r}$",            DG, "-"),
    ("integrated", "t_m2_0_2_r",       r"$t^{(-2)}_{\ell,02;r}$",            DR, "-"),
    ("integrated", "scrS_m4_0_0_r_rp", r"$\mathcal{S}^{(-4)}_{\ell,00}$",    DP, "-"),
    ("base",       "w_0_2_2",          r"$w^{(0)}_{\ell,22}$",               DO, "-"),
]


def _meta_parts(meta_path):
    base = meta_path.replace("_meta.h5", "")
    return sorted(glob.glob(f"{base}_part_*.h5"))


def load_meta(meta_path):
    with h5py.File(meta_path, "r") as h:
        ell = h["grid/ell_values"][:]
        rr = h["grid/rr"][:]
        ell_ranges = h["metadata/ell_ranges"][:].T
    return ell, rr, ell_ranges


def list_array_names(meta_path):
    parts = _meta_parts(meta_path)
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


def compute_l2_per_ell(meta_a, meta_b):
    ell_a, rr_a, eranges_a = load_meta(meta_a)
    ell_b, rr_b, _         = load_meta(meta_b)
    assert np.array_equal(ell_a, ell_b), "ell grids differ"
    assert np.allclose(rr_a, rr_b),       "r grids differ"
    Nell = len(ell_a)

    parts_a = _meta_parts(meta_a)
    parts_b = _meta_parts(meta_b)
    assert len(parts_a) == len(parts_b)

    names = list_array_names(meta_a)
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
                    # Normalize by baseline (B = nR=4097 = denominator)
                    norm_b = np.linalg.norm(B.reshape(B.shape[0], -1), axis=1)
                    norm_d = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
                    ratio = np.where(norm_b > 1e-30, norm_d / norm_b, np.nan)
                    out[(grp, nm)][sl] = ratio
    return ell_a, out


def compute_or_load(meta_a, meta_b, key):
    cache = _cache_load()
    if key in cache:
        d = cache[key]
        ell = np.array(d["ell"], dtype=np.int64)
        ratios = {tuple(k.split("|", 1)): np.asarray(v, dtype=np.float64)
                  for k, v in d["ratios"].items()}
        print(f"  cache hit: {key}")
        return ell, ratios
    print(f"  computing L2 (will cache): {key}")
    ell, ratios = compute_l2_per_ell(meta_a, meta_b)
    cache[key] = {
        "ell": ell.tolist(),
        "ratios": {f"{g}|{n}": np.where(np.isnan(r), None, r).tolist()
                   for (g, n), r in ratios.items()},
    }
    _cache_save(cache)
    return ell, ratios


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
    ax.set_ylabel(r"$\epsilon_\ell = \|I^{n_R{=}5201} - I^{n_R{=}4097}\|_F\,/\,\|I^{n_R{=}4097}\|_F$",
                  fontsize=10)
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

    for el in (2, 5, 10, 20, 100):
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
    for title, ma, mb, slug in PAIRS:
        ell, ratios = compute_or_load(ma, mb, key=slug)
        outpath = f"{OUTDIR}/L2_{slug}.pdf"
        plot_one(title, ell, ratios, outpath)


if __name__ == "__main__":
    main()
