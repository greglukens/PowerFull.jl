#!/usr/bin/env python3
"""
plot_cl_demo.py

Plot the observed C_ell^{ij} for one SphereX-style (sample, zbin) tracer
produced by compute_ClGR.jl.  Companion to examples/spherex_paper_example.jl.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

REPO   = os.path.join(os.path.dirname(__file__), "..")
CL_H5  = os.path.join(os.path.dirname(__file__), "Cl_sample3_binmid_auto_rna.h5")
OUTDIR = os.path.dirname(__file__)

with h5py.File(CL_H5, "r") as f:
    ell = f["ell"][:]
    Cl  = f["Cl"][:]
    prov = {k: f[f"provenance/{k}"][()] for k in f["provenance"].keys()}

z_sample = "SphereX sample 3, bin 50 (z0=4.00, sigma_z=0.15)"

ell_fact = ell * (ell + 1) / (2 * np.pi)

fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1]})

ax = axes[0]
ax.plot(ell, Cl, color="darkblue", lw=1.2, label=r"$C_\ell$")
ax.plot(ell, np.abs(Cl), color="darkblue", lw=0.6, alpha=0.4)
ax.set_yscale("log")
ax.set_ylabel(r"$C_\ell^{ii}$")
ax.set_title(z_sample)
ax.grid(True, which="both", alpha=0.3)

ax = axes[1]
ax.plot(ell, ell_fact * Cl, color="darkred", lw=1.2,
        label=r"$\ell(\ell+1)/(2\pi)\,C_\ell$")
ax.set_yscale("log")
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$\ell(\ell+1)\,C_\ell / (2\pi)$")
ax.grid(True, which="both", alpha=0.3)

for a in axes:
    a.set_xscale("log")

fig.tight_layout()
out_svg = os.path.join(OUTDIR, "Cl_sample3_binmid_auto.svg")
out_png = os.path.join(OUTDIR, "Cl_sample3_binmid_auto.png")
fig.savefig(out_svg)
fig.savefig(out_png, dpi=150)
print(f"wrote {out_svg}")
print(f"wrote {out_png}")

# Quick physical sanity checks
print()
print(f"ell range: {ell[0]}..{ell[-1]}")
print(f"C_ell at ell=2  : {Cl[0]:.6e}")
print(f"C_ell at ell=100: {Cl[ell.tolist().index(100)]:.6e}")
print(f"C_ell at ell=500: {Cl[-1]:.6e}")
imax = int(np.argmax(ell_fact * Cl))
print(f"peak of ell(ell+1)/(2pi) C_ell at ell = {ell[imax]},  value = {(ell_fact*Cl)[imax]:.3e}")

print("\nProvenance:")
for k, v in prov.items():
    s = v.decode() if isinstance(v, bytes) else v
    print(f"  {k:20s} {s}")
