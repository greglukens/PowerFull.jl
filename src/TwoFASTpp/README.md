# TwoFASTpp

Modified fork of [TwoFAST](https://github.com/hsgg/TwoFAST.jl) by
Henry S. Gebhardt, used internally by
[PowerFull](../../README.md).

The original `TwoFAST.jl` implements the FFTLog-based algorithm of
Gebhardt & Jeong (2018) for computing spherical-Bessel integrals
$w^{p}_\ell(r_1, r_2)$.  `TwoFASTpp` extends this to the broader class
of $(p, n, j, j')$ base functions needed for the full relativistic
wide-angle galaxy power spectrum, including the $n=-1$ ($u$) and
$n=-2$ ($v$) branches that carry primordial non-Gaussianity.

The "pp" suffix is ASCII for `++`, following the C/C++ convention;
it denotes an extension of the original library rather than a
replacement.

## Attribution

- Original algorithm and reference implementation: Henry S. Gebhardt
  (`hsggebhardt@gmail.com`),
  [github.com/hsgg/TwoFAST.jl](https://github.com/hsgg/TwoFAST.jl).
- Reference: H. S. Gebhardt & D. Jeong, *"Fast and accurate computation
  of projected two-point functions"*, Phys. Rev. D **97**, 023504 (2018),
  [arXiv:1709.02401](https://arxiv.org/abs/1709.02401).
- Extensions to $(p, n, j, j')$ branches for PowerFull: Donghui Jeong
  and Gregory Lukens.

## Known accuracy limits

### Cancellation in the 9-term reconstruction

`TwoFASTpp` reconstructs the $(j, j') \neq (0, 0)$ derivative bases from
shift-$\ell$ primitives `wldl[s1, s2]` (with $s_1, s_2 \in \{-2, 0, +2\}$
for `oddprimes=false`) via a 9-term linear combination with
$\ell$-dependent rational coefficients (`wljj_from_wldl_allprimes!`,
`wljj_from_wldl!`).  At low $\ell$ with $j = j' = 2$ this combination
suffers catastrophic cancellation in regions where the assembled value
is small relative to the individual primitives.

With production-tier $q$ values (Base 1: $q = 1.30$, see
`scripts/q_sweep_*` and project notes), the worst case is
$w^{0}_{\ell=2, j=2, j'=2}$ on small-$R$ cross pairs:

| Region | Error vs Lucas direct |
|--------|-----------------------|
| Auto pair ($R = 1.00$) | $\lesssim 0.003\%$ |
| $R \gtrsim 0.6$ | $\lesssim 2\%$ |
| $R = 0.4, \chi \approx 3585$ | up to **$24\%$** |

The error is strongly localized in $\ell$: at $\ell = 2$ the cancellation
hot-spot dominates, by $\ell \geq 5$ the residual drops to $\lesssim 0.05\%$
across all $R$, and $\ell \geq 10$ is converged to $\lesssim 10^{-4}$.

### Lucas direct patch (option)

For applications where the $w^{0}_{\ell=2, j=2, j'=2}$ slice must be
accurate at percent level on small-$R$ cross pairs (e.g., standalone use
of $w_{2,2}$ or analyses dominated by off-diagonal bin pairs at
$\ell = 2$), `TwoFASTpp` provides a Lucas-direct override:

`LucasPatch` is **not auto-included** by `using TwoFASTpp` so the parent
project is not forced to depend on `QuadGK`.  Load it on demand:

```julia
using TwoFASTpp
using QuadGK
include(joinpath(pkgdir(TwoFASTpp), "src", "LucasPatch.jl"))
using .LucasPatch

# Replace the (j=jp=2, ell=2) slice with Lucas direct values.
M = lucas_patch_w_0_22_ell2(rr_grid, k -> P_of_k(k); rtol=1e-7)
# M[i, j] = w^0_{ell=2, j=2, jp=2}(rr_grid[i], rr_grid[j])
```

For distributed runs the module *and* `QuadGK` must be loaded on every
worker before calling — otherwise `pmap` will hit `UndefVarError` on
remotes:

```julia
using Distributed
addprocs(24)
@everywhere using QuadGK
@everywhere include(joinpath(pkgdir(Main.TwoFASTpp), "src", "LucasPatch.jl"))
@everywhere using .LucasPatch
```

The patch integrates
$w^{0}_{2,22}(r_1, r_2) = (2/\pi) \int k^2 P(k) j_2''(kr_1) j_2''(kr_2) \, dk$
directly using a closed-form expression for $j_2''(z)$ (no spherical Bessel
calls, only `sin`/`cos` arithmetic), bypassing the 9-term reconstruction
entirely.  Cost is $\sim 2$ s/point with `QuadGK`; for the production
$1155 \times 1155$ chi-grid, ~9 hours on 24 distributed workers.

For faster computation, use the standalone SLURM script
`scripts/lucas_w22_ell2_patch.slurm` in the parent `PowerFull.jl`
repository.  That script calls the more efficient Lucas 1995 quadosc
algorithm (via Henry Gebhardt's `Quad_jar_jbt`, ~0.07 s/point) and
finishes in ~30 minutes.

## Streaming `MlCache` mode (option)

The default disk-backed pipeline (`MlCache(...)` then
`calcwljj_powerfull(...)`) writes the per-$\ell$ $M_\ell$ records to a
binary cache (typically 100–300 GB per $q$, 1.5 TB total at production
$N_r{=}4096$, $n_R{=}4097$, $\ell_{\max}{=}210$) and reads them back in a
second pass.  For machines without that much disk or for one-off runs
where the cache won't be reused, `calcwljj_powerfull_streaming` fuses the
$M_\ell$ backward recurrence and the $\phi \times \mathrm{brfft} \to$
`outfunc` consumer into a single loop, eliminating the on-disk
`MlCache` entirely.

```julia
using TwoFASTpp

# Drop-in replacement for the (calcMljj + calcwljj_powerfull) pair.
# `fell_lmax_file` is the existing F21EllCache directory; that cache is
# still read from disk.  No MlCache.bin is written or read.
rr = calcwljj_powerfull_streaming(
    pkfn, RR;
    ell=aell, kmin=kmin, kmax=kmax, N=Ngrid, r0=chi0, q=q,
    fell_lmax_file=f21ellcache_dir,
    outfunc=outfunc,
    oddprimes=oddprimes,
)
```

In the surrounding `run_twofast.jl` driver this is exposed as the
`--streaming-mlcache` CLI flag (default OFF for safety).

**Verification (2026-05-07).**  In a single Julia process the streaming
function reproduces the disk pipeline bit-identically across all $\ell$
and components $\{00, 02, 20, 22\}$.  Across distributed workers, FFTW's
`MEASURE` planning produces ε-level rounding differences (relative
$\lesssim 2.2 \times 10^{-15}$, 13 orders of magnitude below typical
production sub-percent tolerances).

**Performance.**  At tier-A production parameters
($N_r{=}4096$, $n_R{=}4097$, $\mathrm{dlnR}{=}0.002$, $\ell{=}2{-}199$,
$\ell_{\max,\mathrm{run}}{=}210$, threading 6×9 on a 64-core node):

| Mode | Wall time | Disk write |
|------|-----------|------------|
| disk fresh build (estimate) | ~2 h 24 min | 1.5 TB |
| disk warm cache | ~10 min | 0 (read only) |
| **streaming** | **28 min** | **0** |

The disk-warm-cache row is fastest only when the same `(N_r, n_R,
\mathrm{dlnR}, \ell_{\max,\mathrm{run}}, q\text{-list})` has previously
been built on disk; that case is rare in production sweeps.

## License

MIT, inherited from the upstream TwoFAST.jl.
