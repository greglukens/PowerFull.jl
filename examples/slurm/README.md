# Production 3-tier pipeline — example SLURM workflow

End-to-end example: TwoFAST radial integrand → 3-tier HDF5 build → C_ℓ^GR.
All scripts assume the repository lives at the path resolved by `${REPO}`
(defaults to `$SLURM_SUBMIT_DIR` or the current working directory) and use
the plikHM matterpower file shipped under `data/`.

## Pipeline

```
step1_tier_low.slurm   →   production/results/tier_low/*.jld2  (ell = 2..50,    dlnR=0.002,  nR=4097)
step1_tier_mid.slurm   →   production/results/tier_mid/*.jld2  (ell = 51..200,  dlnR=0.001,  nR=2049)
step1_tier_high.slurm  →   production/results/tier_high/*.jld2 (ell = 201..500, dlnR=0.0005, nR=2049)
        │
        ▼
step2_build_3tier.slurm  →  production/ClGR_production_{meta,part_*}.h5
        │
        ▼
step3_compute_cl.slurm   →  examples/Cl_<tracer>_<bin>.h5
```

## Submitting

```bash
cd /path/to/PowerFull.jl

# Three step1 jobs can run in parallel:
sbatch examples/slurm/step1_tier_low.slurm
sbatch examples/slurm/step1_tier_mid.slurm
sbatch examples/slurm/step1_tier_high.slurm

# Step2 depends on all three step1 jobs.  Use --dependency=afterok:JID1:JID2:JID3
# to chain, or submit manually once step1 completes.
sbatch examples/slurm/step2_build_3tier.slurm

# Step3 reads step2's HDF5.  Edit TRACER / OUTPUT inside the script to match
# your tomographic bin configuration before submitting.
sbatch examples/slurm/step3_compute_cl.slurm
```

## Resource sizing (per node)

| Step          | Cores | Memory | Wall (Nr=4096) |
|---------------|-------|--------|----------------|
| step1_low     | 64    | 200 GB | ~30 min        |
| step1_mid     | 64    | 300 GB | ~1 h           |
| step1_high    | 64    | 400 GB | ~2 h           |
| step2_build   | 24    | 400 GB | ~1 h           |
| step3_cl      | 16    |  40 GB | ~30 min        |

Adjust `-p` (Julia worker count) and `JULIA_NUM_THREADS` for the actual
node.  The default uses 8 workers × 6 threads (= 48 cores).

## Cosmology

All step1 jobs use the **plikHM matterpower** (`data/planck_base_plikHM_…
_matterpower.dat`) and the matching cosmofn table (`data/cosmo_funcr.txt`).
Do not mix with the astropy-normalized power spectrum
(`data/cosmo_funcr_astropy_planck2018.txt`) — the two differ by ~3 % in
`w_0_22` at low ell.  Validation tests in `validation/` are also pinned to
plikHM.

## Output files

- `production/results/tier_*/TwoFAST_*.jld2` — step1 radial integrand cache.
- `production/data_3tier/` — staging symlinks (built by step2).
- `production/ClGR_production_meta.h5` + `_part_NNN.h5` — step2 HDF5 build
  (paper-observable keys: `l_*`, `scrl_*`, `scrL_*`, `scrY_*`, `scrZ_*`).
- `examples/Cl_<tracer>.h5` — step3 final C_ℓ^GR result.

## Notes

- `--streaming-mlcache` in step1 fuses build+consume of the per-ell MlCache
  and avoids writing it to disk.  Peak step1 RAM stays under the node limit.
- step2 uses `build_and_export.jl`'s streaming path; on-disk size of the
  combined 3-tier build is ~150 GB.
- step3's streaming consumer keeps memory low by reading integrals
  tile-by-tile.
