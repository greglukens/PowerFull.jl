#!/usr/bin/env julia
# Inject Lucas direct w_0_2_2 ell=2 slice into a build's base/w_0_2_2 array.
#
# In-place modification (saves backup first).  Argument: meta file path.
# Looks up patch from production/results/lucas_patches/w_0_2_2_ell2_lucas.h5.

using HDF5
using Printf

const PATCH = "/gpfs/djeong/PowerFull.jl/production/results/lucas_patches/w_0_2_2_ell2_lucas.h5"

length(ARGS) >= 1 || error("usage: julia inject_lucas_patch.jl <build_meta.h5>")
const META = ARGS[1]

# Load patch
hp = h5open(PATCH, "r")
M = read(hp["w_0_2_2_ell2"])::Matrix{Float32}
rr_patch = read(hp["rr"])::Vector{Float64}
close(hp)
@printf("Patch loaded: %d × %d matrix, rr ∈ [%.2f, %.2f]\n",
        size(M, 1), size(M, 2), rr_patch[1], rr_patch[end])

# Identify ell=2 part file from meta
hm = h5open(META, "r")
ell_values = read(hm["grid/ell_values"])::Vector{Int}
ell_ranges = read(hm["metadata/ell_ranges"])::Matrix{Int}
part_files = read(hm["part_files"])
rr_meta = read(hm["grid/rr"])::Vector{Float64}
close(hm)

@assert ell_values[1] == 2 "first ell expected to be 2"
@assert all(rr_meta .≈ rr_patch) "r-grid mismatch between build and patch"
@printf("Build r-grid matches patch.\n")

build_dir = dirname(META)
part1_path = joinpath(build_dir, String(part_files[1]))
@printf("ell=2 part file: %s\n", part1_path)

# Backup before modify
backup = part1_path * ".pre_lucas_patch"
if !isfile(backup)
    cp(part1_path, backup)
    @printf("Backup: %s\n", backup)
else
    @printf("Backup already exists, skipping copy\n")
end

# Read original ell=2 slice for diff stats
hb = h5open(part1_path, "r")
orig = read(hb["base/w_0_2_2"])[1, :, :]    # (Nr, Nr) at ell=2
close(hb)
diff = M .- orig
abs_diff = abs.(diff)
denom = max.(abs.(orig), eps(Float32))
rel = abs_diff ./ denom

@printf("\nDiff stats (Lucas patch − stored):\n")
@printf("  L2 norm of diff / L2 norm of stored = %.3e\n",
        sqrt(sum(diff.^2)) / sqrt(sum(orig.^2)))
@printf("  max |diff|                          = %.3e (rel %.3e)\n",
        maximum(abs_diff), maximum(rel[abs.(orig) .> 1e-15]))
@printf("  median rel|diff| (|orig|>1e-15)     = %.3e\n",
        sort(rel[abs.(orig) .> 1e-15])[length(rel[abs.(orig) .> 1e-15]) ÷ 2 + 1])

# Inject — overwrite ell=2 slice
@printf("\nInjecting patch...\n")
hw = h5open(part1_path, "r+")
ds = hw["base/w_0_2_2"]
ds[1, :, :] = M    # ell=2 is index 1
close(hw)
@printf("Done. ell=2 slice replaced with Lucas direct.\n")
