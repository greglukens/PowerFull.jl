#!/usr/bin/env -S julia --project
# =============================================================================
# Generate the full SphereX tracer set: one h5 per (sample, z-bin).
#
# Bins per sample (total 286):
#   s1: 102, s2: 102, s3: 52, s4: 19, s5: 11
#
# Each h5 uses the spherex_paper_example.jl prescription
#   φ_{s,a}(z) ∝ W(z; zmid_s[a], σ_z_s[a]) · sel_func_s_com(z)
# normalized to ∫φ dz = 1.  The output z-grid size is adapted to resolve
# the Gaussian window (~20 pts per σ, clamped to [1024, 8192]).
#
# Outputs:
#   examples/tracer_s{s}_b{idx:03d}.h5   — 286 tracers
#   examples/tracer_list_prod.txt        — 286 paths
#   examples/pairs_full.txt              — 41,041 --pairs spec (i<=j)
# =============================================================================

using HDF5
using Dierckx
using Printf

const REPO       = joinpath(@__DIR__, "..")
const SPHEREX_H5 = joinpath(REPO, "spherex_params_opt_gaussian.h5")
const OUTDIR     = @__DIR__

gauss(z, z0, σ) = exp(-0.5 * ((z - z0) / σ)^2) / (sqrt(2π) * σ)

"""Pick n_z_dense so dz ≤ σ/20 over the full z-span.  Clamped to [1024, 8192]."""
function choose_nz(sigma::Float64, zmin::Float64, zmax::Float64)::Int
    n = ceil(Int, 20 * (zmax - zmin) / sigma)
    return clamp(n, 1024, 8192)
end

function build_tracer_h5(sample_num::Int, bin_index::Int, out_path::String;
                          n_z_dense::Union{Int,Nothing}=nothing)
    h5open(SPHEREX_H5, "r") do f
        ztest    = read(f["ztest"])
        z_trunc  = read(f["z_trunc"])
        bg_v     = read(f["b_$sample_num"])
        sel_com  = read(f["sel_func_$(sample_num)_com"])
        sigmaz_v = read(f["sigmaz_$sample_num"])
        zmid_v   = read(f["zmid$sample_num"])
        be_v     = read(f["b_e"])
        Q_v      = read(f["Q"])

        1 ≤ bin_index ≤ length(zmid_v) ||
            error("bin_index $bin_index out of 1:$(length(zmid_v)) for sample $sample_num")
        z0 = zmid_v[bin_index]; σz = sigmaz_v[bin_index]

        bg_spl  = Spline1D(ztest, bg_v, k=3)
        sel_spl = Spline1D(ztest, sel_com, k=3)
        be_spl  = Spline1D(z_trunc, be_v, k=3)
        Q_spl   = Spline1D(z_trunc, Q_v, k=3)

        zmin = max(ztest[1], z_trunc[1])
        zmax = min(ztest[end], z_trunc[end])
        nz = n_z_dense === nothing ? choose_nz(σz, zmin, zmax) : n_z_dense
        z_out = collect(range(zmin, zmax; length=nz))

        phi_raw = [gauss(z, z0, σz) * sel_spl(z) for z in z_out]
        dz = zeros(length(z_out))
        dz[1]   = 0.5 * (z_out[2] - z_out[1])
        dz[end] = 0.5 * (z_out[end] - z_out[end-1])
        for k in 2:length(z_out)-1
            dz[k] = 0.5 * (z_out[k+1] - z_out[k-1])
        end
        nrm = sum(phi_raw .* dz)
        phi_out = phi_raw ./ nrm
        bg_out  = [bg_spl(z) for z in z_out]
        be_out  = [be_spl(z) for z in z_out]
        Q_out   = [Q_spl(z)  for z in z_out]

        h5open(out_path, "w") do fo
            fo["z"]   = z_out
            fo["bg"]  = bg_out
            fo["be"]  = be_out
            fo["Q"]   = Q_out
            fo["phi"] = phi_out
            fo["sample"]    = sample_num
            fo["bin_index"] = bin_index
            fo["zmid"]      = z0
            fo["sigma_z"]   = σz
        end
    end
    return out_path
end

function main()
    sample_nbins = Dict(1 => 102, 2 => 102, 3 => 52, 4 => 19, 5 => 11)

    tracer_paths = String[]
    for s in 1:5
        nb = sample_nbins[s]
        for bi in 1:nb
            out = joinpath(OUTDIR, @sprintf("tracer_s%d_b%03d.h5", s, bi))
            build_tracer_h5(s, bi, out)
            push!(tracer_paths, out)
        end
        @printf("  sample %d : %3d bins written\n", s, nb)
    end

    n = length(tracer_paths)
    @assert n == 286

    # tracer_list_prod.txt (paths relative to repo root)
    list_path = joinpath(OUTDIR, "tracer_list_prod.txt")
    open(list_path, "w") do io
        println(io, "# 286 SphereX tracers: all (sample, z-bin) combinations")
        println(io, "# s1:102 + s2:102 + s3:52 + s4:19 + s5:11 = 286")
        for p in tracer_paths
            println(io, "examples/" * basename(p))
        end
    end
    println("\nWrote tracer list: $list_path  ($n entries)")

    # pairs_full.txt: 286 * 287 / 2 = 41,041 unique (i<=j) pair specs
    pairs_path = joinpath(OUTDIR, "pairs_full.txt")
    open(pairs_path, "w") do io
        # Comma-separated on a single line; compute_ClGR CLI accepts one
        # giant --pairs= argument.  We keep the whole string on one line so
        # a shell can do `--pairs=$(cat pairs_full.txt)`.
        pieces = String[]
        for i in 1:n, j in i:n
            push!(pieces, "$i-$j")
        end
        print(io, join(pieces, ","))
    end
    npairs = n * (n + 1) ÷ 2
    println("Wrote pairs spec:  $pairs_path  ($npairs unique pairs)")
    println("\nUsage in SLURM (single giant job):")
    println("  julia -t 4 --project src/compute_ClGR.jl meta.h5 out.h5 \\")
    println("      --tracer-list=examples/tracer_list_prod.txt \\")
    println("      --pairs-file=examples/pairs_full.txt --fNL=1.0")
    println()
    println("Wall time at the current sample-grouped kernel (5 samples → 15")
    println("unique sample-pair-type matrices, shared across all 41,041 bin-pairs):")
    println("  ~30–60 min at -t 4 on Nr≈1155 integrals.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
