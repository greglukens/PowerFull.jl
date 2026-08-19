#!/usr/bin/env -S julia --project
# =============================================================================
#
# >> build_and_export.jl <<   (fast variant — default in the release)
#
# Phase 1+2+3 optimized streaming build.  Performs Step 2 of the
# PowerFull pipeline: reads the TwoFAST base functions
# w^{p,n}_{\ell, j j'}(r, R) from JLD2 on a logarithmic R-grid,
# interpolates to the physical (r_1, r_2) grid, accumulates the 1D and
# 2D line-of-sight integrals, and writes the full result to HDF5 using
# the tilde convention (tl_, tscrL, tscrY, tscrZ for lensing-kernel
# bare building blocks; see README).
#
# Optimizations (documented for provenance):
#
#   [Phase 1]
#   (1) Kernel-loop refactor (s, t, l together per ell):
#       - w_integrand JLD2 file loaded ONCE per case.
#       - w_phys interpolated ONCE per ell (variant B, vectorized Dierckx).
#       - prefix_sum_axis1!/axis2! repeated 3x (cheap, O(nr²)).
#   (2) Per-task pre-allocated buffers.
#
#   [Phase 2]
#   (3) Compute-side output layout [nr, nr, nell] (column-major):
#       - Per-ell slice int_r[:, :, ell] is contiguous.
#       - Accumulation write is stride-1 (was stride=nell_total).
#       - One permutedims to [nell, nr, nr] just before HDF5 write to
#         preserve backward-compatible file layout.
#   (4) Eliminate strided prefix_sum_axis2!:
#       - Precompute w_phys_T = permutedims(w_phys, (2,1)) once per ell.
#       - For ;r' direction, run prefix_sum_axis1! on w_phys_T (contiguous),
#         then permutedims the result back to standard layout.
#
#   [Phase 3]
#   (5) @simd annotations on the hot accumulation copy loops.
#   (6) Case-4/6/7 w_integrand file sharing: (p=-4, j=0, jp=0, n=0) is
#       stored separately in case_{4,6,7}.jld2 but content is identical.
#       Load case-4 file per tier and reuse for case 6 and 7 (saves 4/6 of
#       the case-4 family I/O).
#
# Validated:
#   - Nr=1024 debug: 39/39 bit-identical, 1.54× speedup (14.6s → 9.5s)
#   - Nr=4096 ell=2-199: 39/39 bit-identical, 2.65× speedup (363s → 137s)
#
# NOT implemented (evaluated, deferred):
#   - Phase 3.2 pipeline compute/write (complexity vs 1.1× trade).
#   - Phase 4 (triple-kernel fusion, buffer pool, parallel permutedims): at
#     Nr=4096 nr=1155, w_phys fits in L3 so memory-bandwidth savings were
#     neutral. Would help only at Nr=8192. See wiki for design notes.
#
# Output HDF5 is split across multiple part files for large runs.
# 
#
# Usage:
#   julia -t 16 --project src/build_and_export.jl --Nr=4096 --nR=2049
#   julia -t 16 --project src/build_and_export.jl \
#       --tier=0.002,2,199 --tier=0.0005,200,500
#   julia -t 16 --project src/build_and_export.jl --help
#
#  April 2026
#  Donghui Jeong
# =============================================================================

include(joinpath(@__DIR__, "PowerFull.jl"))
using .PowerFull

include(joinpath(@__DIR__, "PowerFull_interp.jl"))
using .PowerFullInterp

include(joinpath(@__DIR__, "cosmofns.jl"))
using .cosmofns

# --- Modified-gravity (local-limit) support -----------------------------------
# This file is a STANDALONE copy of build_and_export.jl (Option C): the GR
# build_and_export.jl is left byte-for-byte untouched.  The only substantive
# differences here are (1) these MG includes, (2) the kernel-array call inside
# build_and_export_streaming uses the MG kernels, and (3) the CLI accepts
# --mu0/--Sigma0 and writes them to the meta.  With μ₀=Σ₀=0 the MG kernels
# reduce EXACTLY to the GR ones, so output is bit-identical to the GR builder.
#
# NOTE: do NOT `include` both build_and_export.jl and build_and_export_mg.jl in
# the same Julia session — they define the same top-level names. Run this file
# as its own program (the runner does exactly that).
include(joinpath(@__DIR__, "mg_cosmo.jl"))
using .MGCosmo
include(joinpath(@__DIR__, "build_and_export_mg_kernels.jl"))

using HDF5
using Base.Threads

# =============================================================================
# Cosmological kernel functions (same as build_and_export.jl)
# =============================================================================

function fs(r::Real, cfns::cosmofn)::Float64
    z, a, H, Om, f, D = cfns(r)
    return a^3 * H^3 * Om * (f - 1) * D
end

function ft(r::Real, cfns::cosmofn)::Float64
    z, a, H, Om, f, D = cfns(r)
    return a^2 * H^2 * Om * D
end

function fl1(r::Real, cfns::cosmofn)::Float64
    return ft(r, cfns) / r
end

function compute_kernel_arrays(rr::Vector{Float64}, cfns::cosmofn)
    nr = length(rr)
    rf_s = Vector{Float64}(undef, nr)
    rf_t = Vector{Float64}(undef, nr)
    rf_l = Vector{Float64}(undef, nr)
    D_r  = Vector{Float64}(undef, nr)
    @inbounds for k in 1:nr
        r = rr[k]
        rf_s[k] = r * fs(r, cfns)
        rf_t[k] = r * ft(r, cfns)
        rf_l[k] = r * fl1(r, cfns)
        D_r[k]  = cfns.fDr(r)
    end
    return rf_s, rf_t, rf_l, D_r
end

# =============================================================================
# Streaming build (Phase 2+3 fast variant)
# =============================================================================

"""
    build_and_export_streaming(; Nr, nR, dlnR, ellmin, ellmax, tiers,
                                      datadir, outname, load_base, use_float32,
                                      max_size_gb)

Phase 2+3 optimized streaming build. Arguments and output format are identical
to `PowerFull.build_and_export_streaming` of the pipeline, so the
output HDF5 meta + part files can be loaded by the same calcClGR readers.
"""
function build_and_export_streaming(;
    Nr::Int=4096,
    nR::Int=2049,
    dlnR::Float64=0.002,
    ellmin::Int=2,
    ellmax::Int=500,
    tiers::Union{Nothing,Vector{@NamedTuple{dlnR::Float64,ellmin::Int,ellmax::Int,nR::Int}}}=nothing,
    datadir::String="./results",
    outname::String="ClGR_integrals",
    load_base::Bool=true,
    lucas_patch_w_0_22_ell2::String="",  # opt-in: HDF5 path; replaces base/w_0_2_2[ell=2] slice
    use_float32::Bool=true,
    max_size_gb::Float64=5.0,
    cosmo_funcr::String=joinpath(@__DIR__, "..", "data", "cosmo_funcr.txt"),
    Rcut_min::Float64=0.0,        # 0.0 = no lower cut
    Rcut_max::Float64=Inf,        # Inf = no upper cut
    base_only::Bool=false,        # Append base arrays to existing meta + part files
    mu0::Float64=0.0,             # MG: μ amplitude (GR = 0)
    Sigma0::Float64=0.0,          # MG: Σ amplitude (GR = 0)
)
    # -------------------------------------------------------------------------
    # base_only: read all parameters from existing meta, leave integrated/ alone
    # -------------------------------------------------------------------------
    if base_only
        meta_file_init = "$(outname)_meta.h5"
        isfile(meta_file_init) || error("base_only mode: meta file not found: $meta_file_init")
        loaded_tiers = h5open(meta_file_init, "r") do f
            Nr = Int(read(f["metadata/Nr"]))
            nR_global = Int(read(f["metadata/nR"]))
            Rcut_min = Float64(read(f["metadata/Rcut_min"]))
            rmx_raw = Float64(read(f["metadata/Rcut_max"]))
            Rcut_max = rmx_raw < 0 ? Inf : rmx_raw
            tier_count = Int(read(f["metadata/tier_count"]))
            tlist = [@NamedTuple{dlnR::Float64,ellmin::Int,ellmax::Int,nR::Int}((
                Float64(read(f["metadata/tier$(t)_dlnR"])),
                Int(read(f["metadata/tier$(t)_ellmin"])),
                Int(read(f["metadata/tier$(t)_ellmax"])),
                haskey(f, "metadata/tier$(t)_nR") ? Int(read(f["metadata/tier$(t)_nR"])) : nR_global))
                for t in 1:tier_count]
            return (Nr, nR_global, Rcut_min, Rcut_max, tlist)
        end
        Nr, nR, Rcut_min, Rcut_max, loaded_tlist = loaded_tiers
        tiers = loaded_tlist
        load_base = true
        println("[base_only] Loaded params from $meta_file_init")
        println("           Nr=$Nr  nR(global)=$nR")
        println("           Rcut_min=$Rcut_min  Rcut_max=$Rcut_max")
        for (i, t) in enumerate(loaded_tlist)
            println("           tier $i: dlnR=$(t.dlnR)  ell=$(t.ellmin)..$(t.ellmax)  nR=$(t.nR)")
        end
    end

    # -------------------------------------------------------------------------
    # Tier list + banner
    # -------------------------------------------------------------------------
    if tiers === nothing
        tiers_list = [@NamedTuple{dlnR::Float64,ellmin::Int,ellmax::Int,nR::Int}((dlnR, ellmin, ellmax, nR))]
    else
        tiers_list = tiers
    end

    println("="^60)
    println("Building IntegralCollection (STREAMING-FAST mode, Phase 2+3)")
    println("="^60)
    println("Parameters:")
    println("  Nr = $Nr, nR(default) = $nR")
    for (tidx, tier) in enumerate(tiers_list)
        println("  Tier $tidx: dlnR=$(tier.dlnR), ell=$(tier.ellmin)-$(tier.ellmax), nR=$(tier.nR)")
    end
    if (Rcut_min > 0.0) || isfinite(Rcut_max)
        println("  Rcut_min = $Rcut_min, Rcut_max = $Rcut_max  (R-cutoff ON)")
    else
        println("  Rcut: OFF  (full R-range)")
    end
    println("  datadir = $datadir")
    println("  load_base = $load_base")
    println("  use_float32 = $use_float32")
    println("  max_size_gb = $max_size_gb")
    println("  threads = $(Threads.nthreads())")
    println()

    # -------------------------------------------------------------------------
    # Step 1: Load grid info from first tier's first w_integrand file
    # (base_only: read rr from existing meta to skip a ~13 GB cache load.)
    # -------------------------------------------------------------------------
    local rr::Vector{Float64}
    local w_integrand_1_t1::Array{Float64,3}
    if base_only
        rr = h5open("$(outname)_meta.h5", "r") do f
            Vector{Float64}(read(f["grid/rr"]))
        end
        w_integrand_1_t1 = Array{Float64,3}(undef, 0, 0, 0)
        println("[base_only] rr grid loaded from meta ($(length(rr)) points)")
    else
        tier1 = tiers_list[1]
        println("Loading grid info from tier 1 case 1 w_integrand file...")
        w_integrand_1_t1, rr, _, _ = PowerFull._load_w_integrand_file(1, Nr, tier1.nR, tier1.dlnR, tier1.ellmin, tier1.ellmax, datadir)
    end
    nr   = length(rr)
    Δlnr = log(rr[2] / rr[1])

    # Per-tier ell counts (cumulative offsets into the combined array)
    tier_nells   = [length(tier.ellmin:tier.ellmax) for tier in tiers_list]
    tier_offsets = cumsum([0; tier_nells])[1:end-1]  # 0-based start offset per tier
    total_nell   = sum(tier_nells)

    aell_combined = Int[]
    for tier in tiers_list
        append!(aell_combined, collect(tier.ellmin:tier.ellmax))
    end

    println("  r grid: $nr points, range [$(round(rr[1], digits=1)), $(round(rr[end], digits=1))] Mpc/h")
    println("  ℓ values: $total_nell, range [$(aell_combined[1]), $(aell_combined[end])]")

    println("Loading cosmological functions from: $cosmo_funcr")
    cfns = cosmofn(cosmo_funcr)
    mg_active = (mu0 != 0.0) || (Sigma0 != 0.0)
    if mg_active
        println("Modified gravity ACTIVE: μ₀ = $mu0, Σ₀ = $Sigma0  (local limit, no scale dependence)")
        mg = build_mg_model(cfns; mu0=mu0, Sigma0=Sigma0)
        println("  Ω_Λ = $(round(mg.OmLambda, digits=5));  solving modified growth ODE for D₀,f₀")
        rf_s, rf_t, rf_l, D_r = compute_kernel_arrays_mg(rr, cfns, mg)
    else
        println("Modified gravity inactive (GR): μ₀ = 0, Σ₀ = 0")
        rf_s, rf_t, rf_l, D_r = compute_kernel_arrays(rr, cfns)
    end

    inv3D = zeros(Float64, nr)
    @inbounds for i in 1:nr
        inv3D[i] = 3.0 / D_r[i]
    end

    function compute_RR_for_tier(tier_dlnR::Float64, tier_nR::Int)::Vector{Float64}
        half = (tier_nR - 1) ÷ 2
        lnRR = collect(range(-half * tier_dlnR, stop=half * tier_dlnR, length=tier_nR))
        return exp.(lnRR)
    end

    # Hard R-cutoff: drop columns of w_int (and matching RR entries) where
    # R is outside [Rcut_min, Rcut_max].  Out-of-range R contributions in
    # the subsequent interpolation become zero (PowerFullInterp boundary).
    # No-op when params are at defaults.
    Rcut_active = (Rcut_min > 0.0) || isfinite(Rcut_max)
    function apply_Rcut(RR::Vector{Float64}, w_int::Array{Float64,3})
        Rcut_active || return RR, w_int
        mask = (RR .>= Rcut_min) .& (RR .<= Rcut_max)
        return RR[mask], w_int[:, mask, :]
    end

    # -------------------------------------------------------------------------
    # Step 2: Estimate split layout, write meta + skeletons
    # (base_only: read existing layout from meta, do NOT rewrite skeletons.)
    # -------------------------------------------------------------------------
    local n_parts::Int
    local ell_ranges::Matrix{Int}
    local part_files::Vector{String}
    local meta_file::String

    if base_only
        meta_file = "$(outname)_meta.h5"
        n_parts, ell_ranges = h5open(meta_file, "r") do f
            np = Int(read(f["metadata/n_parts"]))
            er = Matrix{Int}(read(f["metadata/ell_ranges"]))
            return (np, er)
        end
        part_files = ["$(outname)_part_$(lpad(i, 3, '0')).h5" for i in 1:n_parts]
        for pf in part_files
            isfile(pf) || error("base_only mode: part file missing: $pf")
        end
        println("\n[base_only] Existing layout: $n_parts part files, ell_ranges from meta.")
    else
    n_integrated = 39
    n_base = load_base ? 22 : 0
    n_arrays = n_integrated + n_base

    bytes_per_element = use_float32 ? 4 : 8
    bytes_per_ell = nr * nr * n_arrays * bytes_per_element
    max_bytes = max_size_gb * 1024^3
    ell_per_part = max(1, floor(Int, max_bytes / bytes_per_ell))
    n_parts = ceil(Int, total_nell / ell_per_part)

    println("\nSplit plan: $total_nell ell values → $n_parts parts (~$ell_per_part ell/part)")
    println("Estimated size per part: $(round(bytes_per_ell * ell_per_part / 1024^3, digits=2)) GB")

    ell_ranges = Matrix{Int}(undef, n_parts, 2)
    for i in 1:n_parts
        ell_start = (i - 1) * ell_per_part + 1
        ell_end = min(i * ell_per_part, total_nell)
        ell_ranges[i, :] = [ell_start, ell_end]
    end

    base_name_only = Base.basename(outname)
    part_files     = ["$(outname)_part_$(lpad(i, 3, '0')).h5" for i in 1:n_parts]
    part_filenames = ["$(base_name_only)_part_$(lpad(i, 3, '0')).h5" for i in 1:n_parts]

    meta_file = "$(outname)_meta.h5"
    mkpath(dirname(abspath(meta_file)))
    h5open(meta_file, "w") do f
        g_grid = create_group(f, "grid")
        g_grid["rr"] = rr
        g_grid["ell_values"] = aell_combined

        g_meta = create_group(f, "metadata")
        g_meta["Nr"] = Nr
        g_meta["nR"] = nR
        g_meta["logRmin"] = 0.0
        g_meta["logRmax"] = 0.0
        g_meta["Rcut_min"] = Rcut_min
        g_meta["Rcut_max"] = isfinite(Rcut_max) ? Rcut_max : -1.0
        g_meta["nr_actual"] = nr
        g_meta["nell"] = total_nell
        g_meta["dtype"] = use_float32 ? "float32" : "float64"
        g_meta["compress_level"] = 0
        g_meta["format_version"] = "2.1"
        g_meta["n_parts"] = n_parts
        g_meta["ell_ranges"] = ell_ranges
        # MG provenance (0,0 ⇒ GR).
        g_meta["mu0"] = mu0
        g_meta["Sigma0"] = Sigma0

        g_meta["tier_count"] = length(tiers_list)
        for (tidx, tier) in enumerate(tiers_list)
            g_meta["tier$(tidx)_dlnR"] = tier.dlnR
            g_meta["tier$(tidx)_ellmin"] = tier.ellmin
            g_meta["tier$(tidx)_ellmax"] = tier.ellmax
            g_meta["tier$(tidx)_nR"] = tier.nR
        end

        f["part_files"] = part_filenames
    end
    println("Written: $meta_file")

    for part_idx in 1:n_parts
        h5open(part_files[part_idx], "w") do f
            f["ell_start"] = ell_ranges[part_idx, 1]
            f["ell_end"]   = ell_ranges[part_idx, 2]
            create_group(f, "base")
            create_group(f, "integrated")
        end
    end
    println("Created $n_parts empty part files.")
    end  # if !base_only

    # -------------------------------------------------------------------------
    # Helper: permutedims [nr, nr, nell] → [nell, nr, nr] and write + free.
    # -------------------------------------------------------------------------
    n_written = Ref(0)
    function write_nnl_and_free!(group::String, key_tuple::Tuple,
                                  data_nnl::Array{Float64,3})::Nothing
        # data_nnl has layout [nr, nr, nell]. Write expects [nell, nr, nr].
        data_nlr = permutedims(data_nnl, (3, 1, 2))
        key = PowerFull._make_key(key_tuple...)
        PowerFull._write_array_to_split_h5!(part_files, ell_ranges, group, key, data_nlr, use_float32)
        n_written[] += 1
        return nothing
    end

    # -------------------------------------------------------------------------
    # Per-ell compute cores (write into pre-allocated [nr, nr, nell] slices).
    #
    # Key changes from Phase 1:
    #   - Output arrays use [nr, nr, nell] layout (contiguous per-ell slice).
    #   - ;r' direction uses permutedims + prefix_sum_axis1! + permutedims
    #     (avoids the strided prefix_sum_axis2! access pattern).
    #   - Copy loops annotated with @simd for vectorized accumulation.
    #
    # Per-task buffers (w_phys, w_phys_T, buf, buf_T, buf_final) are allocated
    # per ell-loop iteration; safe under Julia 1.7+ task migration.
    # -------------------------------------------------------------------------

    function _compute_1D_multi_nnl!(int_r_s::AbstractArray{Float64,3},
                                     int_rp_s::AbstractArray{Float64,3},
                                     int_r_t::AbstractArray{Float64,3},
                                     int_rp_t::AbstractArray{Float64,3},
                                     int_r_l::AbstractArray{Float64,3},
                                     int_rp_l::AbstractArray{Float64,3},
                                     w_integrand::Array{Float64,3},
                                     inv3D_vec::Vector{Float64},
                                     RR_tier::Vector{Float64},
                                     rf_s_in::Vector{Float64},
                                     rf_t_in::Vector{Float64},
                                     rf_l_in::Vector{Float64},
                                     asymmetric::Bool)::Nothing
        nell_t = size(w_integrand, 3)

        @threads for ell_idx in 1:nell_t
            # Per-task buffers
            w_phys   = zeros(Float64, nr, nr)
            w_phys_T = zeros(Float64, nr, nr)
            buf      = zeros(Float64, nr, nr)  # prefix_sum output (standard layout)
            buf_T    = zeros(Float64, nr, nr)  # prefix_sum output (transposed layout)
            buf_rp   = zeros(Float64, nr, nr)  # ;r' result in standard layout

            # 1) Interpolate w_phys once
            PowerFullInterp.interpolate_to_physical_grid!(
                w_phys, @view(w_integrand[:, :, ell_idx]), rr, RR_tier)

            # 2) Precompute w_phys_T for ;r' direction (shared by all 3 kernels)
            permutedims!(w_phys_T, w_phys, (2, 1))

            # 3) For each kernel: compute ;r and ;r' and accumulate
            for (rf, int_r_arr, int_rp_arr) in (
                    (rf_s_in, int_r_s, int_rp_s),
                    (rf_t_in, int_r_t, int_rp_t),
                    (rf_l_in, int_r_l, int_rp_l))

                # ;r direction: sym/asym identical.
                # buf[i, j] = Σ_{k≤i} rf[k] * w_phys[k, j]  (axis1, contig)
                # int_r[i, j, ell] = inv3D[i] * buf[i, j]
                PowerFull.prefix_sum_axis1!(buf, Δlnr, rf, w_phys)
                @inbounds for j in 1:nr
                    @simd for i in 1:nr
                        int_r_arr[i, j, ell_idx] = inv3D_vec[i] * buf[i, j]
                    end
                end

                # ;r' direction:
                #   SYM:  result[i,j] = Σ_{k≤j} rf[k] w_phys[i,k]
                #         axis1 on w_phys_T → buf_T[a,b] = Σ_{k≤a} rf[k] w_phys[b,k]
                #         result[i,j] = buf_T[j,i]  →  permutedims back.
                #   ASYM: result[i,j] = Σ_{k≤j} rf[k] w_phys[k,i]  (j, jp swap)
                #         axis1 on w_phys directly → buf[a,b] = Σ_{k≤a} rf[k] w_phys[k,b]
                #         result[i,j] = buf[j,i]  →  permutedims back.
                if asymmetric
                    PowerFull.prefix_sum_axis1!(buf, Δlnr, rf, w_phys)
                    permutedims!(buf_rp, buf, (2, 1))
                else
                    PowerFull.prefix_sum_axis1!(buf_T, Δlnr, rf, w_phys_T)
                    permutedims!(buf_rp, buf_T, (2, 1))
                end

                @inbounds for j in 1:nr
                    @simd for i in 1:nr
                        int_rp_arr[i, j, ell_idx] = inv3D_vec[j] * buf_rp[i, j]
                    end
                end
            end
        end

        return nothing
    end

    # 2D multi-kernel compute with N kernel pairs sharing w_int (and w_phys).
    # Pass 1 (axis1 on w_phys) is already contiguous. Pass 2 (axis2 on buf)
    # is strided; we rewrite via axis1-on-transpose trick for each kernel pair.
    function _compute_2D_multi_nnl!(outs::NTuple{N,<:AbstractArray{Float64,3}},
                                     w_integrand::Array{Float64,3},
                                     inv3D_vec::Vector{Float64},
                                     RR_tier::Vector{Float64},
                                     rf_pairs::NTuple{N,Tuple{Vector{Float64},Vector{Float64}}}
                                     )::Nothing where N
        nell_t = size(w_integrand, 3)

        @threads for ell_idx in 1:nell_t
            w_phys   = zeros(Float64, nr, nr)
            buf1     = zeros(Float64, nr, nr)
            buf1_T   = zeros(Float64, nr, nr)
            buf2     = zeros(Float64, nr, nr)
            buf2_T   = zeros(Float64, nr, nr)

            PowerFullInterp.interpolate_to_physical_grid!(
                w_phys, @view(w_integrand[:, :, ell_idx]), rr, RR_tier)

            for k in 1:N
                rf1, rf2 = rf_pairs[k]

                # Pass 1: axis1 on w_phys with weighted kernel rf1[a] * rf2[j].
                # Original prefix_sum_2D! does this inline; we emulate with axis1
                # using rf1 (scaled by rf2[j] per column inside).
                # buf1[i, j] = Δlnr * [Σ_{k≤i} rf1[k] w_phys[k,j] - half-corrections] * rf2[j]
                # Easier: call prefix_sum_axis1! with rf1 → buf1[i, j] = Σ.. rf1[k] w_phys[k,j],
                # then per-column scale rf2[j].
                PowerFull.prefix_sum_axis1!(buf1, Δlnr, rf1, w_phys)
                @inbounds for j in 1:nr
                    cj = rf2[j]
                    @simd for i in 1:nr
                        buf1[i, j] *= cj
                    end
                end

                # Pass 2: prefix sum along second axis of buf1.
                # Use transpose + axis1 with unit weights to get cache-friendly access.
                #   result[i,j] = Σ_{b≤j} buf1[i, b]
                #   Via: buf1_T[j, i] = buf1[i, j]
                #        axis1 with unit weights on buf1_T:
                #           buf2_T[a, b] = Σ_{k≤a} 1 * buf1_T[k, b]
                #                        = Σ_{k≤a} buf1[b, k]
                #        result[i,j] = buf2_T[j, i]  →  permutedims to buf2 standard.
                permutedims!(buf1_T, buf1, (2, 1))
                _prefix_sum_axis1_unit!(buf2_T, Δlnr, buf1_T)
                permutedims!(buf2, buf2_T, (2, 1))

                # Accumulate into output with inv3D[i] * inv3D[j] scaling
                int_2D = outs[k]
                @inbounds for j in 1:nr
                    cj = inv3D_vec[j]
                    @simd for i in 1:nr
                        int_2D[i, j, ell_idx] = inv3D_vec[i] * cj * buf2[i, j]
                    end
                end
            end
        end

        return nothing
    end

    # -------------------------------------------------------------------------
    # 1D dispatcher: allocate per-case [nr, nr, total_nell] output arrays,
    # fill tier-by-tier into contiguous slices, permutedims to [nell, nr, nr]
    # and write. Supports file sharing via shared_w_int.
    # -------------------------------------------------------------------------
    # NOTE: lensing-related kernel symbols use tilde prefix (:tl, :tscrL, :tscrY,
    # :tscrZ, :tscrl) to distinguish them from the paper's full-observable
    # symbols.  The paper's l, L, Y, Z include a (r-r'')/(r r'') factor and an
    # ell(ell+1)/2 prefactor that our stored integrals do NOT; see
    # pkfull_integrals.f90 header and the Limber validation appendix.  Non-
    # lensing kernels (s, t, S, T, X) match the paper directly and carry no
    # tilde.  HDF5 keys reflect these names: "tl_m2_0_0_r", "tscrL_m4_0_0_r_rp",
    # etc.
    function process_case_tiers_1D_multi!(case_num::Int, mode::Symbol,
                                           p::Int, j_val::Int, jp_val::Int, n_val::Int;
                                           types=(:s, :t, :tl),
                                           shared_w_int::Union{Nothing,Vector{Array{Float64,3}}}=nothing)
        # Pre-allocate full per-case output arrays in [nr, nr, total_nell] layout.
        int_r_s  = zeros(Float64, nr, nr, total_nell)
        int_rp_s = zeros(Float64, nr, nr, total_nell)
        int_r_t  = zeros(Float64, nr, nr, total_nell)
        int_rp_t = zeros(Float64, nr, nr, total_nell)
        int_r_l  = zeros(Float64, nr, nr, total_nell)
        int_rp_l = zeros(Float64, nr, nr, total_nell)

        asymmetric = (mode != :symmetric)

        for (tidx, tier) in enumerate(tiers_list)
            RR_tier = compute_RR_for_tier(tier.dlnR, tier.nR)
            nell_t  = tier_nells[tidx]
            off     = tier_offsets[tidx]
            ell_rng = (off+1):(off+nell_t)

            local w_int::Array{Float64,3}
            if shared_w_int !== nothing
                w_int = shared_w_int[tidx]
            elseif tidx == 1 && case_num == 1
                w_int = w_integrand_1_t1
            else
                w_int, _, _, _ = PowerFull._load_w_integrand_file(case_num, Nr, tier.nR, tier.dlnR, tier.ellmin, tier.ellmax, datadir)
            end

            RR_tier_use, w_int_use = apply_Rcut(RR_tier, w_int)
            _compute_1D_multi_nnl!(
                @view(int_r_s[:, :, ell_rng]),  @view(int_rp_s[:, :, ell_rng]),
                @view(int_r_t[:, :, ell_rng]),  @view(int_rp_t[:, :, ell_rng]),
                @view(int_r_l[:, :, ell_rng]),  @view(int_rp_l[:, :, ell_rng]),
                w_int_use, inv3D, RR_tier_use, rf_s, rf_t, rf_l, asymmetric)
        end

        # Write each kernel × direction, freeing memory per array.
        arrs_r  = (int_r_s,  int_r_t,  int_r_l)
        arrs_rp = (int_rp_s, int_rp_t, int_rp_l)
        for (k, kernel_type) in enumerate(types)
            if mode == :symmetric
                write_nnl_and_free!("integrated", (kernel_type, p, j_val, jp_val, :r),  arrs_r[k])
                write_nnl_and_free!("integrated", (kernel_type, p, j_val, jp_val, :rp), arrs_rp[k])
            else  # asymmetric: j/jp swapped for ;r'
                write_nnl_and_free!("integrated", (kernel_type, p, j_val, jp_val, :r),  arrs_r[k])
                write_nnl_and_free!("integrated", (kernel_type, p, jp_val, j_val, :rp), arrs_rp[k])
            end
        end
        GC.gc()
    end

    # 2D multi-kernel dispatcher (case 6 or 7).
    function process_case_tiers_2D_multi!(case_num::Int,
                                           kernel_types::NTuple{N,Symbol},
                                           rf_pairs::NTuple{N,Tuple{Vector{Float64},Vector{Float64}}};
                                           do_transpose::Bool=false,
                                           shared_w_int::Union{Nothing,Vector{Array{Float64,3}}}=nothing
                                           ) where N
        outs = ntuple(_ -> zeros(Float64, nr, nr, total_nell), N)

        for (tidx, tier) in enumerate(tiers_list)
            RR_tier = compute_RR_for_tier(tier.dlnR, tier.nR)
            nell_t  = tier_nells[tidx]
            off     = tier_offsets[tidx]
            ell_rng = (off+1):(off+nell_t)

            w_int = if shared_w_int !== nothing
                shared_w_int[tidx]
            else
                load_result, _, _, _ = PowerFull._load_w_integrand_file(case_num, Nr, tier.nR, tier.dlnR, tier.ellmin, tier.ellmax, datadir)
                load_result
            end

            RR_tier_use, w_int_use = apply_Rcut(RR_tier, w_int)
            out_views = ntuple(k -> @view(outs[k][:, :, ell_rng]), N)
            _compute_2D_multi_nnl!(out_views, w_int_use, inv3D, RR_tier_use, rf_pairs)
        end

        for k in 1:N
            arr = outs[k]
            write_nnl_and_free!("integrated", (kernel_types[k], -4, 0, 0, :r_rp), arr)

            if do_transpose
                # Transposed form: swap i <-> j in each ell slice.
                arr_T = similar(arr)
                @inbounds for ell_idx in 1:total_nell
                    @simd for j in 1:nr
                        for i in 1:nr
                            arr_T[i, j, ell_idx] = arr[j, i, ell_idx]
                        end
                    end
                end
                write_nnl_and_free!("integrated", (kernel_types[k], -4, 0, 0, :rp_r), arr_T)
            end
        end
        GC.gc()
    end

    # -------------------------------------------------------------------------
    # Step 3: Process cases 1-7 (multi-kernel, multi-tier, [nr, nr, nell])
    # -------------------------------------------------------------------------
    t_total = @elapsed begin

    if !base_only
    println("\n[Streaming-fast] Case 1: (-2,0,0,0) → s,t,l ;r and ;r'...")
    @time process_case_tiers_1D_multi!(1, :symmetric, -2, 0, 0, 0)
    w_integrand_1_t1 = Array{Float64,3}(undef, 0, 0, 0); GC.gc()

    println("[Streaming-fast] Case 2: (-2,0,2,0) → s,t,l ;r and transposed ;r'...")
    @time process_case_tiers_1D_multi!(2, :asymmetric, -2, 0, 2, 0)

    println("[Streaming-fast] Case 3: (-3,0,1,0) → s,t,l ;r and transposed ;r'...")
    @time process_case_tiers_1D_multi!(3, :asymmetric, -3, 0, 1, 0)

    # Cases 4, 6, 7 share the SAME w_integrand (p=-4, j=0, jp=0, n=0).
    # Load once per tier and reuse for all three cases.
    println("[Streaming-fast] Case 4: (-4,0,0,0) → s,t,l ;r and ;r'...")
    println("  Loading case-4 w_integrand files (shared by cases 4, 6, 7)...")
    shared_case4 = Vector{Array{Float64,3}}(undef, length(tiers_list))
    for (tidx, tier) in enumerate(tiers_list)
        w_int, _, _, _ = PowerFull._load_w_integrand_file(4, Nr, tier.nR, tier.dlnR, tier.ellmin, tier.ellmax, datadir)
        shared_case4[tidx] = w_int
    end
    @time process_case_tiers_1D_multi!(4, :symmetric, -4, 0, 0, 0; shared_w_int=shared_case4)

    println("[Streaming-fast] Case 5: (-4,0,0,-1) → scrs,scrt,tscrl ;r and ;r'...")
    @time process_case_tiers_1D_multi!(5, :symmetric, -4, 0, 0, -1; types=(:scrs, :scrt, :tscrl))

    println("[Streaming-fast] Case 6: (-4,0,0,0) 2D → scrS,scrT,tscrL ;r,r' (shared w_int)...")
    @time process_case_tiers_2D_multi!(6, (:scrS, :scrT, :tscrL),
        ((rf_s, rf_s), (rf_t, rf_t), (rf_l, rf_l));
        shared_w_int=shared_case4)

    println("[Streaming-fast] Case 7: (-4,0,0,0) 2D → scrX,tscrY,tscrZ ;r,r' and ;r',r (shared w_int)...")
    @time process_case_tiers_2D_multi!(7, (:scrX, :tscrY, :tscrZ),
        ((rf_s, rf_t), (rf_s, rf_l), (rf_t, rf_l));
        do_transpose=true, shared_w_int=shared_case4)

    # Free case-4 shared buffers
    for tidx in 1:length(tiers_list)
        shared_case4[tidx] = Array{Float64,3}(undef, 0, 0, 0)
    end
    shared_case4 = nothing; GC.gc()
    end  # if !base_only

    # -------------------------------------------------------------------------
    # Step 4: Optional base arrays (w, u, v)
    # -------------------------------------------------------------------------
    if load_base
        println("\n[Streaming-fast] Loading base wpljjprime functions (w, u, v) on physical grid...")

        function load_and_write_base!(type_sym::Symbol, p::Int, j::Int, jp::Int, n::Int)::Nothing
            # Allocate full [nr, nr, total_nell] output
            out = zeros(Float64, nr, nr, total_nell)

            for (tidx, tier) in enumerate(tiers_list)
                RR_tier = compute_RR_for_tier(tier.dlnR, tier.nR)
                tier_ells = collect(tier.ellmin:tier.ellmax)
                nell_tier = length(tier_ells)
                off = tier_offsets[tidx]

                # Load raw w_integrand for this (type, p, j, jp, n) tier
                w_integrand_raw = zeros(Float64, nr, tier.nR, nell_tier)
                for (i, ell) in enumerate(tier_ells)
                    try
                        data_loc = wpljjprime(p, j, jp, n, ell,
                            Nr=Nr, nR=tier.nR, dlnR=tier.dlnR,
                            ellmin=tier.ellmin, ellmax=tier.ellmax, datadir=datadir)
                        w_integrand_raw[:, :, i] = data_loc.wrRl
                    catch e
                        @warn "Failed to load wpljjprime(p=$p, j=$j, jp=$jp, n=$n, ell=$ell): $e"
                    end
                end

                # Apply Rcut (no-op when defaults).
                RR_tier_use, w_integrand_raw_use = apply_Rcut(RR_tier, w_integrand_raw)

                # Interpolate per ell, write into out[:, :, off+i]
                @threads for ell_idx in 1:nell_tier
                    w_phys_2d = zeros(Float64, nr, nr)
                    PowerFullInterp.interpolate_to_physical_grid!(
                        w_phys_2d, @view(w_integrand_raw_use[:, :, ell_idx]), rr, RR_tier_use)
                    @inbounds for j in 1:nr
                        @simd for i in 1:nr
                            out[i, j, off + ell_idx] = w_phys_2d[i, j]
                        end
                    end
                end
                w_integrand_raw = nothing
            end

            write_nnl_and_free!("base", (type_sym, p, j, jp, :none), out)
            GC.gc()
            return nothing
        end

        # The 22 base queries fall across at most 9 TwoFAST output files
        # (keyed by `indx` in PowerFull.LOOKUP_TABLE).  Group them by
        # `indx` and process each group back-to-back so wpljjprime's
        # per-(indx, Nr, nR, dlnR) file cache is reused within the group,
        # then call PowerFull.clear_cache! + GC.gc between groups to
        # release the ~3·13 GB of cached w_integrand before loading the
        # next file.  This caps peak memory during the base phase at
        # roughly (one indx's 3-tier cache + one out array) ≈ 45 GB,
        # which is what lets the full build (load_base=true) fit on a
        # 96 GB RNA node — previously the unbounded cache growth
        # accumulated all 9 × 3 = 27 cache entries (~350 GB).
        w_combos = [
            (0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2),
            (-1, 0, 1), (-1, 1, 0), (-1, 1, 2), (-1, 2, 1),
            (-2, 0, 0), (-2, 0, 2), (-2, 2, 0), (-2, 1, 1),
            (-3, 0, 1), (-3, 1, 0),
            (-4, 0, 0)
        ]
        u_combos = [
            (-2, 0, 0), (-2, 0, 2), (-2, 2, 0),
            (-3, 0, 1), (-3, 1, 0),
            (-4, 0, 0)
        ]
        base_queries = Tuple{Symbol,Int,Int,Int,Int}[]
        for (p, j, jp) in w_combos; push!(base_queries, (:w, p, j, jp, 0));  end
        for (p, j, jp) in u_combos; push!(base_queries, (:u, p, j, jp, -1)); end
        push!(base_queries, (:v, -4, 0, 0, -2))

        # Group by file index via LOOKUP_TABLE.
        groups = Dict{Int, Vector{Tuple{Symbol,Int,Int,Int,Int}}}()
        for q in base_queries
            (ty, p, j, jp, n) = q
            key = (p, j, jp, n)
            haskey(PowerFull.LOOKUP_TABLE, key) ||
                error("wpljjprime key $(key) missing from LOOKUP_TABLE")
            indx, _ = PowerFull.LOOKUP_TABLE[key]
            push!(get!(groups, indx, Tuple{Symbol,Int,Int,Int,Int}[]), q)
        end

        for indx in sort!(collect(keys(groups)))
            for q in groups[indx]
                (ty, p, j, jp, n) = q
                println("  Loading $(ty)^{$p}_{$j,$jp} (indx=$indx) ...")
                load_and_write_base!(ty, p, j, jp, n)
            end
            # Free the wpljjprime file cache for this indx (three tier
            # cache entries per indx, each ~13 GB at Nr=4096 / nR=2049).
            PowerFull.clear_cache!()
            GC.gc()
        end

        # Optional: inject Lucas-direct w^0_{ell=2, j=2, jp=2} slice over the
        # FFTLog-derived value to fix the 9-term cancellation hot-spot at
        # ell=2 small-R cross pairs (~2% off vs Lucas direct, see wiki
        # "Lucas patch w_0_22 ell=2" section).  Standalone use of w_22
        # benefits; production Cl assembly is mostly insensitive (effect
        # is ell=2 only, < few-percent on small-Cl pairs).
        if !isempty(lucas_patch_w_0_22_ell2)
            isfile(lucas_patch_w_0_22_ell2) ||
                error("--lucas-patch-w_0_22 file missing: $lucas_patch_w_0_22_ell2")
            println("\n[Lucas patch] Injecting w_0_2_2 ell=2 slice from $lucas_patch_w_0_22_ell2 ...")
            hp = h5open(lucas_patch_w_0_22_ell2, "r")
            try
                M = read(hp["w_0_2_2_ell2"])::Matrix{Float32}
                rr_patch = read(hp["rr"])::Vector{Float64}
                @assert size(M) == (nr, nr) "patch grid mismatch ($(size(M)) vs ($nr, $nr))"
                @assert all(rr_patch .≈ rr) "patch rr-grid mismatch"
                # ell=2 must be the first ell, in the first part file.
                first_ell = if !isempty(tiers_list); tiers_list[1].ellmin; else 0; end
                @assert first_ell == 2 "first ell must be 2 to apply ell=2 patch (got $first_ell)"
                part1 = part_files[1]
                hw = h5open(part1, "r+")
                try
                    ds = hw["base/w_0_2_2"]
                    ds[1, :, :] = M    # ell=2 slice index = 1
                finally
                    close(hw)
                end
                println("[Lucas patch] Done (replaced base/w_0_2_2[ell=2] in $(basename(part1)))")
            finally
                close(hp)
            end
        end
    end

    # -------------------------------------------------------------------------
    # Step 5: Paper-observable post-pass.
    # The build above writes BARE (tilde) lensing blocks to HDF5: tl_*,
    # tscrl_*, tscrY_*, tscrZ_*, tscrL_*.  The paper's lensing observables
    # carry an additional ell(ell+1)/2 prefactor and an (r-r'')/(r r'')
    # subtraction against a companion block.  Convert each tilde block to
    # its paper observable in place, writing under the paper key
    # (l_*, scrl_*, scrY_*, scrZ_*, scrL_*) and deleting the tilde key.
    # Companions (t, scrt, scrX, scrT) stay; they are used by their own
    # 19-term coefficients.
    #
    # Conversions (paper appendix Eqs. l_jj', l_jj'_png, Y_jj', Z_jj', L_jj'):
    #   l       = (ell(ell+1)/2)   * (tl    - t/r_src)
    #   scrl    = (ell(ell+1)/2)   * (tscrl - scrt/r_src)
    #   scrY    = (ell(ell+1)/2)   * (tscrY - scrX/r_lens)
    #   scrZ    = (ell(ell+1)/2)   * (tscrZ - scrT/r_lens)
    #   scrL    = (ell(ell+1)/2)^2 * (tscrL - tscrZ_rpr/r2 - tscrZ_rrp/r1 + scrT/(r1 r2))
    # Run scrL BEFORE scrZ within each part because scrL consumes raw tscrZ.
    # -------------------------------------------------------------------------
    if !base_only
        println("\n[Streaming-fast] Post-pass: converting tilde lensing blocks → paper observables...")
        t_pp = @elapsed begin
            lens1d_pjj = [(-2, 0, 0), (-2, 2, 0), (-2, 0, 2),
                          (-3, 1, 0), (-3, 0, 1),
                          (-4, 0, 0)]
            for p_idx in 1:n_parts
                part_path = part_files[p_idx]
                ell_lo    = ell_ranges[p_idx, 1]
                ell_hi    = ell_ranges[p_idx, 2]
                nell_part = ell_hi - ell_lo + 1
                ells_in_part = aell_combined[ell_lo:ell_hi]
                ll = [Float64(ell) * Float64(ell+1) / 2.0 for ell in ells_in_part]

                h5open(part_path, "r+") do f
                    g = f["integrated"]

                    # 1D LOS lensing
                    for (p, j_, jp_) in lens1d_pjj, sub in (:r, :rp)
                        tl_key = PowerFull._make_key(:tl, p, j_, jp_, sub)
                        t_key  = PowerFull._make_key(:t,  p, j_, jp_, sub)
                        l_key  = PowerFull._make_key(:l,  p, j_, jp_, sub)
                        haskey(g, tl_key) || continue
                        haskey(g, t_key)  || error("companion $t_key missing for $tl_key in $(basename(part_path))")
                        tl_arr = read(g[tl_key])::Array{Float32,3}
                        t_arr  = read(g[t_key])::Array{Float32,3}
                        l_arr  = similar(tl_arr)
                        @inbounds for jj in 1:nr, i in 1:nr
                            r_src = sub === :r ? rr[i] : rr[jj]
                            inv_r = 1.0 / r_src
                            for e in 1:nell_part
                                l_arr[e, i, jj] = Float32(
                                    ll[e] * (Float64(tl_arr[e, i, jj]) - Float64(t_arr[e, i, jj]) * inv_r))
                            end
                        end
                        g[l_key] = l_arr
                        delete_object(g, tl_key)
                    end

                    # 1D PNG fraktur l
                    for sub in (:r, :rp)
                        tscrl_key = PowerFull._make_key(:tscrl, -4, 0, 0, sub)
                        scrt_key  = PowerFull._make_key(:scrt,  -4, 0, 0, sub)
                        scrl_key  = PowerFull._make_key(:scrl,  -4, 0, 0, sub)
                        haskey(g, tscrl_key) || continue
                        haskey(g, scrt_key)  || error("companion $scrt_key missing for $tscrl_key")
                        tscrl = read(g[tscrl_key])::Array{Float32,3}
                        scrt  = read(g[scrt_key])::Array{Float32,3}
                        scrl  = similar(tscrl)
                        @inbounds for jj in 1:nr, i in 1:nr
                            r_src = sub === :r ? rr[i] : rr[jj]
                            inv_r = 1.0 / r_src
                            for e in 1:nell_part
                                scrl[e, i, jj] = Float32(
                                    ll[e] * (Float64(tscrl[e, i, jj]) - Float64(scrt[e, i, jj]) * inv_r))
                            end
                        end
                        g[scrl_key] = scrl
                        delete_object(g, tscrl_key)
                    end

                    # 2D Y
                    for sub in (:r_rp, :rp_r)
                        tscrY_key = PowerFull._make_key(:tscrY, -4, 0, 0, sub)
                        scrX_key  = PowerFull._make_key(:scrX,  -4, 0, 0, sub)
                        scrY_key  = PowerFull._make_key(:scrY,  -4, 0, 0, sub)
                        haskey(g, tscrY_key) || continue
                        haskey(g, scrX_key)  || error("companion $scrX_key missing for $tscrY_key")
                        tscrY = read(g[tscrY_key])::Array{Float32,3}
                        scrX  = read(g[scrX_key])::Array{Float32,3}
                        scrY  = similar(tscrY)
                        lens_is_i = sub === :rp_r
                        @inbounds for jj in 1:nr, i in 1:nr
                            r_lens = lens_is_i ? rr[i] : rr[jj]
                            inv_r = 1.0 / r_lens
                            for e in 1:nell_part
                                scrY[e, i, jj] = Float32(
                                    ll[e] * (Float64(tscrY[e, i, jj]) - Float64(scrX[e, i, jj]) * inv_r))
                            end
                        end
                        g[scrY_key] = scrY
                        delete_object(g, tscrY_key)
                    end

                    # 2D L (must precede Z so raw tscrZ is still present)
                    tscrL_key = PowerFull._make_key(:tscrL, -4, 0, 0, :r_rp)
                    if haskey(g, tscrL_key)
                        tscrZ_rrp_key = PowerFull._make_key(:tscrZ, -4, 0, 0, :r_rp)
                        tscrZ_rpr_key = PowerFull._make_key(:tscrZ, -4, 0, 0, :rp_r)
                        scrT_key      = PowerFull._make_key(:scrT,  -4, 0, 0, :r_rp)
                        scrL_key      = PowerFull._make_key(:scrL,  -4, 0, 0, :r_rp)
                        (haskey(g, tscrZ_rrp_key) && haskey(g, tscrZ_rpr_key) && haskey(g, scrT_key)) ||
                            error("companion(s) missing for $tscrL_key conversion")
                        tscrL    = read(g[tscrL_key])::Array{Float32,3}
                        tscrZrrp = read(g[tscrZ_rrp_key])::Array{Float32,3}
                        tscrZrpr = read(g[tscrZ_rpr_key])::Array{Float32,3}
                        scrT     = read(g[scrT_key])::Array{Float32,3}
                        scrL = similar(tscrL)
                        @inbounds for jj in 1:nr, i in 1:nr
                            r1_v = rr[i]; r2_v = rr[jj]
                            inv_r1 = 1.0 / r1_v; inv_r2 = 1.0 / r2_v
                            inv_r1r2 = inv_r1 * inv_r2
                            for e in 1:nell_part
                                ll1sq = ll[e] * ll[e]
                                scrL[e, i, jj] = Float32(ll1sq * (
                                    Float64(tscrL[e, i, jj])
                                    - Float64(tscrZrpr[e, i, jj]) * inv_r2
                                    - Float64(tscrZrrp[e, i, jj]) * inv_r1
                                    + Float64(scrT[e, i, jj]) * inv_r1r2))
                            end
                        end
                        g[scrL_key] = scrL
                        delete_object(g, tscrL_key)
                    end

                    # 2D Z (after L; consumes the raw tscrZ that L already used)
                    for sub in (:r_rp, :rp_r)
                        tscrZ_key = PowerFull._make_key(:tscrZ, -4, 0, 0, sub)
                        scrT_key  = PowerFull._make_key(:scrT,  -4, 0, 0, :r_rp)
                        scrZ_key  = PowerFull._make_key(:scrZ,  -4, 0, 0, sub)
                        haskey(g, tscrZ_key) || continue
                        haskey(g, scrT_key)  || error("companion $scrT_key missing for $tscrZ_key")
                        tscrZ = read(g[tscrZ_key])::Array{Float32,3}
                        scrT  = read(g[scrT_key])::Array{Float32,3}
                        scrZ  = similar(tscrZ)
                        lens_is_i = sub === :rp_r
                        @inbounds for jj in 1:nr, i in 1:nr
                            r_lens = lens_is_i ? rr[i] : rr[jj]
                            inv_r = 1.0 / r_lens
                            for e in 1:nell_part
                                scrZ[e, i, jj] = Float32(
                                    ll[e] * (Float64(tscrZ[e, i, jj]) - Float64(scrT[e, i, jj]) * inv_r))
                            end
                        end
                        g[scrZ_key] = scrZ
                        delete_object(g, tscrZ_key)
                    end
                end
            end
        end
        println("[Streaming-fast] Paper-observable post-pass: $(round(t_pp, digits=1)) s")
    end

    end  # @elapsed

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    total_size_mb = filesize(meta_file) / 1024^2 + sum(filesize(f) for f in part_files) / 1024^2
    println("\n" * "="^60)
    println("Streaming-fast export complete!")
    println("="^60)
    println("  Total time: $(round(t_total, digits=1)) s")
    println("  Arrays written: $(n_written[])")
    println("  Total size: $(round(total_size_mb / 1024, digits=2)) GB in $(n_parts + 1) files")
    println("Output files:")
    println("  - $meta_file (metadata)")
    println("  - $(outname)_part_*.h5 (data files)")
    println()
    println("Julia usage:")
    println("  include(\"calcClGR.jl\")")
    println("  using .CalcClGR")
    println("  I = load_integrals_hdf5(\"$meta_file\")")

    return nothing
end

# =============================================================================
# Internal helper: prefix sum along axis 1 with unit weights (for 2D pass 2).
# Same trapezoidal-rule structure as PowerFull.prefix_sum_axis1! but with
# rf[k] ≡ 1. Inner loop is stride-1 contiguous over rows.
#
#   result[i, j] = Δlnr × Σ_{k=1}^{i} M[k, j]  (with half-corrections at both ends)
# =============================================================================
function _prefix_sum_axis1_unit!(result::Matrix{Float64}, Δlnr::Float64,
                                  M::Matrix{Float64})::Nothing
    nr = size(M, 1)
    @assert size(M, 2) == nr "M must be square"
    @assert size(result) == (nr, nr)

    @inbounds for j in 1:nr
        s = 0.0
        f1_half = M[1, j] / 2.0
        @simd for i in 1:nr
            fi = M[i, j]
            s += fi
            result[i, j] = Δlnr * (s - f1_half - fi / 2.0)
        end
    end
    return nothing
end

# =============================================================================
# Command line interface (mirrors build_and_export.jl --streaming)
# =============================================================================

function parse_args()
    args = Dict{Symbol, Any}(
        :Nr => 4096,
        :nR => 2049,
        :dlnR => 0.002,
        :ellmin => 2,
        :ellmax => 500,
        :tiers => nothing,
        :datadir => "./results",
        :outname => "ClGR_integrals",
        :load_base => true,
        :lucas_patch_w_0_22_ell2 => "",
        :use_float32 => true,
        :max_size_gb => 5.0,
        :cosmo_funcr => joinpath(@__DIR__, "..", "data", "cosmo_funcr.txt"),
        :Rcut_min => 0.0,
        :Rcut_max => Inf,
        :base_only => false,
        :mu0 => 0.0,
        :Sigma0 => 0.0,
    )

    for arg in ARGS
        if arg == "--help" || arg == "-h"
            println("""
Usage: julia -t <N> --project src/build_and_export.jl [options]

Phase 2+3 optimized streaming build. Output is bit-compatible with the
original build_and_export_streaming split-HDF5 layout.

Options:
  --Nr=<int>          Number of r grid points (default: 4096)
  --nR=<int>          Number of R grid points (default: 2049)
  --dlnR=<float>      Logarithmic R spacing (default: 0.002, single-tier)
  --ellmin=<int>      Minimum ell (default: 2, single-tier)
  --ellmax=<int>      Maximum ell (default: 500, single-tier)
  --tier=<dlnR,ellmin,ellmax[,nR]>  Add a tier (repeat for multi-tier).
                               Optional 4th field overrides global --nR for this tier.
  --datadir=<path>    Directory with TwoFAST files (default: ./results)
  --outname=<name>    Base name for output files (default: ClGR_integrals)
  --no-base           Don't load base wpljjprime functions
  --lucas-patch-w_0_22=<path>  After base/* is written, replace base/w_0_2_2 ell=2
                               slice with Lucas-direct values from this HDF5
                               (path to scripts/lucas_w22_ell2_patch.jl output).
                               Opt-in: improves standalone w_22 accuracy by ~2%
                               at small-R cross-pairs.  No effect on ell≥3.
  --no-float32        Use Float64 in HDF5 output (larger files)
  --max-size-gb=<f>   Max size per part file in GB (default: 5.0)
  --cosmo-funcr=<p>   Cosmology table path (default: data/cosmo_funcr.txt)
  --help, -h          Show this help message

Examples:
  julia -t 16 --project src/build_and_export.jl --Nr=4096 --nR=2049

  julia -t 16 --project src/build_and_export.jl \\
    --tier=0.002,2,199,4097 --tier=0.0005,200,500,2049
""")
            exit(0)
        elseif startswith(arg, "--Nr=")
            args[:Nr] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--nR=")
            args[:nR] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--dlnR=")
            args[:dlnR] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--ellmin=")
            args[:ellmin] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--ellmax=")
            args[:ellmax] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--tier=")
            parts = split(split(arg, "=")[2], ",")
            tier_dlnR = parse(Float64, parts[1])
            tier_ellmin = parse(Int, parts[2])
            tier_ellmax = parse(Int, parts[3])
            tier_nR = length(parts) >= 4 ? parse(Int, parts[4]) : args[:nR]
            tier_nt = @NamedTuple{dlnR::Float64,ellmin::Int,ellmax::Int,nR::Int}((tier_dlnR, tier_ellmin, tier_ellmax, tier_nR))
            if args[:tiers] === nothing
                args[:tiers] = [tier_nt]
            else
                push!(args[:tiers], tier_nt)
            end
        elseif startswith(arg, "--datadir=")
            args[:datadir] = String(split(arg, "=")[2])
        elseif startswith(arg, "--outname=")
            args[:outname] = String(split(arg, "=")[2])
        elseif arg == "--no-base"
            args[:load_base] = false
        elseif startswith(arg, "--lucas-patch-w_0_22=")
            args[:lucas_patch_w_0_22_ell2] = String(split(arg, "=")[2])
        elseif arg == "--no-float32"
            args[:use_float32] = false
        elseif startswith(arg, "--max-size-gb=")
            args[:max_size_gb] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--cosmo-funcr=")
            args[:cosmo_funcr] = String(split(arg, "=")[2])
        elseif startswith(arg, "--Rcut-min=")
            args[:Rcut_min] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--Rcut-max=")
            args[:Rcut_max] = parse(Float64, split(arg, "=")[2])
        elseif arg == "--base-only"
            args[:base_only] = true
        elseif startswith(arg, "--mu0=")
            args[:mu0] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--Sigma0=")
            args[:Sigma0] = parse(Float64, split(arg, "=")[2])
        else
            @warn "Unknown argument: $arg"
        end
    end

    return args
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_args()
    build_and_export_streaming(
        Nr=args[:Nr],
        nR=args[:nR],
        dlnR=args[:dlnR],
        ellmin=args[:ellmin],
        ellmax=args[:ellmax],
        tiers=args[:tiers],
        datadir=args[:datadir],
        outname=args[:outname],
        load_base=args[:load_base],
        lucas_patch_w_0_22_ell2=args[:lucas_patch_w_0_22_ell2],
        use_float32=args[:use_float32],
        max_size_gb=args[:max_size_gb],
        cosmo_funcr=args[:cosmo_funcr],
        Rcut_min=args[:Rcut_min],
        Rcut_max=args[:Rcut_max],
        base_only=args[:base_only],
        mu0=args[:mu0],
        Sigma0=args[:Sigma0],
    )
end
