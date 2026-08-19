#!/usr/bin/env -S julia --project
# =============================================================================
#
# >> PowerFull.jl <<
#
# Standalone module to read and interpolate TwoFAST output files
#
#  29 November 2024
#  Donghui Jeong
# =============================================================================

module PowerFull

export wpljjprime, clear_cache!

using JLD2
using HDF5
using Dierckx  # for interpolation
using Base.Threads  # for parallel evaluation

# =============================================================================
# Lookup table for wpljjprime data
# =============================================================================

const DjjpΔp = Dict{Int,NTuple{3,Int}}(
    1 => (0,0,0),
    2 => (0,2,0),
    3 => (2,0,0),
    4 => (2,2,0),
    5 => (0,1,-1),
    6 => (1,0,-1),
    7 => (1,1,0),
    8 => (1,2,-1),
    9 => (2,1,-1))

# =============================================================================
# 9-base structure (Apr 2026): Split by (p_eff, n) for optimal q per integrand
# - Bases 1-2: p_base=0,  n=0,  split by Δp (p_eff=0 vs p_eff=-1)
# - Bases 3-4: p_base=-2, n=0,  split by Δp (p_eff=-2 vs p_eff=-3)
# - Base  5:   p_base=-4, n=0   (p_eff=-4)
# - Bases 6-7: p_base=-2, n=-1, split by Δp (p_eff=-2 vs p_eff=-3)
# - Bases 8-9: p_base=-4, n=-1/-2
# =============================================================================
const parray = [0,     0,    -2,    -2,   -4,   -2,   -2,   -4,   -4]
const narray = [0,     0,     0,     0,    0,   -1,   -1,   -1,   -2]
const acases = [
    [1,2,3,4],      # Base 1: p_eff=0,  n=0
    [5,6,8,9],      # Base 2: p_eff=-1, n=0
    [1,2,3,7],      # Base 3: p_eff=-2, n=0
    [5,6],          # Base 4: p_eff=-3, n=0
    [1],            # Base 5: p_eff=-4, n=0
    [1,2,3],        # Base 6: p_eff=-2, n=-1
    [5,6],          # Base 7: p_eff=-3, n=-1
    [1],            # Base 8: p_eff=-4, n=-1
    [1],            # Base 9: p_eff=-4, n=-2
]

# Build reverse lookup table: (p,j,jprime,n) -> (indx, cindx)
function build_lookup_table()
    lookup = Dict{NTuple{4,Int}, NTuple{2,Int}}()
    for (indx, (p_base, n, cases)) in enumerate(zip(parray, narray, acases))
        for (cindx, c) in enumerate(cases)
            jc, jpc, Δp = DjjpΔp[c]
            p_eff = p_base + Δp
            lookup[(p_eff, jc, jpc, n)] = (indx, cindx)
        end
    end
    return lookup
end

const LOOKUP_TABLE = build_lookup_table()

# =============================================================================
# wpljjprime: w^{p,n}_{ell,jj'}(r,r') data structure
# =============================================================================

"""
    wpljjprime

Structure to hold w^{p,n}_{ell,jj'}(r,r') data for a specific (p,j,jprime,n,ell) combination.

# Fields
- `wrRl::Array{Float64,2}`: 2D array [rindx, Rindx] of values
- `rr::Vector{Float64}`: Radii values
- `RR::Vector{Float64}`: Radii ratio values (R = r'/r)
- `ell::Int`: Multipole moment
- `n::Int`: Transfer function power
- `p::Int`: Power index
- `j::Int`: First derivative order
- `jprime::Int`: Second derivative order
"""
struct wpljjprime
    wrRl::Array{Float64,2}
    rr::Vector{Float64}
    RR::Vector{Float64}
    ell::Int
    n::Int
    p::Int
    j::Int
    jprime::Int
end

# Cache for wpljjprime files
# Cache key is (indx, Nr, nR, dlnR)
const _wpljjprime_cache = Dict{Tuple{Int,Int,Int,Float64},Tuple{Array{Float64,4},Vector{Float64},Vector{Float64},Vector{Int}}}()

"""
    wpljjprime(p, j, jprime, n, ell; Nr, nR, dlnR, ellmin, ellmax, ...) -> wpljjprime

Load w^{p,n}_{ell,jj'}(r,r') data for the given (p,j,jprime,n,ell) combination.

# Arguments
- `p::Int`: Power index
- `j::Int`: First derivative order
- `jprime::Int`: Second derivative order
- `n::Int`: Transfer function power (T(k)^n)
- `ell::Int`: Multipole moment value
- `Nr::Int=4096`: Number of r grid points (must match data file)
- `nR::Int=2049`: Number of R grid points (must match data file)
- `dlnR::Float64=0.002`: Logarithmic R spacing (must match data file)
- `ellmin::Int=2`: Minimum ell in the data file
- `ellmax::Int=500`: Maximum ell in the data file
- `use_cache::Bool=true`: Whether to cache loaded files in memory
- `datadir::String="./results"`: Directory where TwoFAST output files are located

# Returns
- `wpljjprime`: Struct with wrRl[rindx, Rindx], rr, RR, and metadata
"""
function wpljjprime(p::Int, j::Int, jprime::Int, n::Int, ell::Int;
                    Nr::Int=4096, nR::Int=2049, dlnR::Float64=0.002,
                    ellmin::Int=2, ellmax::Int=500,
                    use_cache::Bool=true, datadir::String="./results")
    # Lookup the file index and case index
    key = (p, j, jprime, n)
    if !haskey(LOOKUP_TABLE, key)
        available = sort(collect(keys(LOOKUP_TABLE)))
        error("No matching entry found for p=$p, j=$j, jprime=$jprime, n=$n. " *
              "Available combinations: $available")
    end

    indx_found, cindx_found = LOOKUP_TABLE[key]

    # Load file with caching
    cache_key = (indx_found, Nr, nR, dlnR)

    local fullresult::Array{Float64,4}
    local rr::Vector{Float64}
    local RR::Vector{Float64}
    local aell::Vector{Int}

    if use_cache && haskey(_wpljjprime_cache, cache_key)
        fullresult, rr, RR, aell = _wpljjprime_cache[cache_key]
    else
        filename = joinpath(datadir, "TwoFAST_output_nr=$(Nr)_nR=$(nR)_dlnR=$(dlnR)_ell=$(ellmin)-$(ellmax)_$indx_found.jld2")
        if !isfile(filename)
            error("File $filename not found. Run TwoFAST computation first with Nr=$Nr, nR=$nR, dlnR=$dlnR, ellmin=$ellmin, ellmax=$ellmax.")
        end

        data = load(filename)
        if !haskey(data, "fullresult")
            error("Invalid file format: $filename does not contain 'fullresult' key")
        end
        fullresult = data["fullresult"]
        rr = data["rr"]
        RR = data["RR"]
        aell = data["aell"]

        if use_cache
            _wpljjprime_cache[cache_key] = (fullresult, rr, RR, aell)
        end
    end

    # Find the ell index
    ellindx = findfirst(==(ell), aell)
    if isnothing(ellindx)
        error("ell=$ell not found in available ell values. Available range: $(minimum(aell)) to $(maximum(aell))")
    end

    # Extract the 2D array for this combination
    wrRl = fullresult[:, :, cindx_found, ellindx]

    return wpljjprime(wrRl, rr, RR, ell, n, p, j, jprime)
end

# =============================================================================
# Cache management
# =============================================================================

"""
    clear_cache!()

Clear all internal file caches to free memory.
"""
function clear_cache!()
    empty!(_wpljjprime_cache)
    GC.gc()
    return nothing
end

# =============================================================================
# Physical grid functions: (r, R) → (r₁, r₂) conversion and prefix sums
# =============================================================================

"""
    _load_w_integrand_file(case::Int, Nr::Int, nR::Int, dlnR::Float64,
                           ellmin::Int, ellmax::Int, datadir::String)

Load raw w_integrand data from a TwoFAST_w_integrand file.
"""
function _load_w_integrand_file(case::Int, Nr::Int, nR::Int, dlnR::Float64,
                                ellmin::Int, ellmax::Int, datadir::String)
    filename = joinpath(datadir,
        "TwoFAST_w_integrand_nr=$(Nr)_nR=$(nR)_dlnR=$(dlnR)_ell=$(ellmin)-$(ellmax)_$(case).jld2")
    if !isfile(filename)
        error("File $filename not found. Run save_w_integrand first.")
    end

    data = load(filename)
    return data["w_integrand"]::Array{Float64,3}, data["rr"]::Vector{Float64},
           data["RR"]::Vector{Float64}, data["aell"]::Vector{Int}
end

"""
    interpolate_to_physical_grid(w_integrand::Matrix{Float64},
                                  rr::Vector{Float64},
                                  RR::Vector{Float64})::Matrix{Float64}

Convert a single w_integrand[nr, nR] slice from TwoFAST's (r, R) grid to the physical
(r₁, r₂) grid, where R = r₂/r₁.

For each row i (fixed r₁ = rr[i]), builds a 1D spline in ln(R) and evaluates
at R = rr[j]/rr[i] for all j.

Returns w_phys[nr, nr] where w_phys[i,j] = w_integrand(rr[i], rr[j]/rr[i]).
"""
function interpolate_to_physical_grid(w_integrand::AbstractMatrix{Float64},
                                       rr::Vector{Float64},
                                       RR::Vector{Float64})::Matrix{Float64}
    nr = length(rr)
    ln_rr = log.(rr)
    ln_RR = log.(RR)
    ln_R_min = ln_RR[1]
    ln_R_max = ln_RR[end]

    w_phys = zeros(Float64, nr, nr)

    @threads for i in 1:nr
        spl = Spline1D(ln_RR, @view(w_integrand[i, :]), k=3, bc="zero")
        @inbounds for j in 1:nr
            ln_R = ln_rr[j] - ln_rr[i]  # ln(r_j / r_i)
            if ln_R >= ln_R_min && ln_R <= ln_R_max
                w_phys[i, j] = spl(ln_R)
            end
        end
    end
    return w_phys
end

"""
    interpolate_to_physical_grid(w_integrand::Array{Float64,3},
                                  rr::Vector{Float64},
                                  RR::Vector{Float64})::Array{Float64,3}

Batch version: convert w_integrand[nr, nR, nell] → w_phys[nell, nr, nr].
Note: output is [nell, nr, nr] for cache-optimal ell access in column-major.
"""
function interpolate_to_physical_grid(w_integrand::Array{Float64,3},
                                       rr::Vector{Float64},
                                       RR::Vector{Float64})::Array{Float64,3}
    nr, nR, nell = size(w_integrand)
    w_phys = zeros(Float64, nell, nr, nr)

    @threads for ell_idx in 1:nell
        w_phys_2d = interpolate_to_physical_grid(
            @view(w_integrand[:, :, ell_idx]), rr, RR)
        @inbounds for j in 1:nr, i in 1:nr
            w_phys[ell_idx, i, j] = w_phys_2d[i, j]
        end
    end
    return w_phys
end

"""
    prefix_sum_axis1!(result::Matrix{Float64}, Δlnr::Float64,
                       rf::Vector{Float64}, w_phys::Matrix{Float64})

Compute the ;r integral (axis 1 prefix sum) on the physical (r₁, r₂) grid.

For each fixed r₂ = rr[j]:
    result[i,j] = Δlnr × Σ_{k=1}^{i} rf[k] × w_phys[k,j]   (trapezoidal)

rf[k] = rr[k] × f(rr[k]) where f is the kernel (fs, ft, or fl1).
"""
function prefix_sum_axis1!(result::Matrix{Float64}, Δlnr::Float64,
                            rf::Vector{Float64}, w_phys::Matrix{Float64})::Nothing
    nr = size(w_phys, 1)
    @assert size(w_phys, 2) == nr "w_phys must be square on physical grid"
    @assert length(rf) == nr "rf length must match"
    @assert size(result) == (nr, nr) "result must be nr × nr"

    @inbounds for j in 1:nr  # fixed r₂ = rr[j]
        s = 0.0
        f1_half = rf[1] * w_phys[1, j] / 2.0
        for i in 1:nr
            fi = rf[i] * w_phys[i, j]
            s += fi
            result[i, j] = Δlnr * (s - f1_half - fi / 2.0)
        end
    end
    return nothing
end

"""
    prefix_sum_axis2!(result::Matrix{Float64}, Δlnr::Float64,
                       rf::Vector{Float64}, w_phys::Matrix{Float64})

Compute the ;r' integral (axis 2 prefix sum) on the physical (r₁, r₂) grid.

For each fixed r₁ = rr[i]:
    result[i,j] = Δlnr × Σ_{k=1}^{j} rf[k] × w_phys[i,k]   (trapezoidal)

rf[k] = rr[k] × f(rr[k]) where f is the kernel (fs, ft, or fl1).
"""
function prefix_sum_axis2!(result::Matrix{Float64}, Δlnr::Float64,
                            rf::Vector{Float64}, w_phys::Matrix{Float64})::Nothing
    nr = size(w_phys, 1)
    @assert size(w_phys, 2) == nr "w_phys must be square on physical grid"
    @assert length(rf) == nr "rf length must match"
    @assert size(result) == (nr, nr) "result must be nr × nr"

    @inbounds for i in 1:nr  # fixed r₁ = rr[i]
        s = 0.0
        f1_half = rf[1] * w_phys[i, 1] / 2.0
        for j in 1:nr
            fj = rf[j] * w_phys[i, j]
            s += fj
            result[i, j] = Δlnr * (s - f1_half - fj / 2.0)
        end
    end
    return nothing
end

"""
    prefix_sum_2D!(result::Matrix{Float64}, buf::Matrix{Float64},
                    Δlnr::Float64, rf1::Vector{Float64}, rf2::Vector{Float64},
                    w_phys::Matrix{Float64})

Compute the 2D (;r,r') integral on the physical (r₁, r₂) grid using two
sequential 1D prefix sums (trapezoidal rule).

    result[i,j] = Δlnr² × Σ_{a=1}^{i} Σ_{b=1}^{j} rf1[a] × rf2[b] × w_phys[a,b]

Uses `buf` as a scratch buffer to avoid overwriting during pass 2.
"""
function prefix_sum_2D!(result::Matrix{Float64}, buf::Matrix{Float64},
                         Δlnr::Float64, rf1::Vector{Float64}, rf2::Vector{Float64},
                         w_phys::Matrix{Float64})::Nothing
    nr = size(w_phys, 1)
    @assert size(w_phys, 2) == nr "w_phys must be square"
    @assert size(result) == (nr, nr) && size(buf) == (nr, nr)

    # Pass 1: prefix sum over axis 1 (rows) for each column j
    @inbounds for j in 1:nr
        s = 0.0
        f1_half = rf1[1] * rf2[j] * w_phys[1, j] / 2.0
        for i in 1:nr
            fi = rf1[i] * rf2[j] * w_phys[i, j]
            s += fi
            buf[i, j] = Δlnr * (s - f1_half - fi / 2.0)
        end
    end

    # Pass 2: prefix sum over axis 2 (columns) for each row i
    @inbounds for i in 1:nr
        s = 0.0
        g1_half = buf[i, 1] / 2.0
        for j in 1:nr
            gj = buf[i, j]
            s += gj
            result[i, j] = Δlnr * (s - g1_half - gj / 2.0)
        end
    end
    return nothing
end

# =============================================================================
# Coordinate transformation: ;r integrals → ;r' integrals (LEGACY - replaced by transpose)
# =============================================================================

"""
    transform_r_to_rprime(stlint::Array{Float64,4}, rr::Vector{Float64}, RR::Vector{Float64};
                          k::Int=3) -> Array{Float64,4}

Transform ;r integrals to ;r' integrals using the symmetry:
    s^p_{ℓ,jj';r}(r,r') = s^p_{ℓ,j'j;r'}(r',r)

In (r, R) coordinates where R = r'/r:
- The transformation swaps r ↔ r'
- R_new = 1/R (reverse R index for symmetric grid)
- r_new = r × R (requires log-linear interpolation)

Points outside [r_min, r_max] are set to 0.0.

# Arguments
- `stlint::Array{Float64,4}`: Input array [nr, nR, 3, n_ell] (s,t,l components)
- `rr::Vector{Float64}`: r grid (log-uniform)
- `RR::Vector{Float64}`: R grid (log-uniform, symmetric around 1)
- `k::Int=3`: Spline interpolation order

# Returns
- `Array{Float64,4}`: Transformed array with same dimensions

# Note
Apply to 1D integral cases (1-5) and case 7. NOT case 6 (which is symmetric).
"""
function transform_r_to_rprime(stlint::Array{Float64,4}, rr::Vector{Float64}, RR::Vector{Float64};
                                k::Int=3)
    nr, nR, ncomp, nell = size(stlint)
    @assert length(rr) == nr "rr length must match first dimension of stlint"
    @assert length(RR) == nR "RR length must match second dimension of stlint"

    # Output array (initialized to zero - out-of-bounds will remain zero)
    result = zeros(Float64, nr, nR, ncomp, nell)

    # Log coordinates for interpolation
    ln_rr = log.(rr)
    ln_RR = log.(RR)
    ln_r_min = ln_rr[1]
    ln_r_max = ln_rr[end]

    # Pre-compute ALL target coordinates (outside loops)
    # r_old = r_new * R_new  →  ln_r_old = ln_r_new + ln_R_new
    # R_old = 1 / R_new      →  ln_R_old = -ln_R_new
    ln_r_old_mat = zeros(nr, nR)
    ln_R_old_vec = -ln_RR  # log(1/R) = -log(R)

    for iR in 1:nR
        @. ln_r_old_mat[:, iR] = ln_rr + ln_RR[iR]  # log(r * R)
    end

    # Pre-compute validity mask (which points are in bounds)
    in_bounds = Matrix{Bool}(undef, nr, nR)
    for iR in 1:nR
        for ir in 1:nr
            in_bounds[ir, iR] = (ln_r_old_mat[ir, iR] >= ln_r_min) && (ln_r_old_mat[ir, iR] <= ln_r_max)
        end
    end

    # Thread over (iell, icomp) pairs
    tasks = [(iell, icomp) for iell in 1:nell for icomp in 1:ncomp]

    @threads for (iell, icomp) in tasks
        # Create spline for this slice
        spl = Spline2D(ln_rr, ln_RR, @view(stlint[:, :, icomp, iell]); kx=k, ky=k)

        # Evaluate all points (only in-bounds)
        @inbounds for iR in 1:nR
            ln_R_old = ln_R_old_vec[iR]
            for ir in 1:nr
                if in_bounds[ir, iR]
                    result[ir, iR, icomp, iell] = spl(ln_r_old_mat[ir, iR], ln_R_old)
                end
            end
        end
    end

    return result
end

"""
    transform_r_to_rprime(stlint::Array{Float64,3}, rr::Vector{Float64}, RR::Vector{Float64};
                          k::Int=3) -> Array{Float64,3}

Transform a single component array [nr, nR, n_ell] from ;r to ;r' coordinates.
"""
function transform_r_to_rprime(stlint::Array{Float64,3}, rr::Vector{Float64}, RR::Vector{Float64};
                                k::Int=3)
    nr, nR, nell = size(stlint)
    # Reshape to 4D, transform, reshape back
    stlint_4d = reshape(stlint, nr, nR, 1, nell)
    result_4d = transform_r_to_rprime(stlint_4d, rr, RR, k=k)
    return reshape(result_4d, nr, nR, nell)
end

# =============================================================================
# IntegralCollection: Data structure matching paper notation
# =============================================================================

"""
    IntegralCollection

Data structure to hold all TwoFAST integrals with paper-matching notation.

# Access pattern
- `I[type, p, j, jp, sub]` where:
  - `type::Symbol`: :w, :u, :v (base), :s, :t, :l (1D), :scrs, :scrt, :scrl (script 1D),
                    :scrS, :scrT, :scrL, :scrX, :scrY, :scrZ (2D)
  - `p::Int`: Power index
  - `j::Int, jp::Int`: Derivative orders
  - `sub::Symbol`: :r (;r integral), :rp (;r' integral), :r_rp (2D), :none (base function)

# Naming convention based on n parameter:
- n = 0  → w (base), s/t/l (integrals)
- n = -1 → u (base), 𝔰/𝔱/𝔩 (scrs/scrt/scrl integrals)
- n = -2 → v (base)

# Examples
```julia
I[:s, -2, 0, 0, :r]      # s^{-2}_{ℓ,00;r}
I[:s, -2, 0, 0, :rp]     # s^{-2}_{ℓ,00;r'}
I[:s, -2, 2, 0, :rp]     # s^{-2}_{ℓ,20;r'} (from transform of (0,2);r)
I[:scrs, -4, 0, 0, :r]   # 𝔰^{-4}_{ℓ,00;r}
I[:scrS, -4, 0, 0, :r_rp] # 𝒮^{-4}_{ℓ,00;r,r'}
I[:w, 0, 0, 0, :none]    # w^{0}_{ℓ,00}
```
"""
struct IntegralCollection
    data::Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}  # [n_ell, nr, nr] on physical grid
    rr::Vector{Float64}       # physical r grid (same for both axes)
    aell::Vector{Int}
    Nr::Int                   # original TwoFAST Nr parameter
    nR::Int                   # original TwoFAST nR parameter
    logRmin::Float64          # kept for backward compatibility (set to 0.0 for two-tier)
    logRmax::Float64          # kept for backward compatibility (set to 0.0 for two-tier)
    tier_info::Vector{@NamedTuple{dlnR::Float64, ellmin::Int, ellmax::Int}}  # tier metadata
end

# Backward-compatible constructor (logRmin/logRmax only, no tier_info)
function IntegralCollection(data, rr, aell, Nr, nR, logRmin, logRmax)
    IntegralCollection(data, rr, aell, Nr, nR, logRmin, logRmax,
                       [@NamedTuple{dlnR::Float64, ellmin::Int, ellmax::Int}((0.0, minimum(aell), maximum(aell)))])
end

function Base.getindex(ic::IntegralCollection, type::Symbol, p::Int, j::Int, jp::Int, sub::Symbol)
    key = (type, p, j, jp, sub)
    if !haskey(ic.data, key)
        available = sort(collect(keys(ic.data)))
        error("Key $key not found. Available keys: $available")
    end
    return ic.data[key]
end

function Base.setindex!(ic::IntegralCollection, val::Array{Float64,3}, type::Symbol, p::Int, j::Int, jp::Int, sub::Symbol)
    ic.data[(type, p, j, jp, sub)] = val
end

function Base.haskey(ic::IntegralCollection, type::Symbol, p::Int, j::Int, jp::Int, sub::Symbol)
    return haskey(ic.data, (type, p, j, jp, sub))
end

function Base.keys(ic::IntegralCollection)
    return keys(ic.data)
end

"""
    show_available_keys(ic::IntegralCollection)

Print all available keys in the IntegralCollection.
"""
function show_available_keys(ic::IntegralCollection)
    sorted_keys = sort(collect(keys(ic.data)))
    println("Available integrals ($(length(sorted_keys)) total):")
    for k in sorted_keys
        println("  I[$(k[1]), $(k[2]), $(k[3]), $(k[4]), $(k[5])]")
    end
end

"""
    get_rr_grid(; Nr, nR, dlnR, ellmin, ellmax, datadir) -> Vector{Float64}

Load the r grid from the first w_integrand file without loading all data.
"""
function get_rr_grid(; Nr::Int=4096, nR::Int=2049,
                       dlnR::Float64=0.002, ellmin::Int=2, ellmax::Int=500,
                       datadir::String="./results")::Vector{Float64}
    _, rr, _, _ = _load_w_integrand_file(1, Nr, nR, dlnR, ellmin, ellmax, datadir)
    return rr
end

# =============================================================================
# Load all integrals into IntegralCollection
# =============================================================================

"""
    load_all_integrals(rf_s, rf_t, rf_l, D_r;
                       Nr=4096, nR=2049, dlnR=0.002, ellmin=2, ellmax=500,
                       datadir="./results", load_base=false) -> IntegralCollection

Single-tier version: load w_integrand files for one (dlnR, ellmin, ellmax) tier.

    load_all_integrals(rf_s, rf_t, rf_l, D_r;
                       Nr=4096, nR=2049,
                       tiers=[(dlnR=0.002, ellmin=2, ellmax=199),
                              (dlnR=0.0005, ellmin=200, ellmax=500)],
                       datadir="./results", load_base=false) -> IntegralCollection

Two-tier version: load w_integrand files for multiple tiers, interpolate each to
physical grid with tier-specific RR, then concatenate along the ℓ axis.

# Arguments
- `rf_s::Vector{Float64}`: Kernel array `rr[k] * fs(rr[k])` for s-type integrals (length nr)
- `rf_t::Vector{Float64}`: Kernel array `rr[k] * ft(rr[k])` for t-type integrals (length nr)
- `rf_l::Vector{Float64}`: Kernel array `rr[k] * fl1(rr[k])` for l-type integrals (length nr)
- `D_r::Vector{Float64}`:  Growth factor D(r) on the r grid (length nr)
- `Nr::Int=4096`: Number of r grid points (TwoFAST parameter)
- `nR::Int=2049`: Number of R grid points (TwoFAST parameter)
- `dlnR::Float64=0.002`: Logarithmic R spacing
- `ellmin::Int=2`: Minimum ell
- `ellmax::Int=500`: Maximum ell
- `tiers`: Vector of NamedTuples (dlnR, ellmin, ellmax) for multi-tier mode
- `datadir::String="./results"`: Directory with TwoFAST_w_integrand files
- `load_base::Bool=false`: Also load base wpljjprime functions (w, u, v) on physical grid

# Returns
- `IntegralCollection`: Collection of all integrals on physical (r₁, r₂) grid

# Pipeline
1. Load raw w_integrand[nr, nR, nell] per case from TwoFAST_w_integrand files
2. Interpolate from (r, R) → (r₁, r₂) physical grid per ℓ (using tier-specific RR)
3. Compute 1D prefix sums (;r and ;r') with trapezoidal rule
4. For j/j' exchange: use transpose instead of 2D spline interpolation
5. Compute 2D prefix sums for cases 6-7
6. For multi-tier: concatenate results along ℓ axis

# Loaded integrals
| Case | (p,j,j',n) | Types | ;r source | ;r' source |
|------|------------|-------|-----------|------------|
| 1 | (-2,0,0,0) | s,t,l | prefix_sum_axis1 | prefix_sum_axis2 |
| 2 | (-2,0,2,0) | s,t,l | prefix_sum_axis1 (02) | transpose of 02;r (→ 20;r') |
| 3 | (-3,0,1,0) | s,t,l | prefix_sum_axis1 (01) | transpose of 01;r (→ 10;r') |
| 4 | (-4,0,0,0) | s,t,l | prefix_sum_axis1 | prefix_sum_axis2 |
| 5 | (-4,0,0,-1) | scrs,scrt,scrl | prefix_sum_axis1 | prefix_sum_axis2 |
| 6 | (-4,0,0,0) 2D | scrS,scrT,scrL | prefix_sum_2D (symmetric) | — |
| 7 | (-4,0,0,0) 2D | scrX,scrY,scrZ | prefix_sum_2D | transpose for ;r',r |
"""
function load_all_integrals(rf_s::Vector{Float64}, rf_t::Vector{Float64}, rf_l::Vector{Float64},
                            D_r::Vector{Float64};
                            Nr::Int=4096, nR::Int=2049,
                            dlnR::Float64=0.002, ellmin::Int=2, ellmax::Int=500,
                            tiers::Union{Nothing,
                                         Vector{@NamedTuple{dlnR::Float64, ellmin::Int, ellmax::Int}}}=nothing,
                            datadir::String="./results", load_base::Bool=false)

    # Build a single-tier list from the (dlnR, ellmin, ellmax) kwargs if
    # the caller did not supply an explicit multi-tier specification.
    if tiers === nothing
        tiers = [@NamedTuple{dlnR::Float64, ellmin::Int, ellmax::Int}((dlnR, ellmin, ellmax))]
    end

    # Load first tier's first file to get rr grid info
    tier1 = tiers[1]
    println("Loading raw w_integrand for case 1, tier 1 (dlnR=$(tier1.dlnR), ell=$(tier1.ellmin)-$(tier1.ellmax))...")
    w_integrand_1_t1, rr, RR_t1, aell_t1 = @time _load_w_integrand_file(1, Nr, nR, tier1.dlnR, tier1.ellmin, tier1.ellmax, datadir)
    nr = length(rr)
    Δlnr = log(rr[2] / rr[1])

    # Build combined aell from all tiers
    aell_combined = Int[]
    for tier in tiers
        append!(aell_combined, collect(tier.ellmin:tier.ellmax))
    end
    total_nell = length(aell_combined)

    @assert length(rf_s) == nr "rf_s length ($( length(rf_s))) must match nr ($nr)"
    @assert length(rf_t) == nr "rf_t length ($( length(rf_t))) must match nr ($nr)"
    @assert length(rf_l) == nr "rf_l length ($( length(rf_l))) must match nr ($nr)"
    @assert length(D_r)  == nr "D_r length ($(  length(D_r))) must match nr ($nr)"

    # Pre-compute prefactor arrays: 3/D(rᵢ)
    inv3D = zeros(Float64, nr)
    @inbounds for i in 1:nr
        inv3D[i] = 3.0 / D_r[i]
    end

    # Initialize collection
    ic_data = Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}()
    I = IntegralCollection(ic_data, rr, aell_combined, Nr, nR, 0.0, 0.0, copy(tiers))

    # Thread-local scratch buffers for prefix sums (one per thread for parallel ℓ loop)
    nthreads_max = Threads.maxthreadid()
    result_bufs = [zeros(Float64, nr, nr) for _ in 1:nthreads_max]
    buf_2Ds     = [zeros(Float64, nr, nr) for _ in 1:nthreads_max]

    # -------------------------------------------------------------------------
    # Helper: compute 1D ;r and ;r' integrals for a given w_integrand and kernel
    # RR_tier is the R grid specific to this tier
    # Returns (int_r, int_rp) each [nell_tier, nr, nr]
    # -------------------------------------------------------------------------
    function compute_1D_integrals(w_integrand::Array{Float64,3}, rf::Vector{Float64},
                                  inv3D_vec::Vector{Float64}, RR_tier::Vector{Float64})::Tuple{Array{Float64,3}, Array{Float64,3}}
        nell_t = size(w_integrand, 3)
        int_r  = zeros(Float64, nell_t, nr, nr)
        int_rp = zeros(Float64, nell_t, nr, nr)

        @threads for ell_idx in 1:nell_t
            tid = Threads.threadid()
            buf = result_bufs[tid]

            # Interpolate to physical grid for this ℓ using tier-specific RR
            w_phys = interpolate_to_physical_grid(@view(w_integrand[:, :, ell_idx]), rr, RR_tier)

            # ;r integral: prefix sum over axis 1, prefactor 3/D(rᵢ)
            prefix_sum_axis1!(buf, Δlnr, rf, w_phys)
            @inbounds for j in 1:nr, i in 1:nr
                int_r[ell_idx, i, j] = inv3D_vec[i] * buf[i, j]
            end

            # ;r' integral: prefix sum over axis 2, prefactor 3/D(rⱼ)
            prefix_sum_axis2!(buf, Δlnr, rf, w_phys)
            @inbounds for j in 1:nr, i in 1:nr
                int_rp[ell_idx, i, j] = inv3D_vec[j] * buf[i, j]
            end
        end
        return (int_r, int_rp)
    end

    # -------------------------------------------------------------------------
    # Helper: compute 1D ;r and (transposed → j/j' swapped ;r') integrals
    # -------------------------------------------------------------------------
    function compute_1D_integrals_asymmetric(w_integrand::Array{Float64,3}, rf::Vector{Float64},
                                              inv3D_vec::Vector{Float64}, RR_tier::Vector{Float64})::Tuple{Array{Float64,3}, Array{Float64,3}}
        nell_t = size(w_integrand, 3)
        int_r   = zeros(Float64, nell_t, nr, nr)  # (jj');r
        int_rp  = zeros(Float64, nell_t, nr, nr)  # (j'j);r' via transpose

        # Thread-local transpose buffers
        w_phys_Ts = [zeros(Float64, nr, nr) for _ in 1:nthreads_max]

        @threads for ell_idx in 1:nell_t
            tid = Threads.threadid()
            buf = result_bufs[tid]
            w_phys_T = w_phys_Ts[tid]

            w_phys = interpolate_to_physical_grid(@view(w_integrand[:, :, ell_idx]), rr, RR_tier)

            # ;r integral for (j,j'): prefix sum over axis 1, prefactor 3/D(rᵢ)
            prefix_sum_axis1!(buf, Δlnr, rf, w_phys)
            @inbounds for j in 1:nr, i in 1:nr
                int_r[ell_idx, i, j] = inv3D_vec[i] * buf[i, j]
            end

            # (j',j);r' via transpose
            permutedims!(w_phys_T, w_phys, (2, 1))
            prefix_sum_axis2!(buf, Δlnr, rf, w_phys_T)
            @inbounds for j in 1:nr, i in 1:nr
                int_rp[ell_idx, i, j] = inv3D_vec[j] * buf[i, j]
            end
        end
        return (int_r, int_rp)
    end

    # -------------------------------------------------------------------------
    # Helper: compute 2D (;r,r') integrals for a given w_integrand and kernel pair
    # Returns int_2D[nell_tier, nr, nr]
    # -------------------------------------------------------------------------
    function compute_2D_integrals(w_integrand::Array{Float64,3}, rf1::Vector{Float64}, rf2::Vector{Float64},
                                  inv3D_vec::Vector{Float64}, RR_tier::Vector{Float64})::Array{Float64,3}
        nell_t = size(w_integrand, 3)
        int_2D = zeros(Float64, nell_t, nr, nr)

        @threads for ell_idx in 1:nell_t
            tid = Threads.threadid()
            buf = result_bufs[tid]
            buf2 = buf_2Ds[tid]

            w_phys = interpolate_to_physical_grid(@view(w_integrand[:, :, ell_idx]), rr, RR_tier)

            prefix_sum_2D!(buf, buf2, Δlnr, rf1, rf2, w_phys)
            # Prefactor: 9/(D(rᵢ)*D(rⱼ))
            @inbounds for j in 1:nr, i in 1:nr
                int_2D[ell_idx, i, j] = inv3D_vec[i] * inv3D_vec[j] * buf[i, j]
            end
        end
        return int_2D
    end

    # -------------------------------------------------------------------------
    # Helper: compute RR grid from dlnR for a tier
    # -------------------------------------------------------------------------
    function compute_RR_for_tier(tier_dlnR::Float64)::Vector{Float64}
        half = (nR - 1) ÷ 2
        lnRR = collect(range(-half * tier_dlnR, stop=half * tier_dlnR, length=nR))
        return exp.(lnRR)
    end

    # -------------------------------------------------------------------------
    # Helper: process one case across all tiers, concatenate along ℓ axis
    # mode: :symmetric, :asymmetric, :2D_symmetric, :2D_asymmetric
    # -------------------------------------------------------------------------
    function process_case_all_tiers(case_num::Int, mode::Symbol,
                                    rf_primary::Vector{Float64},
                                    rf_secondary::Union{Vector{Float64},Nothing}=nothing)
        tier_results = []

        for (tidx, tier) in enumerate(tiers)
            RR_tier = compute_RR_for_tier(tier.dlnR)

            # Load w_integrand for this case and tier
            # For tier 1, case 1: reuse already-loaded data
            local w_int::Array{Float64,3}
            if tidx == 1 && case_num == 1
                w_int = w_integrand_1_t1
            else
                println("  Loading case $case_num for tier $tidx (dlnR=$(tier.dlnR), ell=$(tier.ellmin)-$(tier.ellmax))...")
                w_int, _, _, _ = @time _load_w_integrand_file(case_num, Nr, nR, tier.dlnR, tier.ellmin, tier.ellmax, datadir)
            end

            if mode == :symmetric
                r_result, rp_result = compute_1D_integrals(w_int, rf_primary, inv3D, RR_tier)
                push!(tier_results, (r_result, rp_result))
            elseif mode == :asymmetric
                r_result, rp_result = compute_1D_integrals_asymmetric(w_int, rf_primary, inv3D, RR_tier)
                push!(tier_results, (r_result, rp_result))
            elseif mode == :twoD
                result_2d = compute_2D_integrals(w_int, rf_primary, rf_secondary::Vector{Float64}, inv3D, RR_tier)
                push!(tier_results, result_2d)
            end
        end

        # Concatenate along ℓ axis (axis 1)
        if mode == :symmetric || mode == :asymmetric
            int_r  = vcat([tr[1] for tr in tier_results]...)
            int_rp = vcat([tr[2] for tr in tier_results]...)
            return (int_r, int_rp)
        else  # :twoD
            return vcat(tier_results...)
        end
    end

    # =========================================================================
    # Case 1: (p,j,j',n) = (-2,0,0,0) — symmetric (j=j')
    # =========================================================================
    println("Case 1: (-2,0,0,0) → s,t,l ;r and ;r'...")
    s_r, s_rp = @time process_case_all_tiers(1, :symmetric, rf_s)
    I[:s, -2, 0, 0, :r] = s_r
    I[:s, -2, 0, 0, :rp] = s_rp

    t_r, t_rp = process_case_all_tiers(1, :symmetric, rf_t)
    I[:t, -2, 0, 0, :r] = t_r
    I[:t, -2, 0, 0, :rp] = t_rp

    l_r, l_rp = process_case_all_tiers(1, :symmetric, rf_l)
    I[:l, -2, 0, 0, :r] = l_r
    I[:l, -2, 0, 0, :rp] = l_rp

    w_integrand_1_t1 = nothing; GC.gc()

    # =========================================================================
    # Case 2: (p,j,j',n) = (-2,0,2,0) — asymmetric (j≠j')
    # ;r gives (0,2);r, transpose gives (2,0);r'
    # =========================================================================
    println("Case 2: (-2,0,2,0) → s,t,l ;r and transposed ;r'...")

    s_r, s_rp = @time process_case_all_tiers(2, :asymmetric, rf_s)
    I[:s, -2, 0, 2, :r] = s_r    # (0,2);r
    I[:s, -2, 2, 0, :rp] = s_rp   # (2,0);r' via transpose

    t_r, t_rp = process_case_all_tiers(2, :asymmetric, rf_t)
    I[:t, -2, 0, 2, :r] = t_r
    I[:t, -2, 2, 0, :rp] = t_rp

    l_r, l_rp = process_case_all_tiers(2, :asymmetric, rf_l)
    I[:l, -2, 0, 2, :r] = l_r
    I[:l, -2, 2, 0, :rp] = l_rp

    GC.gc()

    # =========================================================================
    # Case 3: (p,j,j',n) = (-3,0,1,0) — asymmetric (j≠j')
    # ;r gives (0,1);r, transpose gives (1,0);r'
    # =========================================================================
    println("Case 3: (-3,0,1,0) → s,t,l ;r and transposed ;r'...")

    s_r, s_rp = @time process_case_all_tiers(3, :asymmetric, rf_s)
    I[:s, -3, 0, 1, :r] = s_r
    I[:s, -3, 1, 0, :rp] = s_rp

    t_r, t_rp = process_case_all_tiers(3, :asymmetric, rf_t)
    I[:t, -3, 0, 1, :r] = t_r
    I[:t, -3, 1, 0, :rp] = t_rp

    l_r, l_rp = process_case_all_tiers(3, :asymmetric, rf_l)
    I[:l, -3, 0, 1, :r] = l_r
    I[:l, -3, 1, 0, :rp] = l_rp

    GC.gc()

    # =========================================================================
    # Case 4: (p,j,j',n) = (-4,0,0,0) — symmetric (j=j')
    # =========================================================================
    println("Case 4: (-4,0,0,0) → s,t,l ;r and ;r'...")

    s_r, s_rp = @time process_case_all_tiers(4, :symmetric, rf_s)
    I[:s, -4, 0, 0, :r] = s_r
    I[:s, -4, 0, 0, :rp] = s_rp

    t_r, t_rp = process_case_all_tiers(4, :symmetric, rf_t)
    I[:t, -4, 0, 0, :r] = t_r
    I[:t, -4, 0, 0, :rp] = t_rp

    l_r, l_rp = process_case_all_tiers(4, :symmetric, rf_l)
    I[:l, -4, 0, 0, :r] = l_r
    I[:l, -4, 0, 0, :rp] = l_rp

    GC.gc()

    # =========================================================================
    # Case 5: (p,j,j',n) = (-4,0,0,-1) — symmetric, script font 𝔰,𝔱,𝔩
    # =========================================================================
    println("Case 5: (-4,0,0,-1) → scrs,scrt,scrl ;r and ;r'...")

    s_r, s_rp = @time process_case_all_tiers(5, :symmetric, rf_s)
    I[:scrs, -4, 0, 0, :r] = s_r
    I[:scrs, -4, 0, 0, :rp] = s_rp

    t_r, t_rp = process_case_all_tiers(5, :symmetric, rf_t)
    I[:scrt, -4, 0, 0, :r] = t_r
    I[:scrt, -4, 0, 0, :rp] = t_rp

    l_r, l_rp = process_case_all_tiers(5, :symmetric, rf_l)
    # NOTE: legacy in-memory convention — :scrl holds the BARE integral (no
    # ell(ell+1)/2, no /r subtraction).  Matched by the consumer at the
    # in-memory compute path further below.  The HDF5/streaming path uses a
    # tilde-key on-disk layout and applies _apply_paper_lensing_conversion!
    # at load time; see calcClGR.jl.
    I[:scrl, -4, 0, 0, :r]  = l_r
    I[:scrl, -4, 0, 0, :rp] = l_rp

    GC.gc()

    # =========================================================================
    # Case 6: (p,j,j',n) = (-4,0,0,0) 2D — 𝒮,𝒯,ℒ (scrS,scrT,scrL)
    # NO transpose needed (symmetric in r,r')
    # =========================================================================
    println("Case 6: (-4,0,0,0) 2D → scrS,scrT,scrL ;r,r'...")

    I[:scrS, -4, 0, 0, :r_rp] = @time process_case_all_tiers(6, :twoD, rf_s, rf_s)
    I[:scrT, -4, 0, 0, :r_rp] = process_case_all_tiers(6, :twoD, rf_t, rf_t)
    # Legacy in-memory convention: :scrL holds the BARE 2D integral here.
    I[:scrL, -4, 0, 0, :r_rp] = process_case_all_tiers(6, :twoD, rf_l, rf_l)

    GC.gc()

    # =========================================================================
    # Case 7: (p,j,j',n) = (-4,0,0,0) 2D — 𝒳,𝒴,𝒵 (scrX,scrY,scrZ)
    # Transpose gives (0,0);r',r version
    # =========================================================================
    println("Case 7: (-4,0,0,0) 2D → scrX,scrY,scrZ ;r,r' and ;r',r...")

    scrX = @time process_case_all_tiers(7, :twoD, rf_s, rf_t)
    I[:scrX, -4, 0, 0, :r_rp] = scrX
    scrX_rp_r = similar(scrX)
    @inbounds for ell_idx in 1:total_nell, j in 1:nr, i in 1:nr
        scrX_rp_r[ell_idx, i, j] = scrX[ell_idx, j, i]
    end
    I[:scrX, -4, 0, 0, :rp_r] = scrX_rp_r

    # Legacy in-memory convention: :scrY/:scrZ hold the BARE 2D integrals.
    scrY = process_case_all_tiers(7, :twoD, rf_s, rf_l)
    I[:scrY, -4, 0, 0, :r_rp] = scrY
    scrY_rp_r = similar(scrY)
    @inbounds for ell_idx in 1:total_nell, j in 1:nr, i in 1:nr
        scrY_rp_r[ell_idx, i, j] = scrY[ell_idx, j, i]
    end
    I[:scrY, -4, 0, 0, :rp_r] = scrY_rp_r

    scrZ = process_case_all_tiers(7, :twoD, rf_t, rf_l)
    I[:scrZ, -4, 0, 0, :r_rp] = scrZ
    scrZ_rp_r = similar(scrZ)
    @inbounds for ell_idx in 1:total_nell, j in 1:nr, i in 1:nr
        scrZ_rp_r[ell_idx, i, j] = scrZ[ell_idx, j, i]
    end
    I[:scrZ, -4, 0, 0, :rp_r] = scrZ_rp_r

    GC.gc()

    # =========================================================================
    # Optional: Load base TwoFAST results (w, u, v) on physical grid
    # (only supported for single-tier mode)
    # =========================================================================
    if load_base
        if length(tiers) > 1
            @warn "load_base=true is only supported for single-tier mode. Skipping."
        else
            tier = tiers[1]
            RR_tier = compute_RR_for_tier(tier.dlnR)
            println("Loading base wpljjprime functions (w, u, v) on physical grid...")

            function load_base_array_physical(p::Int, j::Int, jp::Int, n::Int)
                nell_t = length(aell_combined)
                w_integrand_raw = zeros(Float64, nr, nR, nell_t)
                for (ellindx, ell) in enumerate(aell_combined)
                    try
                        data_loc = wpljjprime(p, j, jp, n, ell, Nr=Nr, nR=nR,
                                              dlnR=tier.dlnR, ellmin=tier.ellmin, ellmax=tier.ellmax, datadir=datadir)
                        w_integrand_raw[:, :, ellindx] = data_loc.wrRl
                    catch e
                        @warn "Failed to load wpljjprime(p=$p, j=$j, jp=$jp, n=$n, ell=$ell): $e"
                    end
                end
                return interpolate_to_physical_grid(w_integrand_raw, rr, RR_tier)
            end

            # w (n=0)
            w_combos = [
                (0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2),
                (-1, 0, 1), (-1, 1, 0), (-1, 1, 2), (-1, 2, 1),
                (-2, 0, 0), (-2, 0, 2), (-2, 2, 0), (-2, 1, 1),
                (-3, 0, 1), (-3, 1, 0),
                (-4, 0, 0)
            ]
            for (p, j, jp) in w_combos
                println("  Loading w^{$p}_{$j,$jp} on physical grid...")
                I[:w, p, j, jp, :none] = load_base_array_physical(p, j, jp, 0)
            end

            # u (n=-1)
            u_combos = [
                (-2, 0, 0), (-2, 0, 2), (-2, 2, 0),
                (-3, 0, 1), (-3, 1, 0),
                (-4, 0, 0)
            ]
            for (p, j, jp) in u_combos
                println("  Loading u^{$p}_{$j,$jp} on physical grid...")
                I[:u, p, j, jp, :none] = load_base_array_physical(p, j, jp, -1)
            end

            # v (n=-2)
            println("  Loading v^{-4}_{0,0} on physical grid...")
            I[:v, -4, 0, 0, :none] = load_base_array_physical(-4, 0, 0, -2)
        end
    end

    println("Loaded $(length(I.data)) integral arrays on physical (r₁, r₂) grid.")
    println("  Tiers: $(length(tiers)), total ℓ values: $total_nell")
    for (tidx, tier) in enumerate(tiers)
        println("  Tier $tidx: dlnR=$(tier.dlnR), ℓ=$(tier.ellmin)-$(tier.ellmax)")
    end
    return I
end

# =============================================================================
# Verification function
# =============================================================================

"""
    verify_symmetry_at_diagonal(ic::IntegralCollection, type::Symbol, p::Int, j::Int, jp::Int;
                                 ell_idx::Int=50, tol::Float64=1e-6)

Verify that on the physical grid diagonal (r₁ = r₂), the ;r and ;r' integrals
with swapped j,j' are consistent.  For symmetric cases (j=j'):
    int_{;r}[i,i,ℓ] ≈ int_{;r'}[i,i,ℓ]

# Returns
- `true` if verification passes, `false` otherwise
"""
function verify_symmetry_at_diagonal(ic::IntegralCollection, type::Symbol, p::Int, j::Int, jp::Int;
                                      ell_idx::Int=50, tol::Float64=1e-6)
    nr = length(ic.rr)

    # Get original and transformed
    orig = ic[type, p, j, jp, :r]
    trans = ic[type, p, jp, j, :rp]  # Note: j,jp swapped for transform

    # Compare at diagonal (r₁ = r₂, i.e., i = j)
    orig_diag = [orig[ell_idx, i, i] for i in 1:nr]
    trans_diag = [trans[ell_idx, i, i] for i in 1:nr]

    max_diff = maximum(abs.(orig_diag .- trans_diag))
    rel_diff = max_diff / (maximum(abs.(orig_diag)) + 1e-30)

    if rel_diff < tol
        println("Verification passed: max relative difference at diagonal = $rel_diff")
        return true
    else
        @warn "Verification failed: max relative difference at diagonal = $rel_diff (tol=$tol)"
        return false
    end
end

# =============================================================================
# Export integrals to JLD2 and HDF5
# =============================================================================

# Helper function to encode p values (negative → "m" prefix)
function _encode_p(p::Int)
    if p < 0
        return "m$(abs(p))"
    else
        return string(p)
    end
end

# Helper function to decode p values ("m" prefix → negative)
function _decode_p(p_str::String)
    if startswith(p_str, "m")
        return -parse(Int, p_str[2:end])
    else
        return parse(Int, p_str)
    end
end

# Helper function to create dataset key
function _make_key(type::Symbol, p::Int, j::Int, jp::Int, sub::Symbol)
    p_str = _encode_p(p)
    if sub == :none
        return "$(type)_$(p_str)_$(j)_$(jp)"
    else
        return "$(type)_$(p_str)_$(j)_$(jp)_$(sub)"
    end
end

# Helper function to parse key
function _parse_key(key::String, is_base::Bool)
    parts = split(key, "_")
    type = Symbol(parts[1])
    p = _decode_p(String(parts[2]))
    j = parse(Int, parts[3])
    jp = parse(Int, parts[4])

    if is_base
        sub = :none
    else
        sub_str = join(parts[5:end], "_")
        sub = Symbol(sub_str)
    end

    return (type, p, j, jp, sub)
end

"""
    export_integrals_jld2(I::IntegralCollection, filename::String; use_float32::Bool=true)

Export integrals to JLD2 format with compression (for Julia users).

# Arguments
- `I::IntegralCollection`: Collection of all TwoFAST integrals
- `filename::String`: Output JLD2 filename
- `use_float32::Bool=true`: Convert to Float32 for ~2x size reduction (precision ~10⁻⁷)

# Notes
JLD2 uses built-in compression when compress=true in jldopen.
Float32 provides sufficient precision for most physics applications.
"""
function export_integrals_jld2(I::IntegralCollection, filename::String; use_float32::Bool=true, compress::Bool=false)
    maybe_f32(x) = use_float32 ? Float32.(x) : x

    jldopen(filename, "w"; compress=compress) do f
        # Grid information (keep full precision for coordinates)
        f["grid/rr"] = I.rr
        f["grid/ell_values"] = I.aell

        # Metadata
        f["metadata/Nr"] = I.Nr
        f["metadata/nR"] = I.nR
        f["metadata/logRmin"] = I.logRmin
        f["metadata/logRmax"] = I.logRmax
        f["metadata/nr_actual"] = length(I.rr)
        f["metadata/nell"] = length(I.aell)
        f["metadata/dtype"] = use_float32 ? "float32" : "float64"
        f["metadata/format_version"] = "1.1"

        # Tier metadata
        f["metadata/tier_count"] = length(I.tier_info)
        for (tidx, tier) in enumerate(I.tier_info)
            f["metadata/tier$(tidx)_dlnR"] = tier.dlnR
            f["metadata/tier$(tidx)_ellmin"] = tier.ellmin
            f["metadata/tier$(tidx)_ellmax"] = tier.ellmax
        end

        # Base TwoFAST results (w, u, v) — sub == :none
        n_base = 0
        for ((type, p, j, jp, sub), data) in I.data
            if sub == :none
                key = "base/$(_make_key(type, p, j, jp, sub))"
                f[key] = maybe_f32(data)
                n_base += 1
            end
        end

        # Integrated quantities (s, t, l, scrs, scrS, scrX, etc.) — sub != :none
        n_int = 0
        for ((type, p, j, jp, sub), data) in I.data
            if sub != :none
                key = "integrated/$(_make_key(type, p, j, jp, sub))"
                f[key] = maybe_f32(data)
                n_int += 1
            end
        end

        f["metadata/n_base"] = n_base
        f["metadata/n_integrated"] = n_int
    end

    filesize_mb = filesize(filename) / 1024^2
    println("Exported JLD2: $filename ($(round(filesize_mb, digits=1)) MB)")
    println("  - dtype: $(use_float32 ? "Float32" : "Float64")")
    println("  - $(length(I.data)) arrays total")

    return nothing
end

"""
    export_integrals_hdf5(I::IntegralCollection, filename::String;
                          compress_level::Int=4, use_float32::Bool=true)

Export integrals to HDF5 format with gzip compression (for Python users).

# Arguments
- `I::IntegralCollection`: Collection of all TwoFAST integrals
- `filename::String`: Output HDF5 filename
- `compress_level::Int=4`: gzip compression level (0=none, 1=fast, 9=max)
- `use_float32::Bool=true`: Convert to Float32 for ~2x size reduction

# HDF5 file structure
```
integrals.h5
├── grid/
│   ├── rr          [nr]         Float64  # r grid (Mpc/h), same for both axes
│   └── ell_values  [n_ell]      Int      # ℓ values
├── metadata/
│   ├── Nr, nR, logRmin, logRmax, dtype, ...
├── base/                        Float32, compressed
│   ├── w_0_0_0     [n_ell, nr, nr]   # physical (r₁, r₂) grid
│   ├── u_m2_0_0    [n_ell, nr, nr]   # m2 = -2
│   └── v_m4_0_0    [n_ell, nr, nr]
└── integrated/                  Float32, compressed
    ├── s_m2_0_0_r   [n_ell, nr, nr]
    ├── s_m2_0_0_rp  [n_ell, nr, nr]
    ├── scrS_m4_0_0_r_rp  [n_ell, nr, nr]
    └── ...
```

# Notes
- Negative p values: encoded as 'm' prefix (e.g., -2 → m2)
- Float32 precision (~10⁻⁷) is sufficient for physics
- compress_level=4 is good balance of speed vs size
"""
function export_integrals_hdf5(I::IntegralCollection, filename::String;
                                compress_level::Int=0, use_float32::Bool=true)
    maybe_f32(x) = use_float32 ? Float32.(x) : x

    mkpath(dirname(abspath(filename)))

    h5open(filename, "w") do f
        # Grid information (keep full precision)
        g_grid = create_group(f, "grid")
        g_grid["rr"] = I.rr
        g_grid["ell_values"] = I.aell

        # Metadata
        g_meta = create_group(f, "metadata")
        g_meta["Nr"] = I.Nr
        g_meta["nR"] = I.nR
        g_meta["logRmin"] = I.logRmin
        g_meta["logRmax"] = I.logRmax
        g_meta["nr_actual"] = length(I.rr)
        g_meta["nell"] = length(I.aell)
        g_meta["dtype"] = use_float32 ? "float32" : "float64"
        g_meta["compress_level"] = compress_level
        g_meta["format_version"] = "1.1"  # Updated for tier support

        # Tier metadata
        g_meta["tier_count"] = length(I.tier_info)
        for (tidx, tier) in enumerate(I.tier_info)
            g_meta["tier$(tidx)_dlnR"] = tier.dlnR
            g_meta["tier$(tidx)_ellmin"] = tier.ellmin
            g_meta["tier$(tidx)_ellmax"] = tier.ellmax
        end

        # Base TwoFAST results (w, u, v)
        g_base = create_group(f, "base")
        n_base = 0

        # Integrated quantities
        g_int = create_group(f, "integrated")
        n_int = 0

        # Iterate through all data
        for ((type, p, j, jp, sub), data) in I.data
            key = _make_key(type, p, j, jp, sub)
            arr = maybe_f32(data)

            if sub == :none
                if compress_level > 0
                    g_base[key, compress=compress_level] = arr
                else
                    g_base[key] = arr
                end
                n_base += 1
            else
                if compress_level > 0
                    g_int[key, compress=compress_level] = arr
                else
                    g_int[key] = arr
                end
                n_int += 1
            end
        end

        g_meta["n_base"] = n_base
        g_meta["n_integrated"] = n_int
    end

    filesize_mb = filesize(filename) / 1024^2
    println("Exported HDF5: $filename ($(round(filesize_mb, digits=1)) MB)")
    println("  - dtype: $(use_float32 ? "Float32" : "Float64"), compress=$compress_level")
    println("  - $(length(I.data)) arrays total")

    return nothing
end

# Helper: write a [nell, nr, nr] array to split HDF5 part files (ℓ-range slices)
function _write_array_to_split_h5!(part_files::Vector{String},
                                    ell_ranges::Matrix{Int},
                                    group::String,
                                    key::String,
                                    data::Array{Float64,3},
                                    use_float32::Bool;
                                    compress_level::Int=0,
                                    h5_lock::Union{ReentrantLock,Nothing}=nothing)::Nothing
    maybe_f32(x) = use_float32 ? Float32.(x) : x
    n_parts = size(ell_ranges, 1)

    # Parallelize over parts: each part file is a distinct HDF5 file, so
    # multiple threads can open+write+close them simultaneously.  If the
    # caller passed an h5_lock we still honour it (caller owns serial
    # semantics), but the common default path parallelizes.
    if h5_lock === nothing
        @threads for part_idx in 1:n_parts
            ell_start = ell_ranges[part_idx, 1]
            ell_end   = ell_ranges[part_idx, 2]
            arr_slice = maybe_f32(data[ell_start:ell_end, :, :])
            h5open(part_files[part_idx], "r+") do f
                g = f[group]
                if compress_level > 0
                    g[key, compress=compress_level] = arr_slice
                else
                    g[key] = arr_slice
                end
            end
        end
    else
        for part_idx in 1:n_parts
            ell_start = ell_ranges[part_idx, 1]
            ell_end   = ell_ranges[part_idx, 2]
            arr_slice = maybe_f32(data[ell_start:ell_end, :, :])
            lock(h5_lock) do
                h5open(part_files[part_idx], "r+") do f
                    g = f[group]
                    if compress_level > 0
                        g[key, compress=compress_level] = arr_slice
                    else
                        g[key] = arr_slice
                    end
                end
            end
        end
    end
    return nothing
end

"""
    export_integrals_hdf5_split(I::IntegralCollection, basename::String;
                                 compress_level::Int=0, use_float32::Bool=true,
                                 max_size_gb::Float64=5.0)

Export IntegralCollection to multiple HDF5 files, splitting by ell index.

# Arguments
- `I::IntegralCollection`: Collection to export
- `basename::String`: Base name for output files (without extension)
- `compress_level::Int=0`: HDF5 compression level (0-9)
- `use_float32::Bool=true`: Use Float32 for storage
- `max_size_gb::Float64=5.0`: Target maximum size per file in GB

# Output files
- `{basename}_meta.h5`: Grid, metadata, and file list
- `{basename}_part_001.h5`, `{basename}_part_002.h5`, ...: Data files

# File structure
```
{basename}_meta.h5
├── grid/
│   ├── rr [nr]
│   └── ell_values [n_ell]
├── metadata/
│   ├── Nr, nR, logRmin, logRmax, ...
│   ├── n_parts (number of part files)
│   ├── ell_ranges [n_parts, 2] (start/end ell index for each part)
│   └── format_version = "2.0"
└── part_files [string array of part filenames]

{basename}_part_NNN.h5
├── ell_start, ell_end (ell index range in this file)
├── base/
│   └── {key} [n_ell_in_part, nr, nr]
└── integrated/
    └── {key} [n_ell_in_part, nr, nr]
```
"""
function export_integrals_hdf5_split(I::IntegralCollection, basename::String;
                                      compress_level::Int=0, use_float32::Bool=true,
                                      max_size_gb::Float64=5.0)
    maybe_f32(x) = use_float32 ? Float32.(x) : x

    n_ell = length(I.aell)
    nr = length(I.rr)
    nR = nr  # physical grid: both axes are rr
    n_arrays = length(I.data)

    # Estimate bytes per ell slice
    bytes_per_element = use_float32 ? 4 : 8
    bytes_per_ell = nr * nR * n_arrays * bytes_per_element

    # Calculate ell per part
    max_bytes = max_size_gb * 1024^3
    ell_per_part = max(1, floor(Int, max_bytes / bytes_per_ell))
    n_parts = ceil(Int, n_ell / ell_per_part)

    println("Splitting $n_ell ell values into $n_parts parts (~$ell_per_part ell/part)")
    println("Estimated size per part: $(round(bytes_per_ell * ell_per_part / 1024^3, digits=2)) GB")

    # Compute ell ranges for each part
    ell_ranges = Matrix{Int}(undef, n_parts, 2)
    for i in 1:n_parts
        ell_start = (i - 1) * ell_per_part + 1
        ell_end = min(i * ell_per_part, n_ell)
        ell_ranges[i, :] = [ell_start, ell_end]
    end

    # Generate part filenames (full path for writing, basename for storing in meta)
    base_name_only = Base.basename(basename)  # Remove directory path
    part_files = ["$(basename)_part_$(lpad(i, 3, '0')).h5" for i in 1:n_parts]
    part_filenames = ["$(base_name_only)_part_$(lpad(i, 3, '0')).h5" for i in 1:n_parts]  # Just filenames

    # Write meta file
    meta_file = "$(basename)_meta.h5"
    mkpath(dirname(abspath(meta_file)))
    h5open(meta_file, "w") do f
        # Grid information
        g_grid = create_group(f, "grid")
        g_grid["rr"] = I.rr
        g_grid["ell_values"] = I.aell

        # Metadata
        g_meta = create_group(f, "metadata")
        g_meta["Nr"] = I.Nr
        g_meta["nR"] = I.nR
        g_meta["logRmin"] = I.logRmin
        g_meta["logRmax"] = I.logRmax
        g_meta["nr_actual"] = nr
        g_meta["nell"] = n_ell

        # Tier metadata
        g_meta["tier_count"] = length(I.tier_info)
        for (tidx, tier) in enumerate(I.tier_info)
            g_meta["tier$(tidx)_dlnR"] = tier.dlnR
            g_meta["tier$(tidx)_ellmin"] = tier.ellmin
            g_meta["tier$(tidx)_ellmax"] = tier.ellmax
        end
        g_meta["dtype"] = use_float32 ? "float32" : "float64"
        g_meta["compress_level"] = compress_level
        g_meta["format_version"] = "2.0"  # New split format
        g_meta["n_parts"] = n_parts
        g_meta["ell_ranges"] = ell_ranges

        # Part file list (store only filenames, not full paths)
        f["part_files"] = part_filenames
    end
    println("Written: $meta_file")

    # Write each part file
    for part_idx in 1:n_parts
        ell_start = ell_ranges[part_idx, 1]
        ell_end = ell_ranges[part_idx, 2]
        part_file = part_files[part_idx]

        h5open(part_file, "w") do f
            # Ell range info
            f["ell_start"] = ell_start
            f["ell_end"] = ell_end

            # Base TwoFAST results (w, u, v)
            g_base = create_group(f, "base")

            # Integrated quantities
            g_int = create_group(f, "integrated")

            # Iterate through all data
            for ((type, p, j, jp, sub), data) in I.data
                key = _make_key(type, p, j, jp, sub)
                # Extract ell slice
                arr_slice = maybe_f32(data[ell_start:ell_end, :, :])

                if sub == :none
                    if compress_level > 0
                        g_base[key, compress=compress_level] = arr_slice
                    else
                        g_base[key] = arr_slice
                    end
                else
                    if compress_level > 0
                        g_int[key, compress=compress_level] = arr_slice
                    else
                        g_int[key] = arr_slice
                    end
                end
            end
        end

        part_size_mb = filesize(part_file) / 1024^2
        println("Written: $part_file ($(round(part_size_mb, digits=1)) MB) - ell $ell_start:$ell_end")
    end

    total_size_mb = filesize(meta_file) / 1024^2 + sum(filesize(f) for f in part_files) / 1024^2
    println("\nTotal: $(round(total_size_mb / 1024, digits=2)) GB in $(n_parts + 1) files")

    return part_files
end

"""
    export_all_integrals(I::IntegralCollection, basename::String; use_float32::Bool=true)

Export integrals to both JLD2 and HDF5 formats.
- JLD2 for Julia users (native, fast loading)
- HDF5 for Python users (h5py compatible)

# Arguments
- `I::IntegralCollection`: Collection of all TwoFAST integrals
- `basename::String`: Base filename (without extension)
- `use_float32::Bool=true`: Convert to Float32 for size reduction
"""
function export_all_integrals(I::IntegralCollection, basename::String; use_float32::Bool=true)
    export_integrals_jld2(I, "$(basename).jld2"; use_float32=use_float32)
    export_integrals_hdf5(I, "$(basename).h5"; use_float32=use_float32)

    println("\nExported both formats:")
    println("  Julia:  $(basename).jld2")
    println("  Python: $(basename).h5")
end

"""
    import_integrals_jld2(filename::String) -> IntegralCollection

Import integrals from JLD2 file back into IntegralCollection.
Automatically converts Float32 back to Float64 for calculations.
"""
function import_integrals_jld2(filename::String)
    data = Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}()
    local rr, aell, Nr, nR, logRmin, logRmax
    local tier_info

    jldopen(filename, "r") do f
        # Grid
        rr = Float64.(f["grid/rr"])
        aell = f["grid/ell_values"]

        # Metadata
        Nr = f["metadata/Nr"]
        nR = f["metadata/nR"]
        logRmin = f["metadata/logRmin"]
        logRmax = f["metadata/logRmax"]

        # Tier metadata (v1.1+)
        if haskey(f, "metadata/tier_count")
            n_tiers = f["metadata/tier_count"]
            tier_info = [@NamedTuple{dlnR::Float64, ellmin::Int, ellmax::Int}((
                f["metadata/tier$(i)_dlnR"],
                f["metadata/tier$(i)_ellmin"],
                f["metadata/tier$(i)_ellmax"]
            )) for i in 1:n_tiers]
        else
            tier_info = [@NamedTuple{dlnR::Float64, ellmin::Int, ellmax::Int}((0.0, minimum(aell), maximum(aell)))]
        end

        # Base functions
        if haskey(f, "base")
            for key in keys(f["base"])
                tuple_key = _parse_key(key, true)
                data[tuple_key] = Float64.(f["base/$key"])
            end
        end

        # Integrated quantities
        if haskey(f, "integrated")
            for key in keys(f["integrated"])
                tuple_key = _parse_key(key, false)
                data[tuple_key] = Float64.(f["integrated/$key"])
            end
        end
    end

    println("Imported $(length(data)) arrays from $filename")
    return IntegralCollection(data, rr, aell, Nr, nR, logRmin, logRmax, tier_info)
end

"""
    import_integrals_hdf5(filename::String) -> IntegralCollection

Import integrals from HDF5 file back into IntegralCollection.
Automatically converts Float32 back to Float64 for calculations.
"""
function import_integrals_hdf5(filename::String)
    data = Dict{Tuple{Symbol,Int,Int,Int,Symbol}, Array{Float64,3}}()
    local rr, aell, Nr, nR, logRmin, logRmax
    local tier_info

    h5open(filename, "r") do f
        # Grid
        rr = Float64.(read(f["grid/rr"]))
        aell = read(f["grid/ell_values"])

        # Metadata
        Nr = read(f["metadata/Nr"])
        nR = read(f["metadata/nR"])
        logRmin = read(f["metadata/logRmin"])
        logRmax = read(f["metadata/logRmax"])

        # Tier metadata (v1.1+)
        if haskey(f["metadata"], "tier_count")
            n_tiers = read(f["metadata/tier_count"])
            tier_info = [@NamedTuple{dlnR::Float64, ellmin::Int, ellmax::Int}((
                read(f["metadata/tier$(i)_dlnR"]),
                read(f["metadata/tier$(i)_ellmin"]),
                read(f["metadata/tier$(i)_ellmax"])
            )) for i in 1:n_tiers]
        else
            tier_info = [@NamedTuple{dlnR::Float64, ellmin::Int, ellmax::Int}((0.0, minimum(aell), maximum(aell)))]
        end

        # Base functions
        if haskey(f, "base")
            for key in keys(f["base"])
                tuple_key = _parse_key(key, true)
                data[tuple_key] = Float64.(read(f["base/$key"]))
            end
        end

        # Integrated quantities
        if haskey(f, "integrated")
            for key in keys(f["integrated"])
                tuple_key = _parse_key(key, false)
                data[tuple_key] = Float64.(read(f["integrated/$key"]))
            end
        end
    end

    println("Imported $(length(data)) arrays from $filename")
    return IntegralCollection(data, rr, aell, Nr, nR, logRmin, logRmax, tier_info)
end

"""
    list_hdf5_contents(filename::String)

List the contents of an HDF5 integrals file.
"""
function list_hdf5_contents(filename::String)
    h5open(filename, "r") do f
        println("=== HDF5 Contents: $filename ===\n")

        # Grid info
        println("Grid:")
        rr = read(f["grid/rr"])
        ell = read(f["grid/ell_values"])
        println("  rr: $(length(rr)) points, [$(rr[1]), $(rr[end])]")
        println("  ell_values: $(length(ell)) values, [$(ell[1]), $(ell[end])]")

        # Metadata
        println("\nMetadata:")
        for key in sort(collect(keys(f["metadata"])))
            val = read(f["metadata/$key"])
            println("  $key: $val")
        end

        # Base functions
        if haskey(f, "base")
            base_keys = sort(collect(keys(f["base"])))
            println("\nBase functions ($(length(base_keys)) total):")
            for key in base_keys
                arr = read(f["base/$key"])
                println("  $key: $(size(arr)) $(eltype(arr))")
            end
        end

        # Integrated quantities
        if haskey(f, "integrated")
            int_keys = sort(collect(keys(f["integrated"])))
            println("\nIntegrated quantities ($(length(int_keys)) total):")
            for key in int_keys
                arr = read(f["integrated/$key"])
                println("  $key: $(size(arr)) $(eltype(arr))")
            end
        end

        # File size
        filesize_mb = filesize(filename) / 1024^2
        println("\nFile size: $(round(filesize_mb, digits=1)) MB")
    end
end

"""
    list_jld2_contents(filename::String)

List the contents of a JLD2 integrals file.
"""
function list_jld2_contents(filename::String)
    jldopen(filename, "r") do f
        println("=== JLD2 Contents: $filename ===\n")

        # Grid info
        println("Grid:")
        rr = f["grid/rr"]
        ell = f["grid/ell_values"]
        println("  rr: $(length(rr)) points, [$(rr[1]), $(rr[end])]")
        println("  ell_values: $(length(ell)) values, [$(ell[1]), $(ell[end])]")

        # Metadata
        println("\nMetadata:")
        for key in sort(collect(keys(f["metadata"])))
            val = f["metadata/$key"]
            println("  $key: $val")
        end

        # Base functions
        if haskey(f, "base")
            base_keys = sort(collect(keys(f["base"])))
            println("\nBase functions ($(length(base_keys)) total):")
            for key in base_keys
                arr = f["base/$key"]
                println("  $key: $(size(arr)) $(eltype(arr))")
            end
        end

        # Integrated quantities
        if haskey(f, "integrated")
            int_keys = sort(collect(keys(f["integrated"])))
            println("\nIntegrated quantities ($(length(int_keys)) total):")
            for key in int_keys
                arr = f["integrated/$key"]
                println("  $key: $(size(arr)) $(eltype(arr))")
            end
        end

        # File size
        filesize_mb = filesize(filename) / 1024^2
        println("\nFile size: $(round(filesize_mb, digits=1)) MB")
    end
end

# Export new functions
export transform_r_to_rprime, IntegralCollection, load_all_integrals, get_rr_grid
export show_available_keys, verify_symmetry_at_diagonal
export export_integrals_jld2, export_integrals_hdf5, export_integrals_hdf5_split, export_all_integrals
export import_integrals_jld2, import_integrals_hdf5
export list_jld2_contents, list_hdf5_contents

end # module PowerFull
