#!/usr/bin/env -S julia --project
# =============================================================================
#
# >> PowerFullTwoFAST.jl <<
#
# Julia program to run TwoFAST to compute the components needed for PowerFull.
#
#  18 November 2025
#  Donghui Jeong
#
# Usage: julia -p 8 --project PowerFullTwoFAST.jl --Nr=1024 --nR=1025
# =============================================================================

# Setup distributed computing (workers should be added via julia -p N)
using Distributed

# -- Early CLI parsing: paths must be known before @everywhere const cfns --
# Parse --matterpower=<path> and --cosmo-funcr=<path> here so they can be
# broadcast to workers before the cosmology tables are loaded.  The full
# arg parser lower in the file re-parses the remaining options.
function _early_path_arg(name::AbstractString, default::String)
    for arg in ARGS
        pfx = "--$(name)="
        if startswith(arg, pfx)
            return String(arg[length(pfx)+1:end])
        end
    end
    return default
end

const _default_matterpower = joinpath(@__DIR__, "..", "data",
    "astropy_planck_2018_matterpower.dat")
const _default_cosmo_funcr = joinpath(@__DIR__, "..", "data", "cosmo_funcr_astropy_planck2018.txt")

const MATTERPOWER_PATH = _early_path_arg("matterpower", _default_matterpower)
const COSMO_FUNCR_PATH = _early_path_arg("cosmo-funcr", _default_cosmo_funcr)
@info "Cosmology inputs" MATTERPOWER_PATH COSMO_FUNCR_PATH

# Broadcast the paths to all workers so @everywhere bodies see them.
_mp = MATTERPOWER_PATH
_cf = COSMO_FUNCR_PATH
@everywhere const MATTERPOWER_PATH = $_mp
@everywhere const COSMO_FUNCR_PATH = $_cf

# Load packages on all workers
@everywhere using Dierckx
@everywhere using DelimitedFiles

const TWOFASTPP_SRC = joinpath(@__DIR__, "TwoFASTpp", "src", "TwoFASTpp.jl")

@everywhere include($TWOFASTPP_SRC)
@everywhere using .TwoFASTpp

@everywhere using JLD2
@everywhere using Printf
@everywhere using SharedArrays

# =============================================================================
# Module to compute the power spectrum over exteneded range of wavenumber.
@everywhere module PkSpectra

 export PkSpectrum, transferk
 using Dierckx  # same as used in the TwoFAST tests
 using DelimitedFiles

 struct PkSpectrum
    kk::Vector{Float64}
    pk::Vector{Float64}
    pkspl::Spline1D
    kmin::Float64
    kmax::Float64
    nslo::Float64
    nshi::Float64
    kmin_norm::Float64
    kmax_norm::Float64
 end

 function PkSpectrum(filename=joinpath(@__DIR__, "..", "data", "astropy_planck_2018_matterpower.dat"))
    if !isfile(filename)
        error("Power spectrum file not found: $filename")
    end
    data = readdlm(filename, comments=true)
    kk = data[:,1]
    pk = data[:,2]
    pkspl = Spline1D(kk, pk)

    # fit low-k using derivative
    k0 = kk[1]
    P0 = pkspl(k0)
    Pp0 = derivative(pkspl, k0)
    nslo = k0 * Pp0 / P0
    kmin_norm = P0 / k0^nslo
    kmin = k0

    # fit high-k using derivative
    k0 = kk[end]
    P0 = pkspl(k0)
    Pp0 = derivative(pkspl, k0)
    nshi = 4 + k0 * Pp0 / P0
    kmax_norm = P0 / (k0^(nshi - 4))
    kmax = k0

    PkSpectrum(kk, pk, pkspl, kmin, kmax, nslo, nshi, kmin_norm, kmax_norm)
 end

 function pkspectrum(k, pwr)
    if k < pwr.kmin
        return pwr.kmin_norm * k^pwr.nslo
    elseif k > pwr.kmax
        return pwr.kmax_norm * k^(pwr.nshi - 4)
    else
        return pwr.pkspl(k)
    end
 end

 function transferk(k,pwr)
    pk  = pkspectrum(k,pwr)
    pk0 = pwr.kmin_norm * k^pwr.nslo
    x   = pk / pk0

    if x < 0.0
        if abs(x) < 1e-12
            x = 0.0
        else
            error("transferk got genuinely negative pk/pk0 = $x at k=$k, pk=$pk, pk0=$pk0")
        end
    end

    return sqrt(x)
 end

 (pwr::PkSpectrum)(k) = pkspectrum(k, pwr)

end # of module PkSpectrum
# ============================================================================
@everywhere include(joinpath(@__DIR__, "cosmofns.jl"))
# ============================================================================

# Use the modules defined above on all workers
@everywhere using .PkSpectra
@everywhere using .cosmofns

# cosmology functions to be used in the integration (on all workers)
@info "collecting cosmology functions::"
@everywhere const cfns = cosmofns.cosmofn(COSMO_FUNCR_PATH)

# compute the rmin and rmax corresponding to zmin and zmax
const zmin = 0.01
const zmax = 5.0
const rmin = cfns.frz(zmin)
const rmax = cfns.frz(zmax)

rmin_log = round(rmin, digits=3)
rmax_log = round(rmax, digits=3)
@info("Computation of angular correlation functions",
      rmin = rmin_log,
      rmax = rmax_log,
      zmin = zmin,
      zmax = zmax)

# ===========================================================================
# We compute
#
#   w^{p,n}_{ell,jj'}(r,r')
# = 2/\pi \int dk k^{2+p} Tk^n j_ell^{(a)}(kr) j_ell^{(b)}(kr')
#
# with the bias factor q, which is determined empirically.
# ===========================================================================
# ===========================================================================
# 9-base structure (Apr 2026): Split by (p_eff, n) for optimal q per integrand
# - Bases 1-2: p_base=0,  n=0,  split by Δp (p_eff=0 vs p_eff=-1)
# - Bases 3-4: p_base=-2, n=0,  split by Δp (p_eff=-2 vs p_eff=-3)
# - Base  5:   p_base=-4, n=0   (p_eff=-4)
# - Bases 6-7: p_base=-2, n=-1, split by Δp (p_eff=-2 vs p_eff=-3)
# - Bases 8-9: p_base=-4, n=-1/-2
# q* values: two-stage selection at N=4096 (see Appendix B)
# ===========================================================================
# Define constants on all workers for @distributed access
@everywhere const parray = [0,     0,    -2,    -2,   -4,   -2,   -2,   -4,   -4]
@everywhere const narray = [0,     0,     0,     0,    0,   -1,   -1,   -1,   -2]
@everywhere const qarray = [1.30, 1.50, -0.20, 0.50, -1.92, 0.50, 0.80, -1.08, 0.02]
# Base 1: p_base=0,  n=0,  Δp=0  → p_eff=0,  q=1.30
# Base 2: p_base=0,  n=0,  Δp=-1 → p_eff=-1, q=1.50
# Base 3: p_base=-2, n=0,  Δp=0  → p_eff=-2, q=-0.20
# Base 4: p_base=-2, n=0,  Δp=-1 → p_eff=-3, q=0.50
# Base 5: p_base=-4, n=0         → p_eff=-4, q=-1.92
# Base 6: p_base=-2, n=-1, Δp=0  → p_eff=-2, q=0.50
# Base 7: p_base=-2, n=-1, Δp=-1 → p_eff=-3, q=0.80
# Base 8: p_base=-4, n=-1        → p_eff=-4, q=-1.08
# Base 9: p_base=-4, n=-2        → p_eff=-4, q=0.02

@everywhere const DjjpΔp = Dict{Int,NTuple{3,Int}}(
            1 => (0,0,0),
            2 => (0,2,0),
            3 => (2,0,0),
            4 => (2,2,0),
            5 => (0,1,-1),
            6 => (1,0,-1),
            7 => (1,1,0),
            8 => (1,2,-1),
            9 => (2,1,-1))

@everywhere const acases = [
    [1,2,3,4],      # Base 1: p_eff=0,  n=0
    [5,6,8,9],      # Base 2: p_eff=-1, n=0
    [1,2,3,7],      # Base 3: p_eff=-2, n=0
    [5,6],          # Base 4: p_eff=-3, n=0
    [1],            # Base 5: p_eff=-4, n=0
    [1,2,3],        # Base 6: p_eff=-2, n=-1 (split from old base 6)
    [5,6],          # Base 7: p_eff=-3, n=-1 (split from old base 6)
    [1],            # Base 8: p_eff=-4, n=-1
    [1],            # Base 9: p_eff=-4, n=-2
]

# Build a reverse lookup table for faster access: (p,j,jprime,n) -> (indx, cindx)
@everywhere function build_lookup_table()
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

@everywhere const LOOKUP_TABLE = build_lookup_table()

"""
    validate_output_files(Nr::Int, nR::Int, dlnR::Float64, ellmin::Int, ellmax::Int;
                          outdir::String="./results") -> Bool

Check if all expected TwoFAST output files exist.
Returns true if all files are present, false otherwise.
"""
function validate_output_files(Nr::Int, nR::Int, dlnR::Float64, ellmin::Int, ellmax::Int;
                               outdir::String="./results")
    all_exist = true
    for indx in 1:length(parray)
        filename = joinpath(outdir, "TwoFAST_output_nr=$(Nr)_nR=$(nR)_dlnR=$(dlnR)_ell=$(ellmin)-$(ellmax)_$indx.jld2")
        if !isfile(filename)
            @warn "Missing output file: $filename"
            all_exist = false
        end
    end
    return all_exist
end

"""
    run_TwoFAST_powerfull(; dlnR=0.002, nR=2049, Nr=4096, ellmin=2, ellmax=500, ...)

Run the full TwoFAST computation to generate all output files.

# Arguments
- `dlnR::Float64=0.002`: Logarithmic spacing in ln(R), where R = r'/r
- `nR::Int=2049`: Number of R grid points
- `Nr::Int=4096`: FFTlog grid size
- `ellmin::Int=2`: Minimum multipole moment
- `ellmax::Int=500`: Maximum multipole moment
- `outdir::String="./results"`: Directory where output files will be saved
- `force_recompute::Bool=false`: If true, recompute even if files exist

# R grid construction
R = exp(dlnR × m), m = -half_nR...half_nR, half_nR = (nR-1)÷2

# Returns
- `nothing`

The function generates files TwoFAST_output_nr=..._nR=..._dlnR=..._ell=..._[indx].jld2
containing the computed w^{p,n}_{ell,jj'}(r,r') arrays.
"""
function run_TwoFAST_powerfull(; dlnR::Float64=0.002, nR::Int=2049, Nr::Int=4096,
                                  ellmin::Int=2, ellmax::Int=500,
                                  ellmax_margin::Int=0,
                                  outdir::String="./results", force_recompute::Bool=false,
                                  mlcache_dir::String="",       # "" = default Cacheout/, else override
                                  mlcache_cleanup::Bool=false,  # delete MlCache after calcwljj
                                  max_q_parallel::Int=0,        # 0 = no throttle
                                  streaming_mlcache::Bool=false) # true = no disk MlCache, fuse build+consume

    println("Starting TwoFAST computation...")
    start_time = time()

    # Create output directory if it doesn't exist
    mkpath(outdir)

    # power spectrum (path set via --matterpower CLI, broadcast to workers)
    pk = PkSpectrum(MATTERPOWER_PATH)
    tk(k) = transferk(k,pk)

    # FFTlog parameters
    # grid size
    Ngrid = Nr

    # TwoFASTpp computes the derivative bases `w^p_{ℓ,jj'}(r, R)` via 2F1
    # recurrences in ℓ whose accuracy degrades near the ends of the
    # requested ℓ list (the base `w^0_{ℓ,00}` itself is fine — only the
    # derivatives are affected; see Newtonian-vs-Kaiser diagnostic).
    # Setting `ellmax_margin > ellmax` asks the recurrence to extend to
    # the higher ceiling and then trims the output to `ellmin:ellmax`
    # before saving, pushing the noisy top-of-range points out of the
    # usable ell list.  The naming is historical — the margin kwarg name
    # stays even though the actual recurrence artifact is 2F1's, not
    # Miller's.  `ellmax_margin=0` (default) disables this and the
    # compute range equals the save range.
    ellmax_run = ellmax_margin > ellmax ? ellmax_margin : ellmax

    # Check if outputs already exist
    if !force_recompute && validate_output_files(Ngrid, nR, dlnR, ellmin, ellmax; outdir=outdir)
        println("All output files already exist. Use force_recompute=true to regenerate.")
        return nothing
    end
    # wavenumber intervals
    kmin = 1e-5
    kmax = 1e3
    # chi0 = 1/kmax is the minimum radius
    chi0 = 1e-3

    # ell range for the computation (may extend above ellmax by
    # ellmax_margin for the recurrence-ceiling margin; trimmed back to
    # ellmin:ellmax before save).
    aell   = collect(ellmin:ellmax_run)
    Nell   = length(aell)
    n_save = ellmax - ellmin + 1           # slices kept in the saved file

    # Array for the radii-ratio R = r'/r
    # R = exp(dlnR × m), m = -half_nR ... half_nR
    half_nR = (nR - 1) ÷ 2   # 1024 for nR=2049
    lnRR = collect(range(-half_nR * dlnR, stop=half_nR * dlnR, length=nR))
    RR   = exp.(lnRR)

    Nrr    = Ngrid
    NRR    = length(RR)

    # All cache files go into a tier-specific Cacheout folder.  Cache is
    # keyed by the compute ceiling `ellmax_run`, not the save ellmax, so
    # a run with `ellmax_margin` > 0 gets its own cache dir and a later
    # run with a larger margin doesn't falsely reuse a shorter cache.
    outfolder = "./Cacheout_nR=$(nR)_dlnR=$(dlnR)_ellmax=$(ellmax_run)"
    mkpath(outfolder)

    # MlCache base directory: defaults to outfolder (the "dumb option" — keep
    # 100+ GB per q on /gpfs forever).  Override with mlcache_dir to redirect
    # to /dev/shm or any fast scratch.  Combine with mlcache_cleanup=true to
    # delete each base's MlCache after calcwljj_powerfull is done with it,
    # reclaiming /dev/shm space for the next batch.
    mlcache_outfolder = isempty(mlcache_dir) ? outfolder : mlcache_dir
    mkpath(mlcache_outfolder)
    println("Cache: F21EllCache → $outfolder, MlCache → $mlcache_outfolder",
            mlcache_cleanup ? " (cleanup ON)" : "")

    # precomputing the 2F1 and Ml Cache files (parallelized with @distributed)
    # Pre-compute the full rr array to filter by rmin/rmax (shared across all indices)
    G = log(kmax / kmin)
    rr_full = @. chi0 * exp((0:Ngrid-1) * (G / Ngrid))
    mask = (rmin .<= rr_full .<= rmax)
    valid_ridxs = findall(mask)
    Nrr_filtered = length(valid_ridxs)
    rr_filtered = rr_full[mask]

    # Throttle q-parallelism (default = no throttle = all 9 q parallel = needs ~1 TB
    # MlCache space).  For /dev/shm mode (~252 GB per node), set max_q_parallel=2.
    n_bases = length(parray)
    batch_size = (max_q_parallel <= 0) ? n_bases : min(max_q_parallel, n_bases)
    batches = [collect(b:min(b+batch_size-1, n_bases)) for b in 1:batch_size:n_bases]
    println("Q-parallelism: $batch_size in flight, $(length(batches)) batches")

    for batch_indices in batches
    @sync @distributed for indx in batch_indices
        # Per-base resume: if this base's output file already exists (and
        # the caller did not request force_recompute), skip.  This allows
        # a resubmit to complete only the bases that were missed when a
        # prior worker died mid-run.
        output_file = joinpath(outdir, "TwoFAST_output_nr=$(Nr)_nR=$(nR)_dlnR=$(dlnR)_ell=$(ellmin)-$(ellmax)_$indx.jld2")
        if !force_recompute && isfile(output_file)
            println("Base $indx already exists, skipping: $output_file")
            continue
        end

        p = parray[indx]
        n = narray[indx]
        q = qarray[indx]
        cases = acases[indx]

        # define the Fourier function in this case
        kfn(k) = k^p * pk(k) * tk(k)^n
        # turn on the oddprimes if max(cases)>5
        oddprimes = (maximum(cases) > 4)
        qname = strip(@sprintf "_q=%4.1f" q)
        if oddprimes
            qname = qname*"_oddprimes"
        end

        # calculate M_ll at high ell, result gets saved to a file.
        # On rerun we skip the (slow) F21 and Ml builds if their disk
        # caches already exist, since neither the F21 struct nor the Ml
        # struct is read back in-memory below — only the paths are used
        # by calcwljj_powerfull below.
        fol2F1cache = joinpath(outfolder,"F21EllCache"*qname)
        folMlcache  = joinpath(mlcache_outfolder,"MlCache"*qname)
        fMlcache    = joinpath(folMlcache,"MlCache.bin")

        if !isdir(fol2F1cache) || force_recompute   # was isfile; F21EllCache is a dir
            # Build the 2F1 cache up to the compute ceiling so Miller can
            # reach every ell in `aell` (which runs to `ellmax_run`, not
            # just the save `ellmax`).
            f21cache = F21EllCache(ellmax_run, RR, Ngrid; q=q, kmin=kmin, kmax=kmax, χ0=chi0)
            write(fol2F1cache, f21cache)
        else
            println("F21EllCache cache hit: $fol2F1cache")
        end

        # calculate all M_ll, result gets saved to a file:
        # In streaming mode the disk MlCache is skipped entirely; the per-ell
        # wjj is built and consumed in a fused loop below.
        if !streaming_mlcache
            if !isfile(fMlcache) || force_recompute
                MlCache(aell, fol2F1cache, folMlcache, oddprimes=oddprimes)
            else
                println("MlCache cache hit: $fMlcache")
            end
        else
            println("[streaming] skipping MlCache disk build for $qname")
        end

        # Allocate fullresult with filtered size
        Ncases = length(cases)
        fullresult = Array{Float64,4}(undef, Nrr_filtered, NRR, Ncases, Nell)

        function outfunc(wjj, ell, rr, RR)
            @show ell
            for (cindx,c) in enumerate(cases)
                ellindx = findfirst(==(ell), aell)
                # Store only the filtered r indices
                fullresult[:,:,cindx,ellindx] = wjj[c][valid_ridxs, :]
            end
        end

        # Compute with full FFTlog grid (don't use ridxs to preserve FFT)
        if streaming_mlcache
            rr = calcwljj_powerfull_streaming(kfn, RR; ell=aell, kmin=kmin, kmax=kmax,
                N=Ngrid, r0=chi0, q=q, outfunc=outfunc,
                fell_lmax_file=fol2F1cache, oddprimes=oddprimes)
        else
            rr = calcwljj_powerfull(kfn, RR; ell=aell, kmin=kmin, kmax=kmax, N=Ngrid,
                r0=chi0, q=q, outfunc=outfunc, cachefile=fMlcache, oddprimes=oddprimes)
        end

        # Trim Miller margin (if any) back to the save range ellmin:ellmax.
        # fullresult has `Nell` ell slices along axis 4; we keep the first
        # `n_save` (= ellmax - ellmin + 1), which are the lower ells
        # (ellmin..ellmax), and drop the Miller margin (ellmax+1..ellmax_run).
        fullresult_save = ellmax_run == ellmax ? fullresult :
                          fullresult[:, :, :, 1:n_save]
        aell_save = aell[1:n_save]

        # Save only the filtered results (use rr= to save rr_filtered with key "rr")
        # Filename uses input Nr for easy lookup; actual data dimensions saved as metadata
        output_file = joinpath(outdir, "TwoFAST_output_nr=$(Nr)_nR=$(nR)_dlnR=$(dlnR)_ell=$(ellmin)-$(ellmax)_$indx.jld2")
        @save output_file fullresult=fullresult_save rr=rr_filtered RR aell=aell_save dlnR ellmin ellmax Nr_input=Nr Nr_actual=Nrr_filtered nR
        println("Saved: $output_file")

        # Reclaim MlCache space (esp. when on /dev/shm) so the next batch
        # has room.  Skipped by default — set mlcache_cleanup=true to enable.
        if mlcache_cleanup && isdir(folMlcache)
            rm(folMlcache, recursive=true, force=true)
            println("Cleaned up: $folMlcache")
        end
    end  # @sync @distributed
    end  # for batch_indices

    elapsed = time() - start_time
    println("="^60)
    println("Computation completed in $(elapsed/60) minutes")
    println("Nr_actual = $Nrr_filtered (filtered from input Nr=$Nr)")
    println("="^60)
end
# ===========================================================================
# The portion of the code to unfold the fullresult
@everywhere module PowerFullInt

 using JLD2
 using Distributed
 using SharedArrays
 using ..Main: LOOKUP_TABLE, cfns

 export wpljjprime, clear_cache!, save_w_integrand

 struct wpljjprime
     # store wpljjprime[rindx,Rindx] for the specific ell,n,p,j,jprime
     # wrRl: 2D array of w^{p,n}_{ell,jj'}(r,r') values for a specific ell
     # rr: radii values
     # RR: radii ratio values (r'/r)
     # ell: the specific multipole value for this data
    wrRl::Array{Float64,2}
    rr::Vector{Float64}
    RR::Vector{Float64}
    ell::Int
    n::Int
    p::Int
    j::Int
    jprime::Int
 end

 # Cache for loaded files to avoid repeated I/O
 # Cache key is (indx, Nr, nR, dlnR) to support multiple grid resolutions
 const _file_cache = Dict{Tuple{Int,Int,Int,Float64},Tuple{Array{Float64,4},Vector{Float64},Vector{Float64},Vector{Int}}}()

 """
     wpljjprime(p, j, jprime, n, ell; Nr, nR, dlnR, ellmin, ellmax, ...) -> wpljjprime

 Outer constructor that loads w^{p,n}_{ell,jj'}(r,r') data for given (p,j,jprime,n,ell).

 # Arguments
 - `p::Int`: Power index
 - `j::Int`: First derivative order
 - `jprime::Int`: Second derivative order
 - `n::Int`: Transfer function power (T(k)^n)
 - `ell::Int`: Multipole moment value
 - `Nr::Int=4096`: Number of r grid points (must match the data file)
 - `nR::Int=2049`: Number of R grid points (must match the data file)
 - `dlnR::Float64=0.002`: Logarithmic R spacing (must match the data file)
 - `ellmin::Int=2`: Minimum ell in the data file
 - `ellmax::Int=500`: Maximum ell in the data file
 - `use_cache::Bool=true`: Whether to cache loaded files in memory
 - `datadir::String="./results"`: Directory where TwoFAST output files are located
 """
 function wpljjprime(p::Int, j::Int, jprime::Int, n::Int, ell::Int; Nr::Int=4096, nR::Int=2049,
                     dlnR::Float64=0.002, ellmin::Int=2, ellmax::Int=500,
                     use_cache::Bool=true, datadir::String="./results")
    # Use the precomputed lookup table for O(1) lookup
    key = (p, j, jprime, n)
    if !haskey(LOOKUP_TABLE, key)
        available = sort(collect(keys(LOOKUP_TABLE)))
        error("No matching entry found for p=$p, j=$j, jprime=$jprime, n=$n. " *
              "Available combinations: $available")
    end

    indx_found, cindx_found = LOOKUP_TABLE[key]

    # Load the appropriate file (with caching)
    cache_key = (indx_found, Nr, nR, dlnR)

    local fullresult::Array{Float64,4}
    local rr::Vector{Float64}
    local RR::Vector{Float64}
    local aell::Vector{Int}

    if use_cache && haskey(_file_cache, cache_key)
        fullresult, rr, RR, aell = _file_cache[cache_key]
    else
        filename = joinpath(datadir, "TwoFAST_output_nr=$(Nr)_nR=$(nR)_dlnR=$(dlnR)_ell=$(ellmin)-$(ellmax)_$indx_found.jld2")
        if !isfile(filename)
            error("File $filename not found. Run run_TwoFAST_powerfull(dlnR=$dlnR, nR=$nR, ellmin=$ellmin, ellmax=$ellmax) first with Nr=$Nr.")
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
            _file_cache[cache_key] = (fullresult, rr, RR, aell)
        end
    end

    # Find the index for the requested ell
    ellindx = findfirst(==(ell), aell)
    if isnothing(ellindx)
        error("ell=$ell not found in available ell values. Available range: $(minimum(aell)) to $(maximum(aell))")
    end

    # Extract the specific 2D array for this (p,j,jprime,n,ell) combination
    # fullresult[rindx,Rindx,cindx,ellindx]
    wrRl = fullresult[:,:,cindx_found,ellindx]

    return wpljjprime(wrRl, rr, RR, ell, n, p, j, jprime)
 end

 """
     clear_cache!()

 Clear the internal file cache to free memory.
 """
 function clear_cache!()
    empty!(_file_cache)
    GC.gc()  # Suggest garbage collection
    return nothing
 end
 function save_w_integrand(aell::Vector{Int}; Nr::Int=4096, nR::Int=21,
                            dlnR::Float64=0.002, ellmin::Int=2, ellmax::Int=500,
                            datadir::String="./results")
    # Save raw w_integrand data for each case, to be integrated later on a
    # physical (r₁, r₂) grid in PowerFull.jl.
    #
    # Cases 1-5: 1D integral inputs  (different p,j,j',n)
    # Cases 6-7: 2D integral inputs  (same w^{-4}_{00}, different assembly)
    #
    # Output: w_integrand[nr, nR, nell] per case file
    #   TwoFAST_w_integrand_nr=<Nr>_nR=<nR>_dlnR=<>_ell=<>-<>_<case>.jld2

    parray_int  = [-2,-2,-3,-4,-4, -4,-4]
    jarray_int  = [ 0, 0, 0, 0, 0,  0, 0]
    jparray_int = [ 0, 2, 1, 0, 0,  0, 0]
    narray_int  = [ 0, 0, 0, 0,-1,  0, 0]
    # Cases 6 and 7 use the same w^{-4}_{00} as case 4, but are stored
    # separately for clarity and backward compatibility.

    # prep: load one to get grid info
    data = wpljjprime(-2,0,0,0,aell[1],Nr=Nr,nR=nR,dlnR=dlnR,ellmin=ellmin,ellmax=ellmax,datadir=datadir)
    rr = data.rr
    RR = data.RR
    Nr_actual = length(rr)
    Nell = length(aell)

    # Preallocate the output buffer once.  This step is I/O-bound (reading
    # a ~13 GB JLD2 per case and writing a ~13 GB output), so running the
    # ell loop on the main process only keeps peak memory bounded to
    # ~(one TwoFAST_output file) + (one w_integrand buffer) ≈ 2×file_size.
    # The previous @distributed version loaded the same 13 GB file into
    # every worker's cache simultaneously, which OOMs a 96 GB node for
    # N_workers ≥ 8.
    w_integrand = Array{Float64}(undef, Nr_actual, nR, Nell)

    for icase in 1:length(parray_int)
        p      = parray_int[icase]
        j      = jarray_int[icase]
        jprime = jparray_int[icase]
        n      = narray_int[icase]

        for ell_idx in 1:Nell
            ell = aell[ell_idx]
            data_local = wpljjprime(p, j, jprime, n, ell,
                                    Nr=Nr, nR=nR, dlnR=dlnR, ellmin=ellmin, ellmax=ellmax, datadir=datadir)
            w_integrand[:, :, ell_idx] = data_local.wrRl
        end

        output_file = joinpath(datadir,
            "TwoFAST_w_integrand_nr=$(Nr)_nR=$(nR)_dlnR=$(dlnR)_ell=$(ellmin)-$(ellmax)_$(icase).jld2")
        @save output_file w_integrand rr RR aell Nr nR dlnR ellmin ellmax p j jprime n
        println("Saved w_integrand case $icase (p=$p, j=$j, j'=$jprime, n=$n): $output_file")

        # Release the main-process file cache so the next case starts from
        # scratch and does not accumulate.
        clear_cache!()
    end
 end

end # End of module

# =============================================================================
# CLI argument parsing
# =============================================================================
function parse_args()
    args = Dict{Symbol, Any}(
        :Nr => 4096,
        :nR => 2049,
        :dlnR => 0.002,
        :ellmin => 2,
        :ellmax => 500,
        :ellmax_margin => 0,
        :outdir => "./results",
        :force_recompute => false,
        :mlcache_dir => "",
        :mlcache_cleanup => false,
        :max_q_parallel => 0,
        :streaming_mlcache => false,
    )

    for arg in ARGS
        if arg == "--help" || arg == "-h"
            println("""
Usage: julia -p N --project src/run_twofast.jl [options]

Options:
  --Nr=<int>          FFTlog grid size (default: 4096)
  --nR=<int>          Number of R grid points (default: 2049)
  --dlnR=<float>      Logarithmic R spacing (default: 0.002)
  --ellmin=<int>      Minimum ell (default: 2)
  --ellmax=<int>      Maximum ell (default: 500)
  --outdir=<path>     Output directory (default: ./results)
  --matterpower=<p>   CAMB linear P(k) file (default: data/astropy_planck_2018_matterpower.dat)
  --cosmo-funcr=<p>   Cosmology table r(z),H(r),Ω_m,f_g,D (default: data/cosmo_funcr_astropy_planck2018.txt)
  --force-recompute   Recompute even if output files exist
  --mlcache-dir=<p>   Override location for MlCache (default: same as --outdir)
  --mlcache-cleanup   Delete MlCache after calcwljj completes
  --max-q-parallel=<n>  Throttle q-batch parallelism (0 = no throttle)
  --streaming-mlcache Skip disk MlCache; fuse build+consume per ell.
                      Saves disk + read/write time; bit-identical output.
  --help, -h          Show this help message

R grid: R = exp(dlnR × m), m = -(nR-1)/2 ... (nR-1)/2

Examples (two-tier):
  # Tier 1: low-ell, coarse R
  julia -p 8 --project src/run_twofast.jl --Nr=4096 --nR=2049 --dlnR=0.002 --ellmin=2 --ellmax=199

  # Tier 2: high-ell, fine R
  julia -p 8 --project src/run_twofast.jl --Nr=4096 --nR=2049 --dlnR=0.0005 --ellmin=200 --ellmax=500
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
        elseif startswith(arg, "--ellmax-margin=")
            args[:ellmax_margin] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--outdir=")
            args[:outdir] = String(split(arg, "=")[2])
        elseif startswith(arg, "--matterpower=") || startswith(arg, "--cosmo-funcr=")
            # already consumed by _early_path_arg (MATTERPOWER_PATH / COSMO_FUNCR_PATH)
            nothing
        elseif arg == "--force-recompute"
            args[:force_recompute] = true
        elseif startswith(arg, "--mlcache-dir=")
            args[:mlcache_dir] = String(split(arg, "=")[2])
        elseif arg == "--mlcache-cleanup"
            args[:mlcache_cleanup] = true
        elseif startswith(arg, "--max-q-parallel=")
            args[:max_q_parallel] = parse(Int, split(arg, "=")[2])
        elseif arg == "--streaming-mlcache"
            args[:streaming_mlcache] = true
        else
            @warn "Unknown argument: $arg"
        end
    end

    return args
end

if abspath(PROGRAM_FILE) == @__FILE__
    using .PowerFullInt

    args = parse_args()
    Nr       = args[:Nr]
    nR       = args[:nR]
    dlnR     = args[:dlnR]
    ellmin   = args[:ellmin]
    ellmax   = args[:ellmax]
    outdir   = args[:outdir]

    @info "Run powerfull" Nr nR dlnR ellmin ellmax outdir
    # A Distributed worker occasionally throws a spurious
    # ProcessExitedException during teardown AFTER all 9 output files
    # have been saved.  Catch that and proceed to save_w_integrand if the
    # expected outputs exist on disk; otherwise rethrow.
    try
        @time run_TwoFAST_powerfull(
            dlnR=dlnR, nR=nR, Nr=Nr,
            ellmin=ellmin, ellmax=ellmax,
            ellmax_margin=args[:ellmax_margin],
            outdir=outdir, force_recompute=args[:force_recompute],
            mlcache_dir=args[:mlcache_dir],
            mlcache_cleanup=args[:mlcache_cleanup],
            max_q_parallel=args[:max_q_parallel],
            streaming_mlcache=args[:streaming_mlcache],
        )
    catch err
        if validate_output_files(Nr, nR, dlnR, ellmin, ellmax; outdir=outdir)
            @warn "run_TwoFAST_powerfull threw but all output files exist; continuing to save_w_integrand." exception=err
        else
            rethrow()
        end
    end

    aell = collect(ellmin:ellmax)
    @info "Saving w_integrand" Nr nR dlnR ellmin ellmax
    @time save_w_integrand(aell, Nr=Nr, nR=nR,
                            dlnR=dlnR, ellmin=ellmin, ellmax=ellmax, datadir=outdir)
    clear_cache!()
end

# Interactive usage example:
# julia> include("src/run_twofast.jl")
# julia> run_TwoFAST_powerfull(dlnR=0.002, nR=2049, ellmin=2, ellmax=199)
# julia> result = PowerFullInt.wpljjprime(0, 0, 0, 0, 100, Nr=4096, nR=2049, dlnR=0.002, ellmin=2, ellmax=199)
# julia> result.wrRl  # Access the 2D array [rindx, Rindx]
# julia> PowerFullInt.clear_cache!()  # Clear memory when done
#
# Two-tier usage (CLI):
#   Tier 1: julia -p 8 --project src/run_twofast.jl --dlnR=0.002 --ellmin=2 --ellmax=199
#   Tier 2: julia -p 8 --project src/run_twofast.jl --dlnR=0.0005 --ellmin=200 --ellmax=500
#
# Output files: ./results/TwoFAST_output_nr=..._nR=..._dlnR=..._ell=..._[indx].jld2
# Cache files:  ./Cacheout_dlnR=..._ellmax=.../
