#!/usr/bin/env -S julia --project
using HDF5
using Dierckx
using LinearAlgebra
using Printf
using SHA

const SURVEY_ZMIN = parse(Float64, get(ENV, "CONT_ZMIN", "0.05"))
const SURVEY_ZMAX = parse(Float64, get(ENV, "CONT_ZMAX", "4.6"))

function trap_weights(x::AbstractVector{<:Real})
    n = length(x)
    n >= 2 || error("trap_weights needs at least two nodes")
    w = Vector{Float64}(undef, n)
    w[1] = 0.5 * (x[2] - x[1])
    w[end] = 0.5 * (x[end] - x[end-1])
    @inbounds for k in 2:n-1
        w[k] = 0.5 * (x[k+1] - x[k-1])
    end
    return w
end

trapz(x, y) = sum(trap_weights(x) .* y)

_parse_bool_string(x::AbstractString) = lowercase(strip(x)) in ("1", "true", "yes", "on")

function parse_args()
    env_quad = lowercase(get(ENV, "WINDOW_QUAD", "zproj"))
    env_quad == "zproj" || error(
        "This script is the zproj shot-noise builder; WINDOW_QUAD must be zproj, got '$env_quad'."
    )

    o = Dict{Symbol,Any}(
        :fsky               => parse(Float64, get(ENV, "FSKY", "1.0")),
        :window_nz          => parse(Int, get(ENV, "WINDOW_NZ", "257")),
        :window_zscan       => parse(Int, get(ENV, "WINDOW_ZSCAN", "8193")),
        :window_active_rtol => parse(Float64, get(ENV, "WINDOW_ACTIVE_RTOL", "1e-12")),
        :noise_oversample   => parse(Float64, get(ENV, "NOISE_Z_OVERSAMPLE", "1.0")),
        :write_window_table => _parse_bool_string(get(ENV, "WRITE_WINDOW_TABLE", "0")),
        :params             => nothing,
        :meta               => nothing,
        :out                => nothing,
    )

    pos = String[]
    for a in ARGS
        if startswith(a, "--fsky=")
            o[:fsky] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--window-nz=")
            o[:window_nz] = parse(Int, split(a, "=", limit=2)[2])
        elseif startswith(a, "--window-zscan=")
            o[:window_zscan] = parse(Int, split(a, "=", limit=2)[2])
        elseif startswith(a, "--window-active-rtol=")
            o[:window_active_rtol] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--noise-oversample=")
            o[:noise_oversample] = parse(Float64, split(a, "=", limit=2)[2])
        elseif startswith(a, "--write-window-table=")
            o[:write_window_table] = _parse_bool_string(split(a, "=", limit=2)[2])
        elseif startswith(a, "--nz-true=")
            o[:window_zscan] = parse(Int, split(a, "=", limit=2)[2])
            @warn "--nz-true is deprecated; treating it as --window-zscan"
        elseif startswith(a, "--")
            error("unknown argument: $a")
        else
            push!(pos, a)
        end
    end

    length(pos) >= 3 || error(
        "usage: compute_continuous_shotnoise_zproj.jl " *
        "<params.h5> <tracer_meta_cont.h5> <Nshot_cont_zproj.h5> " *
        "[--fsky=..] [--window-nz=..] [--window-zscan=..] " *
        "[--window-active-rtol=..] [--noise-oversample=..] " *
        "[--write-window-table=0|1]"
    )

    o[:params], o[:meta], o[:out] = pos[1], pos[2], pos[3]

    o[:fsky] > 0 || error("fsky must be positive")
    o[:window_nz] >= 3 || error("window_nz must be >= 3")
    o[:window_zscan] >= 17 || error("window_zscan must be >= 17")
    o[:window_active_rtol] >= 0 || error("window_active_rtol must be >= 0")
    o[:noise_oversample] >= 1 || error("noise_oversample must be >= 1")

    return o
end

function load_nbar_of_z(params_file::String, samples)
    out = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()

    h5open(params_file, "r") do f
        ztest = Float64.(read(f["ztest"]))

        for s in samples
            if haskey(f, "shot_noise_d$s")
                inv_nbar = Float64.(read(f["shot_noise_d$s"]))
                length(inv_nbar) == length(ztest) || error(
                    "shot_noise_d$s length $(length(inv_nbar)) != ztest length $(length(ztest))"
                )
                nbar = [iv > 0 ? 1.0 / iv : 0.0 for iv in inv_nbar]
                out[s] = (ztest, nbar)
                @info "sample $s: using z-dependent shot_noise_d$s"
            elseif haskey(f, "sel_func_$(s)_com") && haskey(f, "shot_noise_$s")
                sel = Float64.(read(f["sel_func_$(s)_com"]))
                Nbar_tot = 1.0 / Float64(read(f["shot_noise_$s"]))
                total = trapz(ztest, sel)
                total > 0 || error("sample $s selection integral is not positive")
                nbar = Nbar_tot .* (sel ./ total)
                out[s] = (ztest, nbar)
                @warn "sample $s: using selection-shape fallback for nbar(z)"
            else
                error(
                    "sample $s lacks shot_noise_d$s and the " *
                    "sel_func_$(s)_com + shot_noise_$s fallback"
                )
            end
        end
    end

    return out
end

struct ZProjWindow
    path::String
    sample::Int
    zobs::Float64
    zmin::Float64
    zmax::Float64
    phi_spl::Spline1D
    source_norm::Float64
    active_lo::Float64
    active_hi::Float64
    local_dz::Float64
    local_norm::Float64
end

@inline function phi_value(w::ZProjWindow, z::Real)
    if z < w.zmin || z > w.zmax || z < w.active_lo || z > w.active_hi
        return 0.0
    end
    return w.phi_spl(z) / w.local_norm
end

function _resolve_tracer_path(meta_path::String, raw_path::String)
    p = strip(raw_path)
    return isabspath(p) ? normpath(p) : normpath(joinpath(dirname(meta_path), p))
end

function build_zproj_window(path::String,
                            sample_expected::Int,
                            zobs_expected::Float64,
                            window_nz::Int,
                            window_zscan::Int,
                            active_rtol::Float64)
    isfile(path) || error("tracer file not found: $path")

    z, phi, sample_file, zobs_file = h5open(path, "r") do f
        haskey(f, "z") || error("tracer is missing /z: $path")
        haskey(f, "phi") || error("tracer is missing /phi: $path")
        z = Float64.(read(f["z"]))
        phi = Float64.(read(f["phi"]))
        s = haskey(f, "sample") ? Int(read(f["sample"])) : sample_expected
        zo = haskey(f, "z_obs") ? Float64(read(f["z_obs"])) : zobs_expected
        (z, phi, s, zo)
    end

    length(z) == length(phi) || error("/z and /phi lengths differ in $path")
    length(z) >= 4 || error("need at least four tracer z nodes in $path")
    all(isfinite, z) || error("non-finite /z values in $path")
    all(isfinite, phi) || error("non-finite /phi values in $path")

    order = sortperm(z)
    z = z[order]
    phi = phi[order]
    all(diff(z) .> 0) || error("tracer /z must be strictly increasing: $path")

    sample_file == sample_expected || error(
        "sample mismatch for $path: meta=$sample_expected, tracer=$sample_file"
    )
    abs(zobs_file - zobs_expected) <= 1e-10 * max(1.0, abs(zobs_expected)) || error(
        "z_obs mismatch for $path: meta=$zobs_expected, tracer=$zobs_file"
    )

    k_spl = min(3, length(z) - 1)
    spl = Spline1D(z, phi, k=k_spl)

    lo = max(SURVEY_ZMIN, first(z))
    hi = min(SURVEY_ZMAX, last(z))
    hi > lo || error("empty survey/tracer overlap for $path")

    zscan = collect(range(lo, hi; length=window_zscan))
    pscan = [spl(zz) for zz in zscan]
    peak = maximum(abs, pscan)
    peak > 0 || error("tracer phi is identically zero: $path")

    active = if active_rtol <= 0
        collect(eachindex(zscan))
    else
        findall(abs.(pscan) .>= active_rtol * peak)
    end
    isempty(active) && error("active support is empty for $path")

    ia = first(active)
    ib = last(active)
    active_lo = zscan[ia]
    active_hi = zscan[ib]

    if active_hi <= active_lo
        ia = max(1, ia - 1)
        ib = min(length(zscan), ib + 1)
        active_lo = zscan[ia]
        active_hi = zscan[ib]
    end
    active_hi > active_lo || error("degenerate active support for $path")

    zlocal = collect(range(active_lo, active_hi; length=window_nz))
    wlocal = trap_weights(zlocal)
    plocal = [spl(zz) for zz in zlocal]
    local_norm = sum(wlocal .* plocal)
    local_norm > 0 || error("non-positive zproj window norm for $path: $local_norm")

    source_norm = trapz(z, phi)
    source_norm > 0 || error("non-positive source-grid phi norm for $path")

    local_dz = (active_hi - active_lo) / (window_nz - 1)

    return ZProjWindow(
        path,
        sample_expected,
        zobs_expected,
        first(z),
        last(z),
        spl,
        source_norm,
        active_lo,
        active_hi,
        local_dz,
        local_norm,
    )
end

function build_noise_block(windows::Vector{ZProjWindow},
                           nbar_spl::Spline1D,
                           fsky::Float64,
                           noise_oversample::Float64;
                           write_table::Bool=false)
    ns = length(windows)
    ns > 0 || error("cannot build an empty sample block")

    zlo = minimum(w.active_lo for w in windows)
    zhi = maximum(w.active_hi for w in windows)
    dz_target = minimum(w.local_dz for w in windows) / noise_oversample
    dz_target > 0 || error("non-positive common-grid target spacing")

    nz = max(3, ceil(Int, (zhi - zlo) / dz_target) + 1)
    z = collect(range(zlo, zhi; length=nz))
    wt = trap_weights(z)

    Phi = zeros(Float64, nz, ns)
    common_mass = Vector{Float64}(undef, ns)

    @inbounds for j in 1:ns
        win = windows[j]
        for k in 1:nz
            Phi[k, j] = phi_value(win, z[k])
        end
        common_mass[j] = sum(wt .* view(Phi, :, j))
    end

    mass_err = maximum(abs.(common_mass .- 1.0))
    mass_err <= 5e-4 || error(
        "common z-grid does not reproduce zproj window normalization: " *
        "max |int phi dz - 1| = $mass_err. Increase NOISE_Z_OVERSAMPLE."
    )

    nbar = Vector{Float64}(undef, nz)
    @inbounds for k in 1:nz
        nbar[k] = max(nbar_spl(z[k]), 0.0)
    end

    sqrtfac = zeros(Float64, nz)
    @inbounds for k in 1:nz
        sqrtfac[k] = nbar[k] > 0 ? sqrt(wt[k] / (fsky * nbar[k])) : 0.0
    end

    B = Phi .* reshape(sqrtfac, :, 1)
    block = Matrix(transpose(B) * B)
    block .= 0.5 .* (block .+ transpose(block))

    table = write_table ? (
        z=z,
        weights=wt,
        phi=Phi,
        nbar=nbar,
        common_mass=common_mass,
    ) : nothing

    return block, table, nz, z[2] - z[1], mass_err
end

function window_hash(windows::Vector{ZProjWindow}, o)
    io = IOBuffer()
    write(io, "window_quad=zproj\n")
    write(io, "window_nz=$(o[:window_nz])\n")
    write(io, "window_zscan=$(o[:window_zscan])\n")
    write(io, "window_active_rtol=$(o[:window_active_rtol])\n")
    write(io, "noise_oversample=$(o[:noise_oversample])\n")

    for w in windows
        write(io, w.path)
        write(io, '\n')
        write(io, string(w.sample, ',', w.zobs, ',', w.active_lo, ',', w.active_hi,
                         ',', w.local_dz, ',', w.local_norm, '\n'))
    end

    return bytes2hex(sha256(take!(io)))
end

function main()
    o = parse_args()

    @info "params=$(o[:params])"
    @info "meta=$(o[:meta])"
    @info "out=$(o[:out])"
    @info "WINDOW_QUAD=zproj"
    @info "WINDOW_NZ=$(o[:window_nz])"
    @info "WINDOW_ZSCAN=$(o[:window_zscan])"
    @info "WINDOW_ACTIVE_RTOL=$(o[:window_active_rtol])"
    @info "NOISE_Z_OVERSAMPLE=$(o[:noise_oversample])"
    @info "fsky=$(o[:fsky])"

    isfile(o[:params]) || error("parameter file not found: $(o[:params])")
    isfile(o[:meta]) || error("tracer metadata file not found: $(o[:meta])")

    sample_of, zobs_of, tracer_paths_raw = h5open(o[:meta], "r") do f
        for key in ("sample", "z_obs", "tracer_paths")
            haskey(f, key) || error("metadata file is missing /$key: $(o[:meta])")
        end
        (
            Int.(read(f["sample"])),
            Float64.(read(f["z_obs"])),
            String.(read(f["tracer_paths"])),
        )
    end

    nalpha = length(sample_of)
    length(zobs_of) == nalpha || error("z_obs length mismatch")
    length(tracer_paths_raw) == nalpha || error("tracer_paths length mismatch")

    tracer_paths = [
        _resolve_tracer_path(o[:meta], p) for p in tracer_paths_raw
    ]

    samples = sort(unique(sample_of))
    @printf("Loaded %d continuous tracers across samples %s\n", nalpha, string(samples))

    @info "Loading exact tracer /phi arrays and constructing zproj supports..."
    windows = Vector{ZProjWindow}(undef, nalpha)

    Threads.@threads for alpha in 1:nalpha
        windows[alpha] = build_zproj_window(
            tracer_paths[alpha],
            sample_of[alpha],
            zobs_of[alpha],
            o[:window_nz],
            o[:window_zscan],
            o[:window_active_rtol],
        )
    end

    src_norm_err = maximum(abs(w.source_norm - 1.0) for w in windows)
    local_norm_shift = maximum(abs(w.local_norm / w.source_norm - 1.0) for w in windows)
    @printf("max |source-grid int phi dz - 1|       = %.3e\n", src_norm_err)
    @printf("max |zproj active norm/source norm-1| = %.3e\n", local_norm_shift)

    nbar_tab = load_nbar_of_z(o[:params], samples)
    nbar_spl = Dict{Int,Spline1D}()
    for s in samples
        ztab, ntab = nbar_tab[s]
        nbar_spl[s] = Spline1D(ztab, ntab, k=min(3, length(ztab)-1))
    end

    N = zeros(Float64, nalpha, nalpha)
    idx_of_sample = Dict(s => findall(==(s), sample_of) for s in samples)

    sample_tables = Dict{Int,Any}()
    common_nz = Dict{Int,Int}()
    common_dz = Dict{Int,Float64}()
    common_mass_err = Dict{Int,Float64}()

    @info "Assembling PSD zproj shot-noise blocks..."
    for s in samples
        idxs = idx_of_sample[s]
        wins = windows[idxs]

        @printf("  sample %d: %d tracers\n", s, length(idxs))
        flush(stdout)

        block, table, nz, dz, mass_err = build_noise_block(
            wins,
            nbar_spl[s],
            o[:fsky],
            o[:noise_oversample];
            write_table=o[:write_window_table],
        )

        N[idxs, idxs] .= block
        common_nz[s] = nz
        common_dz[s] = dz
        common_mass_err[s] = mass_err
        if table !== nothing
            sample_tables[s] = table
        end

        evals = eigvals(Symmetric(block))
        lambda_max = maximum(evals)
        lambda_min = minimum(evals)
        neg_rel = lambda_max > 0 ? min(lambda_min, 0.0) / lambda_max : NaN
        cond_pos = lambda_max / max(
            minimum(evals[evals .> max(lambda_max * 1e-14, 0.0)]),
            eps(Float64),
        )

        d = diag(block)
        corr = block ./ sqrt.(d * transpose(d))
        offmax = length(d) > 1 ? maximum(abs.(corr - I)) : 0.0
        nnmax = length(d) > 1 ? maximum(abs(corr[k, k+1]) for k in 1:length(d)-1) : 0.0

        @printf(
            "    common grid: Nz=%d dz=%.4e max|mass-1|=%.3e\n",
            nz, dz, mass_err,
        )
        @printf(
            "    block: minEig/maxEig=%.3e cond_pos=%.3e max|offcorr|=%.3f NN=%.3f\n",
            lambda_max > 0 ? lambda_min/lambda_max : NaN,
            cond_pos,
            offmax,
            nnmax,
        )
        if neg_rel < -1e-11
            error("sample $s zproj shot-noise block is not PSD: min/max=$neg_rel")
        end
    end

    asym = maximum(abs.(N .- transpose(N))) / max(maximum(abs, N), eps(Float64))
    asym <= 1e-12 || error("global N is asymmetric at relative level $asym")
    all(diag(N) .> 0) || error("global N contains a non-positive diagonal entry")

    whash = window_hash(windows, o)

    h5open(o[:out], "w") do f
        f["N"] = N
        f["sample"] = sample_of
        f["z_obs"] = zobs_of
        f["nbar_eff"] = [N[a,a] > 0 ? 1.0 / N[a,a] : 0.0 for a in 1:nalpha]
        f["fsky"] = o[:fsky]
        f["tracer_paths"] = tracer_paths

        g = create_group(f, "zproj")
        g["active_lo"] = [w.active_lo for w in windows]
        g["active_hi"] = [w.active_hi for w in windows]
        g["local_dz"] = [w.local_dz for w in windows]
        g["source_norm"] = [w.source_norm for w in windows]
        g["local_norm"] = [w.local_norm for w in windows]

        if o[:write_window_table]
            for s in samples
                table = sample_tables[s]
                sg = create_group(g, "sample_$s")
                sg["z"] = table.z
                sg["trap_weights"] = table.weights
                sg["phi"] = table.phi
                sg["nbar"] = table.nbar
                sg["common_mass"] = table.common_mass
                sg["global_indices"] = idx_of_sample[s]
            end
        end

        p = create_group(f, "provenance")
        p["params"] = o[:params]
        p["meta"] = o[:meta]
        p["window_quad"] = "zproj"
        p["window_nz"] = o[:window_nz]
        p["window_zscan"] = o[:window_zscan]
        p["window_active_rtol"] = o[:window_active_rtol]
        p["noise_z_oversample"] = o[:noise_oversample]
        p["phi_source"] = "exact tracer HDF5 /phi"
        p["quadrature"] = "sample-common trapezoid, dz <= minimum local zproj dz"
        p["window_hash"] = whash
        p["write_window_table"] = o[:write_window_table] ? 1 : 0

        for s in samples
            ztab, ntab = nbar_tab[s]
            p["Nbar_sr_$s"] = trapz(ztab, ntab)
            p["common_nz_$s"] = common_nz[s]
            p["common_dz_$s"] = common_dz[s]
            p["common_mass_error_$s"] = common_mass_err[s]
        end
    end

    @printf("\nWrote zproj shot-noise matrix N[%d x %d] to %s\n", nalpha, nalpha, o[:out])
    @printf("window hash: %s\n", whash)
    println("Use this file in the Fisher as Cbar_ell = C_ell + N.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

