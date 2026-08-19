#!/usr/bin/env julia

using DelimitedFiles
using Printf

const REPO = "/storage/group/duj13/default/PowerFull.jl"
include(joinpath(REPO, "scripts", "grid_initialization_powerfull.jl"))

const DELTAS = Dict(
    "Om" => 0.001,
    "Ok" => 0.001,
    "w0" => 0.005,
    "wa" => 0.005,
    "ns" => 0.0005,
    "as" => 0.0005,
)

function getarg(prefix::String, default=nothing)
    for a in ARGS
        startswith(a, prefix) && return split(a, "=", limit=2)[2]
    end
    return default
end

if length(ARGS) < 1
    error("Usage: julia make_powerfull_cosmo_table.jl OUTDIR [--param=x --side=p/n --matterpower=file]")
end

outdir = ARGS[1]
param  = getarg("--param=", "fid")
side   = getarg("--side=", "fid")

mkpath(outdir)

# --- fiducial defaults ------------------------------------------------------
const OM_FID = 0.3111
const OK_FID = 0.0
const W0_FID = -1.0
const WA_FID = 0.0
const OR_FID = 0.0

Om = OM_FID
Ok = OK_FID
w0 = W0_FID
wa = WA_FID
Or = OR_FID

# --- 1) explicit cosmology flags take precedence (what the drivers send) ----
# If any of --Om/--Ok/--w0/--wa is given, use them directly. This is the primary
# path; the --param/--side block is a legacy fallback used only when NO explicit
# cosmology flags are present.
_om = getarg("--Om=", nothing)
_ok = getarg("--Ok=", nothing)
_w0 = getarg("--w0=", nothing)
_wa = getarg("--wa=", nothing)
explicit = any(x -> x !== nothing, (_om, _ok, _w0, _wa))

if explicit
    _om !== nothing && (Om = parse(Float64, _om))
    _ok !== nothing && (Ok = parse(Float64, _ok))
    _w0 !== nothing && (w0 = parse(Float64, _w0))
    _wa !== nothing && (wa = parse(Float64, _wa))
elseif haskey(DELTAS, param) && param in ["Om", "Ok", "w0", "wa"]
    # legacy fallback: shift by internal DELTAS using --param/--side
    sgn = side == "p" ? 1.0 : -1.0
    if param == "Om"
        Om += sgn * DELTAS[param]
    elseif param == "Ok"
        Ok += sgn * DELTAS[param]
    elseif param == "w0"
        w0 += sgn * DELTAS[param]
    elseif param == "wa"
        wa += sgn * DELTAS[param]
    end
end
Ol = 1.0 - Om - Or

# --- 2) optional growth freeze (geometry-only sweep) ------------------------
# With --freeze-growth, D(z) and f(z) come from a SEPARATE (fiducial) cosmology
# while r(z), H(z), Omega_m(z) stay on the shifted one. This severs the growth
# response from the parameter derivative. Defaults make it a no-op at fiducial.
freeze_growth = ("--freeze-growth" in ARGS)
g_om = parse(Float64, getarg("--growth-Om=", string(OM_FID)))
g_ok = parse(Float64, getarg("--growth-Ok=", string(OK_FID)))
g_w0 = parse(Float64, getarg("--growth-w0=", string(W0_FID)))
g_wa = parse(Float64, getarg("--growth-wa=", string(WA_FID)))
g_or = OR_FID
g_ol = 1.0 - g_om - g_or

println("Making PowerFull cosmology table")
println("  outdir = $outdir")
println("  source = ", explicit ? "explicit flags" : "param/side ($param/$side)")
println("  Om     = $Om")
println("  Ok     = $Ok")
println("  Ol     = $Ol")
println("  w0     = $w0")
println("  wa     = $wa")
println("  freeze_growth = $freeze_growth",
        freeze_growth ? "  (growth from Om=$g_om Ok=$g_ok w0=$g_w0 wa=$g_wa)" : "")

b_dummy(z) = 1.0

cosmo = initialize_cosmology(
    Om, Ol, Or, w0, b_dummy;
    wa=wa,
    zmax=10,
    Nz=10000,
    omega_k=Ok,
)

z = collect(range(0.0, 10.0, length=10000))

r   = cosmo["r_of_z"].(z)
H   = cosmo["H_of_z"].(z)
Omz = cosmo["omega_m_of_z"].(z)
a   = 1.0 ./ (1.0 .+ z)

# Growth: by default from the SAME (shifted) cosmology. With --freeze-growth,
# build a separate fiducial-growth cosmology and take f, D from it instead, so
# the geometry (r, H, Omz) stays shifted while growth is held at fiducial.
if freeze_growth
    cosmo_g = initialize_cosmology(
        g_om, g_ol, g_or, g_w0, b_dummy;
        wa=g_wa,
        zmax=10,
        Nz=10000,
        omega_k=g_ok,
    )
    f = cosmo_g["f_of_z"].(z)
    D = cosmo_g["D_of_z"].(z)
else
    f = cosmo["f_of_z"].(z)
    D = cosmo["D_of_z"].(z)
end

const C_OVER_H0 = 2997.92458
H_over_c = H ./ C_OVER_H0

# PowerFull cosmofns.jl expects 7 columns.
table = hcat(r, z, a, H_over_c, Omz, f, D)

outfile = joinpath(outdir, "cosmo_funcr.txt")
writedlm(outfile, table)

println("Wrote: $outfile")
println("Columns: z, r, H_over_c, Omega_m, f, D, a")
