# =============================================================================
#
# >> cosmofns.jl <<
#
# Cosmological background functions from tabulated data.
# Reads cosmo_funcr.txt and returns spline interpolators for:
#   z(r), a(r), H(r), Ω_m(r), f(r), D(r), and r(z).
#
# Used by: PowerFullTwoFAST.jl, build_and_export.jl, compute_ClGR.jl
#
# =============================================================================

module cosmofns

export cosmofn

using Dierckx
using DelimitedFiles

struct cosmofn
    frz::Spline1D
    fzr::Spline1D
    far::Spline1D
    fHr::Spline1D
    fOmr::Spline1D
    ffr::Spline1D
    fDr::Spline1D
end

function cosmofn(filename::String=joinpath(@__DIR__, "..", "data", "cosmo_funcr.txt"))
    if !isfile(filename)
        error("Cosmology function file not found: $filename")
    end
    dd = readdlm(filename)

    # For frz (z -> r), need to remove duplicate z values for valid spline
    z_vals = dd[:,2]
    r_vals = dd[:,1]
    unique_z_idx = unique(i -> z_vals[i], 1:length(z_vals))

    frz  = Spline1D(z_vals[unique_z_idx], r_vals[unique_z_idx], k=3)
    fzr  = Spline1D(dd[:,1], dd[:,2], k=3)
    far  = Spline1D(dd[:,1], dd[:,3], k=3)
    fHr  = Spline1D(dd[:,1], dd[:,4], k=3)
    fOmr = Spline1D(dd[:,1], dd[:,5], k=3)
    ffr  = Spline1D(dd[:,1], dd[:,6], k=3)
    fDr  = Spline1D(dd[:,1], dd[:,7], k=3)

    return cosmofn(frz, fzr, far, fHr, fOmr, ffr, fDr)
end

@inline function (c::cosmofn)(r::Real)
    return (
        c.fzr(r),   # z(r)
        c.far(r),   # a(r)
        c.fHr(r),   # H(r)
        c.fOmr(r),  # Ω_m(r)
        c.ffr(r),   # f(r)
        c.fDr(r)    # D(r)
    )
end

end # module cosmofns
