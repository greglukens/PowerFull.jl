#!/usr/bin/env -S julia --project
# =============================================================================
#
# >> PowerFull_interp.jl <<   (fast variant — default in the release)
#
# R-interpolation helper for the build_and_export pipeline.
#
# Provides a vectorized, in-place R-interpolation routine that:
#   (1) Uses Dierckx Spline1D per row but evaluates all target points
#       at once via a vectorized spl(xs) call.  Bit-identical to the
#       reference interpolate_to_physical_grid in PowerFull.jl, ~1.3-1.6x
#       faster.
#   (2) Writes into a caller-provided w_phys matrix so buffers can be
#       pre-allocated and reused across kernels.
#   (3) Contains NO inner @threads; callers are expected to thread at the
#       outer (ell) level.
#
# Used by: src/build_and_export.jl
# =============================================================================

module PowerFullInterp

using Dierckx

export interpolate_to_physical_grid!, interpolate_to_physical_grid_fast

"""
    interpolate_to_physical_grid!(w_phys, w_integrand, rr, RR) -> Nothing

In-place vectorized Dierckx interpolation onto the physical (r₁, r₂) grid.

For each row i (fixed r₁ = rr[i]), build a cubic spline in ln(R) over
w_integrand[i, :] and evaluate at the target points
ln(rr[j]/rr[i]) = ln_rr[j] - ln_rr[i] for all j in one vectorized call.
Points outside [ln_RR[1], ln_RR[end]] are zeroed (boundary condition "zero").

# Arguments
- `w_phys::AbstractMatrix{Float64}`: pre-allocated (nr, nr) output buffer.
  Will be filled with zero before writing — caller need not pre-zero.
- `w_integrand::AbstractMatrix{Float64}`: input (nr, nR) grid in ln(R).
- `rr::Vector{Float64}`: physical r grid, length nr.
- `RR::Vector{Float64}`: dimensionless R grid, length nR.

# Notes
- NO inner @threads. Thread-safe when called from per-task contexts where
  `w_phys` is task-local.
- Bit-identical output to PowerFull.interpolate_to_physical_grid (variant B
  was verified zero-error against variant A in test_convergence benchmarks).
"""
function interpolate_to_physical_grid!(w_phys::AbstractMatrix{Float64},
                                        w_integrand::AbstractMatrix{Float64},
                                        rr::Vector{Float64},
                                        RR::Vector{Float64})::Nothing
    nr = length(rr)
    @assert size(w_phys) == (nr, nr) "w_phys must be (nr, nr)"
    @assert size(w_integrand, 1) == nr "w_integrand row count must match nr"
    @assert size(w_integrand, 2) == length(RR) "w_integrand col count must match nR"

    ln_rr = log.(rr)
    ln_RR = log.(RR)
    ln_R_min = ln_RR[1]
    ln_R_max = ln_RR[end]

    fill!(w_phys, 0.0)

    @inbounds for i in 1:nr
        spl = Spline1D(ln_RR, @view(w_integrand[i, :]), k=3, bc="zero")
        xs = ln_rr .- ln_rr[i]
        vals = spl(xs)
        for j in 1:nr
            xj = xs[j]
            if xj >= ln_R_min && xj <= ln_R_max
                w_phys[i, j] = vals[j]
            end
        end
    end
    return nothing
end

"""
    interpolate_to_physical_grid_fast(w_integrand, rr, RR) -> Matrix{Float64}

Allocating wrapper around `interpolate_to_physical_grid!` for callers that
do not maintain a pre-allocated buffer. Returns a freshly allocated (nr, nr)
matrix.
"""
function interpolate_to_physical_grid_fast(w_integrand::AbstractMatrix{Float64},
                                            rr::Vector{Float64},
                                            RR::Vector{Float64})::Matrix{Float64}
    nr = length(rr)
    w_phys = zeros(Float64, nr, nr)
    interpolate_to_physical_grid!(w_phys, w_integrand, rr, RR)
    return w_phys
end

end # module PowerFullInterp
