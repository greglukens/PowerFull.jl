# =============================================================================
#
# >> runtests.jl <<
#
# Unit tests that do not require the 152 GB production HDF5.  Covered:
#   1. cosmofn loader parses data/cosmo_funcr.txt and yields sensible r(z).
#   2. Tracer constructor builds a z-space sample with normalized phi.
#   3. tracer_to_clgr_params composes the tracer with cosmology into a
#      ClGRParams struct whose function fields evaluate to finite floats.
#   4. load_tracer_h5 round-trips a tiny tracer h5.
#
# End-to-end integration (compute_Cl_GR_batch / compute_Cl_observed) needs
# the full integrals HDF5; run scripts/rna/compute_cl_streaming.slurm for
# that path.
#
# =============================================================================

using Test
using HDF5
using QuadGK

const SRC = joinpath(@__DIR__, "..", "src")

include(joinpath(SRC, "cosmofns.jl"))
using .cosmofns: cosmofn

include(joinpath(SRC, "calcClGR.jl"))
using .CalcClGR

const COSMO_TXT = joinpath(@__DIR__, "..", "data", "cosmo_funcr.txt")

@testset "PowerFull unit tests" begin

    @testset "cosmofn loader" begin
        @test isfile(COSMO_TXT)
        cf = cosmofn(COSMO_TXT)
        # r grid first row has z = 0.001 → r ≈ 3.33e-4 Mpc/h (not exactly 0)
        # Spline extrapolation at z=0 must give something very small.
        r_at_z0 = cf.frz(0.0)
        @test abs(r_at_z0) < 1.0e-2
        # z at the largest tabulated r must be finite and positive.
        r_max = 99551.2860915850
        z_at_rmax = cf.fzr(r_max)
        @test isfinite(z_at_rmax)
        @test z_at_rmax > 0.0
        # Round-trip sanity: r(z(r)) near a typical r.
        r_test = 3000.0
        z_test = cf.fzr(r_test)
        r_back = cf.frz(z_test)
        @test isapprox(r_back, r_test; rtol=1e-4)
    end

    @testset "Tracer construction + phi normalization" begin
        # Gaussian phi centred at z=1, sigma=0.1; renormalize on [0, 5].
        gauss(z) = exp(-((z - 1.0) / 0.1)^2 / 2) / (0.1 * sqrt(2π))
        norm_raw, _ = quadgk(gauss, 0.0, 5.0; rtol=1e-10)
        phi_fn = z -> gauss(z) / norm_raw

        tracer = CalcClGR.Tracer(
            zz -> 1.5,        # bg(z)
            zz -> 0.2,        # be(z)
            zz -> 0.3,        # Q(z)
            phi_fn,           # phi(z)
            nothing,          # bPhi default
            0.0, 5.0,
        )
        @test tracer.bg(1.0) == 1.5
        @test tracer.be(0.5) == 0.2
        @test tracer.Q(2.0)  == 0.3
        @test tracer.bPhi === nothing

        integrated, _ = quadgk(tracer.phi, 0.0, 5.0; rtol=1e-10)
        @test isapprox(integrated, 1.0; atol=1e-6)
    end

    @testset "tracer_to_clgr_params bridge" begin
        cf = cosmofn(COSMO_TXT)
        gauss(z) = exp(-((z - 1.0) / 0.1)^2 / 2) / (0.1 * sqrt(2π))
        norm_raw, _ = quadgk(gauss, 0.0, 5.0; rtol=1e-10)
        phi_fn = z -> gauss(z) / norm_raw
        tracer = CalcClGR.Tracer(
            zz -> 1.5, zz -> 0.2, zz -> 0.3,
            phi_fn, nothing, 0.0, 5.0,
        )

        params = tracer_to_clgr_params(tracer, cf;
                                       fNL=1.0, Omm0=0.30682, H0=67.78)

        @test params isa ClGRParams
        # 8 function fields.
        for fn in (params.D, params.aH, params.bg, params.β,
                   params.B, params.A, params.Q, params.bPhi)
            @test fn isa Function
        end
        # Scalars set correctly.
        @test params.f_NL == 1.0
        @test params.Omm0 == 0.30682
        @test params.H0   == 67.78

        # Evaluate at a sample r.
        r0 = 3000.0
        @test isfinite(params.D(r0))
        @test isfinite(params.aH(r0))
        @test params.bg(r0) == 1.5
        @test params.Q(r0)  == 0.3
        @test isfinite(params.β(r0))
        @test isfinite(params.B(r0))
        @test isfinite(params.A(r0))
        # Default bPhi = 2 · delta_c · (bg - 1), with delta_c = 1.686 and bg = 1.5.
        @test isapprox(params.bPhi(r0), 2 * 1.686 * (1.5 - 1.0); rtol=1e-12)
    end

    @testset "load_tracer_h5 round-trip" begin
        tmpdir = mktempdir()
        tmph5  = joinpath(tmpdir, "tiny_tracer.h5")

        zz = collect(range(0.0, 5.0; length=201))
        # Normalized Gaussian on the discrete z grid.
        gauss(z) = exp(-((z - 1.0) / 0.1)^2 / 2) / (0.1 * sqrt(2π))
        phi_raw = gauss.(zz)
        # Trapezoidal normalization on the uniform grid.
        dz = zz[2] - zz[1]
        norm_tr = (phi_raw[1] + phi_raw[end]) * 0.5 * dz +
                  sum(phi_raw[2:end-1]) * dz
        phi = phi_raw ./ norm_tr

        bg_val = 1.7
        bg_vec = fill(bg_val, length(zz))
        be_vec = fill(0.1,    length(zz))
        Q_vec  = fill(0.4,    length(zz))

        h5open(tmph5, "w") do f
            f["z"]   = zz
            f["bg"]  = bg_vec
            f["be"]  = be_vec
            f["Q"]   = Q_vec
            f["phi"] = phi
        end

        tracer = load_tracer_h5(tmph5)
        @test tracer.zmin == zz[1]
        @test tracer.zmax == zz[end]
        @test isapprox(tracer.bg(1.0), bg_val; atol=1e-10)
        @test isapprox(tracer.be(2.0), 0.1;    atol=1e-10)
        @test isapprox(tracer.Q(3.0),  0.4;    atol=1e-10)
        @test tracer.bPhi === nothing
    end

end
