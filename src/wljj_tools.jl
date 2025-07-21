using TwoFAST, Dierckx, LinearAlgebra, Base.Threads, Suppressor, Trapz

function determine_scenario(RSD,doppler,GR,lensing,PNG;qvals::Array=[1.1,0.1,-0.1,-0.1,-0.1])
    if qvals != [1.1,0.1,-0.1,-0.1,-0.1]
        println("Warning: Not using default bias parameters, aliasing effects possible!")
    end
    scenario = (RSD,doppler,GR,lensing,PNG)
    if scenario == (true,true,true,true,true)
        println("RSD: on")
        println("Doppler: on")
        println("GR: on")
        println("Lensing: on")
        println("PNG: on")
        pvals = [0,-1,-2,-3,-4]
    elseif scenario == (true,true,true,true,false)
        println("RSD: on")
        println("Doppler: on")
        println("GR: on")
        println("Lensing: on")
        println("PNG: off")
        pvals = [0,-1,-2,-3,-4]
    elseif scenario == (true,true,false,false,true)
        println("RSD: on")
        println("Doppler: on")
        println("GR: off")
        println("Lensing: off")
        println("PNG: on")
        pvals = [0,-1,-2,-3,-4]
    elseif scenario == (true,false,false,false,true)
        println("RSD: on")
        println("Doppler: off")
        println("GR: off")
        println("Lensing: off")
        println("PNG: on")
        pvals = [0]
    elseif scenario == (true,false,false,false,false)
        println("RSD: on")
        println("Doppler: off")
        println("GR: off")
        println("Lensing: off")
        println("PNG: off")
        pvals = [0]
    elseif scenario == (true,true,false,false,false)
        println("RSD: on")
        println("Doppler: on")
        println("GR: off")
        println("Lensing: off")
        println("PNG: off")
        pvals = [0,-2,-4]
    elseif scenario == (true,true,false,false,true)
        println("RSD: on")
        println("Doppler: on")
        println("GR: off")
        println("Lensing: off")
        println("PNG: on")
        pvals = [0,-2,-4]
    elseif scenario == (true,true,false,true,true)
        println("RSD: on")
        println("Doppler: on")
        println("GR: off")
        println("Lensing: on")
        println("PNG: on")
        pvals = [0,-2,-4]
    elseif scenario == (false,false,false,false,false)
        println("RSD: off")
        println("Doppler: off")
        println("GR: off")
        println("Lensing: off")
        println("PNG: off")
        pvals = [0]
    elseif scenario == (false,false,false,true,false)
        println("RSD: off")
        println("Doppler: off")
        println("GR: off")
        println("Lensing: on")
        println("PNG: off")
        pvals = [0,-2,-4]
    elseif scenario == (true,false,false,true,false)
        println("RSD: on")
        println("Doppler: off")
        println("GR: off")
        println("Lensing: on")
        println("PNG: off")
        pvals = [0,-2,-4]
    elseif scenario == (true,false,false,true,true)
        println("RSD: on")
        println("Doppler: off")
        println("GR: off")
        println("Lensing: on")
        println("PNG: on")
        pvals = [0,-2,-4]
    else
        error("Combination of inputs not supported currently...")
    end

    return pvals
end
# necessary functions
function ISW_kernel(r,D_r,H_r,Om_r,z_r,f_r)
    return D_r .* (f_r .- 1) .* Om_r .* H_r .^3 ./ ((1 .+ z_r).^3 ) / 3000^3
end

function time_delay_kernel(r,D_r,H_r,Om_r,z_r)
    return D_r .* Om_r .* H_r .^2 ./ ((1 .+ z_r).^2 ) / 3000^2
end

function lensing_kernel(r,rstar,D_r,H_r,Om_r,z_r)
    return (rstar .- r) ./ (rstar .* r) .* D_r .* Om_r .* H_r .^2 ./ ((1 .+ z_r).^2 ) / 3000^2
end

function bbks(k)
    q = k/(0.6766 * 0.3111) 
    return log(1+2.34*q)/(2.34*q) * (1+3.89*q + (16.1*q)^2 + (5.46*q)^3 + (6.71*q)^4)^(-1/4)
end

function PNG(k)
    M = k^2 * bbks(k)
    return M^(-1)
end

# to run TwoFAST

function calc_Mell(ellmax,Rs,q_in,kmax,kmin,N,cachepath)

    chi0 = 1/kmax
    aell = collect(1:ellmax)

    RR = [Rs;]
    @suppress_out begin
        # calculate M_ll at high ell, result gets saved to a file:
        f21cache = F21EllCache(ellmax, RR, N; q=q_in, kmin=kmin, kmax=kmax, χ0=chi0)
        write("$(cachepath)/F21EllCache", f21cache)

        # calculate all M_ll, result gets saved to a file:
        mlcache = MlCache(aell, "$(cachepath)/F21EllCache", "$(cachepath)/MlCache")
        write("$(cachepath)/MlCache", mlcache)
    end
end


function integration_prep(rr_truncated, RR_truncated, z_of_r, D_of_z, H_of_z, Om_of_z, f_of_z)
    r_R = rr_truncated .* RR_truncated'

    # Background functions evaluated over r and r × R
    z_r = z_of_r.(rr_truncated)
    z_rR = z_of_r.(r_R)

    D_r = D_of_z.(z_r)
    D_rR = D_of_z.(z_rR)

    H_r = H_of_z.(z_r)
    H_rR = H_of_z.(z_rR)

    Om_r = Om_of_z.(z_r)
    Om_rR = Om_of_z.(z_rR)

    f_r = f_of_z.(z_r)
    f_rR = f_of_z.(z_rR)

    # Kernels
    ISW_kernel_rR = ISW_kernel(r_R, D_rR, H_rR, Om_rR, z_rR, f_rR)
    TD_kernel_rR = time_delay_kernel(r_R, D_rR, H_rR, Om_rR, z_rR)

    #l_kernel_R = precompute_l_kernel(RR_truncated)

    DD_rr = D_r .* D_r'

    # Kernel prefactors
    pre_r_factor_lens = @. D_r * Om_r * H_r^2 / (1 + z_r)^2 / 3000^2
    pre_r_factor_td   = pre_r_factor_lens
    pre_r_factor_isw  = @. D_r * Om_r * H_r^3 * (f_r - 1) / (1 + z_r)^3 / 3000^3

    pre_rR_factor_lens = @. D_rR * Om_rR * H_rR^2 / (1 + z_rR)^2 / 3000^2
    pre_rR_factor_td   = pre_rR_factor_lens
    pre_rR_factor_isw  = @. D_rR * Om_rR * H_rR^3 * (f_rR - 1) / (1 + z_rR)^3 / 3000^3

    # Trapezoid integration weights
    log_rr_truncated = log.(rr_truncated)
    log_RR_truncated = log.(RR_truncated)
    trapz_weights_log_RR = compute_trapz_weights(log_RR_truncated)
    trapz_weights_log_rr = compute_trapz_weights(log_rr_truncated)
    trapz_weights_RR     = compute_trapz_weights(RR_truncated)

    return (
    r_R = r_R, z_r=z_r, D_r=D_r, H_r=H_r, Om_r=Om_r, f_r=f_r,
    z_rR=z_rR, D_rR=D_rR, H_rR=H_rR, Om_rR=Om_rR, f_rR=f_rR,
    ISW_kernel_rR=ISW_kernel_rR,
    TD_kernel_rR=TD_kernel_rR,
    pre_r_factor_lens=pre_r_factor_lens,
    pre_r_factor_td=pre_r_factor_td,
    pre_r_factor_isw=pre_r_factor_isw,
    pre_rR_factor_lens=pre_rR_factor_lens,
    pre_rR_factor_td=pre_rR_factor_td,
    pre_rR_factor_isw=pre_rR_factor_isw,
    trapz_weights_log_rr=trapz_weights_log_rr,
    trapz_weights_log_RR=trapz_weights_log_RR,
    trapz_weights_RR=trapz_weights_RR,
    DD_rr = DD_rr
    )
end


# Constructs for necessary outputs of full relativistic C_ell

function real_term(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids=nothing, R_calc_ids=nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    w00 = Array{Float64}(undef, N, length(RR))


    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    w00rx = @views w00[r_calc_ids, R_calc_ids]

    return w00rx
end

function RSD_terms(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    

    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    w00 = Array{Float64}(undef, N, length(RR))
    w02 = Array{Float64}(undef, N, length(RR))
    w20 = Array{Float64}(undef, N, length(RR))
    w22 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
            @views w02[:, :] = wjj[2]
            @views w20[:, :] = wjj[3]
            @views w22[:, :] = wjj[4]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    Rmin = minimum(RR)
    #rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    #r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    #Nr_new = Int(length(rr[1:r_slicing:end]))    

    rr_truncated = @views rr[rr_min_id:rr_max_id]

    #w00rx = @views w00[r_calc_ids, R_calc_ids]
    #w02rx = @views w02[r_calc_ids, R_calc_ids]
    #w20rx = @views w20[r_calc_ids, R_calc_ids]
    #w22rx = @views w22[r_calc_ids, R_calc_ids]

    @views w00_truncated = w00[rr_min_id:rr_max_id, 1:RR_max_id]
    @views w02_truncated = w02[rr_min_id:rr_max_id, 1:RR_max_id]
    @views w20_truncated = w20[rr_min_id:rr_max_id, 1:RR_max_id]
    @views w22_truncated = w22[rr_min_id:rr_max_id, 1:RR_max_id]


    w00_gridded = cl_to_grid(w00_truncated,rr_truncated,RR_truncated)
    w20_gridded = cl_to_grid(w20_truncated,rr_truncated,RR_truncated)
    w02_gridded = cl_to_grid(w02_truncated,rr_truncated,RR_truncated)
    w22_gridded = cl_to_grid(w22_truncated,rr_truncated,RR_truncated)

    w00_gridded = @view w00_gridded[r_calc_ids,r_calc_ids]
    w20_gridded = @view w20_gridded[r_calc_ids,r_calc_ids]
    w02_gridded = @view w02_gridded[r_calc_ids,r_calc_ids]
    w22_gridded = @view w22_gridded[r_calc_ids,r_calc_ids]


    return w00_gridded, w02_gridded, w20_gridded, w22_gridded
end









function lst_02_integrals(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))
    w02 = Array{Float64}(undef, N, length(RR))
    w20 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
            @views w02[:, :] = wjj[2]
            @views w20[:, :] = wjj[3]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    #Nr_new = Int(length(rr[1:r_slicing:end]))    

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]

    

    @views w00_truncated = w00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    @views w02_truncated = w02[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    @views w20_truncated = w20[rr_trunc_low_id:rr_max_id, 1:RR_max_id]

    w10p1_truncated = similar(w00_truncated)
    w12p1_truncated = similar(w00_truncated)
    w01p1_truncated = similar(w00_truncated)
    w21p1_truncated = similar(w00_truncated)

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w00_truncated[rindx,:])
        fthis2 = Spline1D(RR_truncated * rr_truncated[rindx], w20_truncated[rindx,:])
        @views w01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
        @views w21p1_truncated[rindx,:] = derivative(fthis2, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, w00_truncated[:,Rindx])
        fthis2 = Spline1D(rr_truncated, w02_truncated[:,Rindx])
        @views w10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * w01p1_truncated[:,Rindx]
        @views w12p1_truncated[:,Rindx] = derivative(fthis2, rr_truncated, nu=1) .- RR_truncated[Rindx] * w21p1_truncated[:,Rindx]
    end

    #@views w00rx = w00_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w20rx = w20_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w02rx = w02_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w10p1rx = w10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w01p1rx = w01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w12p1rx = w12p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w21p1rx = w21p1_truncated[r_calc_ids_trunc,R_calc_ids]

    l00_truncated = similar(w00_truncated)
    s00_truncated = similar(w00_truncated)
    t00_truncated = similar(w00_truncated)

    l02_truncated = similar(w00_truncated)
    s02_truncated = similar(w00_truncated)
    t02_truncated = similar(w00_truncated)

    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw

    @views rtest = rr_truncated[r_calc_ids_trunc]


    w_row_lst = int_prep.trapz_weights_log_RR

    l_kernel_precomp = precompute_l_kernel_r(RR_truncated, R_calc_ids) 
    
    w00_weighted_l = w_row_lst' .* w00_truncated .* pre_rR_factor_lens
    w00_weighted_s = w_row_lst' .* w00_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated
    w00_weighted_t = w_row_lst' .* w00_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated
    
    w20_weighted_l = w_row_lst' .* w20_truncated .* pre_rR_factor_lens
    w20_weighted_s = w_row_lst' .* w20_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated
    w20_weighted_t = w_row_lst' .* w20_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated

    compute_lensing_cross_integral_r!(r_calc_ids_trunc,R_calc_ids,l_kernel_precomp,w00_weighted_l,l00_truncated)
    compute_lensing_cross_integral_r!(r_calc_ids_trunc,R_calc_ids,l_kernel_precomp,w20_weighted_l,l02_truncated)

    compute_st_cross_integrals_r!(r_calc_ids_trunc,R_calc_ids,w00_weighted_s,w00_weighted_t,s00_truncated,t00_truncated)
    compute_st_cross_integrals_r!(r_calc_ids_trunc,R_calc_ids,w20_weighted_s,w20_weighted_t,s02_truncated,t02_truncated)


    w00_gridded = cl_to_grid(w00_truncated,rr_truncated,RR_truncated)
    w02_gridded = cl_to_grid(w02_truncated,rr_truncated,RR_truncated)
    w20_gridded = cl_to_grid(w20_truncated,rr_truncated,RR_truncated)

    w01p1_gridded = cl_to_grid(w01p1_truncated,rr_truncated,RR_truncated)
    w10p1_gridded = cl_to_grid(w10p1_truncated,rr_truncated,RR_truncated)
    w12p1_gridded = cl_to_grid(w12p1_truncated,rr_truncated,RR_truncated)
    w21p1_gridded = cl_to_grid(w21p1_truncated,rr_truncated,RR_truncated)

    l00_gridded = cl_to_grid(l00_truncated,rr_truncated,RR_truncated)
    s00_gridded = cl_to_grid(s00_truncated,rr_truncated,RR_truncated)
    t00_gridded = cl_to_grid(t00_truncated,rr_truncated,RR_truncated)

    l02_gridded = cl_to_grid(l02_truncated,rr_truncated,RR_truncated)
    s02_gridded = cl_to_grid(s02_truncated,rr_truncated,RR_truncated)
    t02_gridded = cl_to_grid(t02_truncated,rr_truncated,RR_truncated)

    w00_gridded = @view w00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w02_gridded = @view w02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w20_gridded = @view w20_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    w01p1_gridded = @view w01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w10p1_gridded = @view w10p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w12p1_gridded = @view w12p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w21p1_gridded = @view w21p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    l00_gridded = @view l00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    s00_gridded = @view s00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    t00_gridded = @view t00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    l02_gridded = @view l02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    s02_gridded = @view s02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    t02_gridded = @view t02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    return w00_gridded, w02_gridded, w20_gridded, w01p1_gridded, w10p1_gridded, w12p1_gridded, w21p1_gridded, l00_gridded, s00_gridded, t00_gridded, l02_gridded, s02_gridded, t02_gridded
end


function lst_LST_XYZ_integrals(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    @views w00_truncated = w00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    w10p1_truncated = similar(w00_truncated)
    w01p1_truncated = similar(w00_truncated)
    w11p2_truncated = similar(w00_truncated)

    if ell_i <= 100
        Rmin_choice = max(0.5 - (ell_i-1) * 0.01,0.425)
        R_issue_min = Rmin_choice
        R_issue_max = 0.53

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min,R_issue_max)

        R_issue_min2 = 0.69
        R_issue_max2 = 0.715

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min2,R_issue_max2)

        R_issue_min3 = 1/0.705
        R_issue_max3 = 1/0.67

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min3,R_issue_max3)

        R_issue_min4 = 1/0.508
        R_issue_max4 = 1/0.5

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min4,R_issue_max4)

        deltaR_choice = min(0.3,0.025*ell_i)
        R_issue_min5 = 1/0.491
        R_issue_max5 = 1/0.491 + deltaR_choice

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min5,R_issue_max5)

    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w00_truncated[rindx,:])
        @views w01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, w00_truncated[:,Rindx])
        @views w10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * w01p1_truncated[:,Rindx]
    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w10p1_truncated[rindx,:])
        @views w11p2_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    #@views w00rx = w00_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w10p1rx = w10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w01p1rx = w01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w11p2rx = w11p2_truncated[r_calc_ids_trunc,R_calc_ids]


    l00_truncated = similar(w00_truncated)
    s00_truncated = similar(w00_truncated)
    t00_truncated = similar(w00_truncated)
    l01p1_truncated = similar(w00_truncated)
    s01p1_truncated = similar(w00_truncated)
    t01p1_truncated = similar(w00_truncated)

    r_R = int_prep.r_R
    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw
    pre_rr_factor_lens  = int_prep.pre_r_factor_lens
    pre_rr_factor_isw  = int_prep.pre_r_factor_isw
    pre_rr_factor_td  = int_prep.pre_r_factor_td

    #@views rtest = rr_truncated[r_calc_ids_trunc]

    w_row_lst = int_prep.trapz_weights_log_RR
    w_row_rr = int_prep.trapz_weights_log_rr

    l_kernel_precomp = precompute_l_kernel_r(RR_truncated, R_calc_ids) 
    
    w00_weighted_l = w_row_lst' .* w00_truncated .* pre_rR_factor_lens
    w00_weighted_s = w_row_lst' .* w00_truncated .* pre_rR_factor_isw .* r_R
    w00_weighted_t = w_row_lst' .* w00_truncated .* pre_rR_factor_td .* r_R

    w01p1_weighted_l = w_row_lst' .* w01p1_truncated .* pre_rR_factor_lens
    w01p1_weighted_s = w_row_lst' .* w01p1_truncated .* pre_rR_factor_isw .* r_R
    w01p1_weighted_t = w_row_lst' .* w01p1_truncated .* pre_rR_factor_td .* r_R

    compute_lensing_cross_integral_r!(collect(1:length(rr_truncated)),R_calc_ids,l_kernel_precomp,w00_weighted_l,l00_truncated)
    compute_lensing_cross_integral_r!(collect(1:length(rr_truncated)),R_calc_ids,l_kernel_precomp,w01p1_weighted_l,l01p1_truncated)

    compute_st_cross_integrals_r!(collect(1:length(rr_truncated)),R_calc_ids,w00_weighted_s,w00_weighted_t,s00_truncated,t00_truncated)
    compute_st_cross_integrals_r!(collect(1:length(rr_truncated)),R_calc_ids,w01p1_weighted_s,w01p1_weighted_t,s01p1_truncated,t01p1_truncated)

    w00_gridded = cl_to_grid(w00_truncated, rr_truncated, RR)
    w01p1_gridded = cl_to_grid(w01p1_truncated, rr_truncated, RR)
    w10p1_gridded = cl_to_grid(w10p1_truncated, rr_truncated, RR)
    w11p2_gridded = cl_to_grid(w11p2_truncated, rr_truncated, RR)
    w00_gridded = cl_to_grid(w00_truncated, rr_truncated, RR)

    l00_gridded = cl_to_grid(l00_truncated, rr_truncated, RR)
    s00_gridded = cl_to_grid(s00_truncated, rr_truncated, RR)
    t00_gridded = cl_to_grid(t00_truncated, rr_truncated, RR)
    l01p1_gridded = cl_to_grid(l01p1_truncated, rr_truncated, RR)
    s01p1_gridded = cl_to_grid(s01p1_truncated, rr_truncated, RR)
    t01p1_gridded = cl_to_grid(t01p1_truncated, rr_truncated, RR)

    S00_gridded = zeros(length(rr_truncated),length(rr_truncated))
    L00_gridded = similar(S00_gridded)
    T00_gridded = similar(S00_gridded)
    X00_gridded = similar(S00_gridded)
    Y00_gridded = similar(S00_gridded)
    Z00_gridded = similar(S00_gridded)
    
    L00_pre_sum = w_row_rr .* l00_gridded .* pre_rr_factor_lens
    S00_pre_sum = w_row_rr .* s00_gridded .* pre_rr_factor_isw .* rr_truncated
    T00_pre_sum = w_row_rr .* t00_gridded .* pre_rr_factor_td .* rr_truncated
    X00_pre_sum = w_row_rr .* s00_gridded .* pre_rr_factor_td .* rr_truncated
    Y00_pre_sum = w_row_rr .* l00_gridded .* pre_rr_factor_isw .* rr_truncated 
    Z00_pre_sum = w_row_rr .* l00_gridded .* pre_rr_factor_td .* rr_truncated

    L00_gridded = cumsum(L00_pre_sum, dims=1) .- (cumsum(rr_truncated .* L00_pre_sum, dims = 1) ./ rr_truncated)
    S00_gridded = cumsum(S00_pre_sum, dims=1)
    T00_gridded = cumsum(T00_pre_sum, dims=1)
    X00_gridded = cumsum(X00_pre_sum, dims=1)
    Y00_gridded = cumsum(Y00_pre_sum, dims=1)
    Z00_gridded = cumsum(Z00_pre_sum, dims=1)
    
    #X00rx_t = similar(S00rx)
    #Y00rx_t = similar(S00rx)
    #Z00rx_t = similar(S00rx)
    #X00_pre_sum_t = w_row_rr .* t00_gridded .* pre_rr_factor_isw .* rr_truncated
    #Y00_pre_sum_t = w_row_rr .* s00_gridded .* pre_rr_factor_lens
    #Z00_pre_sum_t = w_row_rr .* t00_gridded .* pre_rr_factor_lens
    #X00rx_t = cumsum(X00_pre_sum_t, dims=1)
    #Y00rx_t = cumsum(Y00_pre_sum_t, dims=1) .- (cumsum(rr_truncated .* Y00_pre_sum_t, dims = 1) ./ rr_truncated)
    #Z00rx_t = cumsum(Z00_pre_sum_t, dims=1) .- (cumsum(rr_truncated .* Z00_pre_sum_t, dims = 1) ./ rr_truncated)

    w00_gridded = @view w00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w01p1_gridded = @view w01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w10p1_gridded = @view w10p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w11p2_gridded = @view w11p2_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    l00_gridded = @view l00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    s00_gridded = @view s00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    t00_gridded = @view t00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    l01p1_gridded = @view l01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    s01p1_gridded = @view s01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    t01p1_gridded = @view t01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    L00_gridded = @view L00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    S00_gridded = @view S00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    T00_gridded = @view T00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    X00_gridded = @view X00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    Y00_gridded = @view Y00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    Z00_gridded = @view Z00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    return w00_gridded, w01p1_gridded, w10p1_gridded, w11p2_gridded, l00_gridded, s00_gridded, t00_gridded, l01p1_gridded, s01p1_gridded, t01p1_gridded, L00_gridded, S00_gridded, T00_gridded, X00_gridded, Y00_gridded, Z00_gridded   #X00rx_t, Y00rx_t, Z00rx_t
end







function fnl_02_terms(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    u00 = Array{Float64}(undef, N, length(RR))
    u02 = Array{Float64}(undef, N, length(RR))
    u20 = Array{Float64}(undef, N, length(RR))


    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views u00[:, :] = wjj[1]
            @views u02[:, :] = wjj[2]
            @views u20[:, :] = wjj[3]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    #Nr_new = Int(length(rr[1:r_slicing:end]))    

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]

    #w00rx = @views w00[r_calc_ids, R_calc_ids]
    #w02rx = @views w02[r_calc_ids, R_calc_ids]
    #w20rx = @views w20[r_calc_ids, R_calc_ids]
    #w22rx = @views w22[r_calc_ids, R_calc_ids]

    @views u00_truncated = u00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    @views u02_truncated = u02[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    @views u20_truncated = u20[rr_trunc_low_id:rr_max_id, 1:RR_max_id]


    u00_gridded = cl_to_grid(u00_truncated,rr_truncated,RR_truncated)
    u20_gridded = cl_to_grid(u20_truncated,rr_truncated,RR_truncated)
    u02_gridded = cl_to_grid(u02_truncated,rr_truncated,RR_truncated)
    

    u00_gridded = @view u00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    u20_gridded = @view u20_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    u02_gridded = @view u02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
   

    return u00_gridded, u02_gridded, u20_gridded
end

function fnl_01_integrals(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    u00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views u00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    @views u00_truncated = u00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    u10p1_truncated = similar(u00_truncated)
    u01p1_truncated = similar(u00_truncated)
   

    if ell_i <= 30
        R_issue_min = 0.5
        R_issue_max = 0.51

        smooth_over_alias_features!(u00_truncated,RR_truncated,R_issue_min,R_issue_max)
    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], u00_truncated[rindx,:])
        @views u01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, u00_truncated[:,Rindx])
        @views u10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * u01p1_truncated[:,Rindx]
    end


    #@views u00rx = u00_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views u10p1rx = u10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views u01p1rx = u01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    

    lfnl00_truncated = similar(u00_truncated)
    sfnl00_truncated = similar(u00_truncated)
    tfnl00_truncated = similar(u00_truncated)
    
    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw

    @views rtest = rr_truncated[r_calc_ids_trunc]

    w_row_lst = int_prep.trapz_weights_log_RR

    l_kernel_precomp = precompute_l_kernel_r(RR_truncated, R_calc_ids) 
    
    u00_weighted_l = w_row_lst' .* u00_truncated .* pre_rR_factor_lens
    u00_weighted_s = w_row_lst' .* u00_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated
    u00_weighted_t = w_row_lst' .* u00_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated

    compute_lensing_cross_integral_r!(r_calc_ids_trunc,R_calc_ids,l_kernel_precomp,u00_weighted_l,lfnl00_truncated)
    compute_st_cross_integrals_r!(r_calc_ids_trunc,R_calc_ids,u00_weighted_s,u00_weighted_t,sfnl00_truncated,tfnl00_truncated)
    

    u00_gridded = cl_to_grid(u00_truncated,rr_truncated,RR_truncated)
    u10p1_gridded = cl_to_grid(u10p1_truncated,rr_truncated,RR_truncated)
    u01p1_gridded = cl_to_grid(u01p1_truncated,rr_truncated,RR_truncated)

    lfnl00_gridded = cl_to_grid(lfnl00_truncated,rr_truncated,RR_truncated)
    sfnl00_gridded = cl_to_grid(sfnl00_truncated,rr_truncated,RR_truncated)
    tfnl00_gridded = cl_to_grid(tfnl00_truncated,rr_truncated,RR_truncated)

    u00_gridded = @view u00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    u10p1_gridded = @view u10p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    u01p1_gridded = @view u01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    lfnl00_gridded = @view lfnl00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    sfnl00_gridded = @view sfnl00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    tfnl00_gridded = @view tfnl00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    return u00_gridded, u01p1_gridded, u10p1_gridded, lfnl00_gridded, sfnl00_gridded, tfnl00_gridded
end

function fnl_auto_term(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    v00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views v00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    if ell_i <= 30
        R_issue_min = 0.5
        R_issue_max = 0.51

        smooth_over_alias_features!(v00,RR,R_issue_min,R_issue_max)
    end

    @views v00_truncated = v00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]

    v00_gridded = cl_to_grid(v00_truncated,rr_truncated,RR_truncated)
    v00_gridded = @view v00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    return v00_gridded
end






@inline function trapz_fast(x::AbstractVector, y::AbstractVector)
    acc = zero(eltype(x))
    @inbounds @simd for i in 1:length(x)-1
        dx = x[i+1] - x[i]
        acc += dx * (y[i] + y[i+1]) * 0.5
    end
    return acc
end



function compute_lensing_cross_integral!(
    r_calc_ids, R_calc_ids, l_kernel, w_weighted_l, lrx_R)
    N_r = length(r_calc_ids)
    N_R = length(R_calc_ids)


    Threads.@threads for idx in 1:(N_r * N_R)
        rindx = fld(idx - 1, N_R) + 1
        Rindx = mod(idx - 1, N_R) + 1

        rindx_adj = r_calc_ids[rindx]
        R_calc_indx = R_calc_ids[Rindx]

        w_l = @views w_weighted_l[rindx_adj, 1:R_calc_indx] .* l_kernel[Rindx,1:R_calc_indx]

        # Kernel-weighted integrals
        lrx_R[rindx, Rindx]  = sum(w_l)
    end
end



function compute_st_cross_integrals!(
    r_calc_ids, R_calc_ids, w_weighted_s, w_weighted_t, srx_R, trx_R)
    N_r = length(r_calc_ids)
    N_R = length(R_calc_ids)

    Threads.@threads for idx in 1:(N_r * N_R)
        rindx = fld(idx - 1, N_R) + 1
        Rindx = mod(idx - 1, N_R) + 1

        rindx_adj = r_calc_ids[rindx]
        R_calc_indx = R_calc_ids[Rindx]

        w_s = @views w_weighted_s[rindx_adj, 1:R_calc_indx] 
        w_t = @views w_weighted_t[rindx_adj, 1:R_calc_indx]

        # Kernel-weighted integrals
        srx_R[rindx, Rindx]   = sum(w_s)
        trx_R[rindx, Rindx]   = sum(w_t)
    end
end



function compute_lensing_cross_integral_r!(
    r_calc_ids, R_calc_ids, l_kernel, w_weighted_l, lrx_r)
    N_r = length(r_calc_ids)
    N_R = size(w_weighted_l,2)
    N_R_calc = length(R_calc_ids)
    
    Threads.@threads for idx in 1:(N_r * N_R_calc)
        rindx = fld(idx - 1, N_R_calc) + 1
        Rindx = mod(idx - 1, N_R_calc) + 1

        rindx_adj = r_calc_ids[rindx]
        R_calc_indx = N_R - R_calc_ids[Rindx] + 1
        iRindx = N_R_calc - Rindx + 1

        acc = 0.0
        @inbounds @simd for k in 1:R_calc_indx
            acc += @views w_weighted_l[rindx_adj, k] * l_kernel[Rindx, k]
        end
        lrx_r[rindx, iRindx] = acc
    end
end



function compute_st_cross_integrals_r!(r_calc_ids, R_calc_ids, w_weighted_s, w_weighted_t, srx_r, trx_r)
    N_r = length(r_calc_ids)
    N_R = size(w_weighted_s,2)
    N_R_calc = length(R_calc_ids)


    Threads.@threads for idx in 1:(N_r * N_R_calc)
        rindx = fld(idx - 1, N_R_calc) + 1
        Rindx = mod(idx - 1, N_R_calc) + 1

        rindx_adj = r_calc_ids[rindx]
        R_calc_indx = N_R - R_calc_ids[Rindx] + 1
        iRindx = N_R_calc - Rindx + 1

        w_s = @view w_weighted_s[rindx_adj, 1:R_calc_indx] 
        w_t = @view w_weighted_t[rindx_adj, 1:R_calc_indx]

        # Kernel-weighted integrals
        srx_r[rindx, iRindx]   = sum_fast!(w_s)
        trx_r[rindx, iRindx]   = sum_fast!(w_t)
    end
end

function mul_fast!(out, a, b)
    @inbounds @simd for i in 1:length(out)
        out[i] = a[i] * b[i]
    end
end


function sum_fast!(x)
    s = 0.0
    @inbounds @simd for i in 1:length(x)
        s += x[i]
    end
    return s
end

function sum_pairwise_simd(x::AbstractVector{Float64}; chunk_size::Int = 1024)
    N = length(x)
    s_chunks = zeros(Float64, cld(N, chunk_size))  # preallocate partial sums

    @inbounds Threads.@threads for c in 1:length(s_chunks)
        start = (c - 1) * chunk_size + 1
        stop  = min(c * chunk_size, N)
        s = 0.0
        @simd for i in start:stop
            s += x[i]
        end
        s_chunks[c] = s
    end

    # Final sum using pairwise to combine partials
    return pairwise_sum(s_chunks)
end

# Simple recursive pairwise summation
function pairwise_sum(x::Vector{Float64})
    N = length(x)
    if N == 0
        return 0.0
    elseif N == 1
        return x[1]
    elseif N == 2
        return x[1] + x[2]
    else
        mid = N ÷ 2
        return pairwise_sum(view(x, 1:mid)) + pairwise_sum(view(x, mid+1:N))
    end
end



function lensing_kernel!(out, r, rmax, D, H, Om, z)
    @inbounds @simd for i in eachindex(r)
        out[i] = (rmax - r[i])/(rmax * r[i]) / 3000^2 * D[i] * H[i]^2 / (1+z[i])^2 * Om[i]
    end
end



@inline function compute_lensing_kernel_fast(r, r_star, Rr_star, Rrprime)
    numer = (r_star - r) * (Rr_star - Rrprime)
    denom = r * r_star * Rr_star * Rrprime
    return numer / denom
end



@inline function compute_YZ_kernel_fast(r, r_star)
    numer = (r_star - r)
    denom = r * r_star
    return numer / denom
end

function compute_trapz_weights(x::AbstractVector)
    w = zeros(eltype(x), length(x))
    w[1] = (x[2] - x[1]) / 2
    for i in 2:length(x)-1
        w[i] = (x[i+1] - x[i-1]) / 2
    end
    w[end] = (x[end] - x[end-1]) / 2
    return w
end


function precompute_YZ_kernel(rr_truncated, rtest, r_calc_ids)
    N_r = length(r_calc_ids)
    N_rr = length(rr_truncated)
    YZ_kernel_precomp = Array{Float64}(undef, N_r, N_rr)

    Threads.@threads for rindx in eachindex(r_calc_ids)
        r_star = rtest[rindx]

        @inbounds for i in 1:N_rr
            r = rr_truncated[i]
            YZ_kernel_precomp[rindx, i] = (r_star - r) / (r_star * r)
        end
    end

    return YZ_kernel_precomp
end


function precompute_l_kernel(RR_truncated, R_calc_ids)
    N_RR = length(RR_truncated)
    N_R = length(R_calc_ids)
    l_kernel_precomp = Array{Float64}(undef, N_R, N_RR)

    Threads.@threads for Rindx in eachindex(R_calc_ids)
        R_star = RR_truncated[Rindx]

        @inbounds for i in 1:N_RR
            R = RR_truncated[i]
            l_kernel_precomp[Rindx, i] = (R_star - R) / (R_star)
        end
    end

    return l_kernel_precomp
end


function precompute_l_kernel_r(RR_truncated, R_calc_ids)
    N_RR = length(RR_truncated)
    N_R = length(R_calc_ids)
    
    l_kernel_precomp = Array{Float64}(undef, N_R, N_RR)

    Threads.@threads for Rindx in eachindex(R_calc_ids)
        R_star = RR_truncated[Rindx]

        @inbounds for i in 1:N_RR
            R = RR_truncated[i]
            l_kernel_precomp[Rindx, i] = (1 - R_star * R)
        end
    end

    return l_kernel_precomp
end


function precompute_L_kernel_r(rr_truncated, rtest, r_calc_ids)
    N_r = length(r_calc_ids)
    N_rr = length(rr_truncated)
    L_kernel_precomp_r = Array{Float64}(undef, N_r, N_rr)

    Threads.@threads for rindx in eachindex(r_calc_ids)
        r_idx = r_calc_ids[rindx]
        r_star = rtest[rindx]

        @inbounds for i in 1:N_rr
            r = rr_truncated[i]
            # compute_YZ_kernel_fast is (r_star - r) / (r_star * r)
            L_kernel_precomp_r[rindx, i] = (r_star - r) / (r_star * r)
        end
    end

    return L_kernel_precomp_r
end

#L_kernel_precomp_r = precompute_L_kernel_r(rr_truncated,rtest,r_calc_ids)





function compute_prefix_sum_2d(A::Matrix{T}) where T
    P = copy(A)
    m, n = size(P)

    # Cumulative sum over rows
    for j in 1:n
        for i in 2:m
            @inbounds P[i,j] += P[i-1,j]
        end
    end

    # Cumulative sum over columns
    for i in 1:m
        for j in 2:n
            @inbounds P[i,j] += P[i,j-1]
        end
    end

    return P
end


function compute_ST_integrals!(
    S00rx_lt, T00rx_lt,
    P_S, P_T,
    r_calc_ids::Vector{Int}, R_calc_ids::Vector{Int}
)
    Threads.@threads for rindx in eachindex(r_calc_ids)
        r_idx = r_calc_ids[rindx]

        for Rindx in eachindex(R_calc_ids)
            R_idx = R_calc_ids[Rindx]

            @inbounds S00rx_lt[rindx, Rindx] = P_S[r_idx, R_idx]
            @inbounds T00rx_lt[rindx, Rindx] = P_T[r_idx, R_idx]
        end
    end
end


function compute_YZ_integrals!(Y00rx_lt, Z00rx_lt, w00_weighted_Y, w00_weighted_Z,
    YZ_kernel_precomp, r_calc_ids, R_calc_ids
)
    Threads.@threads for rindx in eachindex(r_calc_ids)
        r_idx = r_calc_ids[rindx]
        @views YZ_kernel = YZ_kernel_precomp[rindx, 1:r_idx]

        for Rindx in eachindex(R_calc_ids)
            R_idx = R_calc_ids[Rindx]

            # views to avoid allocations
            @views wY_col = w00_weighted_Y[1:r_idx, R_idx]
            @views wZ_col = w00_weighted_Z[1:r_idx, R_idx]

            acc_Y = 0.0
            acc_Z = 0.0

            @inbounds @simd for i in 1:r_idx
                acc_Y += wY_col[i] * YZ_kernel[i]
                acc_Z += wZ_col[i] * YZ_kernel[i]
            end

            Y00rx_lt[rindx, Rindx] = acc_Y
            Z00rx_lt[rindx, Rindx] = acc_Z
        end
    end
end


function compute_L_integral!(
    L00rx_lt,
    rtest, rr_truncated,
    w00_weighted_L, rR, Rr_star_precomp, 
    r_calc_ids::Vector{Int}, R_calc_ids::Vector{Int}
)
    @inbounds Threads.@threads for rindx in eachindex(r_calc_ids)
        cutoff_idx = r_calc_ids[rindx]
        r_star = rtest[rindx]

        @views rvals = rr_truncated[1:cutoff_idx]

        @inbounds for Rindx in eachindex(R_calc_ids)
            R_idx = R_calc_ids[Rindx]
            Rr_star = Rr_star_precomp[rindx, Rindx]

            @views Rr_block = rR[1:cutoff_idx, 1:R_idx]
            @views wL_block = w00_weighted_L[1:cutoff_idx, 1:R_idx]

            acc_L = 0.0
            @inbounds for j in 1:R_idx
                inner = 0.0
                @simd for i in 1:cutoff_idx
                    krn = compute_lensing_kernel_fast(rvals[i], r_star, Rr_star, Rr_block[i, j])
                    inner += krn * wL_block[i, j]
                end
                acc_L += inner
            end

            L00rx_lt[rindx, Rindx] = acc_L
        end
    end
end

function lensing_kernel!(out, r, rmax, D, H, Om, z)
    @inbounds @simd for i in eachindex(r)
        out[i] = (rmax - r[i])/(rmax * r[i])
    end
end

@inline function lensing_kernel_R(r, r_star)
    numer = (r_star - r)
    denom = r * r_star
    return numer / denom
end



@inline function lensing_diag_test(r, r_star, Rrprime, Rstar)
    numer = (r_star - r) * (Rstar * r_star - Rrprime)
    denom = r * r_star^2 * Rrprime * Rstar
    return numer / denom
end



function compute_L_auto_test!(
    L00diag,
    rtest, rr_truncated,
    w00_neg4_weighted, rR, 
    r_calc_ids::Vector{Int}, R_diag_id::Int
)

    @inbounds Threads.@threads for rindx in eachindex(r_calc_ids)
        cutoff_idx = r_calc_ids[rindx]
        r_star = rtest[rindx]

        @views rvals = rr_truncated[1:cutoff_idx]


        @views Rr_block = rR[1:cutoff_idx, 1:R_diag_id]
        @views wL_block = w00_neg4_weighted[1:cutoff_idx, 1:R_diag_id]
        acc_L = 0.0
        @inbounds for j in 1:R_diag_id
            inner = 0.0
            @simd for i in 1:cutoff_idx
                krn = lensing_diag_test(rvals[i], r_star, Rr_block[i, j])
                inner += krn * wL_block[i,j] 
            end
            
            acc_L += inner
        end

        L00diag[rindx] = acc_L
    end
end

function compute_L_auto_manual!(
    L00diag,
    rtest, rr_truncated,
    w00_neg4_weighted, rR, 
    r_calc_ids::Vector{Int}, R_diag_id::Int
)

    @inbounds Threads.@threads for rindx in eachindex(r_calc_ids)
        cutoff_idx = r_calc_ids[rindx]
        r_star = rtest[rindx]

        @views rvals = rr_truncated[1:cutoff_idx]


        @views Rr_block = rR[1:cutoff_idx, 1:R_diag_id]
        @views wL_block = w00_neg4_weighted[1:cutoff_idx, 1:R_diag_id]
        acc_L = 0.0
        @inbounds for j in 1:R_diag_id
            inner = 0.0
            @simd for i in 1:cutoff_idx
                krn = lensing_diag_test(rvals[i], r_star, Rr_block[i, j])
                inner += krn * wL_block[i,j] 
            end
            
            acc_L += inner
        end

        L00diag[rindx] = acc_L
    end
end



function lensing_test(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:,:] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    rr_max_id = maximum(r_calc_ids)
    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[1:rr_max_id]


    @views w00_truncated = w00[1:rr_max_id, 1:RR_max_id]
    @views w00rx = w00_truncated[r_calc_ids,R_calc_ids]

    RR_truncated_lower = RR_truncated[RR_truncated .<= 1]
    R_calc_ids_lower = R_calc_ids[RR_truncated[R_calc_ids] .<= 1]
    diag_id = maximum(R_calc_ids_lower)


    r_R = int_prep.r_R
    # --- Prefactors for 1D integrals over r ---
    pre_r_factor_lens = int_prep.pre_r_factor_lens

    # --- Prefactors for 2D integrals over r × R ---
    pre_rR_factor_lens = int_prep.pre_rR_factor_lens[:,1:diag_id]

    # --- Trapezoidal integration weights ---
    trapz_weights_log_rr = int_prep.trapz_weights_log_rr
    trapz_weights_RR     = int_prep.trapz_weights_RR

    @views rtest = rr_truncated[r_calc_ids]

    w_row = trapz_weights_log_rr .* rr_truncated.^2
    w_col = trapz_weights_RR[1:diag_id]


    # Combine row and column weights
    @views w_rc = w_row .* w_col'  # (N_r, N_R)
    
    w00_truncated_lt = w00_truncated[:,1:diag_id]

    L00diag = zeros(length(r_calc_ids))
    
    # Preweight the base w00 for each kernel type:
    w00_weighted_L = w_rc .* w00_truncated_lt .* pre_r_factor_lens .* pre_rR_factor_lens

    compute_L_auto_test!(L00diag,rtest,rr_truncated,w00_weighted_L,r_R,r_calc_ids,diag_id)

    return w00rx,  L00diag
end


function lensing_test_manual(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep, rr_trunc_min_id, R_slicing_l, Rval; r_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:,:] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    #rr_max_id = maximum(r_calc_ids)

    RR_truncated = RR[1:R_slicing_l:end]
    Rval_id = minimum(findall(RR .>= Rval))

    rr_truncated = @views rr[rr_trunc_min_id:rr_max_id]


    @views w00_truncated = w00[rr_trunc_min_id:rr_max_id, 1:R_slicing_l:end]

    # --- Prefactors for 1D integrals over r ---
    pre_r_factor_lens = int_prep.pre_r_factor_lens

    # --- Prefactors for 2D integrals over r × R ---
    pre_rR_factor_lens = int_prep.pre_rR_factor_lens[:, 1:Rval_id]

    @views rtest = rr_truncated[r_calc_ids .- rr_trunc_min_id]


    L00diag, l00_r_int = compute_Cl_lensing_test(rr_truncated,r_calc_ids .- rr_trunc_min_id,Rval,RR_truncated[1:Rval_id],w00_truncated,pre_r_factor_lens,pre_rR_factor_lens)

    return L00diag, l00_r_int, w00_truncated
end


function lensing_test_manual_no_w(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep, rr_trunc_min_id, R_slicing_l, Rval; r_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:,:] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    #rr_max_id = maximum(r_calc_ids)

    RR_truncated = RR[1:R_slicing_l:end]
    Rval_id = minimum(findall(RR .>= Rval))

    rr_truncated = @views rr[rr_trunc_min_id:rr_max_id]


    @views w00_truncated = w00[rr_trunc_min_id:rr_max_id, 1:R_slicing_l:end]

    # --- Prefactors for 1D integrals over r ---
    pre_r_factor_lens = int_prep.pre_r_factor_lens

    # --- Prefactors for 2D integrals over r × R ---
    pre_rR_factor_lens = int_prep.pre_rR_factor_lens[:, 1:Rval_id]

    @views rtest = rr_truncated[r_calc_ids .- rr_trunc_min_id]


    L00diag, l00_r_int = compute_Cl_lensing_test(rr_truncated,r_calc_ids .- rr_trunc_min_id,Rval,RR_truncated[1:Rval_id],w00_truncated,pre_r_factor_lens,pre_rR_factor_lens)

    return L00diag, l00_r_int
end

function compute_Cl_lensing_test(
    rr_truncated,         # full χ grid (fine)
    r_calc_ids,  
    R_calc,             # indices to evaluate Cl at
    R_vals,               # R grid
    w00_truncated,            # (N_χ, N_R)
    pref_r,               # (1+z)D(χ)
    pref_rR              # (1+z)D(Rχ)
)

    N_r_full = length(rr_truncated)
    N_R = length(R_vals)
    log_r_vals = log.(rr_truncated)
    rR = rr_truncated .* R_vals'          # (N_χ, N_R)

    N_out = length(r_calc_ids)
    Cl_vals = zeros(Float64, N_out)
    integral_vals = zeros(Float64, N_out, N_R)

    @inbounds Threads.@threads for outidx in 1:N_out
        rindx = r_calc_ids[outidx]
        r_star = rr_truncated[rindx] 

        acc_R = zeros(Float64, N_R)

        for j in 1:N_R
            inner_integrand = similar(rr_truncated)
            @inbounds for i in 1:N_r_full
                χ = rr_truncated[i]
                Rχ = rR[i, j]

                kernel = lensing_diag_test(χ, r_star, Rχ, R_calc)
                integrand = χ^2 * pref_r[i] * pref_rR[i, j] * kernel * w00_truncated[i, j]
                inner_integrand[i] = integrand
            end

            acc_R[j] = trapz_fast(log_r_vals[1:rindx], inner_integrand[1:rindx])
            
        end
        integral_vals[outidx,:] .= acc_R
        Cl_vals[outidx] = 2 * trapz_fast(R_vals, acc_R)
    end

    return Cl_vals, integral_vals
end



function cl_to_grid(cl, rr, RR;r_grid_ids=nothing)
    Nr = length(rr)
    NR = length(RR)

    if r_grid_ids == nothing
        r_grid_ids = collect(1:Nr)
    end

    @assert length(r_grid_ids) <= Nr
    @assert size(cl,1) == Nr
    @assert size(cl,2) == NR
    

    rR_grid = rr[r_grid_ids] .* RR'
    r_grid = rr[r_grid_ids]

    gridded_cl = zeros(length(r_grid_ids),length(r_grid_ids))

    @inbounds @Threads.threads for i = eachindex(r_grid_ids)
        r_id = r_grid_ids[i]
        rR_grid_i = @views rR_grid[i,:]
        cl_i = @views cl[r_id,:]

        spl_i = Spline1D(rR_grid_i,cl_i)
        @views gridded_cl[i,:] .= spl_i.(r_grid) 
    end

    return gridded_cl
end


function cl_lt_to_grid(cl_lt, rr, RR; r_grid_ids=nothing)
    Nr = length(rr)
    NR = length(RR)

    if r_grid_ids == nothing
        r_grid_ids = collect(1:Nr)
    end

    rR_grid_lt = rr[r_grid_ids] .* RR'
    r_grid = rr[r_grid_ids]

    @assert length(r_grid_ids) <= Nr
    @assert size(cl_lt,1) == Nr
    @assert size(cl_lt,2) == NR

    gridded_cl = zeros(length(r_grid_ids),length(r_grid_ids))

    @inbounds @Threads.threads for i = eachindex(r_grid_ids)
        r_id = r_grid_ids[i]
        rR_grid_i = @views rR_grid_lt[i,:]
        cl_i = @views cl_lt[r_id,:]
        r_grid_i = @views r_grid[1:i]

        spl_i = Spline1D(rR_grid_i,cl_i)
        @views gridded_cl[i,1:i] .= spl_i.(r_grid_i) 
    end

    symmetrize_from_lower!(gridded_cl)

    return gridded_cl
end


function symmetrize_from_lower!(A::AbstractMatrix)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    N = size(A, 1)
    @inbounds for i in 1:N
        for j in i+1:N
            A[i, j] = A[j, i]
        end
    end
    return A
end




function smooth_over_alias_features!(w,RR,Rmin,Rmax)
    Nr = size(w,1)
    NR = size(w,2)

    @assert length(RR) == NR
    
    Rmin_id = maximum(findall(RR .<= Rmin))
    Rmax_id = minimum(findall(RR .>= Rmax))

    unsmoothed_ids = collect(Rmin_id:Rmax_id)
    smooth_ids = setdiff(collect(1:NR),unsmoothed_ids)

    RR_smooth = RR[smooth_ids]
    RR_unsmoothed = RR[unsmoothed_ids]

    w_smooth = @views w[:,smooth_ids]
    @inbounds @Threads.threads for i = 1:Nr
        smoothed_i = Spline1D(RR_smooth,w_smooth[i,:])
        @views w[i,unsmoothed_ids] .= smoothed_i.(RR_unsmoothed)
    end
end








###################################################
###################################################
###################################################
##################### ARCHIVE #####################
###################################################
###################################################
###################################################

function compute_00_10_R_integrals!(
    rtest, r_calc_ids, R_calc_ids, RR_truncated,
    r_R, D_rR, H_rR, Om_rR, z_rR, ISW_kernel_rR, TD_kernel_rR,
    w00_truncated, w10p1_truncated,
    l00rx_R, s00rx_R, t00rx_R, l10p1rx_R, s10p1rx_R, t10p1rx_R
)
    N_r = length(rtest)
    N_R = length(R_calc_ids)

    max_cutoff = size(r_R, 2)

    tmp_lens_kernel_pool = [zeros(max_cutoff) for _ in 1:Threads.nthreads()]

    Threads.@threads for idx in 1:(N_r * N_R)
        rindx = fld(idx - 1, N_R) + 1
        Rindx = mod(idx - 1, N_R) + 1

        rindx_adj = r_calc_ids[rindx]
        R_calc_indx = R_calc_ids[Rindx]

        # Precompute slices just once
        rR        = @view r_R[rindx_adj, 1:R_calc_indx]
        D_rR_i    = @view D_rR[rindx_adj, 1:R_calc_indx]
        H_rR_i    = @view H_rR[rindx_adj, 1:R_calc_indx]
        Om_rR_i   = @view Om_rR[rindx_adj, 1:R_calc_indx]
        z_rR_i    = @view z_rR[rindx_adj, 1:R_calc_indx]
        RR_i      = @view RR_truncated[1:R_calc_indx]

        # Kernels
        ISW_kernel_R  = @view ISW_kernel_rR[rindx_adj, 1:R_calc_indx]
        TD_kernel_R   = @view TD_kernel_rR[rindx_adj, 1:R_calc_indx]
        lens_kernel_R = @view tmp_lens_kernel_pool[Threads.threadid()][1:R_calc_indx]
        lensing_kernel!(lens_kernel_R, rR, rR[end], D_rR_i, H_rR_i, Om_rR_i, z_rR_i)


        w00 = @view w00_truncated[rindx_adj, 1:R_calc_indx]
        w10 = @view w10p1_truncated[rindx_adj, 1:R_calc_indx]


        # Kernel-weighted integrals
        l00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, lens_kernel_R)
        s00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, ISW_kernel_R)
        t00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, TD_kernel_R)

        l10p1rx_R[rindx, Rindx] = trapz_mul(rR, w10, lens_kernel_R)
        s10p1rx_R[rindx, Rindx] = trapz_mul(rR, w10, ISW_kernel_R)
        t10p1rx_R[rindx, Rindx] = trapz_mul(rR, w10, TD_kernel_R)
    end
end



function compute_00_20_R_integrals!(
    rtest, r_calc_ids, R_calc_ids, RR_truncated,
    r_R, D_rR, H_rR, Om_rR, z_rR, ISW_kernel_rR, TD_kernel_rR,
    w00_truncated, w20_truncated,
    l00rx_R, s00rx_R, t00rx_R, l20rx_R, s20rx_R, t20rx_R
)
    N_r = length(rtest)
    N_R = length(R_calc_ids)

    max_cutoff = size(r_R, 2)

    tmp_lens_kernel_pool = [zeros(max_cutoff) for _ in 1:Threads.nthreads()]

    Threads.@threads for idx in 1:(N_r * N_R)
        rindx = fld(idx - 1, N_R) + 1
        Rindx = mod(idx - 1, N_R) + 1

        rindx_adj = r_calc_ids[rindx]
        R_calc_indx = R_calc_ids[Rindx]

        # Precompute slices just once
        rR        = @view r_R[rindx_adj, 1:R_calc_indx]
        D_rR_i    = @view D_rR[rindx_adj, 1:R_calc_indx]
        H_rR_i    = @view H_rR[rindx_adj, 1:R_calc_indx]
        Om_rR_i   = @view Om_rR[rindx_adj, 1:R_calc_indx]
        z_rR_i    = @view z_rR[rindx_adj, 1:R_calc_indx]
        RR_i      = @view RR_truncated[1:R_calc_indx]

        # Kernels
        ISW_kernel_R  = @view ISW_kernel_rR[rindx_adj, 1:R_calc_indx]
        TD_kernel_R   = @view TD_kernel_rR[rindx_adj, 1:R_calc_indx]
        lens_kernel_R = @view tmp_lens_kernel_pool[Threads.threadid()][1:R_calc_indx]
        lensing_kernel!(lens_kernel_R, rR, rR[end], D_rR_i, H_rR_i, Om_rR_i, z_rR_i)


        w00 = @view w00_truncated[rindx_adj, 1:R_calc_indx]
        w20 = @view w20_truncated[rindx_adj, 1:R_calc_indx]


        # Kernel-weighted integrals
        l00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, lens_kernel_R)
        s00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, ISW_kernel_R)
        t00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, TD_kernel_R)

        l20rx_R[rindx, Rindx] = trapz_mul(rR, w20, lens_kernel_R)
        s20rx_R[rindx, Rindx] = trapz_mul(rR, w20, ISW_kernel_R)
        t20rx_R[rindx, Rindx] = trapz_mul(rR, w20, TD_kernel_R)
    end
end



function compute_00_R_integrals!(
    rtest, r_calc_ids, R_calc_ids, RR_truncated,
    r_R, D_rR, H_rR, Om_rR, z_rR, ISW_kernel_rR, TD_kernel_rR,
    w00_truncated, l00rx_R, s00rx_R, t00rx_R)
    N_r = length(rtest)
    N_R = length(R_calc_ids)

    max_cutoff = size(r_R, 2)

    tmp_lens_kernel_pool = [zeros(max_cutoff) for _ in 1:Threads.nthreads()]

    Threads.@threads for idx in 1:(N_r * N_R)
        rindx = fld(idx - 1, N_R) + 1
        Rindx = mod(idx - 1, N_R) + 1

        rindx_adj = r_calc_ids[rindx]
        R_calc_indx = R_calc_ids[Rindx]

        # Precompute slices just once
        rR        = @view r_R[rindx_adj, 1:R_calc_indx]
        D_rR_i    = @view D_rR[rindx_adj, 1:R_calc_indx]
        H_rR_i    = @view H_rR[rindx_adj, 1:R_calc_indx]
        Om_rR_i   = @view Om_rR[rindx_adj, 1:R_calc_indx]
        z_rR_i    = @view z_rR[rindx_adj, 1:R_calc_indx]
        RR_i      = @view RR_truncated[1:R_calc_indx]

        # Kernels
        ISW_kernel_R  = @view ISW_kernel_rR[rindx_adj, 1:R_calc_indx]
        TD_kernel_R   = @view TD_kernel_rR[rindx_adj, 1:R_calc_indx]
        lens_kernel_R = @view tmp_lens_kernel_pool[Threads.threadid()][1:R_calc_indx]
        lensing_kernel!(lens_kernel_R, rR, rR[end], D_rR_i, H_rR_i, Om_rR_i, z_rR_i)


        w00 = @view w00_truncated[rindx_adj, 1:R_calc_indx]
        w10 = @view w10p1_truncated[rindx_adj, 1:R_calc_indx]


        # Kernel-weighted integrals
        l00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, lens_kernel_R)
        s00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, ISW_kernel_R)
        t00rx_R[rindx, Rindx]   = trapz_mul(rR, w00, TD_kernel_R)
    end
end



function compute_lensing_integrals_flat!(
    rtest, r_calc_ids, log_rr_truncated, RR_truncated, r_R, D_rR, H_rR, Om_rR, z_rR,
    w00_truncated, ISW_kernel_rR, TD_kernel_rR,
    f2, f3, f4, f5, f6,
    S00_int, T00_int, X00_int, Y00_int, Z00_int
)
    N_r = length(rtest)
    N_R = length(RR_truncated)  # Adjusted range
    max_cutoff = size(log_rr_truncated, 1)

    # Thread-local temporary buffer pool
    tmp_lens_rr_pool = [zeros(max_cutoff) for _ in 1:Threads.nthreads()]

    Threads.@threads for idx in 1:(N_r * N_R)
        rindx = fld(idx - 1, N_R) + 1
        Rindx = mod(idx - 1, N_R) + 1

        cutoff_idx = r_calc_ids[rindx]

        log_rr_i = @view log_rr_truncated[1:cutoff_idx]

        rR        = @view r_R[1:cutoff_idx, Rindx]
        D_rR_i    = @view D_rR[1:cutoff_idx, Rindx]
        H_rR_i    = @view H_rR[1:cutoff_idx, Rindx]
        Om_rR_i   = @view Om_rR[1:cutoff_idx, Rindx]
        z_rR_i    = @view z_rR[1:cutoff_idx, Rindx]
        rR_i      = RR_truncated[Rindx] * rtest[rindx]
        
        lens_kernel_R = @view tmp_lens_rr_pool[Threads.threadid()][1:cutoff_idx]  # reuse temp buffer!
        lensing_kernel!(lens_kernel_R, rR, rR_i, D_rR_i, H_rR_i, Om_rR_i, z_rR_i)
        ISW_kernel_R  = @view ISW_kernel_rR[1:cutoff_idx, Rindx]
        TD_kernel_R   = @view TD_kernel_rR[1:cutoff_idx, Rindx]

        f2ij = @view f2[1:cutoff_idx, Rindx]
        f3ij = @view f3[1:cutoff_idx, Rindx]
        f4ij = @view f4[1:cutoff_idx, Rindx]
        f5ij = @view f5[1:cutoff_idx, Rindx]
        f6ij = @view f6[1:cutoff_idx, Rindx]

        @inbounds S00_int[rindx, Rindx] = trapz_mul(log_rr_i, f2ij, ISW_kernel_R)
        @inbounds T00_int[rindx, Rindx] = trapz_mul(log_rr_i, f3ij, TD_kernel_R)
        @inbounds X00_int[rindx, Rindx] = trapz_mul(log_rr_i, f4ij, TD_kernel_R)
        @inbounds Y00_int[rindx, Rindx] = trapz_mul(log_rr_i, f5ij, lens_kernel_R)
        @inbounds Z00_int[rindx, Rindx] = trapz_mul(log_rr_i, f6ij, lens_kernel_R)
    end
end



# For calculating C_ell on grid

function rR_to_grid(wjjrx_l,wjjrx_u,r,rR,rgrid)
    @assert length(r) == size(rR,1)

    @assert size(wjjrx_l,1) == size(rR,1)
    @assert size(wjjrx_l,2) == size(rR,2)

    @assert size(wjjrx_u,1) == length(rR,1)
    @assert size(wjjrx_u,2) == length(rR,2)


    wjj_grid_Rl = zeros(Float64,size(rR,1),length(rgrid))
    wjj_grid_Ru = zeros(Float64,size(rR,1),length(rgrid))

    wjj_grid_rR = zeros(Float64,length(rgrid),length(rgrid))

    @inbounds @Threads.threads for rindx = 1:size(rR,1) # aligning (r,rR) data to (r,r_grid)
        @views R_slice = rR[rindx,:]
        @views wjjrx_l_slice = wjjrx_l[rindx,:]
        @views wjjrx_u_slice = wjjrx_u[rindx,:]

        end_id = Int(maximum(findall(rgrid .<= maximum(R_slice))))
        @views rgrid_R = rgrid[1:end_id]

        fthisl = Spline1D(R_slice,wjjrx_l_slice)
        fthisu = Spline1D(R_slice,wjjrx_u_slice)

        @views wjj_grid_Rl[rindx,1:end_id] .= fthisl.(rgrid_R)
        @views wjj_grid_Ru[rindx,1:end_id] .= fthisu.(rgrid_R)
    end


    @inbounds @Threads.threads for grididx = 1:size(rR,2) # aligning (r,r_grid) data to (r_grid,r_grid)
        start_id = Int(minimum(findall(r .>= rgrid[grididx] .- 1e-8)))
        @views r_slice = r[start_id:end]
        @views wjj_grid_Rl_slice = wjj_grid_Rl[start_id:end,grididx]
        @views wjj_grid_Ru_slice = wjj_grid_Ru[start_id:end,grididx]

        fthisl = Spline1D(r_slice,wjj_grid_Rl_slice)
        fthisu = Spline1D(r_slice,wjj_grid_Ru_slice)

        @views rgrid_r = rgrid[grididx:end]

        @views wjj_grid_rR[1:start_id,grididx] .= fthisl.(rgrid_r)
        @views wjj_grid_rR[grididx,1:start_id] .= fthisu.(rgrid_r)
    end

    return wjj_grid_rR
end


function trapz_mul(x::AbstractVector, a::AbstractVector, b::AbstractVector)
    acc = zero(eltype(x))
    @inbounds @simd for i in 1:length(x)-1
        dx = x[i+1] - x[i]
        avg = (a[i]*b[i] + a[i+1]*b[i+1]) / 2
        acc += dx * avg
    end
    return acc
end






function RSD_GR_cross_and_lensing_auto_terms(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))
    w02 = Array{Float64}(undef, N, length(RR))
    w20 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
            @views w02[:, :] = wjj[2]
            @views w20[:, :] = wjj[3]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    Nr_new = Int(length(rr[1:r_slicing:end]))    

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]

    

    @views w00_truncated = w00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    @views w02_truncated = w02[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    @views w20_truncated = w20[rr_trunc_low_id:rr_max_id, 1:RR_max_id]

    w10p1_truncated = similar(w00_truncated)
    w12p1_truncated = similar(w00_truncated)
    w01p1_truncated = similar(w00_truncated)
    w21p1_truncated = similar(w00_truncated)

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w00_truncated[rindx,:])
        fthis2 = Spline1D(RR_truncated * rr_truncated[rindx], w20_truncated[rindx,:])
        @views w01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
        @views w21p1_truncated[rindx,:] = derivative(fthis2, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, w00_truncated[:,Rindx])
        fthis2 = Spline1D(rr_truncated, w02_truncated[:,Rindx])
        @views w10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * w01p1_truncated[:,Rindx]
        @views w12p1_truncated[:,Rindx] = derivative(fthis2, rr_truncated, nu=1) .- RR_truncated[Rindx] * w21p1_truncated[:,Rindx]
    end

    @views w00rx = w00_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w20rx = w20_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w02rx = w02_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w10p1rx = w10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w01p1rx = w01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w12p1rx = w12p1_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w21p1rx = w21p1_truncated[r_calc_ids_trunc,R_calc_ids]

    #l00rx_R = similar(w00rx)
    #s00rx_R = similar(w00rx)
    #t00rx_R = similar(w00rx)
    #l20rx_R = similar(w00rx)
    #s20rx_R = similar(w00rx)
    #t20rx_R = similar(w00rx)

    l00rx_r = similar(w00rx)
    s00rx_r = similar(w00rx)
    t00rx_r = similar(w00rx)
    l02rx_r = similar(w00rx)
    s02rx_r = similar(w00rx)
    t02rx_r = similar(w00rx)

    r_R = int_prep.r_R

    # --- 2D Background quantities (functions of r * R) ---
    #z_rR  = int_prep.z_rR
    #D_rR  = int_prep.D_rR
    #H_rR  = int_prep.H_rR
    #Om_rR = int_prep.Om_rR
    #f_rR  = int_prep.f_rR

    # --- ISW and time delay kernels ---
    #ISW_kernel_rR = int_prep.ISW_kernel_rR
    #TD_kernel_rR  = int_prep.TD_kernel_rR
    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw

    @views rtest = rr_truncated[r_calc_ids_trunc]


    w_row_lst = int_prep.trapz_weights_log_RR

    l_kernel_precomp = precompute_l_kernel_r(RR_truncated, R_calc_ids) 
    
    w00_weighted_l = w_row_lst' .* w00_truncated .* pre_rR_factor_lens
    w00_weighted_s = w_row_lst' .* w00_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated
    w00_weighted_t = w_row_lst' .* w00_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated
    
    w20_weighted_l = w_row_lst' .* w20_truncated .* pre_rR_factor_lens
    w20_weighted_s = w_row_lst' .* w20_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated
    w20_weighted_t = w_row_lst' .* w20_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated

    #compute_00_20_R_integrals!(rtest, r_calc_ids_trunc, R_calc_ids, RR_truncated, r_R, D_rR, H_rR, Om_rR, z_rR, ISW_kernel_rR, TD_kernel_rR, w00_truncated, w20_truncated, l00rx_R, s00rx_R, t00rx_R, l20rx_R, s20rx_R, t20rx_R)
    compute_lensing_cross_integral_r!(r_calc_ids_trunc,R_calc_ids,l_kernel_precomp,w00_weighted_l,l00rx_r)
    compute_lensing_cross_integral_r!(r_calc_ids_trunc,R_calc_ids,l_kernel_precomp,w20_weighted_l,l02rx_r)

    compute_st_cross_integrals_r!(r_calc_ids_trunc,R_calc_ids,w00_weighted_s,w00_weighted_t,s00rx_r,t00rx_r)
    compute_st_cross_integrals_r!(r_calc_ids_trunc,R_calc_ids,w20_weighted_s,w20_weighted_t,s02rx_r,t02rx_r)
    

    return w00rx, w02rx, w20rx, w10p1rx, w01p1rx, w12p1rx, w21p1rx, l00rx_r, s00rx_r, t00rx_r, l02rx_r, s02rx_r, t02rx_r
end



function doppler_and_potential_cross_terms(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    w00 = Array{Float64}(undef, N, length(RR))
    w02 = Array{Float64}(undef, N, length(RR))
    w20 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:,:] = wjj[1]
            @views w02[:,:] = wjj[2]
            @views w20[:,:] = wjj[3]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i;], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))


    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    rr_max_id = maximum(r_calc_ids)
    RR_max_id = maximum(R_calc_ids)
    
    RR_truncated = @views RR[1:RR_max_id]    

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]

    w00_truncated = @views w00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    w02_truncated = @views w02[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    w20_truncated = @views w20[rr_trunc_low_id:rr_max_id, 1:RR_max_id]

    w10p1_truncated = similar(w00_truncated)
    w12p1_truncated = similar(w00_truncated)
    w01p1_truncated = similar(w00_truncated)
    w21p1_truncated = similar(w00_truncated)
    w11p2_truncated = similar(w00_truncated)

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w00_truncated[rindx,:])
        fthis2 = Spline1D(RR_truncated * rr_truncated[rindx], w20_truncated[rindx,:])
        @views w01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
        @views w21p1_truncated[rindx,:] = derivative(fthis2, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, w00_truncated[:,Rindx])
        fthis2 = Spline1D(rr_truncated, w02_truncated[:,Rindx])
        @views w10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * w01p1_truncated[:,Rindx]
        @views w12p1_truncated[:,Rindx] = derivative(fthis2, rr_truncated, nu=1) .- RR_truncated[Rindx] * w21p1_truncated[:,Rindx]
    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w10p1_truncated[rindx,:])
        @views w11p2_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

   
    @views w00rx = w00_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w02rx = w02_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w20rx = w20_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w10p1rx = w10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w01p1rx = w01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w12p1rx = w12p1_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w21p1rx = w21p1_truncated[r_calc_ids_trunc,R_calc_ids]
    @views w11p2rx = w11p2_truncated[r_calc_ids_trunc,R_calc_ids]

    return w00rx, w02rx, w20rx, w10p1rx, w01p1rx, w12p1rx, w21p1rx, w11p2rx
end



function fnl_xt_neg2_test(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    u00 = Array{Float64}(undef, N, length(RR))
    u02 = Array{Float64}(undef, N, length(RR))
    u20 = Array{Float64}(undef, N, length(RR))


    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views u00[:, :] = wjj[1]
            @views u02[:, :] = wjj[2]
            @views u20[:, :] = wjj[3]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    u00_truncated = @views u00[rr_trunc_low_id:rr_max_id, :]
    u02_truncated  = @views u02[rr_trunc_low_id:rr_max_id, :]
    u20_truncated  = @views u20[rr_trunc_low_id:rr_max_id, :]

    u00_gridded = cl_to_grid(u00_truncated, rr_truncated, RR)
    u20_gridded = cl_to_grid(u20_truncated, rr_truncated, RR)
    u02_gridded = cl_to_grid(u02_truncated, rr_truncated, RR)

    return u00_gridded, u02_gridded, u20_gridded
end

function fnl_xt_neg4_test(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]


    RR = [R;]
    chi0 = 1 / kmax

    u00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views u00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    @views u00_truncated = u00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]

    u00_gridded = cl_to_grid(v00_truncated, rr_truncated, RR)
    return u00_gridded
end

function fnl_old_auto(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    v00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views v00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end
    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))


    v00_truncated = @views v00[rr_trunc_low_id:rr_max_id, :]

    v00_gridded = cl_to_grid(v00_truncated, rr_truncated, RR)
    return v00_gridded
end












function LST_integrals(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:,:] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]
 

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]

    @views w00_truncated = w00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]

    if ell_i <= 20
        R_issue_min = 0.5
        R_issue_max = 0.55

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min,R_issue_max)
    end


    RR_truncated_lower = RR_truncated[RR_truncated .<= 1]
    R_calc_ids_lower = R_calc_ids[RR_truncated[R_calc_ids] .<= 1]
    diag_id = maximum(R_calc_ids_lower)

    r_R = int_prep.r_R

    # --- Prefactors for 1D integrals over r ---
    pre_r_factor_lens = int_prep.pre_r_factor_lens
    pre_r_factor_td   = int_prep.pre_r_factor_td
    pre_r_factor_isw  = int_prep.pre_r_factor_isw

    # --- Prefactors for 2D integrals over r × R ---
    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw

    # --- Trapezoidal integration weights ---
    trapz_weights_log_rr = int_prep.trapz_weights_log_rr
    trapz_weights_RR     = int_prep.trapz_weights_RR

    @views rtest = rr_truncated[r_calc_ids_trunc]

    w_row = trapz_weights_log_rr .* rr_truncated.^2
    w_col = trapz_weights_RR[1:diag_id]


    # Combine row and column weights
    @views w_rc = w_row .* w_col'  # (N_r, N_R)
    
    w00_truncated_lt = w00_truncated[:,1:diag_id]

    L00rx_lt = zeros(length(r_calc_ids_trunc),length(R_calc_ids_lower))
    S00rx_lt = similar(L00rx_lt)
    T00rx_lt = similar(L00rx_lt)

    # Preweight the base w00 for each kernel type:
    w00_weighted_L = w_rc .* w00_truncated_lt .* pre_r_factor_lens .* pre_rR_factor_lens[:,1:diag_id]
    w00_weighted_S = w_rc .* w00_truncated_lt .* pre_r_factor_isw .* pre_rR_factor_isw[:,1:diag_id]
    w00_weighted_T = w_rc .* w00_truncated_lt .* pre_r_factor_td  .* pre_rR_factor_td[:,1:diag_id]


    P_S = compute_prefix_sum_2d(w00_weighted_S)
    P_T = compute_prefix_sum_2d(w00_weighted_T)


    Rr_star_precomp = [rtest[rindx] * RR_truncated_lower[R_calc_ids[Rindx]] for rindx in eachindex(r_calc_ids_trunc), Rindx in eachindex(R_calc_ids_lower)]
 
    compute_ST_integrals!(S00rx_lt,T00rx_lt,P_S,P_T,r_calc_ids_trunc,R_calc_ids_lower)
    compute_L_integral!(L00rx_lt,rtest,rr_truncated,w00_weighted_L,r_R,Rr_star_precomp,r_calc_ids_trunc,R_calc_ids_lower)
    
    return w00_weighted_S, L00rx_lt, S00rx_lt, T00rx_lt
end

function lst_01_integrals(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    @views w00_truncated = w00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    w10p1_truncated = similar(w00_truncated)
    w01p1_truncated = similar(w00_truncated)
    w11p2_truncated = similar(w00_truncated)

    if ell_i <= 30
        R_issue_min = 0.5
        R_issue_max = 0.51

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min,R_issue_max)
    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w00_truncated[rindx,:])
        @views w01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, w00_truncated[:,Rindx])
        @views w10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * w01p1_truncated[:,Rindx]
    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w10p1_truncated[rindx,:])
        @views w11p2_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    #@views w00rx = w00_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w10p1rx = w10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w01p1rx = w01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w11p2rx = w11p2_truncated[r_calc_ids_trunc,R_calc_ids]


    l00rx_r = similar(w00_truncated)
    s00rx_r = similar(w00_truncated)
    t00rx_r = similar(w00_truncated)
    l01p1rx_r = similar(w00_truncated)
    s01p1rx_r = similar(w00_truncated)
    t01p1rx_r = similar(w00_truncated)

    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw

    @views rtest = rr_truncated[r_calc_ids_trunc]

    w_row_lst = int_prep.trapz_weights_log_RR

    l_kernel_precomp = precompute_l_kernel_r(RR_truncated, R_calc_ids) 
    
    w00_weighted_l = w_row_lst' .* w00_truncated .* pre_rR_factor_lens
    w00_weighted_s = w_row_lst' .* w00_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated
    w00_weighted_t = w_row_lst' .* w00_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated
    
    w10p1_weighted_l = w_row_lst' .* w10p1_truncated .* pre_rR_factor_lens
    w10p1_weighted_s = w_row_lst' .* w10p1_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated
    w10p1_weighted_t = w_row_lst' .* w10p1_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated

    compute_lensing_cross_integral_r!(r_calc_ids_trunc,R_calc_ids,l_kernel_precomp,w00_weighted_l,l00rx_r)
    compute_lensing_cross_integral_r!(r_calc_ids_trunc,R_calc_ids,l_kernel_precomp,w10p1_weighted_l,l01p1rx_r)

    compute_st_cross_integrals_r!(r_calc_ids_trunc,R_calc_ids,w00_weighted_s,w00_weighted_t,s00rx_r,t00rx_r)
    compute_st_cross_integrals_r!(r_calc_ids_trunc,R_calc_ids,w10p1_weighted_s,w10p1_weighted_t,s01p1rx_r,t01p1rx_r)
    

    return w00rx, w10p1rx, w01p1rx, w11p2rx, l00rx_r, s00rx_r, t00rx_r, l01p1rx_r, s01p1rx_r, t01p1rx_r
end




# Estimate tangents using central differences
function estimate_tangents!(m, x, y)
    N = length(x)
    @inbounds begin
        m[1] = (y[2] - y[1]) / (x[2] - x[1])
        m[N] = (y[N] - y[N-1]) / (x[N] - x[N-1])
        @simd for i in 2:N-1
            dx1 = x[i] - x[i-1]
            dx2 = x[i+1] - x[i]
            m[i] = ((y[i+1] - y[i]) / dx2 + (y[i] - y[i-1]) / dx1) / 2
        end
    end
end

# Vectorized Hermite interpolation over evaluation grid
function hermite_interp_vec!(out, x, y, m, x_eval)
    @inbounds for k in eachindex(x_eval)
        x0 = x_eval[k]
        if x0 <= x[1]
            out[k] = y[1]
        elseif x0 >= x[end]
            out[k] = y[end]
        else
            i = searchsortedfirst(x, x0)
            x1, x2 = x[i-1], x[i]
            y1, y2 = y[i-1], y[i]
            m1, m2 = m[i-1], m[i]
            h = x2 - x1
            t = (x0 - x1) / h
            t2 = t * t
            t3 = t2 * t
            h00 = 2t3 - 3t2 + 1
            h10 = t3 - 2t2 + t
            h01 = -2t3 + 3t2
            h11 = t3 - t2
            out[k] = h00*y1 + h10*h*m1 + h01*y2 + h11*h*m2
        end
    end
end

# Full threaded gridding function
function cl_to_grid_fast!(gridded_cl, cl, rr, RR; r_grid_ids=nothing)
    if r_grid_ids === nothing
        r_grid_ids = 1:length(rr)
    end

    r_grid = rr[r_grid_ids]
    rR_grid = r_grid .* RR'
    N = length(r_grid)

    Threads.@threads for i in 1:N
        r_id = r_grid_ids[i]
        xvals = @view rR_grid[i, :]
        yvals = @view cl[r_id, :]
        outrow = @view gridded_cl[i, :]
        m = similar(xvals)  # local buffer for tangents

        estimate_tangents!(m, xvals, yvals)
        hermite_interp_vec!(outrow, xvals, yvals, m, r_grid)
    end
end



function linear_interp(x::Vector{Float64}, y::Vector{Float64}, x0::Float64)
    N = length(x)
    if x0 <= x[1]
        return y[1]
    elseif x0 >= x[end]
        return y[end]
    end
    i = searchsortedfirst(x, x0)
    x1, x2 = x[i-1], x[i]
    y1, y2 = y[i-1], y[i]
    return y1 + (x0 - x1) * (y2 - y1) / (x2 - x1)
end






function RSD_terms_f(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    

    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    w00 = Array{Float64}(undef, N, length(RR))
    w02 = Array{Float64}(undef, N, length(RR))
    w20 = Array{Float64}(undef, N, length(RR))
    w22 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
            @views w02[:, :] = wjj[2]
            @views w20[:, :] = wjj[3]
            @views w22[:, :] = wjj[4]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    Rmin = minimum(RR)
    #rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    #r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    #Nr_new = Int(length(rr[1:r_slicing:end]))    

    rr_truncated = @views rr[rr_min_id:rr_max_id]

    r_calc_ids_new = r_calc_ids .- (rr_min_id - 1)

    #w00rx = @views w00[r_calc_ids, R_calc_ids]
    #w02rx = @views w02[r_calc_ids, R_calc_ids]
    #w20rx = @views w20[r_calc_ids, R_calc_ids]
    #w22rx = @views w22[r_calc_ids, R_calc_ids]

    @views w00_truncated = w00[rr_min_id:rr_max_id, 1:RR_max_id]
    @views w02_truncated = w02[rr_min_id:rr_max_id, 1:RR_max_id]
    @views w20_truncated = w20[rr_min_id:rr_max_id, 1:RR_max_id]
    @views w22_truncated = w22[rr_min_id:rr_max_id, 1:RR_max_id]

    w00_gridded = zeros(length(rr_truncated),length(rr_truncated))
    w02_gridded = similar(w00_gridded)
    w20_gridded = similar(w00_gridded)
    w22_gridded = similar(w00_gridded)

    cl_to_grid_fast!(w00_gridded, w00_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(w20_gridded, w20_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(w02_gridded, w02_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(w22_gridded, w22_truncated,rr_truncated,RR_truncated)

    w00_gridded = @view w00_gridded[r_calc_ids_new,r_calc_ids_new]
    w20_gridded = @view w20_gridded[r_calc_ids_new,r_calc_ids_new]
    w02_gridded = @view w02_gridded[r_calc_ids_new,r_calc_ids_new]
    w22_gridded = @view w22_gridded[r_calc_ids_new,r_calc_ids_new]


    return w00_gridded, w02_gridded, w20_gridded, w22_gridded
end









function lst_02_integrals_f(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))
    w02 = Array{Float64}(undef, N, length(RR))
    w20 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
            @views w02[:, :] = wjj[2]
            @views w20[:, :] = wjj[3]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    #Nr_new = Int(length(rr[1:r_slicing:end]))    

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]

    

    @views w00_truncated = w00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    @views w02_truncated = w02[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    @views w20_truncated = w20[rr_trunc_low_id:rr_max_id, 1:RR_max_id]

    w10p1_truncated = similar(w00_truncated)
    w12p1_truncated = similar(w00_truncated)
    w01p1_truncated = similar(w00_truncated)
    w21p1_truncated = similar(w00_truncated)

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w00_truncated[rindx,:])
        fthis2 = Spline1D(RR_truncated * rr_truncated[rindx], w20_truncated[rindx,:])
        @views w01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
        @views w21p1_truncated[rindx,:] = derivative(fthis2, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, w00_truncated[:,Rindx])
        fthis2 = Spline1D(rr_truncated, w02_truncated[:,Rindx])
        @views w10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * w01p1_truncated[:,Rindx]
        @views w12p1_truncated[:,Rindx] = derivative(fthis2, rr_truncated, nu=1) .- RR_truncated[Rindx] * w21p1_truncated[:,Rindx]
    end

    #@views w00rx = w00_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w20rx = w20_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w02rx = w02_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w10p1rx = w10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w01p1rx = w01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w12p1rx = w12p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w21p1rx = w21p1_truncated[r_calc_ids_trunc,R_calc_ids]

    l00_truncated = similar(w00_truncated)
    s00_truncated = similar(w00_truncated)
    t00_truncated = similar(w00_truncated)

    l02_truncated = similar(w00_truncated)
    s02_truncated = similar(w00_truncated)
    t02_truncated = similar(w00_truncated)

    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw

    @views rtest = rr_truncated[r_calc_ids_trunc]

    w_row_lst = int_prep.trapz_weights_log_RR

    l_kernel_precomp = precompute_l_kernel_r(RR_truncated, R_calc_ids) 
    
    w00_weighted_l = w_row_lst' .* w00_truncated .* pre_rR_factor_lens * 3/2 * ell_i * (ell_i + 1)
    w00_weighted_s = w_row_lst' .* w00_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated * 3
    w00_weighted_t = w_row_lst' .* w00_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated * 3
    
    w20_weighted_l = w_row_lst' .* w20_truncated .* pre_rR_factor_lens * 3/2 * ell_i * (ell_i + 1)
    w20_weighted_s = w_row_lst' .* w20_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated * 3
    w20_weighted_t = w_row_lst' .* w20_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated * 3

    compute_lensing_cross_integral_r!(collect(1:length(rr_truncated)),R_calc_ids,l_kernel_precomp,w00_weighted_l,l00_truncated)
    compute_lensing_cross_integral_r!(collect(1:length(rr_truncated)),R_calc_ids,l_kernel_precomp,w20_weighted_l,l02_truncated)

    compute_st_cross_integrals_r!(collect(1:length(rr_truncated)),R_calc_ids,w00_weighted_s,w00_weighted_t,s00_truncated,t00_truncated)
    compute_st_cross_integrals_r!(collect(1:length(rr_truncated)),R_calc_ids,w20_weighted_s,w20_weighted_t,s02_truncated,t02_truncated)

    w00_gridded = zeros(length(rr_truncated),length(rr_truncated))
    w02_gridded = similar(w00_gridded)
    w20_gridded = similar(w00_gridded)

    w01p1_gridded = similar(w00_gridded)
    w10p1_gridded = similar(w00_gridded)
    w12p1_gridded = similar(w00_gridded)
    w21p1_gridded = similar(w00_gridded)

    l00_gridded = similar(w00_gridded)
    s00_gridded = similar(w00_gridded)
    t00_gridded = similar(w00_gridded)

    l02_gridded = similar(w00_gridded)
    s02_gridded = similar(w00_gridded)
    t02_gridded = similar(w00_gridded)

    cl_to_grid_fast!(w00_gridded, w00_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(w02_gridded, w02_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(w20_gridded, w20_truncated,rr_truncated,RR_truncated)

    cl_to_grid_fast!(w01p1_gridded, w01p1_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(w10p1_gridded, w10p1_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(w12p1_gridded, w12p1_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(w21p1_gridded, w21p1_truncated,rr_truncated,RR_truncated)

    cl_to_grid_fast!(l00_gridded, l00_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(s00_gridded, s00_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(t00_gridded, t00_truncated,rr_truncated,RR_truncated)

    cl_to_grid_fast!(l02_gridded, l02_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(s02_gridded, s02_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(t02_gridded, t02_truncated,rr_truncated,RR_truncated)


    w00_gridded = @view w00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w02_gridded = @view w02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w20_gridded = @view w20_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    w01p1_gridded = @view w01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w10p1_gridded = @view w10p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w12p1_gridded = @view w12p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w21p1_gridded = @view w21p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    l00_gridded = @view l00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    s00_gridded = @view s00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    t00_gridded = @view t00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    l02_gridded = @view l02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    s02_gridded = @view s02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    t02_gridded = @view t02_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    return w00_gridded, w02_gridded, w20_gridded, w01p1_gridded, w10p1_gridded, w12p1_gridded, w21p1_gridded, l00_gridded, s00_gridded, t00_gridded, l02_gridded, s02_gridded, t02_gridded
end


function lst_LST_XYZ_integrals_f(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    @views w00_truncated = w00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    w10p1_truncated = similar(w00_truncated)
    w01p1_truncated = similar(w00_truncated)
    w11p2_truncated = similar(w00_truncated)

    if ell_i <= 100
        Rmin_choice = max(0.5 - (ell_i-1) * 0.01,0.425)
        R_issue_min = Rmin_choice
        R_issue_max = 0.53

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min,R_issue_max)

        R_issue_min2 = 0.69
        R_issue_max2 = 0.715

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min2,R_issue_max2)

        R_issue_min3 = 1/0.705
        R_issue_max3 = 1/0.67

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min3,R_issue_max3)

        R_issue_min4 = 1/0.508
        R_issue_max4 = 1/0.5

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min4,R_issue_max4)

        deltaR_choice = min(0.3,0.025*ell_i)
        R_issue_min5 = 1/0.491
        R_issue_max5 = 1/0.491 + deltaR_choice

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min5,R_issue_max5)

    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w00_truncated[rindx,:])
        @views w01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, w00_truncated[:,Rindx])
        @views w10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * w01p1_truncated[:,Rindx]
    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w10p1_truncated[rindx,:])
        @views w11p2_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    #@views w00rx = w00_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w10p1rx = w10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w01p1rx = w01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views w11p2rx = w11p2_truncated[r_calc_ids_trunc,R_calc_ids]


    l00_truncated = similar(w00_truncated)
    s00_truncated = similar(w00_truncated)
    t00_truncated = similar(w00_truncated)
    l01p1_truncated = similar(w00_truncated)
    s01p1_truncated = similar(w00_truncated)
    t01p1_truncated = similar(w00_truncated)

    r_R = int_prep.r_R

    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw
    pre_rr_factor_lens  = int_prep.pre_r_factor_lens
    pre_rr_factor_isw  = int_prep.pre_r_factor_isw
    pre_rr_factor_td  = int_prep.pre_r_factor_td

    #@views rtest = rr_truncated[r_calc_ids_trunc]

    w_row_lst = int_prep.trapz_weights_log_RR
    w_row_rr = int_prep.trapz_weights_log_rr

    l_kernel_precomp = precompute_l_kernel_r(RR_truncated, R_calc_ids) 
    
    w00_weighted_l = w_row_lst' .* w00_truncated .* pre_rR_factor_lens * 3/2 * ell_i * (ell_i + 1)
    w00_weighted_s = w_row_lst' .* w00_truncated .* pre_rR_factor_isw .* r_R * 3
    w00_weighted_t = w_row_lst' .* w00_truncated .* pre_rR_factor_td .* r_R * 3

    w10p1_weighted_l = w_row_lst' .* w10p1_truncated .* pre_rR_factor_lens * 3/2 * ell_i * (ell_i + 1)
    w10p1_weighted_s = w_row_lst' .* w10p1_truncated .* pre_rR_factor_isw .* r_R * 3
    w10p1_weighted_t = w_row_lst' .* w10p1_truncated .* pre_rR_factor_td .* r_R * 3

    compute_lensing_cross_integral_r!(collect(1:length(rr_truncated)),R_calc_ids,l_kernel_precomp,w00_weighted_l,l00_truncated)
    compute_lensing_cross_integral_r!(collect(1:length(rr_truncated)),R_calc_ids,l_kernel_precomp,w10p1_weighted_l,l01p1_truncated)

    compute_st_cross_integrals_r!(collect(1:length(rr_truncated)),R_calc_ids,w00_weighted_s,w00_weighted_t,s00_truncated,t00_truncated)
    compute_st_cross_integrals_r!(collect(1:length(rr_truncated)),R_calc_ids,w10p1_weighted_s,w10p1_weighted_t,s01p1_truncated,t01p1_truncated)

    w00_gridded = zeros(length(rr_truncated),length(rr_truncated))
    w01p1_gridded = similar(w00_gridded)
    w10p1_gridded = similar(w00_gridded)
    w11p2_gridded = similar(w00_gridded)

    l00_gridded = similar(w00_gridded)
    s00_gridded = similar(w00_gridded)
    t00_gridded = similar(w00_gridded)

    l01p1_gridded = similar(w00_gridded)
    s01p1_gridded = similar(w00_gridded)
    t01p1_gridded = similar(w00_gridded)

    S00_gridded = zeros(length(rr_truncated),length(rr_truncated))
    L00_gridded = similar(S00_gridded)
    T00_gridded = similar(S00_gridded)
    X00_gridded = similar(S00_gridded)
    Y00_gridded = similar(S00_gridded)
    Z00_gridded = similar(S00_gridded)

    cl_to_grid_fast!(w00_gridded, w00_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(w01p1_gridded, w01p1_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(w10p1_gridded, w10p1_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(w11p2_gridded, w11p2_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(w00_gridded, w00_truncated, rr_truncated, RR_truncated)

    cl_to_grid_fast!(l00_gridded, l00_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(s00_gridded, s00_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(t00_gridded, t00_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(l01p1_gridded, l01p1_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(s01p1_gridded, s01p1_truncated, rr_truncated, RR_truncated)
    cl_to_grid_fast!(t01p1_gridded, t01p1_truncated, rr_truncated, RR_truncated)

    
    
    L00_pre_sum = w_row_rr .* l00_gridded .* pre_rr_factor_lens * 3/2 * ell_i * (ell_i + 1)
    S00_pre_sum = w_row_rr .* s00_gridded .* pre_rr_factor_isw .* rr_truncated * 3
    T00_pre_sum = w_row_rr .* t00_gridded .* pre_rr_factor_td .* rr_truncated * 3
    X00_pre_sum = w_row_rr .* s00_gridded .* pre_rr_factor_td .* rr_truncated * 3
    Y00_pre_sum = w_row_rr .* l00_gridded .* pre_rr_factor_isw .* rr_truncated * 3
    Z00_pre_sum = w_row_rr .* l00_gridded .* pre_rr_factor_td .* rr_truncated * 3

    L00_gridded = cumsum(L00_pre_sum, dims=1) .- (cumsum(rr_truncated .* L00_pre_sum, dims = 1) ./ rr_truncated)
    cumsum_dim1!(S00_gridded, S00_pre_sum)
    cumsum_dim1!(T00_gridded, T00_pre_sum)
    cumsum_dim1!(X00_gridded, X00_pre_sum)
    cumsum_dim1!(Y00_gridded, Y00_pre_sum)
    cumsum_dim1!(Z00_gridded, Z00_pre_sum)
    
    #X00rx_t = similar(S00rx)
    #Y00rx_t = similar(S00rx)
    #Z00rx_t = similar(S00rx)
    #X00_pre_sum_t = w_row_rr .* t00_gridded .* pre_rr_factor_isw .* rr_truncated
    #Y00_pre_sum_t = w_row_rr .* s00_gridded .* pre_rr_factor_lens
    #Z00_pre_sum_t = w_row_rr .* t00_gridded .* pre_rr_factor_lens
    #X00rx_t = cumsum(X00_pre_sum_t, dims=1)
    #Y00rx_t = cumsum(Y00_pre_sum_t, dims=1) .- (cumsum(rr_truncated .* Y00_pre_sum_t, dims = 1) ./ rr_truncated)
    #Z00rx_t = cumsum(Z00_pre_sum_t, dims=1) .- (cumsum(rr_truncated .* Z00_pre_sum_t, dims = 1) ./ rr_truncated)

    w00_gridded = @view w00_gridded[r_calc_ids_trunc,r_calc_ids_trunc] 
    w01p1_gridded = @view w01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc] 
    w10p1_gridded = @view w10p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    w11p2_gridded = @view w11p2_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    l00_gridded = @view l00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    s00_gridded = @view s00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    t00_gridded = @view t00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    l01p1_gridded = @view l01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    s01p1_gridded = @view s01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    t01p1_gridded = @view t01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    L00_gridded = @view L00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    S00_gridded = @view S00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    T00_gridded = @view T00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    X00_gridded = @view X00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    Y00_gridded = @view Y00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    Z00_gridded = @view Z00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    return w00_gridded, w01p1_gridded, w10p1_gridded, w11p2_gridded, l00_gridded, s00_gridded, t00_gridded, l01p1_gridded, s01p1_gridded, t01p1_gridded, L00_gridded, S00_gridded, T00_gridded, X00_gridded, Y00_gridded, Z00_gridded   #X00rx_t, Y00rx_t, Z00rx_t
end







function fnl_02_terms_f(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    u00 = Array{Float64}(undef, N, length(RR))
    u02 = Array{Float64}(undef, N, length(RR))
    u20 = Array{Float64}(undef, N, length(RR))


    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views u00[:, :] = wjj[1]
            @views u02[:, :] = wjj[2]
            @views u20[:, :] = wjj[3]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    #Nr_new = Int(length(rr[1:r_slicing:end]))    

    rr_truncated = @views rr[rr_min_id:rr_max_id]


    r_calc_ids_new = r_calc_ids .- (rr_min_id - 1)
    #w00rx = @views w00[r_calc_ids, R_calc_ids]
    #w02rx = @views w02[r_calc_ids, R_calc_ids]
    #w20rx = @views w20[r_calc_ids, R_calc_ids]
    #w22rx = @views w22[r_calc_ids, R_calc_ids]

    @views u00_truncated = u00[rr_min_id:rr_max_id, 1:RR_max_id]
    @views u02_truncated = u02[rr_min_id:rr_max_id, 1:RR_max_id]
    @views u20_truncated = u20[rr_min_id:rr_max_id, 1:RR_max_id]

    u00_gridded = zeros(length(rr_truncated),length(rr_truncated))
    u20_gridded = similar(u00_gridded)
    u02_gridded = similar(u00_gridded)

    cl_to_grid_fast!(u00_gridded, u00_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(u20_gridded, u20_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(u02_gridded, u02_truncated,rr_truncated,RR_truncated)
    

    u00_gridded = @view u00_gridded[r_calc_ids_new,r_calc_ids_new]
    u20_gridded = @view u20_gridded[r_calc_ids_new,r_calc_ids_new]
    u02_gridded = @view u02_gridded[r_calc_ids_new,r_calc_ids_new]
   

    return u00_gridded, u02_gridded, u20_gridded
end

function fnl_01_integrals_f(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    u00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views u00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    @views u00_truncated = u00[rr_trunc_low_id:rr_max_id, 1:RR_max_id]
    u10p1_truncated = similar(u00_truncated)
    u01p1_truncated = similar(u00_truncated)
   

    if ell_i <= 30
        R_issue_min = 0.5
        R_issue_max = 0.52

        smooth_over_alias_features!(u00_truncated,RR_truncated,R_issue_min,R_issue_max)
    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], u00_truncated[rindx,:])
        @views u01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, u00_truncated[:,Rindx])
        @views u10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * u01p1_truncated[:,Rindx]
    end


    #@views u00rx = u00_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views u10p1rx = u10p1_truncated[r_calc_ids_trunc,R_calc_ids]
    #@views u01p1rx = u01p1_truncated[r_calc_ids_trunc,R_calc_ids]
    

    lfnl00_truncated = similar(u00_truncated)
    sfnl00_truncated = similar(u00_truncated)
    tfnl00_truncated = similar(u00_truncated)
    
    pre_rR_factor_lens = int_prep.pre_rR_factor_lens
    pre_rR_factor_td   = int_prep.pre_rR_factor_td
    pre_rR_factor_isw  = int_prep.pre_rR_factor_isw

    @views rtest = rr_truncated[r_calc_ids_trunc]

    w_row_lst = int_prep.trapz_weights_log_RR
    
    l_kernel_precomp = precompute_l_kernel_r(RR_truncated, R_calc_ids) 
    
    u00_weighted_l = w_row_lst' .* u00_truncated .* pre_rR_factor_lens * 3/2 * ell_i * (ell_i + 1)
    u00_weighted_s = w_row_lst' .* u00_truncated .* pre_rR_factor_isw .* RR_truncated' .* rr_truncated * 3
    u00_weighted_t = w_row_lst' .* u00_truncated .* pre_rR_factor_td .* RR_truncated' .* rr_truncated * 3

    compute_lensing_cross_integral_r!(collect(1:length(rr_truncated)),R_calc_ids,l_kernel_precomp,u00_weighted_l,lfnl00_truncated)
    compute_st_cross_integrals_r!(collect(1:length(rr_truncated)),R_calc_ids,u00_weighted_s,u00_weighted_t,sfnl00_truncated,tfnl00_truncated)
    
    u00_gridded = zeros(length(rr_truncated),length(rr_truncated))
    u10p1_gridded = similar(u00_gridded)
    u01p1_gridded = similar(u00_gridded)

    lfnl00_gridded = similar(u00_gridded)
    sfnl00_gridded = similar(u00_gridded)
    tfnl00_gridded = similar(u00_gridded)

    cl_to_grid_fast!(u00_gridded, u00_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(u10p1_gridded, u10p1_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(u01p1_gridded, u01p1_truncated,rr_truncated,RR_truncated)

    cl_to_grid_fast!(lfnl00_gridded, lfnl00_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(sfnl00_gridded, sfnl00_truncated,rr_truncated,RR_truncated)
    cl_to_grid_fast!(tfnl00_gridded, tfnl00_truncated,rr_truncated,RR_truncated)


    u00_gridded = @view u00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    u10p1_gridded = @view u10p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    u01p1_gridded = @view u01p1_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    lfnl00_gridded = @view lfnl00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    sfnl00_gridded = @view sfnl00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]
    tfnl00_gridded = @view tfnl00_gridded[r_calc_ids_trunc,r_calc_ids_trunc]

    return u00_gridded, u01p1_gridded, u10p1_gridded, lfnl00_gridded, sfnl00_gridded, tfnl00_gridded
end

function fnl_auto_term_f(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    v00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views v00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    #r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = @views RR[1:RR_max_id]

    rr_truncated = @views rr[rr_min_id:rr_max_id]
    
    r_calc_ids_new = r_calc_ids .- (rr_min_id - 1)

    if ell_i <= 30
        R_issue_min = 0.5
        R_issue_max = 0.52

        smooth_over_alias_features!(v00,RR,R_issue_min,R_issue_max)
    end

    @views v00_truncated = v00[rr_min_id:rr_max_id, 1:RR_max_id]
    v00_gridded = zeros(length(rr_truncated),length(rr_truncated))
    cl_to_grid_fast!(v00_gridded,v00_truncated,rr_truncated,RR_truncated)
    v00_gridded = @view v00_gridded[r_calc_ids_new,r_calc_ids_new]

    return v00_gridded
end



function cumsum_dim1!(out::Matrix{Float64}, A::Matrix{Float64})
    N, M = size(A)
    @inbounds for j in 1:M
        out[1, j] = A[1, j]
        @simd for i in 2:N
            out[i, j] = out[i - 1, j] + A[i, j]
        end
    end
end



function cl_to_grid!(gridded_cl, cl, rr, RR;r_grid_ids=nothing)
    Nr = length(rr)
    NR = length(RR)

    if r_grid_ids == nothing
        r_grid_ids = collect(1:Nr)
    end

    @assert length(r_grid_ids) <= Nr
    @assert size(cl,1) == Nr
    @assert size(cl,2) == NR
    

    rR_grid = rr[r_grid_ids] .* RR'
    r_grid = rr[r_grid_ids]

    @inbounds @Threads.threads for i = eachindex(r_grid_ids)
        r_id = r_grid_ids[i]
        rR_grid_i = @views rR_grid[i,:]
        cl_i = @views cl[r_id,:]

        spl_i = Spline1D(rR_grid_i,cl_i)
        @views gridded_cl[i,:] .= spl_i.(r_grid) 
    end
end



function wcheck(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    w00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views w00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)



    RR_truncated = RR

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    @views w00_truncated = w00[rr_trunc_low_id:rr_max_id, :]
    w10p1_truncated = similar(w00_truncated)
    w01p1_truncated = similar(w00_truncated)
    w11p2_truncated = similar(w00_truncated)
    
    if ell_i <= 100
        Rmin_choice = max(0.5 - (ell_i-1) * 0.01,0.425)
        R_issue_min = Rmin_choice
        R_issue_max = 0.53

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min,R_issue_max)

        R_issue_min2 = 0.69
        R_issue_max2 = 0.715

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min2,R_issue_max2)

        R_issue_min3 = 1/0.705
        R_issue_max3 = 1/0.67

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min3,R_issue_max3)

        R_issue_min4 = 1/0.508
        R_issue_max4 = 1/0.5

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min4,R_issue_max4)

        deltaR_choice = min(0.3,0.025*ell_i)
        R_issue_min5 = 1/0.491
        R_issue_max5 = 1/0.491 + deltaR_choice

        smooth_over_alias_features!(w00_truncated,RR_truncated,R_issue_min5,R_issue_max5)

    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w00_truncated[rindx,:])
        @views w01p1_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    @inbounds @Threads.threads for Rindx = eachindex(RR_truncated)
        fthis1 = Spline1D(rr_truncated, w00_truncated[:,Rindx])
        @views w10p1_truncated[:,Rindx] = derivative(fthis1, rr_truncated, nu=1) .- RR_truncated[Rindx] * w01p1_truncated[:,Rindx]
    end

    @inbounds @Threads.threads for rindx = eachindex(rr_truncated)
        fthis1 = Spline1D(RR_truncated * rr_truncated[rindx], w10p1_truncated[rindx,:])
        @views w11p2_truncated[rindx,:] = derivative(fthis1, RR_truncated * rr_truncated[rindx], nu=1)
    end

    return w00_truncated, w01p1_truncated, w10p1_truncated, w11p2_truncated
end


function vcheck(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    chi0 = 1 / kmax
    RR = [R;]

    v00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views v00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    #r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = RR

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    
    r_calc_ids_new = r_calc_ids .- (rr_min_id - 1)

    if ell_i <= 30
        R_issue_min = 0.5
        R_issue_max = 0.52

        smooth_over_alias_features!(v00,RR,R_issue_min,R_issue_max)
    end

    @views v00_truncated = v00[rr_trunc_low_id:rr_max_id, :]

    return v00_truncated
end


function ucheck(pk, ell_i, R, q_in, kmax, kmin, zmax, zmin, N, cachepath, cosmology, integration_prep; r_calc_ids = nothing, R_calc_ids = nothing)
    r_of_z = cosmology["r_of_z"]

    int_prep = integration_prep


    RR = [R;]
    chi0 = 1 / kmax

    u00 = Array{Float64}(undef, N, length(RR))

    function outfunc(wjj, ell, rr, RR)
        @inbounds if ell == ell_i
            @views u00[:, :] = wjj[1]
        end
    end

    rr = calcwljj(pk, RR; ell=[ell_i], kmin=kmin, kmax=kmax, N=N, r0=chi0, q=q_in, outfunc=outfunc, cachefile="$(cachepath)/MlCache/MlCache.bin")

    rr_max_id = Int(minimum(findall(rr .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr .<= r_of_z(zmin))))

    if r_calc_ids == nothing
        r_calc_ids = collect(rr_min_id:rr_max_id)
    end

    Rmin = minimum(RR)
    rr_trunc_low_id = minimum(findall(rr * Rmin .>= 1/kmax))

    r_calc_ids_trunc = r_calc_ids .- (rr_trunc_low_id - 1)

    if R_calc_ids == nothing
        R_calc_ids = collect(1:length(RR))
    end

    RR_max_id = maximum(R_calc_ids)

    RR_truncated = RR

    rr_truncated = @views rr[rr_trunc_low_id:rr_max_id]
    

    @views u00_truncated = u00[rr_trunc_low_id:rr_max_id, :]

   

    if ell_i <= 30
        R_issue_min = 0.5
        R_issue_max = 0.52

        smooth_over_alias_features!(u00_truncated,RR_truncated,R_issue_min,R_issue_max)
    end


    return u00_truncated
end