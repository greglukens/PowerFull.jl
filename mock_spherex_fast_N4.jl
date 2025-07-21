using Dierckx
using DelimitedFiles
using HDF5 
using Base.Threads
using LegendrePolynomials

include("./src/grid_initialization.jl")
include("./src/wljj_tools.jl")

ellcut_rs = ["0","10","20","50","100"]
ellcut = ellcut_rs[1]

fnl_fid = 0.1
ns_fid = 0.9665
As_fid = 2.105209e-9
delta_ns = 0.0005
delta_as = 0.0005

Q = round.(vcat(LinRange(0,0.25,26),LinRange(0.26,1,38),LinRange(1.02,2,15)),digits = 2)

ellmax = 500

Nbins = 11

num_params = 4
params = ["fnl","ns","nrun","As"]
#zmin_id = 2
ellmax = 500
sample_num = 4

path_root = "/storage/home/gql5196/scratch"
infile_root = "Cl_NR9001_full_GR_lmax$(ellmax)_sample$(sample_num)"

data = h5open("$(path_root)/spherex_params_new.h5","r")
ztest = data["ztest"][]
zmid = data["zmid"][]
sigmaz = data["sigmaz_$(sample_num)"][]
shot_noise_i = data["shot_noise_$(sample_num)"][]
sel_func_normed = data["sel_func_$(sample_num)_norm"][]
lcuts = data["lcut_$(ellcut)"][]
close(data)


data_gr1 = h5open("$(path_root)/$(infile_root)_Q1.h5","r")

ell = data_gr1["l"][]
r = data_gr1["r"][]
z = data_gr1["z"][]

sel_func = Spline1D(ztest,sel_func_normed,s=0)(z)


shot_noise_l = Diagonal(shot_noise_i)
shot_noise_matrix = zeros(length(zmid),length(zmid),length(ell))

for i in 1:length(ell)
    @views shot_noise_matrix[:,:,i] .= shot_noise_l
end



function normalization_factor(f, z)
    f_spline_z = Spline1D(z, f, s=0)
    return 1 / integrate(f_spline_z)
end

# Precompute window functions
function precompute_windows(binned_z, z, sigmaz)
    Nbins = length(binned_z)
    window_matrix = zeros(Nbins, length(z))

    @inbounds for i in 1:Nbins
        window_matrix[i, :] .= exp.(-((z .- binned_z[i]).^2) ./ (2 * sigmaz[i]^2))
    end

    for i in 1:Nbins
        norm = integrate(Spline1D(z, window_matrix[i, :], s=0), z[1], z[end])
        window_matrix[i, :] ./= norm
    end

    return window_matrix
end

# Optimized integral computation with parallelization
function integral_over_z_parallel(f, weights_i, weights_j, z)
    weighted_cl_i = zeros(length(z))
    Threads.@threads for j in eachindex(z)
        f_col = @view f[:, j]
        integrand_i = @views weights_i .* f_col 
        @views weighted_cl_i[j] = trapz_fast(z, integrand_i)
    end

    @views integrand_j = weights_j .* weighted_cl_i
    @views weighted_cl_ij = trapz_fast(z, integrand_j)
    return weighted_cl_ij
end

function compute_binned_cl(cl, z, binned_z, ellmax, selection_function, sigmaz; zmin=z[1], zmax=z[end])
    Nbins = length(binned_z)
    
    # Precompute windows and weights
    window_matrix = precompute_windows(binned_z, z, sigmaz)
    weights = zeros(Nbins, length(z))

    @inbounds for i in 1:Nbins
        window_i = @view(window_matrix[i, :])
        @views weights[i, :] .= window_i .* selection_function .* normalization_factor(window_i .* selection_function, z, zmin, zmax)
    end

    # Compute binned Cl in parallel
    binned_Cl = [integral_over_z_parallel(cl[:, :, m], weights[i, :], weights[j, :], z)
                 for i in 1:Nbins, j in 1:Nbins, m in 1:ellmax]

    return binned_Cl
end



function covariance_matrix(Cl, ellmin, ellmax, f_sky)
    num_ell = ellmax - ellmin + 1
    @assert num_ell <= size(Cl, 3) "Mismatch between ell range and Cl dimensions"
    num_r_bins = size(Cl, 1)
    @assert num_r_bins == size(Cl, 2) "Cl must be square in r bins"

    cov_matrix = zeros(Float64, num_r_bins^2, num_r_bins^2, num_ell)

    println("Computing covariance matrix...")
    @time begin
        for (ℓ_idx, ell) in enumerate(ellmin:ellmax)
            cov_block = zeros(Float64, num_r_bins^2, num_r_bins^2)
            for (a, b) in Iterators.product(1:num_r_bins, 1:num_r_bins)
                for (c, d) in Iterators.product(1:num_r_bins, 1:num_r_bins)
                    cov_block[(a-1)*num_r_bins + b, (c-1)*num_r_bins + d] = 
                        (Cl[a, c, ℓ_idx] * Cl[b, d, ℓ_idx] + Cl[a, d, ℓ_idx] * Cl[b, c, ℓ_idx]) / 
                        ((2 * ell + 1) * f_sky)
                end
            end
            cov_matrix[:, :, ℓ_idx] .= cov_block
        end
    end
    return cov_matrix
end



function compute_binned_cl_fast(cl, z, binned_z, ellmax, selection_function, sigmaz)
    Nbins = length(binned_z)
    Nz = length(z)

    # Precompute selection weights
    window_matrix = precompute_windows(binned_z, z, sigmaz)
    weights = zeros(Nbins, Nz)
    norm_factors = zeros(Nbins)

    @inbounds for i in 1:Nbins
        @views w = window_matrix[i, :] .* selection_function
        @views norm_factors[i] = trapz_fast(z,w)
        @views weights[i, :] .= w ./ norm_factors[i]
    end

    # Output array
    binned_Cl = Array{Float64}(undef, Nbins, Nbins, ellmax)

    @inbounds Threads.@threads for m in 1:ellmax
        cl_m = @view cl[:, :, m]

        for i in 1:Nbins
            wi = @view weights[i, :]
            weighted_cl_i = zeros(Nz)

            for jz in 1:Nz
                integrand = @views wi .* cl_m[:, jz]
                @views weighted_cl_i[jz] = trapz_fast(z, integrand)
            end

            for j in 1:Nbins
                wj = @view weights[j, :]
                integrand2 = @views wj .* weighted_cl_i
                @views binned_Cl[i, j, m] = trapz_fast(z, integrand2)
            end
        end
    end

    return binned_Cl
end

function symmetrize_matrix(matrix::Array{Float64, 3})
    # Get the size of the 3D matrix
    n = size(matrix, 1)  # Number of rows (and columns, since it's square)
    N_ell = size(matrix, 3)  # Number of ell modes (third dimension)

    # Loop over each slice for each ell
    @inbounds for ell_idx in 1:N_ell
        @views matrix[:, :, ell_idx] .= 0.5 * (matrix[:, :, ell_idx] .+ matrix[:, :, ell_idx]')
    end

    return matrix
end



function compute_fisher_matrix(C_ell, Cov, dC_dparams, ellmin, ellmax, f_sky, delta_l, epsilon)
    num_r = size(C_ell, 1)
    num_params = size(dC_dparams, 4)
    fisher_matrix_per_thread = [zeros(Float64, num_params, num_params) for _ in 1:Threads.nthreads()]
    
    @inbounds Threads.@threads for ell in ellmin:ellmax
        ℓ_idx = ell - ellmin + 1
        tid = Threads.threadid()
        cov_regularized = Symmetric(Cov[:, :, ℓ_idx]) .+ epsilon * mean(Cov[:,:, ℓ_idx]) * Matrix(I, num_r^2, num_r^2)
        @views Cov_inv = inv(cov_regularized)
        
        # Flatten all derivatives at this ell
        @views dC_vecs = [reshape(dC_dparams[:, :, ℓ_idx, p], num_r^2) for p in 1:num_params]
        
        # Compute all entries (i,j) of the Fisher matrix
        for i in 1:num_params
            for j in i:num_params
                @views fisher_ij = dC_vecs[i]' * Cov_inv * dC_vecs[j]
                @views fisher_matrix_per_thread[tid][i,j] += (2 * ell + 1) * f_sky * delta_l * fisher_ij
                if i != j
                    @views fisher_matrix_per_thread[tid][j,i] += (2 * ell + 1) * f_sky * delta_l * fisher_ij
                end
            end
        end
    end

    fisher_matrix = reduce(+, fisher_matrix_per_thread)
    return fisher_matrix
end


function compute_bias(fiducial_C, model_C, dC_dparams, fisher_matrix, ellmin, ellmax, f_sky, delta_l)
    num_r = size(fiducial_C, 1)
    num_params = size(dC_dparams, 4)
    
    delta_C = model_C .- fiducial_C
    bias_per_thread = [zeros(Float64, num_params) for _ in 1:Threads.nthreads()]

    @inbounds Threads.@threads for ell in ellmin:ellmax
        ℓ_idx = ell - ellmin + 1
        tid = Threads.threadid()
        
        # Flatten the arrays
        @views delta_C_i = reshape(delta_C[:,:,ℓ_idx], num_r^2)
        @views fiducial_C_i = reshape(fiducial_C[:,:,ℓ_idx], num_r^2)
        
        delta_C_over_C2 = delta_C_i ./ (fiducial_C_i.^2)
        
        for p in 1:num_params
            @views dC_dparam = reshape(dC_dparams[:, :, ℓ_idx, p], num_r^2)
            @views fisher_term = delta_C_over_C2' * dC_dparam
            @views bias_per_thread[tid][p] += (2 * ell + 1) * f_sky * delta_l * fisher_term
        end
    end

    total_bias = reduce(+, bias_per_thread)
    
    # Normalize: bias = Fisher^{-1} × (bias vector)
    bias_vector = fisher_matrix \ total_bias
    
    return bias_vector
end


function fisher_brute_force(cl,dcl_dparams,ellmin,ellmax, delta_l, f_sky)
    num_params = size(dcl_dparams,4)

    fisher_matrix_per_thread = [zeros(Float64, num_params, num_params) for _ in 1:Threads.nthreads()]

    cl2 = cl .^2
    @Threads.threads for ell = ellmin:ellmax
        tid = Threads.threadid()
        @views dcl_lij = dcl_dparams[:,:,ell,:]
        @views cl_l2 = cl2[:,:,ell]

        @inbounds for i = 1:num_params
            @views dlncl2_li = dcl_lij[:,:,i] ./ cl_l2
            for j = 1:num_params
                @views dcl2_lj = dcl_lij[:,:,j]
                @views traced_quantity = dcl2_lj[:,:,j] .* dlncl2_li
                @views fisher_matrix_per_thread[tid][i,j] += (2*ell + 1)/2 * f_sky * delta_l * tr(traced_quantity) 
            end
        end
    end

    fisher_matrix = reduce(+, fisher_matrix_per_thread)
    return fisher_matrix
end




function bias_brute_force(cl,test_cl, dcl_dparams,fisher,ellmin,ellmax, delta_l, f_sky)
    num_params = size(dcl_dparams,4)

    delta_cl = test_cl .- cl
    bias_per_thread = [zeros(Float64, num_params) for _ in 1:Threads.nthreads()]

        
    delta_cl_over_cl2 = delta_cl ./ cl .^2

    @Threads.threads for ell = ellmin:ellmax
        tid = Threads.threadid()
        @views delta_cl_over_cl2_l = delta_cl_over_cl2[:,:,ell]
        @views dcl_l = dcl_dparams[:,:,ell,:]

        @inbounds for i = 1:num_params
            @views traced_quantity = delta_cl_over_cl2_l .* dcl_dparams[:,:,i]
            @views bias_per_thread[tid][i] += (2*ell + 1)/2 * f_sky * delta_l * tr(traced_quantity) 
        end
    end

    bias = reduce(+, bias_per_thread)
    return fisher \ bias
end

function fisher_brute_force_full(cl,dcl_dparams,ellmin,ellmax,ellcuts,delta_l, f_sky)
    num_params = size(dcl_dparams,4)

    fisher_matrix_per_thread = [zeros(Float64, num_params, num_params) for _ in 1:Threads.nthreads()]

    @Threads.threads for ell = ellmin:ellmax
        bins_used = findall(ellcuts .> ell)
        tid = Threads.threadid()
        @views cl_inv = inv(Matrix(cl[bins_used,bins_used,ell]))
        @views dcl_lij = dcl_dparams[bins_used,bins_used,ell,:]

        @inbounds for i = 1:num_params
            @views dcl_i = dcl_lij[:,:,i]
            @views cl_inv_times_dcl_i = cl_inv * dcl_i
            for j = 1:num_params
                @views dcl_j = dcl_lij[:,:,j]
                @views cl_inv_times_dcl_j = cl_inv * dcl_j
                @views traced_quantity = cl_inv_times_dcl_i * cl_inv_times_dcl_j
                @views fisher_matrix_per_thread[tid][i,j] += (2*ell + 1)/2 * f_sky * delta_l * tr(traced_quantity) 
            end
        end
    end

    fisher_matrix = reduce(+, fisher_matrix_per_thread)
    return fisher_matrix
end




function bias_brute_force_full(cl,test_cl, dcl_dparams,fisher,ellmin,ellmax,ellcuts, delta_l, f_sky)
    num_params = size(dcl_dparams,4)

    delta_cl = test_cl .- cl
    bias_per_thread = [zeros(Float64, num_params) for _ in 1:Threads.nthreads()]


    @Threads.threads for ell = ellmin:ellmax
        bins_used = findall(ellcuts .> ell)
        tid = Threads.threadid()
        @views cl_inv = inv(Matrix(cl[bins_used,bins_used,ell]))
        @views dcl_l = dcl_dparams[bins_used,bins_used,ell,:]
        @views delta_cl_l = delta_cl[bins_used,bins_used,ell]
        @views delta_cl_times_cl_inv_l = delta_cl_l * cl_inv

        @inbounds for i = 1:num_params
            @views dcl_i = dcl_l[:,:,i]
            @views dcl_i_times_cl_inv_l = dcl_i * cl_inv
            @views traced_quantity = dcl_i_times_cl_inv_l * delta_cl_times_cl_inv_l
            @views bias_per_thread[tid][i] += (2*ell + 1)/2 * f_sky * delta_l * tr(traced_quantity) 
        end
    end

    bias = reduce(+, bias_per_thread)
    return fisher \ bias
end





outpath = "$(path_root)/spherex_Cl_Nbins$(Nbins)_lmax$(ellmax)_sample$(sample_num)_NQ$(length(Q))_Npar$(num_params)_lcut$(ellcut).h5"

@time begin
    println("Unpacking and rebinning all Q-dependent quantities: ")

    data_ns1 = h5open("$(path_root)/$(infile_root)_ns_fine_Q1.h5","r")
    data_as1 = h5open("$(path_root)/$(infile_root)_as_fine_Q1.h5","r")

    gr = zeros(length(Q),length(zmid),length(zmid),ellmax)
    der_fnl = zeros(length(Q),length(zmid),length(zmid),ellmax)
    der_ns = zeros(length(Q),length(zmid),length(zmid),ellmax)
    der_as = zeros(length(Q),length(zmid),length(zmid),ellmax)
    der_vec = zeros(length(Q),length(zmid),length(zmid),ellmax,num_params)

    @views cl_ns_p_ff = fnl_fid^2 * data_ns1["ff"][]
    @views cl_as_p_ff = fnl_fid^2 * data_as1["ff"][]
    @views cl_ff = fnl_fid^2 * data_gr1["ff"][]
    @views fnl_der_gr_ff = 2 * cl_ff / (fnl_fid)
    @views cl_kaiser = data_gr1["cl_kaiser"][] .+ fnl_fid * data_gr1["fi_k"][] .+ cl_ff
    @views cl_newtonian = data_gr1["cl_newtonian"][] .+ fnl_fid * data_gr1["fi_n"][] .+ cl_ff

    close(data_gr1)
    close(data_ns1)
    close(data_as1)

    @inbounds for i = 1:length(Q)
        @time begin
            println("Q = $(Q[i]): ")
            flush(stdout)
            data_gr = h5open("$(path_root)/$(infile_root)_Q$(i).h5","r")
            data_ns = h5open("$(path_root)/$(infile_root)_ns_fine_Q$(i).h5","r")
            data_as = h5open("$(path_root)/$(infile_root)_as_fine_Q$(i).h5","r")

            @views fnl_der_gr_fi = data_gr["fi"][] 
            @views cl_gr = data_gr["cl_gr"][] .+ fnl_fid * fnl_der_gr_fi

            @views cl_ns_p_fi = data_ns["cl_gr"][] .+ fnl_fid * data_ns["fi"][] 
            @views cl_as_p_fi = data_as["cl_gr"][] .+ fnl_fid * data_as["fi"][] 

            raw_ns_der = ((cl_ns_p_fi .+ cl_ns_p_ff) .- (cl_gr .+ cl_ff)) / delta_ns
            raw_as_der = ((cl_as_p_fi .+ cl_as_p_ff) .- (cl_gr .+ cl_ff)) / delta_as

            gr_q = compute_binned_cl_fast(cl_gr .+ cl_ff , z, zmid, ellmax, sel_func, sigmaz)
            @views gr[i,:,:,:] .= symmetrize_matrix(gr_q)
            
            fnl_der_q = compute_binned_cl_fast(fnl_der_gr_fi .+ fnl_der_gr_ff, z, zmid, ellmax, sel_func, sigmaz)
            @views der_fnl[i,:,:,:] .= symmetrize_matrix(fnl_der_q)

            ns_der_q = compute_binned_cl_fast(raw_ns_der , z, zmid, ellmax, sel_func, sigmaz)
            @views der_ns[i,:,:,:] .= symmetrize_matrix(ns_der_q)

            as_der_q = compute_binned_cl_fast(raw_as_der , z, zmid, ellmax, sel_func, sigmaz)
            @views der_as[i,:,:,:] .= symmetrize_matrix(as_der_q)

            der_As = gr_q / As_fid

            @views der_vec[i,:,:,:,1] .= der_fnl[i,:,:,:]
            @views der_vec[i,:,:,:,2] .= der_ns[i,:,:,:]
            @views der_vec[i,:,:,:,3] .= der_as[i,:,:,:]
            @views der_vec[i,:,:,:,4] .= der_As


            flush(stdout)

            close(data_gr)
            close(data_ns)
            close(data_as)

            GC.gc()
        end
    end
end


kaiser_g = @time compute_binned_cl_fast(cl_kaiser,z,zmid,ellmax,sel_func,sigmaz)
kaiser_g = symmetrize_matrix(kaiser_g)
flush(stdout)
GC.gc()

newtonian_g = @time compute_binned_cl_fast(cl_newtonian,z,zmid,ellmax,sel_func,sigmaz)
newtonian_g = symmetrize_matrix(newtonian_g)
flush(stdout)
GC.gc()

cl_kaiser = nothing
cl_newtonian = nothing
cl_gr = nothing # for memory cleanup
GC.gc()



ellmaxes = collect(2:500)

fisher_gr_marg = zeros(length(collect(2:500)),num_params,num_params,length(Q))

bias_newt_gr_marg = zeros(length(collect(2:500)),num_params,length(Q))
bias_kaiser_gr_marg = zeros(length(collect(2:500)),num_params,length(Q))

err_gr_marg = zeros(length(collect(2:500)),num_params,length(Q))
err_gr_nomarg = zeros(length(collect(2:500)),num_params,length(Q))

bias_newt_gr_nomarg = zeros(length(collect(2:500)),num_params,length(Q))
bias_kaiser_gr_nomarg = zeros(length(collect(2:500)),num_params,length(Q))

fiducial_newt = newtonian_g .+ shot_noise_matrix
fiducial_kaiser = kaiser_g  .+ shot_noise_matrix



@inbounds for q=1:length(Q)
    @time begin
        println("Q = $(Q[q]): ")

        @views gr_q = gr[q,:,:,:]
        fiducial_q = gr_q .+ shot_noise_matrix
        @views der_vec_q = der_vec[q,:,:,:,:]

    
        @inbounds for l = 7:500
        
            #print("ell = $l:")
            
            fisher_marg_q_l = fisher_brute_force_full(fiducial_q,der_vec_q,2,l,lcuts,1,1)
            @views fisher_gr_marg[l-1,:,:,q] = fisher_marg_q_l
            @views err_gr_marg[l-1,:,q] = sqrt.(abs.(diag(inv(fisher_marg_q_l))))

            @views fisher_fnl_q_l = fisher_brute_force_full(fiducial_q,der_vec_q[:,:,:,1],2,l,lcuts,1,1)[1]
            @views fisher_ns_q_l = fisher_brute_force_full(fiducial_q,der_vec_q[:,:,:,2],2,l,lcuts,1,1)[1]
            @views fisher_as_q_l = fisher_brute_force_full(fiducial_q,der_vec_q[:,:,:,3],2,l,lcuts,1,1)[1]
            @views fisher_As_q_l = fisher_brute_force_full(fiducial_q,der_vec_q[:,:,:,4],2,l,lcuts,1,1)[1]

            @views err_gr_nomarg[l-1,1,q] = 1/sqrt.(abs.(fisher_fnl_q_l))
            @views err_gr_nomarg[l-1,2,q] = 1/sqrt.(abs.(fisher_ns_q_l))
            @views err_gr_nomarg[l-1,3,q] = 1/sqrt.(abs.(fisher_as_q_l))
            @views err_gr_nomarg[l-1,4,q] = 1/sqrt.(abs.(fisher_As_q_l))

            @views bias_newt_gr_marg[l-1,:,q] = bias_brute_force_full(fiducial_q,fiducial_newt,der_vec_q,fisher_marg_q_l,2,l,lcuts,1,1)
            @views bias_kaiser_gr_marg[l-1,:,q] = bias_brute_force_full(fiducial_q,fiducial_kaiser,der_vec_q,fisher_marg_q_l,2,l,lcuts,1,1)

            @views bias_newt_gr_nomarg[l-1,1,q] = bias_brute_force_full(fiducial_q,fiducial_newt,der_vec_q[:,:,:,1],fisher_fnl_q_l,2,l,lcuts,1,1)[1]
            @views bias_newt_gr_nomarg[l-1,2,q] = bias_brute_force_full(fiducial_q,fiducial_newt,der_vec_q[:,:,:,2],fisher_ns_q_l,2,l,lcuts,1,1)[1]
            @views bias_newt_gr_nomarg[l-1,3,q] = bias_brute_force_full(fiducial_q,fiducial_newt,der_vec_q[:,:,:,3],fisher_as_q_l,2,l,lcuts,1,1)[1]
            @views bias_newt_gr_nomarg[l-1,4,q] = bias_brute_force_full(fiducial_q,fiducial_newt,der_vec_q[:,:,:,4],fisher_As_q_l,2,l,lcuts,1,1)[1]


            @views bias_kaiser_gr_nomarg[l-1,1,q] = bias_brute_force_full(fiducial_q,fiducial_kaiser,der_vec_q[:,:,:,1],fisher_fnl_q_l,2,l,lcuts,1,1)[1]
            @views bias_kaiser_gr_nomarg[l-1,2,q] = bias_brute_force_full(fiducial_q,fiducial_kaiser,der_vec_q[:,:,:,2],fisher_ns_q_l,2,l,lcuts,1,1)[1]
            @views bias_kaiser_gr_nomarg[l-1,3,q] = bias_brute_force_full(fiducial_q,fiducial_kaiser,der_vec_q[:,:,:,3],fisher_as_q_l,2,l,lcuts,1,1)[1]
            @views bias_kaiser_gr_nomarg[l-1,4,q] = bias_brute_force_full(fiducial_q,fiducial_kaiser,der_vec_q[:,:,:,4],fisher_As_q_l,2,l,lcuts,1,1)[1]

        end

        GC.gc()
    end
end

println("==========================================================================================")
println("Storing result at $(outpath)...")
@time begin
    h5open(outpath,"w") do outfile 
        outfile["l"] = collect(1:ellmax)
        outfile["z"] = [z;]
        outfile["zmid"] = [zmid;]
        outfile["lcut"] = lcuts
        outfile["sigmaz"] = [sigmaz;]
        outfile["sample_num"] = [sample_num;]
        outfile["sel_func"] = [sel_func;]
        outfile["Q"] = Q

        outfile["kaiser"] = kaiser_g
        outfile["newtonian"] = newtonian_g
        outfile["gr"] = gr

        outfile["der_vec"] = der_vec
        outfile["shot_noise"] = shot_noise_matrix

        outfile["err_gr_marg"] = err_gr_marg
        outfile["err_gr_nomarg"] = err_gr_nomarg

        outfile["fisher_gr_marg"] = fisher_gr_marg

        outfile["bias_newt_gr_marg"] = bias_newt_gr_marg
        outfile["bias_kaiser_gr_marg"] = bias_kaiser_gr_marg

        outfile["bias_newt_gr_nomarg"] = bias_newt_gr_nomarg
        outfile["bias_kaiser_gr_nomarg"] = bias_kaiser_gr_nomarg
    end
end
