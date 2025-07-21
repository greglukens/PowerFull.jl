include("./src/grid_initialization.jl")
include("./src/PkSpectra.jl")
include("./src/wljj_tools.jl")

using TwoFAST
using Dierckx
using DelimitedFiles
using HDF5 
using Suppressor
using Statistics
using Base.Threads
using .PkSpectra

Nr = 4096
NR = 5001

ellmax = 500

zmax = 5
zmin = 0.01

kmax = 1e4
kmin = 1e-5

r_slicing = 1
R_slicing = 1

pk_path = "/storage/home/gql5196/work/tamred/data/power_spec/planck_2018_cosmology_power_hires_as_fine.dat"
cachepath = "/storage/home/gql5196/scratch/caches/Nr$(Nr)NR$(NR)lmax1000"
outpath = "/storage/home/gql5196/scratch/Cl_full_Nr$(Nr)NR$(NR)lmax$(ellmax)_as_fine.h5"

println("Initializing cosmology: ")
@time begin
    b = 2 
    omega_m = 0.3111
    omega_l = 1-omega_m
    omega_r = 0
    w = -1
    cosmology = initialize_cosmology(omega_m,omega_l,omega_r,w,b)
    z_of_r = cosmology["z_of_r"]
    r_of_z = cosmology["r_of_z"]
end
flush(stdout)



function calculate_wljj(ellmax,kmax,kmin,zmax,zmin,Nr,pk_path::String,cachepath::String,cosmology;RSD::Bool=true,doppler::Bool=true,GR::Bool=true,lensing::Bool=true,png::Bool=true,q_vals::Array=[1.1,1.1,0.5,0.1,-0.1],r_slicing=1,R_slicing=1)
    #@assert length(q_vals) == 5
    r_of_z = cosmology["r_of_z"]
    D_of_z = cosmology["D_of_z"]
    H_of_z = cosmology["H_of_z"]
    Om_of_z = cosmology["omega_m_of_z"]
    f_of_z = cosmology["f_of_z"]
    z_of_r = cosmology["z_of_r"]

    println("=========================================================================================================")
    println("Computing wljj'(r,rR) with the following settings:")
    pvals = determine_scenario(RSD,doppler,GR,lensing,png;qvals=q_vals)
    println("=========================================================================================================")
    @time begin
        print("Loading in power spectrum files:")
        pk = PkSpectrum(pk_path)

        if -1 in pvals
            pk_neg1 = Spline1D(pk.kk, pk.pk ./ pk.kk)
        end
        if -2 in pvals
            pk_neg2 = Spline1D(pk.kk, pk.pk ./ pk.kk .^2)
        end
        if -3 in pvals
            pk_neg3 = Spline1D(pk.kk, pk.pk ./ pk.kk .^3)
        end
        if -4 in pvals
            pk_neg4 = Spline1D(pk.kk, pk.pk ./ pk.kk .^4)
        end
        if png
            pk_fnl_xt_neg2 = Spline1D(pk.kk, PNG.(pk.kk) .* pk.pk)
            pk_fnl_xt_neg4 = Spline1D(pk.kk, PNG.(pk.kk) .* pk.pk ./ pk.kk .^2)
            pk_fnl_ft_neg4 = Spline1D(pk.kk, PNG.(pk.kk) .^2 .* pk.pk)
        end
    end
    flush(stdout)
    
    println("Cache files already exist.")
    cname_base = "$(cachepath)"
 
    
    @time begin
        print("Initializing (r,R) grid:")
        rr_cache = [readdlm("$(cname_base)/pk_neg4/MlCache/rr.tsv",'\t')[:,1];]
        rr_max_id = Int(minimum(findall(rr_cache .>= r_of_z(zmax))))
        rr_min_id = Int(maximum(findall(rr_cache .<= r_of_z(zmin))))
        rr = rr_cache[rr_min_id:rr_max_id]
        RR = [readdlm("$(cname_base)/pk_neg4/MlCache/RRatio.tsv",'\t')[:,1];]

        NR = length(RR)

        Rmin = minimum(RR)

        Nr_new = Int(length(rr[1:r_slicing:end]))
        NR_new = length(RR[R_slicing:R_slicing:end-R_slicing+1])
        
        rr_trunc_low_id = minimum(findall(rr_cache * Rmin .>= 1/kmax))

        r_calc_ids = collect(1:length(rr))[1:r_slicing:end] .+ (rr_min_id - 1)

        if R_slicing == 1
            R_calc_ids = collect(1:length(RR))
        else
            R_calc_ids = collect(1:length(RR))[R_slicing:R_slicing:end-R_slicing+1]
        end


        RR_truncated = RR[1:maximum(R_calc_ids)]

        rr_truncated = @views rr_cache[rr_trunc_low_id:rr_max_id]

        int_prep = integration_prep(rr_truncated, RR_truncated, z_of_r, D_of_z, H_of_z, Om_of_z, f_of_z)
    end

    flush(stdout)

    function alloc_dset(file, name, dims)
        create_dataset(file, name, Float64, dims)
    end

    zz = z_of_r.(rr)

    dims = (Nr_new, Nr_new, ellmax)
    
    h5open(outpath, "w") do h5file
        @time begin
            print("Initializing HDF5 file at $(outpath):")

            h5file["l"] = collect(1:ellmax)
            h5file["rr"] = [rr[1:r_slicing:end];]
            h5file["RR"] = [RR[R_slicing:R_slicing:end-R_slicing+1];]
            h5file["z"] = [zz[1:r_slicing:end];]

            # Preallocate all datasets
            fields = [
                "w00_0", "w20_0", "w02_0", "w22_0",
                "w00_neg2", "w20_neg2", "w02_neg2", "w00_neg4",
                "w10_neg1", "w01_neg1", "w12_neg1", "w21_neg1",
                "w10_neg3", "w01_neg3", "w11_neg2",
                "l00_neg2", "s00_neg2", "t00_neg2",
                "l02_neg2", "s02_neg2", "t02_neg2",
                "l00_neg4", "s00_neg4", "t00_neg4",
                "l01_neg3", "s01_neg3", "t01_neg3",
                "lfnl00_neg4", "sfnl00_neg4", "tfnl00_neg4",
                "u00_neg4", "u10_neg3", "u01_neg3", "v00_neg4",
                "u00_neg2", "u20_neg2", "u02_neg2", "L00_neg4", "S00_neg4", "T00_neg4", "X00_neg4", "Y00_neg4", "Z00_neg4"
            ]

            for name in fields
                alloc_dset(h5file, name, dims)
            end
        end
        println("===============================================================")
        println("")
        println("Preallocation complete. Starting loop over ell from 1 to $(ellmax).")

        @inbounds for l = 1:ellmax
            
            @time begin
                println("")
                println("===============================================================")
                println("")
                println("ell = $(l)")
                println("")

                @time begin
                    print("RSD terms:")
                    w00_0, w02_0, w20_0, w22_0 = @suppress_out RSD_terms_f(pk,l,RR,1.1, kmax, kmin, zmax, zmin, Nr, "$(cachepath)/pk_0", cosmology; r_calc_ids = r_calc_ids, R_calc_ids = R_calc_ids)
                end
                flush(stdout)
                @time begin
                    print("Integrated terms:")
                    w00_neg2, w02_neg2, w20_neg2, w01_neg1, w10_neg1, w12_neg1, w21_neg1, l00_neg2, s00_neg2, t00_neg2, l02_neg2, s02_neg2, t02_neg2 = @suppress_out lst_02_integrals_f(pk_neg2,l,RR,0.5, kmax, kmin, zmax, zmin, Nr, "$(cachepath)/pk_neg2", cosmology, int_prep; r_calc_ids = r_calc_ids, R_calc_ids = R_calc_ids)
                    w00_neg4, w01_neg3, w10_neg3, w11_neg2, l00_neg4, s00_neg4, t00_neg4, l01_neg3, s01_neg3, t01_neg3, L00_neg4, S00_neg4, T00_neg4, X00_neg4, Y00_neg4, Z00_neg4 = @suppress_out lst_LST_XYZ_integrals_f(pk_neg4,l,RR,-1.26, kmax, kmin, zmax, zmin, Nr, "$(cachepath)/pk_neg4", cosmology, int_prep; r_calc_ids = r_calc_ids, R_calc_ids = R_calc_ids)
                end
                flush(stdout)
                @time begin 
                    print("PNG terms:")
                    u00_neg2, u02_neg2, u20_neg2 = @suppress_out fnl_02_terms_f(pk_fnl_xt_neg2,l,RR,0.5, kmax, kmin, zmax, zmin, Nr, "$(cachepath)/pk_fnl_xt_neg2", cosmology; r_calc_ids = r_calc_ids, R_calc_ids = R_calc_ids)           
                    u00_neg4, u01_neg3, u10_neg3, lfnl00_neg4, sfnl00_neg4, tfnl00_neg4 = @suppress_out fnl_01_integrals_f(pk_fnl_xt_neg4,l,RR,-0.9, kmax, kmin, zmax, zmin, Nr, "$(cachepath)/pk_fnl_xt_neg4", cosmology, int_prep; r_calc_ids = r_calc_ids, R_calc_ids = R_calc_ids)
                    v00_neg4 = @suppress_out fnl_auto_term_f(pk_fnl_ft_neg4,l,RR,0.3, kmax, kmin, zmax, zmin, Nr, "$(cachepath)/pk_fnl_ft_neg4", cosmology; r_calc_ids = r_calc_ids, R_calc_ids = R_calc_ids)
                end
                flush(stdout)
                @time begin
                    print("Writing to file:")
                    # Write everything for this ℓ directly to the HDF5 file
                    
                    @views h5file["w00_0"][:,:,l] = w00_0
                    @views h5file["w02_0"][:,:,l] = w02_0
                    @views h5file["w20_0"][:,:,l] = w20_0
                    @views h5file["w22_0"][:,:,l] = w22_0

                    @views h5file["w00_neg2"][:,:,l] = w00_neg2
                    @views h5file["w02_neg2"][:,:,l] = w02_neg2
                    @views h5file["w20_neg2"][:,:,l] = w20_neg2
                    @views h5file["w00_neg4"][:,:,l] = w00_neg4

                    @views h5file["w10_neg1"][:,:,l] = w10_neg1
                    @views h5file["w01_neg1"][:,:,l] = w01_neg1
                    @views h5file["w12_neg1"][:,:,l] = w12_neg1
                    @views h5file["w21_neg1"][:,:,l] = w21_neg1

                    @views h5file["w10_neg3"][:,:,l] = w10_neg3
                    @views h5file["w01_neg3"][:,:,l] = w01_neg3
                    @views h5file["w11_neg2"][:,:,l] = w11_neg2

                    @views h5file["l00_neg2"][:,:,l] = l00_neg2
                    @views h5file["s00_neg2"][:,:,l] = s00_neg2
                    @views h5file["t00_neg2"][:,:,l] = t00_neg2

                    @views h5file["l02_neg2"][:,:,l] = l02_neg2
                    @views h5file["s02_neg2"][:,:,l] = s02_neg2
                    @views h5file["t02_neg2"][:,:,l] = t02_neg2

                    @views h5file["l00_neg4"][:,:,l] = l00_neg4
                    @views h5file["s00_neg4"][:,:,l] = s00_neg4
                    @views h5file["t00_neg4"][:,:,l] = t00_neg4

                    @views h5file["l01_neg3"][:,:,l] = l01_neg3
                    @views h5file["s01_neg3"][:,:,l] = s01_neg3
                    @views h5file["t01_neg3"][:,:,l] = t01_neg3

                    @views h5file["lfnl00_neg4"][:,:,l] = lfnl00_neg4
                    @views h5file["sfnl00_neg4"][:,:,l] = sfnl00_neg4
                    @views h5file["tfnl00_neg4"][:,:,l] = tfnl00_neg4

                    @views h5file["u00_neg4"][:,:,l] = u00_neg4
                    @views h5file["u10_neg3"][:,:,l] = u10_neg3
                    @views h5file["u01_neg3"][:,:,l] = u01_neg3

                    @views h5file["v00_neg4"][:,:,l] = v00_neg4

                    @views h5file["u00_neg2"][:,:,l] = u00_neg2
                    @views h5file["u20_neg2"][:,:,l] = u20_neg2
                    @views h5file["u02_neg2"][:,:,l] = u02_neg2

                    @views h5file["L00_neg4"][:,:,l] = L00_neg4
                    @views h5file["S00_neg4"][:,:,l] = S00_neg4
                    @views h5file["T00_neg4"][:,:,l] = T00_neg4

                    @views h5file["X00_neg4"][:,:,l] = X00_neg4
                    @views h5file["Y00_neg4"][:,:,l] = Y00_neg4
                    @views h5file["Z00_neg4"][:,:,l] = Z00_neg4

                    flush(h5file)  # optional, ensures periodic disk writes
                end

                print("Total time elapsed:")
                flush(stdout)   
            end
        end
    end
end


calculate_wljj(ellmax,kmax,kmin,zmax,zmin,Nr,pk_path,cachepath,cosmology;RSD=true,doppler=true,GR=false,lensing=false,png=true, r_slicing = r_slicing, R_slicing = R_slicing)



