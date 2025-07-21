include("./src/grid_initialization.jl")
include("./src/PkSpectra.jl")
include("./src/wljj_tools.jl")

using HDF5, Dierckx, Base.Threads


create_file = true 

ellmax = 500
chunk_size = 250
ell_chunks = [(i:min(i+chunk_size-1, ellmax)) for i in 1:chunk_size:ellmax]

H0 = 67.66

sample_num = 4

omega_m = 0.3111
omega_l = 1 - omega_m
omega_r = 0
w = -1
delta_c = 1.686

Q_test = round.(vcat(LinRange(0,0.25,26),LinRange(0.26,1,38),LinRange(1.02,2,15)),digits = 2)

Nq = length(Q_test)

path_root = "/storage/home/gql5196/scratch"
inpath = "$(path_root)/Cl_full_Nr4096NR5001lmax500_f.h5"
outpath_root = "$(path_root)/Cl_full_GR_lmax$(ellmax)_sample$(sample_num)_large"

@time begin
    print("Initializing cosmology:")
    b_file = h5open("$(path_root)/spherex_params_new.h5","r")
    b_g_spl = Spline1D(b_file["ztest"][],b_file["b_$(sample_num)"][],s=0)
    close(b_file)    



    cosmology = initialize_cosmology(omega_m, omega_l, omega_r, w, b_g_spl)
    D_of_z = cosmology["D_of_z"]
    z_of_r = cosmology["z_of_r"]
    f_of_z = cosmology["f_of_z"]
    r_of_z = cosmology["r_of_z"]
    alpha1_of_z = cosmology["α1_of_z"]
    alpha2_of_z = cosmology["α2_of_z"]
    alpha3_of_z = cosmology["α3_of_z"]
    omega_m_of_z = cosmology["omega_m_of_z"]
    H_of_z = cosmology["H_of_z"]
end
flush(stdout)

@time begin
    print("Setting up grid:")
    data = h5open("$(inpath)", "r")
    r = data["rr"][:]
    z = data["z"][:]
    close(data)

    b_g = b_g_spl.(z)
    z = z_of_r.(r)
    Nr = length(r)

    b_e_eq = delta_c * (f_of_z.(z)) .* (b_g .- 1)

    #h = H0 / 100

    H_z = H_of_z.(z)
    Omega_m_z = omega_m_of_z.(z)
    f_z = f_of_z.(z)
    D_z = D_of_z.(z)

    a = 1 ./ (1 .+ z)

    b_phi = 2 * (b_g .- 1) * delta_c
    OmH2 = omega_m * H0^2 / (9e10)
end
flush(stdout)

l = collect(1:ellmax)

function alloc_dset(file, name, dims)
    create_dataset(file, name, Float64, dims)
end

dims_3D   = (Nr, Nr, ellmax)     # for gg, ff, fi_n, fi_k, cl_kaiser, cl_newtonian


GC.gc()

gg_array = zeros(Nr,Nr,chunk_size)
cl_kaiser_array = similar(gg_array)
cl_newtonian_array = similar(gg_array)
fi_n_array = similar(gg_array)
fi_k_array = similar(gg_array)

ff_array = similar(gg_array)
cl_gr_array = zeros(Nr,Nr,chunk_size)
fi_array = similar(cl_gr_array)

println("==========================================================================================")
println("Calculating C_ell's...")


for ell_range in ell_chunks
    @time begin
    print("Processing ell = $(first(ell_range)) to $(last(ell_range)):")
    data_i = h5open("$(inpath)", "r")
    @views begin
        w00_0 = data_i["w00_0"][:,:,ell_range]
        w20_0 = data_i["w20_0"][:,:,ell_range]
        w02_0 = data_i["w02_0"][:,:,ell_range]
        w22_0 = data_i["w22_0"][:,:,ell_range]
        w10_neg1 = data_i["w10_neg1"][:,:,ell_range]
        w01_neg1 = data_i["w01_neg1"][:,:,ell_range]
        w11_neg2 = data_i["w11_neg2"][:,:,ell_range]
        w21_neg1 = data_i["w21_neg1"][:,:,ell_range]
        w12_neg1 = data_i["w12_neg1"][:,:,ell_range]

        w00_neg2 = data_i["w00_neg2"][:,:,ell_range]
        w20_neg2 = data_i["w20_neg2"][:,:,ell_range]
        w02_neg2 = data_i["w02_neg2"][:,:,ell_range]
        w10_neg3 = data_i["w10_neg3"][:,:,ell_range]
        w01_neg3 = data_i["w01_neg3"][:,:,ell_range]
        w00_neg4  = data_i["w00_neg4"][:,:,ell_range]

        s00_neg2_r1 = data_i["s00_neg2"][:,:,ell_range]
        s00_neg2_r2 = permutedims(s00_neg2_r1, (2,1,3))

        l00_neg2_r1 = data_i["l00_neg2"][:,:,ell_range]
        l00_neg2_r2 = permutedims(l00_neg2_r1, (2,1,3))

        t00_neg2_r1 = data_i["t00_neg2"][:,:,ell_range]
        t00_neg2_r2 = permutedims(t00_neg2_r1, (2,1,3))

        l02_neg2_r1 = data_i["l02_neg2"][:,:,ell_range]
        l20_neg2_r2 = permutedims(l02_neg2_r1, (2,1,3))

        t02_neg2_r1 = data_i["t02_neg2"][:,:,ell_range]
        t20_neg2_r2 = permutedims(t02_neg2_r1, (2,1,3))

        s02_neg2_r1 = data_i["s02_neg2"][:,:,ell_range]
        s20_neg2_r2 = permutedims(s02_neg2_r1, (2,1,3))

        l01_neg3_r1 = data_i["l01_neg3"][:,:,ell_range]
        l10_neg3_r2 = permutedims(l01_neg3_r1, (2,1,3))

        t01_neg3_r1 = data_i["t01_neg3"][:,:,ell_range]
        t10_neg3_r2 = permutedims(t01_neg3_r1, (2,1,3))

        s01_neg3_r1 = data_i["s01_neg3"][:,:,ell_range]
        s10_neg3_r2 = permutedims(s01_neg3_r1, (2,1,3))

        l00_neg4_r1 = data_i["l00_neg4"][:,:,ell_range]
        l00_neg4_r2 = permutedims(l00_neg4_r1, (2,1,3))

        t00_neg4_r1 = data_i["t00_neg4"][:,:,ell_range]
        t00_neg4_r2 = permutedims(t00_neg4_r1, (2,1,3))

        s00_neg4_r1 = data_i["s00_neg4"][:,:,ell_range]
        s00_neg4_r2 = permutedims(s00_neg4_r1, (2,1,3))

        lfnl00_neg4_r1 = data_i["lfnl00_neg4"][:,:,ell_range]
        lfnl00_neg4_r2 = permutedims(lfnl00_neg4_r1, (2,1,3))

        tfnl00_neg4_r1 = data_i["tfnl00_neg4"][:,:,ell_range]
        tfnl00_neg4_r2 = permutedims(tfnl00_neg4_r1, (2,1,3))

        sfnl00_neg4_r1 = data_i["sfnl00_neg4"][:,:,ell_range]
        sfnl00_neg4_r2 = permutedims(sfnl00_neg4_r1, (2,1,3))

        u00_neg2 = data_i["u00_neg2"][:,:,ell_range]
        u00_neg4 = data_i["u00_neg4"][:,:,ell_range]
        u10_neg3 = data_i["u10_neg3"][:,:,ell_range]
        u01_neg3 = data_i["u01_neg3"][:,:,ell_range]
        u02_neg2 = data_i["u02_neg2"][:,:,ell_range]
        u20_neg2 = data_i["u20_neg2"][:,:,ell_range]
        v00_neg4 = data_i["v00_neg4"][:,:,ell_range]

        X00_neg4_r12 = data_i["X00_neg4"][:,:,ell_range]
        X00_neg4_r21 = permutedims(X00_neg4_r12, (2,1,3))

        Y00_neg4_r21 = data_i["Y00_neg4"][:,:,ell_range]
        Y00_neg4_r12 = permutedims(Y00_neg4_r21, (2,1,3))

        Z00_neg4_r21 = data_i["Z00_neg4"][:,:,ell_range]
        Z00_neg4_r12 = permutedims(Z00_neg4_r21, (2,1,3))

        S00_neg4 = data_i["S00_neg4"][:,:,ell_range]
        T00_neg4 = data_i["T00_neg4"][:,:,ell_range]
        L00_neg4 = data_i["L00_neg4"][:,:,ell_range]

    end
    end
    close(data_i)
    flush(stdout)

    
    @inbounds for q in 1:Nq
        outpath_q = "$(outpath_root)_Q$(q).h5"
        println("Working on Q = $(Q_test[q])...")
        
        if ell_range == 1:chunk_size
            h5open(outpath_q, "w") do h5file
                @time begin
                    print("Initializing output at $(outpath_q): ")
        
                    h5file["l"] = l
                    h5file["r"] = r
                    h5file["z"] = z
                    h5file["b_g"] = b_g
                    h5file["b_phi"] = b_phi
                    h5file["b_e"] = b_e_eq
                    h5file["D"] = D_z
                    h5file["f"] = f_z
                    h5file["H"] = H_z * H0
                    h5file["Om"] = Omega_m_z
                    h5file["a1"] = alpha1_of_z.(z)
                    h5file["a2"] = alpha2_of_z.(z)
                    h5file["a3"] = alpha3_of_z.(z)
                    h5file["Q"] = [Q_test[q];]
                    h5file["Q_total"] = [Q_test;]
            
                    for name in ("cl_gr", "fi", "gg", "ff", "fi_n", "fi_k", "cl_kaiser", "cl_newtonian")
                        create_dataset(h5file, name, Float64, dims_3D)
                    end
                end
        
            end
            flush(stdout)
        end

        @time begin
            print("Calculating Cl: ")
            @views Q = fill(Q_test[q], length(r))
            @views C = 3/2 * Omega_m_z .- 2 ./ (a .* H_z .* r / 3000) .* (1 .- Q) .- (2 .* Q)
            @views A = 3/2 * Omega_m_z .* (b_e_eq .* (1 .- 2 .* f_z ./ (3 * Omega_m_z)) .+ 1 .+ 2 .* f_z ./ Omega_m_z .+ C .- f_z .- (2 .* Q))
            @views B = f_z .* (b_e_eq .+ C .- 1)
    
            flush(stdout)

            @inbounds Threads.@threads for i in eachindex(r)
                @views r_i = r[i]
                @views a_i = a[i]
                @views H_i = H_z[i]
                @views b_i = b_g[i]
                @views z_x = z_of_r(r_i)
                @views alpha_x = alpha1_of_z(z_x) + alpha2_of_z(z_x)
                @views A_x = A[i]
                @views f_x = f_of_z(z_x)
                @views B_x = B[i]
                @views Bf_x = B[i] / f_x
                @views D_x = D_of_z(z_x)
                @views beta_x = f_x / b_i
                @views Q_x = Q[i]
                @views aH_x = a_i * H_i / 3000
                @views png_x = b_phi[i] * 3/2 * OmH2 / D_x

                for j in eachindex(r)
                    @views r_j = r[j]
                    @views a_j = a[j]
                    @views H_j = H_z[j]
                    @views b_j = b_g[j]
                    @views z_y = z_of_r(r_j)
                    @views alpha_y = alpha1_of_z(z_y) + alpha2_of_z(z_y)
                    @views A_y = A[j]
                    @views f_y = f_of_z(z_y)
                    @views B_y = B[j]
                    @views Bf_y = B[j] / f_y
                    @views D_y = D_of_z(z_y)
                    @views beta_y = f_y / b_j
                    @views Q_y = Q[j]
                    @views aH_y = a_j * H_j / 3000
                    @views png_y = b_phi[j] * 3/2 * OmH2 / D_y

                    @views bD_xy = b_i * b_j * D_x * D_y
                    @views D_xy = D_x * D_y
                    @views QQ_xy = (1 - Q_x) * (1 - Q_y)
                    @views b_xQ_y = b_i * (1 - Q_y) * D_xy
                    @views b_yQ_x = b_j * (1 - Q_x) * D_xy
                    
                    
                    @views begin
                        gg = bD_xy * w00_0[i,j,:]
                        rg = bD_xy * (- beta_x * w20_0[i,j,:])
                        gr = bD_xy * (- beta_y * w02_0[i,j,:])
                        rr = bD_xy * (beta_x * beta_y * w22_0[i,j,:])                    
                        
                        vg_n = bD_xy * - beta_x * alpha_x/r_i * w10_neg1[i,j,:]
                        gv_n = bD_xy * - beta_y * alpha_y/r_j * w01_neg1[i,j,:]

                        ################### bugged ##########
                        vr_n = bD_xy * beta_x*beta_y * alpha_x/r_i * w21_neg1[j,i,:]
                        rv_n = bD_xy * beta_x * beta_y * alpha_y/r_j * w21_neg1[i,j,:]

                        vv_n = bD_xy * beta_x * beta_y * alpha_x * alpha_y/(r_i*r_j) * w11_neg2[i,j,:]

                        gv = b_i * D_xy * aH_y * B_y * w01_neg1[i,j,:]
                        vg = b_j * D_xy * aH_x * B_x * w10_neg1[i,j,:]
                        rv = -D_xy * aH_y * f_x * B_y * w21_neg1[i,j,:]
                        vr = -D_xy * aH_x * f_y * B_x * w21_neg1[j,i,:]

                        ###################
                        vv = D_xy * aH_x * aH_y * B_x * B_y * w11_neg2[i,j,:]
                        gp = bD_xy * aH_y^2 / b_j * A_y * w00_neg2[i,j,:]
                        pg = bD_xy * aH_x^2 / b_i * A_x * w00_neg2[i,j,:]
                        rp = -bD_xy * aH_y^2 / b_j * A_y * beta_x * w20_neg2[i,j,:]
                        pr = -bD_xy * aH_x^2 / b_i * A_x * beta_y * w02_neg2[i,j,:]
                        vp = D_xy * aH_y^2 * aH_x * A_y * B_x * w10_neg3[i,j,:]
                        pv = D_xy * aH_x^2 * aH_y * A_x * B_y * w01_neg3[i,j,:]
                        pp = D_xy * aH_x^2 * aH_y^2 * A_x * A_y * w00_neg4[i,j,:]
                        sg = bD_xy / D_x  * Bf_x * s00_neg2_r1[i,j,:]
                        gs = bD_xy / D_y  * Bf_y * s00_neg2_r2[i,j,:]
                        sr = -bD_xy / D_x  * Bf_x * beta_y * s02_neg2_r1[i,j,:]
                        rs = -bD_xy / D_y  * Bf_y * beta_x * s20_neg2_r2[i,j,:]
                        vs = D_x * aH_x * B_x * Bf_y * b_j * s10_neg3_r2[i,j,:]
                        sv = D_y * aH_y * B_y * Bf_x * b_i * s01_neg3_r1[i,j,:]
                        ps = D_x * aH_x^2 * A_x * Bf_y * b_j * s00_neg4_r2[i,j,:]
                        sp = D_y * aH_y^2 * A_y * Bf_x * b_i * s00_neg4_r1[i,j,:]
                        gt = -2 * b_xQ_y / D_y  / r_j * t00_neg2_r2[i,j,:]
                        tg = -2 * b_yQ_x / D_x  / r_i * t00_neg2_r1[i,j,:]
                        rt = 2 * b_xQ_y / D_y  * beta_x / r_j * t20_neg2_r2[i,j,:]
                        tr = 2 * b_yQ_x / D_x  * beta_y / r_i * t02_neg2_r1[i,j,:]
                        gl = -2 * b_xQ_y / D_y  * l00_neg2_r2[i,j,:]
                        lg = -2 * b_yQ_x / D_x  * l00_neg2_r1[i,j,:]
                        rl = 2 * b_xQ_y / D_y  * beta_x * l20_neg2_r2[i,j,:]
                        lr = 2 * b_yQ_x / D_x  * beta_y * l02_neg2_r1[i,j,:]
                        vt = -2 * b_xQ_y / D_y  * aH_x * B_x / b_i / r_j * t10_neg3_r2[i,j,:]
                        tv = -2 * b_yQ_x / D_x  * aH_y * B_y / b_j / r_i * t01_neg3_r1[i,j,:]
                        vl = -2 * b_xQ_y / D_y  * aH_x * B_x / b_i * l10_neg3_r2[i,j,:]
                        lv = -2 * b_yQ_x / D_x  * aH_y * B_y / b_j * l01_neg3_r1[i,j,:]
                        pt = -2 * b_xQ_y / D_y  * aH_x^2 / b_i * A_x / r_j * t00_neg4_r2[i,j,:]
                        tp = -2 * b_yQ_x / D_x  * aH_y^2 / b_j * A_y / r_i * t00_neg4_r1[i,j,:]
                        pl = -2 * b_xQ_y / D_y  * aH_x^2 / b_i * A_x * l00_neg4_r2[i,j,:]
                        lp = -2 * b_yQ_x / D_x * aH_y^2 / b_j * A_y * l00_neg4_r1[i,j,:] 
                        st = -2 * Bf_x * (1 - Q_y) / r_j * X00_neg4_r12[i,j,:]
                        ts = -2 * Bf_y * (1 - Q_x) / r_i * X00_neg4_r21[i,j,:]
                        sl = -2 * Bf_x * (1 - Q_y) * Y00_neg4_r12[i,j,:]
                        ls = -2 * Bf_y * (1 - Q_x) * Y00_neg4_r21[i,j,:]
                        tl = 4 * QQ_xy / r_i * Z00_neg4_r12[i,j,:]
                        lt = 4 * QQ_xy / r_j * Z00_neg4_r21[i,j,:]
                        S = Bf_x * Bf_y * S00_neg4[i,j,:]
                        T = 4 * QQ_xy / (r_i * r_j) * T00_neg4[i,j,:]
                        L = 4 * QQ_xy * L00_neg4[i,j,:]

                        fg = D_xy * png_x * b_j * u00_neg2[i,j,:]
                        gf = D_xy * png_y * b_i * u00_neg2[i,j,:]

                        fr = - D_xy * png_x * f_y * u02_neg2[i,j,:]
                        rf = - D_xy * png_y * f_x * u20_neg2[i,j,:]

                        fv_n = D_xy * png_x * - f_y * alpha_y/r_j  * u01_neg3[i,j,:]
                        vf_n = D_xy * png_y * - f_x * alpha_x/r_i  * u10_neg3[i,j,:]

                        fv = D_xy * png_x * aH_y * B_y * u01_neg3[i,j,:]
                        vf = D_xy * png_y * aH_x * B_x * u10_neg3[i,j,:]

                        pf = D_xy * png_y * aH_x^2 * A_x * u00_neg4[i,j,:]
                        fp = D_xy * png_x * aH_y^2 * A_y * u00_neg4[i,j,:]
                        
                        fs = D_x * png_x * Bf_y * sfnl00_neg4_r2[i,j,:]
                        sf = D_y * png_y * Bf_x * sfnl00_neg4_r1[i,j,:]

                        ft = -2 * D_x * (1 - Q_y) * png_x / r_j * tfnl00_neg4_r2[i,j,:]
                        tf = -2 * D_y * (1 - Q_x) * png_y / r_i * tfnl00_neg4_r1[i,j,:]

                        fl = -2 * D_x * (1 - Q_y) * png_x * lfnl00_neg4_r2[i,j,:]
                        lf = -2 * D_y * (1 - Q_x) * png_y * lfnl00_neg4_r1[i,j,:]

                        ff = D_xy * png_x * png_y * v00_neg4[i,j,:]
                    end

                    

                    @views gg_array[i,j,:] .= gg
                    @views cl_kaiser_array[i,j,:] .= gg .+ rg .+ gr .+ rr
                    @views cl_newtonian_array[i,j,:] .= gg .+ rg .+ gr .+ rr .+ vg_n .+ gv_n .+ vr_n .+ rv_n .+ vv_n
                    @views fi_n_array[i,j,:] .= fg .+ gf .+ fr .+ rf .+ fv_n .+ vf_n
                    @views fi_array[i,j,:] .= fg .+ gf .+ fr .+ rf .+ fv .+ vf .+ fp .+ pf .+ fs .+ sf .+ ft .+ tf .+ fl .+ lf
                    @views ff_array[i,j,:] .= ff


                    @views cl_gr_array[i,j,:] .= gg .+ rg .+ gr .+ rr .+ gv .+ vg .+ rv .+ vr .+ vv .+ gp .+ pg .+ rp .+ pr .+ vp .+ pv .+ pp .+ sg .+ gs .+ sr .+ rs .+ vs .+ sv .+ ps .+ sp .+ gt .+ tg .+ rt .+ tr .+ gl .+ lg .+ rl .+ lr .+ vt .+ tv .+ lv .+ vl .+ pt .+ tp .+ pl .+ lp .+ st .+ ts .+ sl .+ ls .+ tl .+ lt .+ S .+ T .+ L
                    @views fi_k_array[i,j,:] .= fg .+ gf .+ fr .+ rf
                end
                flush(stdout)

            end
        end

        @time begin
            print("Writing to file: ")
            h5open(outpath_q,"r+") do h5file
    
                @views h5file["gg"][:,:,ell_range] = gg_array
                @views h5file["cl_kaiser"][:,:,ell_range] = cl_kaiser_array
                @views h5file["cl_newtonian"][:,:,ell_range] = cl_newtonian_array
                @views h5file["fi_n"][:,:,ell_range] = fi_n_array
                @views h5file["fi_k"][:,:,ell_range] = fi_k_array
                @views h5file["ff"][:,:,ell_range] = ff_array
    
                @views h5file["cl_gr"][:,:,ell_range] = cl_gr_array
                @views h5file["fi"][:,:,ell_range] = fi_array
            end
        end
    end
end