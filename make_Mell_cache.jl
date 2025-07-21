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

make_caches = true

ellmax = 1000
kmax = 1e4
kmin = 1e-5
NR = 5001
Nr = 4096
zmax = 5
R_min = 0.005

q0 = 1.1
qn2 = 0.5
qn4 = -1.26

qxt_n2 = 0.5
qxt_n4 = -0.9
qft_n4 = 0.3



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

zmin = z_of_r(r_of_z(zmax) * R_min)
N = Nr

@time begin 
    print("Initializing (r,R) grid: ")
    rr_cache = exp.(range(log(1/kmax),log(1/kmin),N+1))
    rr_max_id = Int(minimum(findall(rr_cache .>= r_of_z(zmax))))
    rr_min_id = Int(maximum(findall(rr_cache .<= r_of_z(zmin))))
    rr = rr_cache[rr_min_id:rr_max_id]
    Rmin = rr_cache[rr_min_id] / rr_cache[rr_max_id]
    RR = exp.(range(log(Rmin),log(1/Rmin),NR))

    if !iseven(NR)
        RR[Int(ceil(NR/2))] = 1
    end
end

q_map =(; pk_0=q0,pk_neg2=qn2,pk_neg4=qn4,pk_fnl_xt_neg2=qxt_n2,pk_fnl_xt_neg4=qxt_n4,pk_fnl_ft_neg4=qft_n4)

cachepath = "/storage/home/gql5196/scratch/caches/Nr$(Nr)NR$(NR)lmax$(ellmax)"
mkpath(cachepath)

println("Making cache files...")
flush(stdout)
@time begin
    print("P(k) with q = $(q_map.pk_0) and $(NR) log-spaced R bins in [$(round(Rmin,digits=3)),$(round(1/Rmin,digits=2))]: ")
    calc_Mell(ellmax,RR,q_map.pk_0,kmax,kmin,N,"$(cachepath)/log_RR/pk_0")
end
flush(stdout)
@time begin
    print("P(k)/k^2 with q = $(q_map.pk_neg2) and $(NR) log-spaced R bins in [$(round(Rmin,digits=3)),$(round(1/Rmin,digits=2))]: ")
    calc_Mell(ellmax,RR,q_map.pk_neg2,kmax,kmin,N,"$(cachepath)/log_RR/pk_neg2")
end
flush(stdout)
@time begin
    print("P(k)/k^4 with q = $(q_map.pk_neg4) and $(NR) log-spaced R bins in [$(round(Rmin,digits=3)),$(round(1/Rmin,digits=2))]: ")
    calc_Mell(ellmax,RR,q_map.pk_neg4,kmax,kmin,N,"$(cachepath)/log_RR/pk_neg4")
end
flush(stdout)
@time begin
    print("P(k)/[k^2 T(k)] with q = $(q_map.pk_fnl_xt_neg2) and $(NR) log-spaced R bins in [$(round(Rmin,digits=3)),$(round(1/Rmin,digits=2))]: ")
    calc_Mell(ellmax,RR,q_map.pk_fnl_xt_neg2,kmax,kmin,N,"$(cachepath)/log_RR/pk_fnl_xt_neg2")
end
flush(stdout)
@time begin
    print("P(k)/[k^4 T(k)] with q = $(q_map.pk_fnl_xt_neg4) and $(NR) log-spaced R bins in [$(round(Rmin,digits=3)),$(round(1/Rmin,digits=2))]: ")
    calc_Mell(ellmax,RR,q_map.pk_fnl_xt_neg4,kmax,kmin,N,"$(cachepath)/log_RR/pk_fnl_xt_neg4")
end
flush(stdout)
@time begin
    print("P(k)/[k^4 T(k)^2] with q = $(q_map.pk_fnl_ft_neg4) and $(NR) log-spaced R bins in [$(round(Rmin,digits=3)),$(round(1/Rmin,digits=2))]: ")
    calc_Mell(ellmax,RR,q_map.pk_fnl_ft_neg4,kmax,kmin,N,"$(cachepath)/log_RR/pk_fnl_ft_neg4")
end
flush(stdout)


