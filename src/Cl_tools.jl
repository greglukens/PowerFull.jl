using Healpix, Statistics, QuadGK, NearestNeighbors, DataFrames, Random

function solid_angle(θ1, θ2, φ1, φ2)
    # Define the integrand function
    integrand(θ) = sin(θ)
    
    # Perform the double integral
    result, err = quadgk(θ -> integrand(θ), θ1, θ2)

    return result * (φ2 - φ1)
end


function shell_resolution(L::Real,Ngrid::Int,Nshells::Int,los_info::Dict;r_max::Real=maximum(los_info["r_obs"]))
    r_obs = los_info["r_obs"]
    DEC = los_info["DEC"]
    RA = los_info["RA"]

    dL = L/Ngrid

    r_shells = LinRange{Float64}(minimum(r_obs[Int(Ngrid/2)+1,Int(Ngrid/2)+1,:])+2*dL,r_max,Nshells)
    nside = zeros(Int,length(r_shells))
    f_sky = zeros(length(r_shells))

    for i in range(1, length(r_shells))
        points_in_ring = findall(r_obs .>= r_shells[i] - dL .&& r_obs .< r_shells[i] + dL)
        fraction_of_sky = solid_angle(minimum(DEC[points_in_ring]), maximum(DEC[points_in_ring]), minimum(RA[points_in_ring]), maximum(RA[points_in_ring])) / (4 * pi)
        N_tot = length(points_in_ring) / fraction_of_sky
        f_sky[i] = fraction_of_sky

        nside[i] = Int(2^ceil(Int, log(2, sqrt(N_tot / 12))))
    end

    return nside,[r_shells;],f_sky
end

function get_HealpixXYZ(Nside,radius)
    Npixels = nside2npix(Nside)
    HealpixRes = Resolution(Nside)

    # Preallocate arrays for RA and Dec to improve performance
    axyz = Array{Float64,2}(undef, 3, Npixels)
    xyz  = Array{Float64,1}(undef, 3)

    # Loop over each pixel index to get its RA and Dec
    for ipix = 1:Npixels
        theta, phi = pix2angRing(HealpixRes, ipix)
        # angle to 3D vector
        xyz = ang2vec(theta,phi)
        @. axyz[:,ipix] = xyz
    end
    broadcast!(*,axyz,axyz,radius)
    return axyz
end



function resolution_window_function(alm::Alm{ComplexF64, Vector{ComplexF64}},Nside::Int)
    kn = 2*pi/nside2resol(Nside) 
    ell = [tuple[1] for tuple in each_ell_m(alm)]
    m = [tuple[2] for tuple in each_ell_m(alm)]

    k1 = m
    k2 = sqrt.(ell.^2 - m.^2)

    #return sqrt.((1 .- 2/3*(sin.(pi/2 * k1/kn)).^2).*(1 .- 2/3*(sin.(pi/2 * k2/kn)).^2))
    return (sinc.(pi/2 * k1/kn) .* sinc.(pi/2 * k2/kn)).^2
end


function NGP_interpolation(axyz,Npixels,Ngrid,dL)
    grid_ids = Vector{CartesianIndex{3}}()
    pixel_ids = Vector{Int64}()
    @inbounds for pindx = 1:Npixels
        x = axyz[1,pindx]
        y = axyz[2,pindx]
        z = axyz[3,pindx]

        ix = floor(Int64,x/dL) + 1
        iy = floor(Int64,y/dL) + 1
        iz = floor(Int64,z/dL) + 1

        if (ix>Ngrid) || (iy>Ngrid) || (iz>Ngrid)
            error("points outside of the box!")
        end

        push!(grid_ids,CartesianIndex(ix,iy,iz))
        push!(pixel_ids,Int(pindx))
    end

    return grid_ids, pixel_ids
end


function CIC_interpolation(axyz,Npixels,Ngrid,dL)
    grid_ids = Vector{CartesianIndex{3}}()
    weights = Vector{Float64}()
    pixel_ids = Vector{Int64}()

    @inbounds for pindx = 1:Npixels
        x = axyz[1,pindx]
        y = axyz[2,pindx]
        z = axyz[3,pindx]

        xx  = x/dL + 0.5
        yy  = y/dL + 0.5
        zz  = z/dL + 0.5

        # weighting for left (w1,w2,w3) and right (w1p,w2p,w3p) points
        fx = floor(xx)
        fy = floor(yy)
        fz = floor(zz)

        wx = xx-fx; wxp = 1.0 - wx
        wy = yy-fy; wyp = 1.0 - wy
        wz = zz-fz; wzp = 1.0 - wz

        # find the left grid point (this can be 0)
        ix = Int(fx); iy = Int(fy); iz = Int(fz)
        ipx = ix +1 ; ipy = iy +1 ; ipz = iz +1
        
        if 1<=ix<=Ngrid  && 1<=iy<=Ngrid && 1<=iz<=Ngrid 
            push!(weights,wxp*wyp*wzp)
            push!(grid_ids,CartesianIndex(ix,iy,iz))
            push!(pixel_ids,Int(pindx))
        else
            error("Need bigger box!")
        end
        if 1<=ipx<=Ngrid  && 1<=iy<=Ngrid  && 1<=iz<=Ngrid 
            push!(weights,wx*wyp*wzp)
            push!(grid_ids,CartesianIndex(ipx,iy,iz))
            push!(pixel_ids,Int(pindx))
        else
            error("Need bigger box!")
        end
        if 1<=ix<=Ngrid  && 1<=ipy<=Ngrid  && 1<=iz<=Ngrid 
            push!(weights,wxp*wy*wzp)
            push!(grid_ids,CartesianIndex(ix,ipy,iz))
            push!(pixel_ids,Int(pindx))
        else
            error("Need bigger box!")
        end
        if 1<=ix<=Ngrid  && 1<=iy<=Ngrid  && 1<=ipz<=Ngrid 
            push!(weights,wxp*wyp*wz)
            push!(grid_ids,CartesianIndex(ix,iy,ipz))
            push!(pixel_ids,Int(pindx))
        else
            error("Need bigger box!")
        end
        if 1<=ix<=Ngrid  && 1<=ipy<=Ngrid  && 1<=ipz<=Ngrid 
            push!(weights,wxp*wy*wz)
            push!(grid_ids,CartesianIndex(ix,ipy,ipz))
            push!(pixel_ids,Int(pindx))
        else
            error("Need bigger box!")
        end
        if 1<=ipx<=Ngrid  && 1<=iy<=Ngrid  && 1<=ipz<=Ngrid 
            push!(weights,wx*wyp*wz)
            push!(grid_ids,CartesianIndex(ipx,iy,ipz))
            push!(pixel_ids,Int(pindx))
        else
            error("Need bigger box!")
        end
        if 1<=ipx<=Ngrid  && 1<=ipy<=Ngrid  && 1<=iz<=Ngrid 
            push!(weights,wx*wy*wzp)
            push!(grid_ids,CartesianIndex(ipx,ipy,iz))
            push!(pixel_ids,Int(pindx))
        else
            error("Need bigger box!")
        end
        if 1<=ipx<=Ngrid  && 1<=ipy<=Ngrid  && 1<=ipz<=Ngrid 
            push!(weights,wx*wy*wz)
            push!(grid_ids,CartesianIndex(ipx,ipy,ipz))
            push!(pixel_ids,Int(pindx))
        else
            error("Need bigger box!")
        end

        
    end

    return grid_ids,weights,pixel_ids
end


function calculate_weights(L::Real,Ngrid::Int,Nside::AbstractArray,r_shells::AbstractArray,observer_position::AbstractArray)
    dL = L/Ngrid
    Nshells = length(r_shells)

    grid_ids = Vector{Any}(undef,Nshells) 
    weights = Vector{Any}(undef,Nshells) 
    pixel_ids = Vector{Any}(undef,Nshells)

    @time begin
        Threads.@threads for i in range(1,Nshells)
            r_i = r_shells[i]

            aXYZ = get_HealpixXYZ(Nside[i],r_i)
            @. aXYZ = aXYZ + observer_position
            Npixels = size(aXYZ)[2]

            grid_ids_r, weights_r, pixels_r = CIC_interpolation(aXYZ,Npixels,Ngrid,dL)
            
            grid_ids[i] = grid_ids_r
            weights[i] = weights_r
            pixel_ids[i] = pixels_r
        end
    end
    
    return weights, grid_ids, pixel_ids
end



function calculate_weights_NGP(L::Real,Ngrid::Int,Nside::AbstractArray,r_shells::AbstractArray,observer_position::AbstractArray)
    dL = L/Ngrid
    Nshells = length(r_shells)

    grid_ids = Vector{Any}(undef,Nshells) 
    pixel_ids = Vector{Any}(undef,Nshells)

    @time begin
        Threads.@threads for i in range(1,Nshells)
            r_i = r_shells[i]

            aXYZ = get_HealpixXYZ(Nside[i],r_i)
            @. aXYZ = aXYZ + observer_position
            Npixels = size(aXYZ)[2]

            grid_ids_r, pixels_r = NGP_interpolation(aXYZ,Npixels,Ngrid,dL)
            
            grid_ids[i] = grid_ids_r
            pixel_ids[i] = pixels_r
        end
    end
    
    return grid_ids, pixel_ids
end





function random_sky_maps(Nside::AbstractArray)
    Nshells = length(Nside)
    map_list = Vector{HealpixMap}(undef,Nshells)

    Threads.@threads for i in range(1,Nshells)
        @inbounds map_i = HealpixMap{Float64,RingOrder}(Nside[i])
        map_i.pixels[:] .= rand(length(map_i)) # set all pixels = 0
        @inbounds map_list[i] = map_i
    end

    return map_list
end


function calculate_sky_maps_CIC(δ_s::AbstractArray, weights::AbstractArray, grid_ids::AbstractArray, pixel_ids::AbstractArray, Nside::AbstractArray, r_shells::AbstractArray)
    Nshells = length(r_shells)
    map_list = Vector{HealpixMap}(undef,Nshells)
    
    Threads.@threads for i in range(1,Nshells) #length(r_bins) - 1

        @inbounds map_i = HealpixMap{Float64,RingOrder}(Nside[i])
        
        map_i.pixels[:] .= 0 # set all pixels = 0
        
        grid_ids_i = grid_ids[i]
        weights_i = weights[i]
        pixel_ids_i = pixel_ids[i]
        pids_i = Int.(unique(pixel_ids_i))

        @inbounds δ_s_i = δ_s[grid_ids_i] 

        δ_s_weighted_i = δ_s_i .* weights_i

        δ_s_i_df = DataFrame(pixel_ids=pixel_ids_i, densities=δ_s_weighted_i)
        grouped_df = combine(groupby(δ_s_i_df, :pixel_ids), :densities => sum)
        δ_s_interp_i = Matrix(grouped_df[:,2:end])

        map_i.pixels[pids_i] = δ_s_interp_i

        @inbounds map_list[i] = map_i
    end

    return map_list
end


function calculate_sky_maps_NGP(δ_s::AbstractArray, grid_ids::AbstractArray, pixel_ids::AbstractArray, Nside::AbstractArray, r_shells::AbstractArray)
    Nshells = length(r_shells)
    map_list = Vector{HealpixMap}(undef,Nshells)
    
    Threads.@threads for i in range(1,Nshells) #length(r_bins) - 1

        @inbounds map_i = HealpixMap{Float64,RingOrder}(Nside[i])
        
        map_i.pixels[:] .= 0 # set all pixels = 0
        
        grid_ids_i = grid_ids[i]
        pixel_ids_i = pixel_ids[i]
        pids_i = Int.(unique(pixel_ids_i))

        @inbounds δ_s_i = δ_s[grid_ids_i] 

        δ_s_i_df = DataFrame(pixel_ids=pixel_ids_i, densities=δ_s_i)
        grouped_df = combine(groupby(δ_s_i_df, :pixel_ids), :densities => sum)
        δ_s_interp_i = Matrix(grouped_df[:,2:end])

        map_i.pixels[pids_i] = δ_s_interp_i

        @inbounds map_list[i] = map_i
    end

    return map_list
end



function calculate_mask_C_l(map_list::AbstractVector,Nside::AbstractArray)
    C_l_mask_list = zeros(Float64,length(map_list),Int(3*maximum(Nside)))
    
    for i in range(1,length(map_list))
        map_i = map_list[i]
        mask_ids = findall(map_i.pixels .== 0)
        unmasked_ids = findall(map_i.pixels .!= 0)

        map_i.pixels[mask_ids] .= 0
        map_i.pixels[unmasked_ids] .= 1

        C_l_mask_list[i,1:Int(3*Nside[i])] = anafast(map_i,lmax = 3*Nside[i] - 1)
    end
    return C_l_mask_list
end

function compute_C_l_stats(n_files::Int, inname::String, lmax::Int, outpath::String, N_shells::Int)

	# initializing the sum arrays
	
	C_l_sum = zeros(Float64,N_shells-1,Int(l_max + 1)) 		# initialize the final sum array
    C_l_sum2 = zeros(Float64,N_shells-1,Int(l_max + 1))
	N = 0 	# this will be the total number of realizations across all files

	# the usual handling of diagnostic_mode initialization

	# looping over all of the files
    println("============================================================================================")
    println("Computing C_l statistics across $(n_files) files...")
    println("============================================================================================")

    @time begin

        Threads.@threads for n in range(1,n_files)
            
            infile = h5open("$(inname)_$(n).h5","r")

            # reading in the nth file's results
            N += infile["N"][] # N_realizations in the nth file
            C_l_sum .+= infile["C_l_sum"][]
            C_l_sum2 .+= infile["C_l_sum2"][]
            close(infile)
        end
    end

    println("============================================================================================")
    println("Computed C_l mean and variance for $(N) realizations")
    println("============================================================================================")

	# computing the mean and variance (of the mean)
	
	C_l_mean = C_l_sum / N
	C_l_var = C_l_sum2 / N .- C_l_mean.^2

	# return more things if diagnostic_mode is on, so need a conditional 
	println("Writing to file: ")
	
	@time begin
		h5open("$(outpath)_N$(N).h5","w") do outfile
			outfile["N"] = N
			outfile["C_l_mean"] = C_l_mean
			outfile["C_l_var"] = C_l_var
		end
	end
end

function calculate_weights_old(L::Real,Ngrid::Int,Nside::AbstractArray,r_shells::AbstractArray,los_info::Dict)
    r_obs = los_info["r_obs"]
    X_obs = los_info["X_obs"]
    Y_obs = los_info["Y_obs"]
    Z_obs = los_info["Z_obs"]
    DEC = los_info["DEC"]
    RA = los_info["RA"]

    X_flat = vec(X_obs)
    Y_flat = vec(Y_obs)
    Z_flat = vec(Z_obs)
    R_flat = vec(r_obs)
    DEC_flat = vec(DEC)
    RA_flat = vec(RA)

    ids = collect(CartesianIndices(r_obs))
    #ids_1 = getindex.(ids,1)
    #ids_2 = getindex.(ids,2)
    #ids_3 = getindex.(ids,3)

    grid_points = transpose(hcat(X_flat,Y_flat,Z_flat))

    dL = L/Ngrid

    Nshells = length(r_shells) - 1 

    grid_ids = Vector{Any}(undef,Nshells) 
    pixel_ids = Vector{Any}(undef,Nshells)
    weights = Vector{Any}(undef,Nshells) 

    @time begin
        Threads.@threads for i in range(1,Nshells)

            Npixels_i = nside2npix(Nside[i])
            R_i = mean(r_shells[i:i+1])

            close_points = findall((R_flat .<= R_i + sqrt(2)*dL) .& (R_flat .>= R_i - sqrt(2)*dL))
            kdtree_close = KDTree(grid_points[:,close_points])

            padding = 0.01

            RA_min = minimum(RA_flat[close_points]) - padding 
            RA_max = maximum(RA_flat[close_points]) + padding 
            DEC_min = minimum(DEC_flat[close_points]) - padding 
            DEC_max = maximum(DEC_flat[close_points]) + padding 

            DEC_i = zeros(Float64,Npixels_i)
            RA_i = zeros(Float64,Npixels_i)

            for j in range(1,Npixels_i)
                DEC_i[j], RA_i[j] = pix2angRing(Resolution(Nside[i]),j)
            end

            close_pixels = findall((RA_i .>= RA_min) .&& (RA_i .<= RA_max) .&& (DEC_i .>= DEC_min) .&& (DEC_i .<= DEC_max))
            
            grid_ids[i] = Vector{Any}(undef,length(close_pixels))
            weights[i] = Vector{Any}(undef,length(close_pixels)) 

            @inbounds for j in close_pixels

                X_j = R_i * sin(DEC_i[j]) * cos(RA_i[j])
                Y_j = R_i * cos(DEC_i[j])
                Z_j = - R_i * sin(DEC_i[j]) * sin(RA_i[j])
                
                nearest_ids, dists = knn(kdtree_close, [X_j,Y_j,Z_j], 8)

                @inbounds dx = abs.(X_j .- X_flat[close_points[nearest_ids]])
                @inbounds dy = abs.(Y_j .- Y_flat[close_points[nearest_ids]])
                @inbounds dz = abs.(Z_j .- Z_flat[close_points[nearest_ids]])

                wx = zeros(8)
                wy = zeros(8)
                wz = zeros(8)
                
                check_range_x = findall(dx .< dL)
                check_range_y = findall(dy .< dL)
                check_range_z = findall(dz .< dL)

                if length(check_range_x) > 0
                    @inbounds wx[check_range_x] .= 1 .- dx[check_range_x]/dL
                end

                if length(check_range_y) > 0
                    @inbounds wy[check_range_y] .= 1 .- dy[check_range_y]/dL
                end

                if length(check_range_z) > 0
                    @inbounds wz[check_range_z] .= 1 .- dz[check_range_z]/dL
                end
                
                @inbounds grid_ids[i][findall(close_pixels .== j)[1]] = ids[close_points[nearest_ids]]
                @inbounds pixel_ids[i] = close_pixels
                @inbounds weights[i][findall(close_pixels .== j)[1]] = Float32.(wx .* wy .* wz)
                
            end
        end

    end
    
    return Dict("grid_ids" => grid_ids, "pixel_ids" => pixel_ids, "weights" => weights)
end

function calculate_weights_older(L::Real,Ngrid::Int,Npoints::Int,Nside::AbstractArray,r_shells::AbstractArray,los_info::Dict)
    r_obs = los_info["r_obs"]
    X_obs = los_info["X_obs"]
    Y_obs = los_info["Y_obs"]
    Z_obs = los_info["Z_obs"]

    DEC = los_info["DEC"]
    RA = los_info["RA"]

    dL = L/Ngrid

    Nshells = length(r_shells) 

    max_bins_half = Int(ceil(dL/2/(r_shells[2]-r_shells[1])))

    shells = zeros(Int16,Ngrid,Ngrid,Ngrid,1+2*max_bins_half)
    weights = zeros(Float16,Ngrid,Ngrid,Ngrid,1+2*max_bins_half,4)
    pixel_ids = zeros(Int64,Ngrid,Ngrid,Ngrid,1+2*max_bins_half,4)

    map_list = Vector{Any}()

    @time begin
        Threads.@threads for i in range(1,Ngrid)
            #println("Working on z-plane $(i)...")
            @inbounds for j in range(1,Ngrid)
                @inbounds for k in range(1,Ngrid)
                    
                    central_position = [X_obs[i,j,k],Y_obs[i,j,k],Z_obs[i,j,k]]
                    central_r = r_obs[i,j,k]

                    #println(central_r)

                    Δx = zeros(Npoints)
                    Δy = zeros(Npoints)
                    Δz = zeros(Npoints)
                    
                    @inbounds for l in range(1,Npoints)
                        Δx[l] = 2*rand() * dL - dL
                        Δy[l] = 2*rand() * dL - dL
                        Δz[l] = 2*rand() * dL - dL
                    end

                    x_positions_ijk = Δx .+ central_position[1]
                    y_positions_ijk = Δy .+ central_position[2]
                    z_positions_ijk = Δz .+ central_position[3]

                    r_positions_ijk = sqrt.(x_positions_ijk.^2 + y_positions_ijk.^2 + z_positions_ijk.^2)

                    check_for_lower_bound = findall(central_r .- r_shells .< 0)

                    if length(check_for_lower_bound) == 0
                        central_bin_idxs_lower = Int(maximum(findall(central_r .- r_shells .> 0)))
                    else
                        central_bin_idxs_lower = Int(minimum(findall(central_r .- r_shells .< 0)) - 1)
                    end

                    
                    if central_bin_idxs_lower <= max_bins_half
                        m_begin = 1
                        m_end = 1+2*max_bins_half 
                    elseif central_bin_idxs_lower > Nshells - max_bins_half
                        m_end = Nshells 
                        m_begin = Nshells - 2*max_bins_half 
                    else
                        m_begin = central_bin_idxs_lower - max_bins_half
                        m_end = central_bin_idxs_lower + max_bins_half 
                    end

                    
                    @inbounds for m in range(m_begin, m_end-1)
                        points_in_bin_m = findall(r_positions_ijk .>= r_shells[m] .&& r_positions_ijk .< r_shells[m+1])
                        N_in_bin_m = length(points_in_bin_m)
                        
                        weights_r_ijkm = Float16(N_in_bin_m / Npoints)
                        shells[i,j,k,m-m_begin+1] = m

                        pixels_ijkm, weights_ang_ijkm = getinterpolRing(Resolution(Nside[m]), DEC[i,j,k], RA[i,j,k])
                        
                        weights_ijkm = weights_r_ijkm .* weights_ang_ijkm

                        weights[i,j,k,m-m_begin+1,:] = weights_ijkm
                        pixel_ids[i,j,k,m-m_begin+1,:] = pixels_ijkm
                    end
                    
                end
            end
        end

    end
    
    
    return Dict("shell_ids" => shells, "pixel_ids" => pixel_ids, "weights" => weights)
end

function calculate_sky_maps_old(δ_s::AbstractArray, weighting::Dict, Nside::AbstractArray, r_shells::AbstractArray, los_info::Dict, n_bar::Real)
    weights = weighting["weights"]
    grid_ids = weighting["grid_ids"]
    pixel_ids = weighting["pixel_ids"]

    map_list = Vector{HealpixMap}(undef,length(r_shells)-1)
    
    Threads.@threads for i in range(1, length(r_shells)-1) #length(r_bins) - 1
        #println("Working on bin $(i)")

        @inbounds map_i = HealpixMap{Float64,RingOrder}(Nside[i])
        
        map_i.pixels[:] .= 0 # set all pixels = 0
        
        grid_ids_i = grid_ids[i]
        pixel_ids_i = pixel_ids[i]
        weights_i = weights[i]

        @inbounds δ_s_i = δ_s[grid_ids_i] 
        #ρ_weighted_i = (1 .+ δ_s_i) .* weights_i

        #map_i.pixels[pixel_ids_i] .= -1 # set weighted pixels = -1 to begin with 

        #ρ_weighted_sum_i = ρ_weighted_i[1,:]
        
        #@inbounds for j in range(2,8)
        #    ρ_weighted_sum_i .+= ρ_weighted_i[j,:]
        #end

        δ_s_weighted_i = δ_s_i .* weights_i
        δ_s_weighted_i_sum = δ_s_weighted_i[1,:]

        @inbounds for j in range(2,8)
            δ_s_weighted_i_sum .+= δ_s_weighted_i[j,:]
        end
        #δ_s_weighted_i = 1/(n_bar*(2*pi)^3) * ρ_weighted_sum_i .- 1 #confused on this pre-factor

        map_i.pixels[pixel_ids_i] = δ_s_weighted_i_sum
        map_i.pixels[findall(map_i.pixels .== -1)] .= 0

        @inbounds map_list[i] = map_i
    end

    return map_list
end

function calculate_sky_maps_older(δ_s::AbstractArray, weighting::Dict, Nside::AbstractArray, r_shells::AbstractArray, los_info::Dict)
    weights = weighting["weights"]
    pixel_ids = weighting["pixel_ids"]
    shell_ids = weighting["shell_ids"]

    r_obs = los_info["r_obs"]
    DEC = los_info["DEC"]
    RA = los_info["RA"]

    map_list = Vector{HealpixMap}(undef,length(r_shells)-1)

    
    Threads.@threads for i in range(1, length(r_shells)-1) #length(r_bins) - 1
        #println("Working on bin $(i)")

        @inbounds map_i = HealpixMap{Float64,RingOrder}(Nside[i])
        
        map_i.pixels[:] .= 0 # set all pixels = 0

        points_i = findall(shell_ids .== i)
        
        grid_idxs_i = Vector{CartesianIndex{3}}()

        @inbounds for j in range(1, length(points_i))
            push!(grid_idxs_i, CartesianIndex(points_i[j][1], points_i[j][2], points_i[j][3]))
        end

        @inbounds δ_s_i = δ_s[grid_idxs_i] 

        #DEC_i = DEC[grid_idxs_i]
        #RA_i = RA[grid_idxs_i]

        @inbounds pixel_ids_i = @view(pixel_ids[points_i,:])
        @inbounds weights_i = @view(weights[points_i,:])
        ρ_weighted_i = (1 .+ δ_s_i) .* weights_i

        unique_pixels_i = unique(pixel_ids_i)

        map_i.pixels[unique_pixels_i] .= -1

        
        @inbounds for j in range(1,length(δ_s_i))
            map_i.pixels[pixel_ids_i[j,:]] .+= ρ_weighted_i[j,:] ##(1 .+ δ_s_i[j]) .* weights_i[j,:]
        end
        
        #old way

        #@time begin
            #Threads.@threads for j in range(1,length(unique_pixels_i))
            #    ids_j = findall(pixel_ids_i .== unique_pixels_i[j])
            #    δ_s_weighted_ij = sum(δ_s_weighted_i[ids_j])
            #    map_i.pixels[unique_pixels_i[j]] = δ_s_weighted_ij
            #end
        #end

        @inbounds map_list[i] = map_i
    end

    return map_list
end


function shell_resolution_old(L::Real,Ngrid::Int,Nshells::Int,los_info::Dict;r_max::Real=maximum(los_info["r_obs"]))
    r_obs = los_info["r_obs"]
    DEC = los_info["DEC"]
    RA = los_info["RA"]

    dL = L/Ngrid

    r_shells = LinRange{Float64}(minimum(r_obs[Int(Ngrid/2)+1,Int(Ngrid/2)+1,:])+2*dL,r_max,Nshells+1)
    nside = zeros(Int,length(r_shells) - 1)
    f_sky = zeros(length(r_shells)-1)

    for i in range(1, length(r_shells) - 1)
        points_in_ring = findall(r_obs .>= r_shells[i] .&& r_obs .< r_shells[i+1])
        fraction_of_sky = solid_angle(minimum(DEC[points_in_ring]), maximum(DEC[points_in_ring]), minimum(RA[points_in_ring]), maximum(RA[points_in_ring])) / (4 * pi)
        N_tot = length(points_in_ring) / fraction_of_sky
        f_sky[i] = fraction_of_sky

        nside[i] = Int(2^ceil(Int, log(2, sqrt(N_tot / 12))))
    end

    return nside,[r_shells;],f_sky
end