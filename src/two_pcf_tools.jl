using HDF5, Statistics

"""
MULTI-THREADED IMPLEMENTATION OF calc_2D_2pcf FUNCTION

The below function is used to compute the redshift space two-point correlation function about a central point in a regular grid, using GridSPT realization data
The function returns the sum and sum^2 of the 2PCF at every point in the grid 

Inputs
- L: box size of the grid
- Ngrid: number of grid points along each dimension (assuming a cube)
- central_redshift: the redshift of the central grid point (i.e., index [Ngrid/2 + 1,Ngrid/2 + 1,Ngrid/2 + 1])
- b: the linear galaxy bias
- inpath: the input file path (where the realization files are)
- outpath: the output file path (for the result)
- r_bins: bin edges for x_perp binning (for binning along constant z slices)
- start_realization: the starting realization number
- end_realization: the ending realization number
- omega_m: the matter density parameter (at present)
- omega_l: the dark energy density parameter (at present)
- omega_r: the radiation density parameter (at present)
- w: the dark energy equation of state (assuming wCDM cosmologies)
- wide_angle: if true: wide angle effect, if false: flat-sky approximation
- full_newtonian: if true: all cosmological evolution effects, if false: simple Kaiser expression
- diagnostic_mode: if true (and wide_angle + full_newtonian true as well): return component 2PCF values as well

Outputs (as HDF5 file)
- two_pcf_sum: sum of 2PCF across all realizations at every grid point
- two_pcf_sum2: sum^2 of 2PCF across all realizations at every grid point

   - Additionally:
    if diagnostic_mode + wide_angle + full_newtonian are true:
        - dd_sum: the sum of the real-space galaxy 2PCF (delta_g-delta_g) across all realizations at every grid point 
        - dd_sum2: the sum^2 of the real-space galaxy 2PCF (delta_g-delta_g) across all realizations at every grid point
        - ddv_sum: the sum of the real-space galaxy density contrast - linear velocity derivative cross correlation function across all realizations at every grid point
        - ddv_sum2: the sum^2 of the real-space galaxy density contrast - linear velocity derivative cross correlation function across all realizations at every grid point
        - dvd_sum: the sum of the real-space linear velocity derivative - galaxy density contrast cross correlation function across all realizations at every grid point
        - dvd_sum2: the sum^2 of the real-space linear velocity derivative - galaxy density contrast cross correlation function across all realizations at every grid point
        - dvdv_sum: the sum of the real-space linear velocity derivative auto-correlation function across all realizations at every grid point
        - dvdv_sum2: the sum^2 of the real-space linear velocity derivative auto-correlation function across all realizations at every grid point
"""

function delta_to_delta_s(Ngrid::Int,central_redshift::Real,b::Real,δ::AbstractArray,uz::AbstractArray,uy::AbstractArray,ux::AbstractArray,los_info::Dict,cosmology::Dict,wide_angle::Bool,full_newtonian::Bool,diagnostic_mode::Bool)
	@assert Ngrid > 0
	@assert central_redshift > 0 
	@assert b > 0
	@assert size(δ) == size(ux) == size(uy) == size(uz)
	
	# input the matrices of geometric info from observer_grid_geometry function

	
	# all lines of sight (wide angle)
	
    x_par = los_info["x_par"]
    x_perp = los_info["x_perp"]
    f_obs = los_info["f_obs"]
    D_obs = los_info["D_obs"]	
    r_obs = los_info["r_obs"]
    α1_obs = los_info["α1_obs"]
    α2_obs = los_info["α2_obs"]
    α3_obs = los_info["α3_obs"]

	# one line of sight (central; small-sky)
	
	f_z = los_info["f_z"]
    D_z = los_info["D_z"]	
    r_z = los_info["r_z"]
    α1_z = los_info["α1_z"]
    α2_z = los_info["α2_z"]
    α3_z = los_info["α3_z"]

	
	# relevant splines for the function, inputted from initialize_cosmology function
	
    f_of_z = cosmology["f_of_z"]
    D_of_z = cosmology["D_of_z"]


	# small-sky only (so z-hat is LoS)
	
    if wide_angle == false

        ∂uz_z = zeros(Float64,Ngrid,Ngrid,Ngrid) 	# initialize z-derivative of z-velocity array

        Threads.@threads for i in range(1,Ngrid) 	# looping over X
            for j in range(1,Ngrid) 	# looping over Y
                @inbounds uz_zcol = uz[i,j,:] 	# column of constant X and Y (i.e., a column of u_z points along z)
                @inbounds ∂uz_z[i,j,:] = derivative(Spline1D(x_par,uz_zcol,s=0),x_par,nu=1) 	# computing derivative of spline of column as a function of x_parallel (or z)
            end
        end

        if full_newtonian == false 		# if just Kaiser formula, just follow top case of eq (13) in documentation (alpha = 0, no evolution effects)

            δ_s = b*δ   +   f_of_z(central_redshift)* ∂uz_z 	
            δ_s .*= D_of_z(central_redshift) 	# rescale by D_c at end

			
			# if diagnostic mode is on: compute components of redshift space 2PCF for Kaiser effect only -> obtain derivative term of expression by itself 

            if diagnostic_mode == true
                ∂v_term = f_of_z(central_redshift) * D_of_z(central_redshift) * ∂uz_z
            end
        end

        if full_newtonian == true 		# if full Newtonian, follow bottom case of eq (13) in documentation (all evolution effects and alpha)

            α_obs = α1_obs + α3_obs 	# relativistic effects (alpha_1 and alpha_3); can use whatever you want to account for here (combo of alpha's)

            δ_s = b*δ   +   f_obs .* α_obs ./ r_obs .* uz   +   f_obs .* ∂u_z
            δ_s .*= D_obs
        end
    end

	
	# wide angle (z-hat is now n-hat to each galaxy/point in grid)

    if wide_angle == true

        ur = ux .* los_info["n_hat_x"] + uy .* los_info["n_hat_y"] + uz .* los_info["n_hat_z"] 		# compute radial velocity

        ∂ur_x = zeros(Float64,Ngrid,Ngrid,Ngrid) 	# initialize derivative arrays
        ∂ur_y = zeros(Float64,Ngrid,Ngrid,Ngrid)
        ∂ur_z = zeros(Float64,Ngrid,Ngrid,Ngrid)

        Threads.@threads for i in range(1,Ngrid) 	# loop over X, Y, or Z (depending on which array)
            for j in range(1,Ngrid) 	# loop over Y, Z, or X (depending on which array)
                @inbounds ur_zcol = ur[i,j,:] 	# column of u_r points across z (so constant X and Y)
                @inbounds ur_ycol = ur[i,:,j] 	# column of u_r points across y (so constant X and Z)
                @inbounds ur_xcol = ur[:,i,j] 	# column of u_r points across x (so constant Y and Z)

                @inbounds ∂ur_z[i,j,:] = derivative(Spline1D(x_par,ur_zcol,s=0),x_par,nu=1) 	# interpolated derivative of column along z (du_r/dz)
                @inbounds ∂ur_y[i,:,j] = derivative(Spline1D(x_perp,ur_ycol,s=0),x_perp,nu=1) 	# interpolated derivative of column along y (du_r/dy)
                @inbounds ∂ur_x[:,i,j] = derivative(Spline1D(x_perp,ur_xcol,s=0),x_perp,nu=1)		# interpolated derivative of column along x (du_r/dx)
            end
        end

		# compute du_r/dr now, using spherical coordinates
		
        ∂ur_r = los_info["cos_φ"] .* los_info["sin_θ"] .* ∂ur_x   +    los_info["sin_φ"] .* los_info["sin_θ"] .* ∂ur_y    +    los_info["cos_θ"] .* ∂ur_z 


		# Kaiser effect only, this is top case of eq (14)
		
        if full_newtonian == false

            δ_s = b*δ   +   f_of_z(central_redshift) * ∂ur_r
            δ_s .*= D_of_z(central_redshift)
        end

		# all effects, this is bottom case of eq (14)
		
        if full_newtonian == true

            α_obs = α1_obs + α3_obs

            δ_s = b*δ   +   f_obs .* α_obs ./ r_obs .* ur   +   f_obs .* ∂ur_r
            δ_s .*= D_obs
        end
    end

	# returning both delta_s and the v-term (since we already have real-space delta outside of the function) if diagnostic mode is on
	
    if diagnostic_mode == true && full_newtonian == false && wide_angle == false
        return δ_s,∂v_term
    else
        return δ_s 		# otherwise just need delta_s
    end
end

function calc_2D_2pcf(L::Real,Ngrid::Int,central_redshift::Real,b::Real,inpath::String,outpath::String,r_bins::AbstractArray,start_realization::Int,end_realization::Int,cosmology::Dict,los_info::Dict,wide_angle::Bool,full_newtonian::Bool,diagnostic_mode::Bool)
	@assert L > 0
	@assert Ngrid > 0
	@assert central_redshift > 0
	@assert b > 0
	@assert length(r_bins) > 0
	@assert start_realization <= end_realization
		
    r_proj = los_info["r_proj"] 	# import projected radius (i.e., x_perp) -> will be binned over 

	
    bin_idxs = Vector{Any}() 	# setup array of arrays that will provide indices of points in each bin; this is the same for each plane of points

    for j in range(1,length(r_bins)-1)
        jth_bin = findall(r_proj[:,:,1] .>= r_bins[j] .&& r_proj[:,:,1] .< r_bins[j+1]) 	# include lower bound in binning (i.e., bin_i <= x < bin_i+1)
        push!(bin_idxs,jth_bin) 	# use push! function here, since each array contains a different number of indices
    end

	
    ξ_sum = zeros(Float64,Ngrid,length(r_bins)-1) 		# initialize the final sum array
    ξ_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)		# initialize the final sum^2 array


	# setup sum arrays for diagnostic correlation functions (if using Kaiser + small sky formula)
	
    if diagnostic_mode == true && full_newtonian == false && wide_angle == false
        δδ_sum = zeros(Float64,Ngrid,length(r_bins)-1)
        δδ_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)
        
        δ∂v_sum = zeros(Float64,Ngrid,length(r_bins)-1)
        δ∂v_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)
        
        ∂vδ_sum = zeros(Float64,Ngrid,length(r_bins)-1)
        ∂vδ_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)

        ∂v∂v_sum = zeros(Float64,Ngrid,length(r_bins)-1)
        ∂v∂v_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)
    end


	####################################################################
	
	##### Compute the 2PCF in redshift space across N realizations ##### 
	
	####################################################################
	
	
    for n in range(start_realization,end_realization)
	println("====================================================================")
        println("Ngrid = $Ngrid, L = $L, Realization = $n")
        println("====================================================================")

		println("Loading in realization files:")
		@time begin
	        δ_s = zeros(Float64,Ngrid,Ngrid,Ngrid) 		# initialize the redshift space density contrast array
	        ξ = zeros(Float64,Ngrid,length(r_bins)-1) 		# initialize 2PCF array
			
			# initialize dianogistic 2PCF arrays
			
	        if diagnostic_mode == true && full_newtonian == false && wide_angle == false
	            δδ = zeros(Float64,Ngrid,length(r_bins)-1)
	            δ∂v = zeros(Float64,Ngrid,length(r_bins)-1)
	            ∂vδ = zeros(Float64,Ngrid,length(r_bins)-1)
	            ∂v∂v = zeros(Float64,Ngrid,length(r_bins)-1)
	        end
		end

        println("Converting δ to redshift space:")

        @time begin 	# here we use @time to keep track of how long our code takes to convert to redshift space

			# loading in the GridSPT data
			
            infile = h5open("$(inpath)$(n)_$(Ngrid).h5","r")

            δ = infile["deltax1"][]
            uz = infile["v1x1"][]
            uy = infile["v2x1"][]
            ux = infile["v3x1"][]
            
            close(infile)

			# computing the "v_term" (the ∂_z v_z term) for the diagnostic 2PCFs

            if diagnostic_mode == true && full_newtonian == false && wide_angle == false
                δ_s, ∂v_term = delta_to_delta_s(Ngrid,central_redshift,b,δ,uz,uy,ux,los_info,cosmology,wide_angle,full_newtonian,diagnostic_mode) 
                central_δ_g = b*δ[Int(Ngrid/2)-1,Int(Ngrid/2)-1,Int(Ngrid/2)-1] 	# central galaxy density contrast (real space)
                central_∂v = ∂v_term[Int(Ngrid/2)-1,Int(Ngrid/2)-1,Int(Ngrid/2)-1] 	# central v_term
			else
				δ_s = delta_to_delta_s(Ngrid,central_redshift,b,δ,uz,uy,ux,los_info,cosmology,wide_angle,full_newtonian,diagnostic_mode)	# converting to redshift space
			end

			central_δ_s = δ_s[Int(Ngrid/2)-1,Int(Ngrid/2)-1,Int(Ngrid/2)-1] 	# finding the central density contrast
        
        end        

        println("Computing 2D ξ in redshift space:")

        @time begin 	# keeping track of how long it takes to compute 2PCFs

            Threads.@threads for i in range(1,Ngrid) 	# looping over x_parallel (i.e., along z)

                @inbounds δ_s_plane = δ_s[:,:,i] 		# each plane of constant x_parallel (or z)
				

				# if we care about diagnostic 2PCFs, we also setup the planes for the real space galaxy density contrast and v_term
				
                if diagnostic_mode == true && full_newtonian == false && wide_angle == false
                    @inbounds δ_g_plane = b*δ[:,:,i]
                    @inbounds ∂v_plane = ∂v_term[:,:,i]
                end

				# looping over bins of x_perp  /  r_projected
				
                @inbounds for j in range(1,length(r_bins)-1)
                    ring_idxs = bin_idxs[j] 	# these are the indices of the points in the jth bin
                    
                    δ_s_in_ring = δ_s_plane[ring_idxs] 		# these are the points in the plane inside the jth bin
                    @inbounds ξ[i,j] = central_δ_s * mean(δ_s_in_ring) 	# averaging the points in the bin and multiplying by central density contrast (gives us 2PCF)

					# following suit for the diagnostic 2PCFs
					
                    if diagnostic_mode == true && full_newtonian == false && wide_angle == false
						δ_g_in_ring = δ_g_plane[ring_idxs] 		# do the same thing for real space galaxy density contrast
                        ∂v_in_ring = ∂v_plane[ring_idxs] 	# do the same thing for the v_term

                        @inbounds δδ[i,j] = central_δ_g * mean(δ_g_in_ring) 		# again, averaging and multiplying by central value
                        @inbounds δ∂v[i,j] = central_δ_g * mean(∂v_in_ring)
                        @inbounds ∂vδ[i,j] = central_∂v * mean(δ_g_in_ring)
                        @inbounds ∂v∂v[i,j] = central_∂v * mean(∂v_in_ring)
                    end
                end
            end

			# now we are done with the nth realization, throw the results into the sum arrays

            ξ_sum .+= ξ
            ξ_sum2 .+= ξ.^2 

			#if diagnostic is on, same thing, just with those 2PCFs
        
            if diagnostic_mode == true && full_newtonian == false && wide_angle == false
                δδ_sum .+= δδ
                δδ_sum2 .+= δδ.^2
                
                δ∂v_sum .+= δ∂v
                δ∂v_sum2 .+= δ∂v.^2
                
                ∂vδ_sum .+= ∂vδ
                ∂vδ_sum2 .+= ∂vδ.^2
        
                ∂v∂v_sum .+= ∂v∂v
                ∂v∂v_sum2 .+= ∂v∂v.^2
            end

        end
    end

	# after finishing all N realizations, we output the results to an .h5 file; we save N and the 2PCF sums

	println("")
	println("Writing to file:")
	
	@time begin
		h5open(outpath,"w") do outfile
			outfile["N"] = end_realization - start_realization + 1
			outfile["two_pcf_sum"] = ξ_sum
			outfile["two_pcf_sum2"] = ξ_sum2

			if diagnostic_mode == true && full_newtonian == false && wide_angle == false
				outfile["dd_sum"] = δδ_sum
				outfile["dd_sum2"] = δδ_sum2
					
				outfile["ddv_sum"] = δ∂v_sum
				outfile["ddv_sum2"] = δ∂v_sum2
					
				outfile["dvd_sum"] = ∂vδ_sum
				outfile["dvd_sum2"] = ∂vδ_sum2
			
				outfile["dvdv_sum"] = ∂v∂v_sum
				outfile["dvdv_sum2"] = ∂v∂v_sum2
			end
		end
	end
	
end


"""
The below function computes the statistics of the 2PCF (i.e., mean and variance of the mean) at every point in a regular grid.
It takes sum and sum^2 information from the calc_2D_2pcf function and returns the statistics.

Inputs 
- n_files: number of 2PCF files inputted
- Ngrid: number of grid points along each dimension (assuming a cube)
- inpath: the input file path (where the realization files are)
- r_bins: bin edges for x_perp binning (for binning along constant z slices)
- wide_angle: if true: wide angle effect, if false: flat-sky approximation
- full_newtonian: if true: all cosmological evolution effects, if false: simple Kaiser expression
- diagnostic_mode: if true (and wide_angle + full_newtonian true as well): return component 2PCF values as well

Outputs (as a Dictionary)
- two_pcf_mean: the 2PCF mean at every grid point
- two_pcf_var: the 2PCF variance (of the mean) at every grid point

   - Additionally:
    if diagnostic_mode + wide_angle + full_newtonian are true:
        - dd_mean: the mean of the real-space galaxy 2PCF (delta_g-delta_g) at every grid point  
        - dd_var: the variance of the real-space galaxy 2PCF (delta_g-delta_g) at every grid point at every grid point
        - ddv_mean: the mean of the real-space galaxy density contrast - linear velocity derivative cross correlation function at every grid point
        - ddv_var: the variance of the real-space galaxy density contrast - linear velocity derivative cross correlation function at every grid point
        - dvd_mean: the mean of the real-space linear velocity derivative - galaxy density contrast cross correlation function at every grid point
        - dvd_var: the variance of the real-space linear velocity derivative - galaxy density contrast cross correlation function at every grid point
        - dvdv_mean: the mean of the real-space linear velocity derivative auto-correlation function at every grid point
        - dvdv_mean: the variance of the real-space linear velocity derivative auto-correlation function at every grid point
"""

function compute_2pcf_stats(n_files, inpath, Ngrid, r_bins, wide_angle::Bool,full_newtonian::Bool,diagnostic_mode::Bool)

	# initializing the sum arrays
	
	ξ_sum = zeros(Float64,Ngrid,length(r_bins)-1) 		
    ξ_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)	
	N = 0 	# this will be the total number of realizations across all files

	# the usual handling of diagnostic_mode initialization
	if diagnostic_mode == true && full_newtonian == false && wide_angle == false 
		δδ_sum = zeros(Float64,Ngrid,length(r_bins)-1)
        δδ_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)
        
        δ∂v_sum = zeros(Float64,Ngrid,length(r_bins)-1)
        δ∂v_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)
        
        ∂vδ_sum = zeros(Float64,Ngrid,length(r_bins)-1)
        ∂vδ_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)

        ∂v∂v_sum = zeros(Float64,Ngrid,length(r_bins)-1)
        ∂v∂v_sum2 = zeros(Float64,Ngrid,length(r_bins)-1)
	end

	# looping over all of the files
	for n in range(1,n_files)
		
		infile = h5open("$(inpath)$(n).h5","r")

		# reading in the nth file's results
		N += infile["N"][] # N_realizations in the nth file
		ξ_sum .+= infile["two_pcf_sum"][]
		ξ_sum2 .+= infile["two_pcf_sum2"][]

		# diagnostic_mode results if true
		if diagnostic_mode == true && full_newtonian == false && wide_angle == false 
			
			δδ_sum .+= infile["dd_sum"][]
			δδ_sum2 .+= infile["dd_sum2"][]

			δ∂v_sum .+= infile["ddv_sum"][]
			δ∂v_sum2 .+= infile["ddv_sum2"][]

			∂vδ_sum .+= infile["dvd_sum"][]
			∂vδ_sum2 .+= infile["dvd_sum2"][]

			∂v∂v_sum .+= infile["dvdv_sum"][]
			∂v∂v_sum2 .+= infile["dvdv_sum2"][]
			
		end
		
		close(infile)
	end

	# computing the mean and variance (of the mean)
	
	ξ_mean = ξ_sum / N
	ξ_var = ξ_sum2 / N .- ξ_mean.^2

	# diagnostic mean and variance
	
	if diagnostic_mode == true && full_newtonian == false && wide_angle == false 

		δδ_mean = δδ_sum / N
		δδ_var = δδ_sum2 / N .- δδ_mean.^2

		δ∂v_mean = δ∂v_sum / N
		δ∂v_var = δ∂v_sum2 / N .- δ∂v_mean.^2

		∂vδ_mean = ∂vδ_sum / N
		∂vδ_var = ∂vδ_sum2 / N .- ∂vδ_mean.^2

		∂v∂v_mean = ∂v∂v_sum / N
		∂v∂v_var = ∂v∂v_sum2 / N .- ∂v∂v_mean.^2
	end

	# return more things if diagnostic_mode is on, so need a conditional 
	
	if diagnostic_mode == true && full_newtonian == false && wide_angle == false
		return Dict("two_pcf_mean" => ξ_mean, "two_pcf_var" => ξ_var, "dd_mean" => δδ_mean, "dd_var" => δδ_var, "ddv_mean" => δ∂v_mean, "ddv_var" => δ∂v_var, "dvd_mean" => ∂vδ_mean, "dvd_var" => ∂vδ_var, "dvdv_mean" => ∂v∂v_mean, "dvdv_var" => ∂v∂v_var, "N" => N)
	else
		return Dict("two_pcf_mean" => ξ_mean, "two_pcf_var" => ξ_var, "N" => N)
	end
end