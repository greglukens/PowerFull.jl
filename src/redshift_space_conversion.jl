using Dierckx, HDF5, Statistics

"""
MULTI-THREADED IMPLEMENTATION OF delta_to_delta_s FUNCTION

The below function converts real space density contrast on a regular grid (i.e., from GridSPT) to redshift space

Inputs 
- Ngrid: number of grid points along each dimension (assuming a cube)
- central_redshift: the redshift of the central grid point (i.e., index [Ngrid/2 + 1,Ngrid/2 + 1,Ngrid/2 + 1])
- b: the linear galaxy bias
- δ: linear density contrast on a regular grid (from GridSPT)
- uz: linear z-velocity field on a regular grid (from GridSPT)
- uy: linear y-velocity field on a regular grid (from GridSPT)
- ux: linear x-velocity field on a regular grid (from GridSPT)
- los_info: grid geometry from the observer_grid_geometry function
- cosmology: a dictionary of cosmological splines for a given cosmology, the output of the initialize_cosmology function
- wide_angle: if true: wide angle effect, if false: flat-sky approximation
- full_newtonian: if true: all cosmological evolution effects, if false: simple Kaiser expression
- diagnostic_mode: if true (and wide_angle + full_newtonian true as well): return component 2PCF values as well

Outputs (as a Dictionary)
- δ_s - the redshift space density contrast at every point in the grid

    - Additionally:
    if diagnostic_mode + wide_angle + full_newtonian are true:
        - ∂v_term - the velocity derivative term of the redshift space conversion expression at every point
    """

function delta_to_delta_s(Ngrid::Int,central_redshift::Real,b::Real,δ::AbstractArray,uz::AbstractArray,uy::AbstractArray,ux::AbstractArray,los_info::Dict,cosmology::Dict,wide_angle::Bool,full_newtonian::Bool)
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
                @inbounds uz_zcol = @view(uz[i,j,:]) 	# column of constant X and Y (i.e., a column of u_z points along z)
                @inbounds ∂uz_z[i,j,:] = derivative(Spline1D(x_par,uz_zcol,s=0),x_par,nu=1) 	# computing derivative of spline of column as a function of x_parallel (or z)
            end
        end

        if full_newtonian == false 		# if just Kaiser formula, just follow top case of eq (13) in documentation (alpha = 0, no evolution effects)

            δ_s = b*δ   +   f_of_z(central_redshift)* ∂uz_z 	
            δ_s .*= D_of_z(central_redshift) 	# rescale by D_c at end

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
                @inbounds ur_zcol = @view(ur[i,j,:])	# column of u_r points across z (so constant X and Y)
                @inbounds ur_ycol = @view(ur[i,:,j]) 	# column of u_r points across y (so constant X and Z)
                @inbounds ur_xcol = @view(ur[:,i,j]) 	# column of u_r points across x (so constant Y and Z)

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
    
    return δ_s 		# otherwise just need delta_s
end


function delta_to_delta_s_fullsky(Ngrid::Int,b::Real,δ::AbstractArray,uz::AbstractArray,uy::AbstractArray,ux::AbstractArray,los_info::Dict,cosmology::Dict)
    @assert Ngrid > 0
    @assert b > 0
    @assert size(δ) == size(ux) == size(uy) == size(uz)
    
    # all lines of sight (wide angle)
    
    x_par = los_info["x_par"]
    x_perp = los_info["x_perp"]
    f_obs = los_info["f_obs"]
    D_obs = los_info["D_obs"]	

    # wide angle (z-hat is now n-hat to each galaxy/point in grid)


    ur = ux .* los_info["n_hat_x"] + uy .* los_info["n_hat_y"] + uz .* los_info["n_hat_z"] 		# compute radial velocity

    check_for_nans = findall(isnan,ur)
    ur[check_for_nans] .= 0

    ∂ur_x = zeros(Float64,Ngrid,Ngrid,Ngrid) 	# initialize derivative arrays
    ∂ur_y = zeros(Float64,Ngrid,Ngrid,Ngrid)
    ∂ur_z = zeros(Float64,Ngrid,Ngrid,Ngrid)

    Threads.@threads for i in range(1,Ngrid) 	# loop over X, Y, or Z (depending on which array)
        for j in range(1,Ngrid) 	# loop over Y, Z, or X (depending on which array)
            @inbounds ur_zcol = @view(ur[i,j,:])	# column of u_r points across z (so constant X and Y)
            @inbounds ur_ycol = @view(ur[i,:,j]) 	# column of u_r points across y (so constant X and Z)
            @inbounds ur_xcol = @view(ur[:,i,j]) 	# column of u_r points across x (so constant Y and Z)

            @inbounds ∂ur_z[i,j,:] = derivative(Spline1D(x_par,ur_zcol,s=0),x_par,nu=1) 	# interpolated derivative of column along z (du_r/dz)
            @inbounds ∂ur_y[i,:,j] = derivative(Spline1D(x_perp,ur_ycol,s=0),x_perp,nu=1) 	# interpolated derivative of column along y (du_r/dy)
            @inbounds ∂ur_x[:,i,j] = derivative(Spline1D(x_perp,ur_xcol,s=0),x_perp,nu=1)		# interpolated derivative of column along x (du_r/dx)
        end
    end

    # compute du_r/dr now, using spherical coordinates
    
    ∂ur_r = los_info["cos_φ"] .* los_info["sin_θ"] .* ∂ur_x   +    los_info["sin_φ"] .* los_info["sin_θ"] .* ∂ur_y    +    los_info["cos_θ"] .* ∂ur_z 

    δ_s = b*δ   .+   f_obs .* ∂ur_r
    δ_s .*= D_obs
    
    # returning both delta_s and the v-term (since we already have real-space delta outside of the function) if diagnostic mode is on
    
    return δ_s 		# otherwise just need delta_s
end


function delta_to_delta_s_velocity(Ngrid::Int,b::Real,δ::AbstractArray,uz::AbstractArray,uy::AbstractArray,ux::AbstractArray,los_info::Dict,cosmology::Dict)
    @assert Ngrid > 0
    @assert b > 0
    @assert size(δ) == size(ux) == size(uy) == size(uz)
    
    # all lines of sight (wide angle)
    
    x_par = los_info["x_par"]
    x_perp = los_info["x_perp"]

    # wide angle (z-hat is now n-hat to each galaxy/point in grid)


    ur = (ux .* los_info["n_hat_x"]) + (uy .* los_info["n_hat_y"]) + (uz .* los_info["n_hat_z"]) 		# compute radial velocity

    check_for_nans = findall(isnan,ur)
    ur[check_for_nans] .= 0

    ∂ur_x = zeros(Float64,Ngrid,Ngrid,Ngrid) 	# initialize derivative arrays
    ∂ur_y = zeros(Float64,Ngrid,Ngrid,Ngrid)
    ∂ur_z = zeros(Float64,Ngrid,Ngrid,Ngrid)

    Threads.@threads for i in range(1,Ngrid) 	# loop over X, Y, or Z (depending on which array)
        for j in range(1,Ngrid) 	# loop over Y, Z, or X (depending on which array)
            @inbounds ur_zcol = @view(ur[i,j,:])	# column of u_r points across z (so constant X and Y)
            @inbounds ur_ycol = @view(ur[i,:,j]) 	# column of u_r points across y (so constant X and Z)
            @inbounds ur_xcol = @view(ur[:,i,j]) 	# column of u_r points across x (so constant Y and Z)

            @inbounds ∂ur_z[i,j,:] = derivative(Spline1D(x_par,ur_zcol,s=0),x_par,nu=1) 	# interpolated derivative of column along z (du_r/dz)
            @inbounds ∂ur_y[i,:,j] = derivative(Spline1D(x_perp,ur_ycol,s=0),x_perp,nu=1) 	# interpolated derivative of column along y (du_r/dy)
            @inbounds ∂ur_x[:,i,j] = derivative(Spline1D(x_perp,ur_xcol,s=0),x_perp,nu=1)		# interpolated derivative of column along x (du_r/dx)
        end
    end

    # compute du_r/dr now, using spherical coordinates
    
    ∂ur_r = los_info["cos_φ"] .* los_info["sin_θ"] .* ∂ur_x   +    los_info["sin_φ"] .* los_info["sin_θ"] .* ∂ur_y    +    los_info["cos_θ"] .* ∂ur_z 
    
    # returning both delta_s and the v-term (since we already have real-space delta outside of the function) if diagnostic mode is on
    
    return ∂ur_r 		# otherwise just need delta_s
end

function delta_to_delta_s_velocity_z(Ngrid::Int,b::Real,δ::AbstractArray,uz::AbstractArray,uy::AbstractArray,ux::AbstractArray,los_info::Dict,cosmology::Dict)
    @assert Ngrid > 0
    @assert b > 0
    @assert size(δ) == size(ux) == size(uy) == size(uz)
    
    # all lines of sight (wide angle)
    
    x_par = los_info["x_par"]

    # wide angle (z-hat is now n-hat to each galaxy/point in grid)


    check_for_nans = findall(isnan,uz)
    uz[check_for_nans] .= 0

    ∂uz_z = zeros(Float64,Ngrid,Ngrid,Ngrid)

    Threads.@threads for i in range(1,Ngrid) 	# loop over X, Y, or Z (depending on which array)
        for j in range(1,Ngrid) 	# loop over Y, Z, or X (depending on which array)
            @inbounds uz_zcol = @view(uz[i,j,:])	# column of u_z points across z (so constant X and Y)

            @inbounds ∂uz_z[i,j,:] = derivative(Spline1D(x_par,uz_zcol,s=0),x_par,nu=1) 	# interpolated derivative of column along z (du_r/dz)
        end
    end

    # compute du_r/dr now, using spherical coordinates
    
    
    # returning both delta_s and the v-term (since we already have real-space delta outside of the function) if diagnostic mode is on
    
    return ∂uz_z 		# otherwise just need delta_s
end