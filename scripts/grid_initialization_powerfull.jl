using PyCall, Dierckx, LinearAlgebra

"""
The below function creates a dictionary of splines for important cosmological parameters (e.g., linear growth rate, growth factor, etc.) given a cosmology 

Inputs
- omega_m: the matter density parameter (at present)
- omega_l: the dark energy density parameter (at present)
- omega_r: the radiation density parameter (at present)
- w: the dark energy equation of state (assuming wCDM cosmologies)
- b: the linear galaxy bias

Outputs (as Dictionary)
- r_of_z: spline of comoving distance R as a function of redshift z
- z_of_r: spline of redshift z as a function of comoving distance r
- H_of_z: spline of H(z)/c in h/Mpc? No: returns dimensionless E(z); convert to E(z)/(c/H0) for PowerFull table
- f_of_z: spline of linear growth rate f as a function of redshift z
- D_of_z: spline of linear growth factor D as a function of redshift z
- α1_of_z: spline of α1 as a function of redshift z
- α2_of_z: spline of α2 as a function of redshift z
- α3_of_z: spline of α3 as a function of redshift z
"""

function initialize_cosmology(omega_m::Real,omega_l::Real,omega_r::Real,w0::Real,b;wa::Real = 0,zmax::Real=10,Nz::Real=10000,omega_k::Real=0)
	
	#function that takes Ω_m (present day matter density parameter), Ω_Λ (present day dark energy density parameter), Ω_r (present day radiation density parameter), w (dark energy equation of state), and b_g (linear galaxy bias) as inputs
	
	# already had this function lying around in Python, no real reason to have to do it in Python though
	# NOTE TO REVIEWER: this is sloppy work, if I haven't converted to Julia yet here, I will soon, so don't worry too much about this function


	@assert omega_m >= 0
	@assert omega_l >= 0
	@assert omega_r >= 0
	

	py"""
    import numpy as np
    from scipy.integrate import solve_ivp, quad

    # ---------------------------------------------------------------------
    # Constants (keep H0=1 so H(a) is dimensionless; distances in c/H0 units)
    # ---------------------------------------------------------------------
    H_0 = 1.0
    c_over_H0 = 299792.458 / 100.0  # 2997.92458 Mpc/h (exact c / (100 km/s/Mpc))
    delta_c = 1.686

    # ---------------------------
    # CPL and Komatsu-2009 pieces
    # ---------------------------
    def w_CPL(a, w0, wa):
        return w0 + wa*(1.0 - a)

    def w_eff_CPL(a, w0, wa):
        a = np.atleast_1d(a)
        la = np.log(a)
        # Avoid division by zero at a=1
        weff = np.empty_like(a)
        mask = np.abs(la) < 1e-8
        weff[mask] = w0
        weff[~mask] = w0 + wa * ((la[~mask] - a[~mask] + 1.0) / la[~mask])
        return weff if weff.size > 1 else weff.item()


    def E2(a, Om, Ode, Or, Ok, w0, wa):
        weff = w_eff_CPL(a, w0, wa)
        de_term = Ode * a**(-3.0*(1.0 + weff))
        return Or*a**(-4.0) + Om*a**(-3.0) + Ok*a**(-2.0) + de_term

    def Omega_m_a(a, Om, Ode, Or, Ok, w0, wa):
        return Om*a**(-3.0) / E2(a, Om, Ode, Or, Ok, w0, wa)

    def Omega_de_a(a, Om, Ode, Or, Ok, w0, wa):
        weff = w_eff_CPL(a, w0, wa)
        return (Ode * a**(-3.0*(1.0 + weff))) / E2(a, Om, Ode, Or, Ok, w0, wa)

    def Omega_k_a(a, Om, Ode, Or, Ok, w0, wa):
        return Ok*a**(-2.0) / E2(a, Om, Ode, Or, Ok, w0, wa)

    def H_of_a(a, Om, Ode, Or, Ok, w0, wa):
        return H_0 * np.sqrt(E2(a, Om, Ode, Or, Ok, w0, wa))

    # -----------------------------------------------------
    # Growth equation for g(a) from Komatsu et al. (2009):
    # d^2 g / d(ln a)^2 + A(a) dg/d(ln a) + B(a) g = 0,
    # where
    #  A = 5/2 + 1/2 [Ω_k(a) - 3 w_eff(a) Ω_de(a)],
    #  B = 2 Ω_k(a) + 3/2 [1 - w_eff(a)] Ω_de(a).
    # -----------------------------------------------------
    def _g_system(ln_a, y, Om, Ode, Or, Ok, w0, wa):
        a = np.exp(ln_a)
        g, gp = y
        weff = w_eff_CPL(a, w0, wa)
        Ok_a = Omega_k_a(a, Om, Ode, Or, Ok, w0, wa)
        Ode_a = Omega_de_a(a, Om, Ode, Or, Ok, w0, wa)
        A = 2.5 + 0.5*(Ok_a - 3.0*weff*Ode_a)
        B = 2.0*Ok_a + 1.5*(1.0 - weff)*Ode_a
        gpp = -A*gp - B*g
        return [gp, gpp]

    def _solve_g(a_grid, Om, Ode, Or, Ok, w0, wa):
        a_ini = a_grid[0]
        # Matter era ICs: D ~ a => g=D/a ~ const, so g' ~ 0
        g_ini  = 1.0
        gp_ini = 0.0
        sol = solve_ivp(
            fun=lambda x,y: _g_system(x,y,Om,Ode,Or,Ok,w0,wa),
            t_span=(np.log(a_ini), 0.0),
            y0=[g_ini, gp_ini],
            t_eval=np.log(a_grid),
            method="BDF", rtol=1e-8, atol=1e-10
        )
        g  = sol.y[0]
        gp = sol.y[1]                      # dg/d ln a
        # Reconstruct D and f
        D = a_grid * g
        D /= D[-1]                         # normalize D(a=1)=1
        f = 1.0 + gp/g                     # f = d ln D / d ln a
        # For dlog f we need g' and g'': g'' from ODE RHS at each a_grid:
        dlog_f = np.empty_like(g)
        for k, a in enumerate(a_grid):
            weff = w_eff_CPL(a, w0, wa)
            Ok_a = Omega_k_a(a, Om, Ode, Or, Ok, w0, wa)
            Ode_a= Omega_de_a(a, Om, Ode, Or, Ok, w0, wa)
            A = 2.5 + 0.5*(Ok_a - 3.0*weff*Ode_a)
            B = 2.0*Ok_a + 1.5*(1.0 - weff)*Ode_a
            gpp = -A*gp[k] - B*g[k]
            # df/dln a = (g'' g - g'^2)/g^2
            df = (gpp*g[k] - gp[k]**2) / (g[k]**2)
            dlog_f[k] = df / f[k]
        return g, D, f, dlog_f

    # -------------------------------------------
    # Comoving distance r(z)
    # -------------------------------------------
    def _chi_of_z(z, Om, Ode, Or, Ok, w0, wa):
        def integrand(zp):
            a = 1.0/(1.0+zp)
            return c_over_H0 / np.sqrt(E2(a, Om, Ode, Or, Ok, w0, wa))
        return quad(integrand, 0.0, z, epsrel=1e-7, epsabs=0.0)[0]


    def ODE_solver(omega_m, omega_l, omega_r, w0, wa, zmax, Na, omega_k):
        Om  = float(omega_m)
        Ode = float(omega_l)
        Or  = float(omega_r)
        Ok  = float(omega_k)

        # Grid
        a = np.logspace(np.log10(1.0/(1.0+zmax)), 0.0, Na)
        z = 1.0/a - 1.0

        # Growth (g, D, f, dlog f)
        g, D, f, dlog_f = _solve_g(a, Om, Ode, Or, Ok, w0, wa)

        # Background arrays
        E2_ = E2(a, Om, Ode, Or, Ok, w0, wa)
        H  = H_0*np.sqrt(E2_)

        # Distances
        r = np.array([_chi_of_z(zz, Om, Ode, Or, Ok, w0, wa) for zz in z])

        # Omegas and alpha-terms (with w_eff, consistent with the g-equation)
        weff = np.array([w_eff_CPL(ai, w0, wa) for ai in a])
        omega_m_a = Omega_m_a(a, Om, Ode, Or, Ok, w0, wa)
        omega_de_a= Omega_de_a(a, Om, Ode, Or, Ok, w0, wa)
        omega_k_a = Omega_k_a(a, Om, Ode, Or, Ok, w0, wa)

        alpha_3_omegas = 1.0 - 0.5*(3.0*omega_m_a + 3.0*(1.0+weff)*omega_de_a + 2.0*omega_k_a)

        # Same α pieces you used
        alpha1_part = a * H * r * delta_c * f / c_over_H0
        alpha2      = -a * H * r * (f + dlog_f) / c_over_H0
        alpha3      = -a * H * r * alpha_3_omegas / c_over_H0

        # Match your exact return format/order (increasing z)
        return [
            np.flip(z),
            np.flip(g),
            np.flip(D/np.nanmax(D)),
            np.flip(f),
            np.flip(r),
            np.flip(H),
            np.flip(dlog_f),
            np.flip(alpha1_part),
            np.flip(alpha2),
            np.flip(alpha3),
            np.flip(omega_m_a),
        ]

    """

	# solve the ODE using the inputted cosmological parameters 
	
    cosmology = py"ODE_solver"(omega_m,omega_l,omega_r,w0,wa,zmax,Nz,omega_k)

	# obtain r(z), z(r), H(z), f(z), D(z), and alpha's(z) 
    r_of_z = Spline1D(cosmology[1],cosmology[5],s=0) #r(0) == 0
    z_of_r = Spline1D(cosmology[5],cosmology[1],s=0) 

    H_of_z = Spline1D(cosmology[1],cosmology[6],s=0) # dimensionless E(z); PowerFull table needs E(z)/(c/H0), not aH
    f_of_z = Spline1D(cosmology[1],cosmology[4],s=0)
    D_of_z = Spline1D(cosmology[1],cosmology[3],s=0) #normalize to now
    
    if b isa Real
        α1_of_z = Spline1D(cosmology[1],2 .- cosmology[8] * (b-1) ,s=0)
    else
        α1_of_z = Spline1D(cosmology[1],2 .- cosmology[8] .* (b.(cosmology[1]) .- 1) ,s=0)
    end

    
    α2_of_z = Spline1D(cosmology[1],cosmology[9],s=0)
    α3_of_z = Spline1D(cosmology[1],cosmology[10],s=0)

    omega_m_of_z = Spline1D(cosmology[1],cosmology[11],s=0)



	# use dictionary to make handling output much easier
	
    return Dict("r_of_z" => r_of_z, "z_of_r" => z_of_r, "H_of_z" => H_of_z, "f_of_z" => f_of_z, "D_of_z" => D_of_z, "α1_of_z" => α1_of_z, "α2_of_z" => α2_of_z, "α3_of_z" => α3_of_z, "omega_m_of_z" => omega_m_of_z)
end







"""
The below function initializes the grid geometry for an observer at the origin and a regular grid of points centered at redshift z_c

Inputs
- L: box size of the grid
- Ngrid: number of grid points along each dimension (assuming a cube)
- central_redshift: the redshift of the central grid point (i.e., index [Ngrid/2 + 1,Ngrid/2 + 1,Ngrid/2 + 1])
- cosmology: a dictionary of cosmological splines for a given cosmology, the output of the initialize_cosmology function

Outputs (as Dictionary)
- x_par: line-of-sight separation from central grid point (array)
- x_perp: perpendicular to the line-of-sight separation from central grid point (array)
- r_proj: the projected separation of every grid point from the central point relative to the observer 
- cos_θ: cosine of the θ spherical angle of every grid point
- sin_θ: sine of the θ spherical angle of every grid point
- cos_φ: cosine of the φ spherical angle of every grid point
- sin_φ: sine of the φ spherical angle of every grid point
- r_obs: comoving distance from observer to every grid point
- f_obs: linear growth rate at every grid point relative to observer
- D_obs: linear growth factor at every grid point relative to observer
- α1_obs: α1(r) at every grid point relative to observer
- α2_obs: α2(r) at every grid point relative to observer
- α3_obs: α3(r) at every grid point relative to observer
- n_hat_x: x-component of line-of-sight direction of every grid point
- n_hat_y: y-component of line-of-sight direction of every grid point
- n_hat_z: z-component of line-of-sight direction of every grid point
- r_z: comoving distance from observer to every grid point assuming flat-sky approximation
- f_z: linear growth rate at every grid point relative to observer assuming flat-sky approximation
- D_z: linear growth factor at every grid point relative to observer assuming flat-sky approximation
- α1_z: α1(z) at every grid point relative to observer assuming flat-sky approximation
- α2_z: α2(z) at every grid point relative to observer assuming flat-sky approximation
- α3_z: α3(z) at every grid point relative to observer assuming flat-sky approximation
- RA: φ but for rotation around y-axis for Healpix usage
- DEC: θ but for rotation around y-axis for Healpix usage
"""


function observer_grid_geometry(L::Real,Ngrid::Int,cosmology::Dict,central_redshift::Real;observer_offset::AbstractArray=[0,0,cosmology["r_of_z"](central_redshift)])
    @assert L > 0
    @assert Ngrid > 0
    
    # inputting the splines from the initialize_cosmology function
    
    #r_of_z = cosmology["r_of_z"]
    z_of_r = cosmology["z_of_r"]
    f_of_z = cosmology["f_of_z"]
    D_of_z = cosmology["D_of_z"]
    α1_of_z = cosmology["α1_of_z"]
    α2_of_z = cosmology["α2_of_z"]
    α3_of_z = cosmology["α3_of_z"]

    # setting up position arrays

    positions_xyz = LinRange{Float64}(0,L,Ngrid) 	# this is general X,Y,Z position in grid, with one corner set as the origin
    
    central_position = zeros(Float64,3) .+ positions_xyz[Int(Ngrid/2)+1] 	# this is the position of the "central" point (not quite the center, since even number typically)
    
    observer_position = central_position .- observer_offset 	# setting the position of the observer as centered on grid but r(z_c) comoving distance away along z

    println("Observer centered at: $(observer_position) Mpc/h in the grid...")

    check_x_pos = findall(positions_xyz .== observer_position[1])
    check_y_pos = findall(positions_xyz .== observer_position[2])
    check_z_pos = findall(positions_xyz .== observer_position[3])

    if length(check_x_pos) > 0 && length(check_y_pos) > 0 && length(check_z_pos) > 0
        observer_on_grid = true
    else
        observer_on_grid = false
    end


    # setting up x_parallel and x_perp, the separation of grid points from the central point along and perpendicular to the central line-of-sight, respectively
    
    x_par = LinRange{Float64}(-L/2,L/2,Ngrid) .- (central_position[1]-L/2)
    x_perp = x_par 		# both should span from -L/2 to L/2 (although, x_perp really only goes from 0 to L/2, but this is just a choice)

    
    # initializing the projected separation from the center axis (i.e., |r_perp|)
    
    r_proj = zeros(Float64,Ngrid,Ngrid,Ngrid)
    
    r_slice = zeros(Float64,Ngrid,Ngrid) 	# just a dummy array; the projected distances should be the same for each slice along z
    
    for i in range(1,Ngrid) 	# loop over X
        for j in range(1,Ngrid) 	# loop over Y
            p = [positions_xyz[i],positions_xyz[j]] - central_position[1:2]
            r_slice[i,j] = sqrt(p[1]^2 + p[2]^2) 	# projected position of (X,Y) point
        end
    end

    for i in range(1,Ngrid) 	# loop over Z
        r_proj[:,:,i] = r_slice
    end
    
    r_obs = zeros(Float64,Ngrid,Ngrid,Ngrid) 	# matrix of comoving distance from observer to each point

    for i in range(1,Ngrid) 	# loop over Z
        r_obs[:,:,i] = sqrt.(r_slice.^2 .+ (x_par[i] .+ observer_offset[3]).^2) 	# observed comoving distance is just sqrt[r_proj^2 + r_parallel^2]
    end

    # initializing X, Y, Z positions relative to observer (all comoving distances)
    
    X_obs = zeros(Float64,Ngrid,Ngrid,Ngrid)
    Y_obs = zeros(Float64,Ngrid,Ngrid,Ngrid)
    Z_obs = zeros(Float64,Ngrid,Ngrid,Ngrid)

    for i in range(1,Ngrid) 	# loop over X,Y, or Z
        for j in range(1,Ngrid) 	# loop over Y, Z, or X
            X_obs[:,i,j] = x_perp .+ observer_offset[1] 	# find X_grid - X_observer 
            Y_obs[i,:,j] = x_perp .+ observer_offset[2] 	# find Y_grid - X_observer
            Z_obs[i,j,:] = x_par .+ observer_offset[3]  	# find Z_grid - Z_observer
        end
    end

    # now that have X,Y,Z, can compute angles θ and φ
    
    cos_θ = Z_obs ./ r_obs 		# find cos θ

    if observer_on_grid == true
        cos_θ[check_x_pos[1],check_y_pos[1],check_z_pos[1]] = 0
    end

    check_for_gtr_one = findall(cos_θ .> 1)
    check_for_less_one = findall(cos_θ .< -1)

    if length(check_for_gtr_one) > 0
        cos_θ[check_for_gtr_one] .= 1
    end
        
    if length(check_for_less_one) > 0
        cos_θ[check_for_less_one] .= -1
    end

    #cos_θ[Int(Ngrid/2)+1, Int(Ngrid/2)+1, :] .= 1 	# assert that along central line-of-sight, cos θ = 1 (if not, have some rounding/floating point errors that cause domain error) 
    
    sin_θ = sin.(acos.(cos_θ)) 		# find sin θ

    φ = atan.(Y_obs,X_obs) 		# find φ (using the arctan function that is in -pi to pi domain)

    cos_φ = cos.(φ) 	# find cos φ
    sin_φ = sin.(φ)		# find sin φ

    
    # initialize n_hat component arrays

    n_hat_x = zeros(Float64,Ngrid,Ngrid,Ngrid)
    n_hat_y = zeros(Float64,Ngrid,Ngrid,Ngrid)
    n_hat_z = zeros(Float64,Ngrid,Ngrid,Ngrid)

    for i in range(1,Ngrid) 	# loop over X
        for j in range(1,Ngrid) 	# loop over Y
            for k in range(1,Ngrid) 	# loop over Z
                n = [positions_xyz[i],positions_xyz[j],positions_xyz[k]] - observer_position 	# compute n
                n_hat = n/norm(n) 	# compute n_hat
                n_hat_x[i,j,k] = n_hat[1] 	# compute x-component of n_hat
                n_hat_y[i,j,k] = n_hat[2] 	# compute y-component of n_hat
                n_hat_z[i,j,k] = n_hat[3] 	# compute z-component of n_hat
            end
        end
    end

    # create matrices that correspond to a given cosmological parameter at each grid point as a function of the comoving distance to each point
    
    f_obs = f_of_z.(z_of_r.(r_obs))
    D_obs = D_of_z.(z_of_r.(r_obs))
    α1_obs = α1_of_z.(z_of_r.(r_obs))
    α2_obs = α2_of_z.(z_of_r.(r_obs))
    α3_obs = α3_of_z.(z_of_r.(r_obs))

    

    # here, assuming z-hat is LoS to all grid points (so, small-sky version of above)
    
    r_z = zeros(Float64,Ngrid,Ngrid,Ngrid) 		# initializing matrix that gives Z-direction distance to each Z-plane of points, defined around CENTER
    
    if length(check_x_pos) > 0 && length(check_y_pos) > 0
        for i in range(1,Ngrid)
            r_z[:,:,i] .= r_obs[check_x_pos[1], check_y_pos[1], i]	# this is ill-defined if observer is inside grid, but will calculate regardless (will add warning later)
        end
    else
        for i in range(1,Ngrid)
            r_z[:,:,i] .= r_obs[Int(Ngrid/2) + 1, Int(Ngrid/2) + 1, i]	
        end
    end

    
    f_z = f_of_z.(z_of_r.(r_z))
    D_z = D_of_z.(z_of_r.(r_z))
    α1_z = α1_of_z.(z_of_r.(r_z))
    α2_z = α2_of_z.(z_of_r.(r_z))
    α3_z = α3_of_z.(z_of_r.(r_z))



    # sky coordinates

    cos_DEC = Y_obs ./ r_obs
    
    if observer_on_grid == true
        cos_DEC[check_x_pos[1],check_y_pos[1],check_z_pos[1]] = 0
    end

    check_for_gtr_one = findall(cos_DEC .> 1)
    check_for_less_one = findall(cos_DEC .< -1)

    if length(check_for_gtr_one) > 0
        cos_DEC[check_for_gtr_one] .= 1
    end
        
    if length(check_for_less_one) > 0
        cos_DEC[check_for_less_one] .= -1
    end


    DEC = acos.(cos_DEC)

    RA = atan.(-Z_obs,X_obs) .+ pi     # add 2π to get RA in range [0,2π]


    # use a dictionary to make handling output much easier
    
    return Dict("x_par" => x_par, "x_perp" => x_perp, "r_proj" => r_proj, "cos_θ" => cos_θ, "sin_θ" => sin_θ, "cos_φ" => cos_φ, "sin_φ" => sin_φ, "r_obs" => r_obs, "f_obs" => f_obs, "D_obs" => D_obs, "α1_obs" => α1_obs,"α2_obs" => α2_obs,"α3_obs" => α3_obs, "n_hat_x" => n_hat_x, "n_hat_y" => n_hat_y, "n_hat_z" => n_hat_z, "X_obs" => X_obs, "Y_obs" => Y_obs, "Z_obs" => Z_obs, "r_z" => r_z, "f_z" => f_z, "D_z" => D_z, "α1_z" => α1_z,"α2_z" => α2_z,"α3_z" => α3_z, "RA" => RA, "DEC" => DEC, "obs_pos" => observer_position)
end



function prep_dims_for_gridspt(los_info)
    cos_φ = los_info["cos_φ"]
    sin_φ = los_info["sin_φ"]
    cos_θ = los_info["cos_θ"]
    sin_θ = los_info["sin_θ"]
    n_hat_x = los_info["n_hat_x"]
    n_hat_y = los_info["n_hat_y"]
    n_hat_z = los_info["n_hat_z"]

    cos_φ_spt = permutedims(los_info["cos_φ"],[3,2,1])
    sin_φ_spt = permutedims(los_info["sin_φ"],[3,2,1])
    cos_θ_spt = permutedims(los_info["cos_θ"],[3,2,1])
    sin_θ_spt = permutedims(los_info["sin_θ"],[3,2,1])
    n_hat_x_spt = permutedims(los_info["n_hat_x"],[3,2,1])
    n_hat_y_spt = permutedims(los_info["n_hat_y"],[3,2,1])
    n_hat_z_spt = permutedims(los_info["n_hat_z"],[3,2,1])

    los_info["cos_φ"] = cos_φ 
    los_info["sin_φ"] = sin_φ  
    los_info["cos_θ"] = cos_θ  
    los_info["sin_θ"] = sin_θ 
    los_info["n_hat_x"] = n_hat_x 
    los_info["n_hat_y"] = n_hat_y 
    los_info["n_hat_z"] = n_hat_z 

    return los_info
end