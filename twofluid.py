#!/usr/bin/env python
# title          : Two-fluid plasma model in cylindrical coordinates
#description     : Tracks evoluton of heated plasma as it expands and moves
#                  in the axial and radial directions
#author          : Martin Goycoolea
#notes           : Adapted from Fortran 90 code provided by Anna Perona and
#                  Prof. Francesco Giammanco
#python_version  : python 3.3.1 , numpy 1.7.1
#==========================================================================
import numpy as np
empty = np.empty
vsum = np.sum
vexp = np.exp
dot = np.dot
where = np.where
vsqrt = np.sqrt
vabs = np.abs
zeros_like = np.zeros_like
empty_like = np.empty_like
fmax = np.fmax
fmin = np.fmin
vsign = np.sign

##########################
# System Input Variables #
##########################

rdomain = 5
rpoints = 10
zdomain = 10
zpoints = 10

dt = 1e-12
nstep = int(1e0)
save_freq = int(1e0)
scheme = 'heun' #either euler, heun, or rk4

T_0 = 0.1
max_den_0 = 1e14
radius_den_0 = 2
Bz_ext = 0


c = 3e10 #speed of light
gas_gamma = 5/3
number_saves = nstep//save_freq

####################
# Helper Functions #
####################


def create_grids(rdomain, rpoints, zdomain, zpoints):
    ''' Return grids where r(z) domain is the physical length of R (Z)
    and r (z) points is the number of grid points. The first
    point is a distance dr (dz) * 1/2 from the origin.'''

    R = np.linspace(0, rdomain, rpoints)
    dr = R[1] - R[0]
    R += dr/2
    Z = np.linspace(0, zdomain, zpoints)
    dz = Z[1] - Z[0]
    Z += dz/2
    Rgrid, Zgrid = np.meshgrid(R, Z, indexing='ij')
    return R, dr, Z, dz, Rgrid, Zgrid



####################################
# Conservative Evolution Equations #
####################################    

def radial_corrections(R):
    Rsize = np.size(R) + 2
    idomain = np.arange(0, Rsize) + 1/2
    i = idomain[1:-1]
    ip = idomain[2:]
    im = idomain[:-2]

    curvature_mod = 1/(6*idomain[:-1])
    centered = 1 - 1/(12*ip*im)
    forward =  1 - 1/(12*i*ip)
    backward =  1 - 1/(12*i*im)
    
    ones = np.ones(zpoints)
    curvature_mod = np.meshgrid(curvature_mod, ones, indexing='ij')[0]
    backward = np.meshgrid(backward, ones, indexing='ij')[0]
    centered = np.meshgrid(centered, ones, indexing='ij')[0]
    forward = np.meshgrid(forward, ones, indexing='ij')[0]

    return curvature_mod, backward, centered, forward

                        
def PM_states(var, r_bc ='open', z_bci='open', z_bcf='open'):


    #R-dir: create an array with ghost cells for boundary conditions
    # open: open ended, same values
    # inv: the value must change signs e.g. r-momentum in reflective boundary on r
    # ref: value must be reflected yet not made negative e.g. pressure
    var_r_bc = empty((rpoints+2, zpoints))
    var_r_bc[:-2,:] = var 
    if r_bc=='open':
         var_r_bc[-2,:] = var_r_bc[-3,:]
         var_r_bc[-1,:] = var_r_bc[-3,:]
    if r_bc=='inv':
        var_r_bc[-2,:] = -var_r_bc[-3,:]
        var_r_bc[-1,:] = -var_r_bc[-4,:]
    if r_bc=='ref':
        var_r_bc[-2,:] = var_r_bc[-3,:]
        var_r_bc[-1,:] = var_r_bc[-4,:]

    # Differences that will be used as input the slope limiter with corrections
    # Note that the indexing starts from the second point (the origin volume is treated separately)
    r_diff = np.diff(var_r_bc, axis=0)
    r_forw = r_diff[1:,:]
    r_back = r_diff[:-1,:]

    U_rf = 2*r_forw / forward
    U_rb = 2*r_back / backward
    U_rc = 1/2*(r_forw+r_back) / centered

    # Computing the slope
    signUr = vsign(U_rf)
    r_slope = empty_like(r_diff)
    r_slope[1:,:] = fmax(0,fmin(vabs(U_rf), fmin(U_rb*signUr, U_rc*signUr) ) )
    #The first point is special because it can't do central differences, take forward instead.
    # (A backward differencing in the second point is the same as forward differencing on first)
    r_slope[0,:] = r_back[0,:] / backward[0,:] 
  
    # The pm states for r!
    rm = var_r_bc[:-2,:] + 1/2*r_slope[:-1,:]*(1 - curvature_mod[:-1,:])
    rp = var_r_bc[1:-1,:] - 1/2*r_slope[1:,:]*(1 + curvature_mod[1:,:])

    
    # Z-direction, same definitions as r-dir, just two boundaries instead of one. 
    var_z_bc = empty((rpoints, zpoints+4))
    var_z_bc[:,2:-2] = var
    if z_bci=='open':
         var_z_bc[:,1] = var_z_bc[:,2]
         var_z_bc[:,0] = var_z_bc[:,2]
    if z_bci=='inv':
        var_z_bc[:,1] = -var_z_bc[:,2]
        var_z_bc[:,0] = -var_z_bc[:,3]
    if z_bci=='ref':
        var_z_bc[:,1] = var_z_bc[:,2]
        var_z_bc[:,0] = var_z_bc[:,3]

    if z_bcf=='open':
         var_z_bc[:,-1] = var_z_bc[:,-3]
         var_z_bc[:,-2] = var_z_bc[:,-3]
    if z_bcf=='inv':
        var_z_bc[:,-1] = -var_z_bc[:,-4]
        var_z_bc[:,-2] = -var_z_bc[:,-3]
    if z_bcf=='ref':
        var_z_bc[:,-1] = var_z_bc[:,-4]
        var_z_bc[:,-2] = var_z_bc[:,-3]

    # Computing differences and slopes
    z_diff = np.diff(var_z_bc, axis=1)
    z_forw = z_diff[:,1:]
    z_back = z_diff[:,:-1]

    U_zf = 2*z_forw 
    U_zb = 2*z_back
    U_zc = 1/2*(z_forw+z_back) 
    signUz = vsign(U_zf) 
    z_slope = fmax(0,fmin(vabs(U_zf), fmin(U_zb*signUz, U_zc*signUz) ) )

    zm = var_z_bc[:,1:-2] + 1/2*z_slope[:,:-1]
    zp = var_z_bc[:,2:-1] - 1/2*z_slope[:,1:]
    
    return rm, rp, zm, zp
    


def compute_next_step(var, r_flux, z_flux, sources):
    
    next_var = var - dt * ( ((Rgrid+1/2*dr)*r_flux[1:,:] - (Rgrid-1/2*dr)*r_flux[:-1,:])/(Rgrid*dr) +
                            (z_flux[:,1:] - z_flux[:,:-1])/dz + sources)

    return next_var

def flux_function(flux_m, flux_p, var_m, var_p, axis):

    if axis == 'r':
        flux = empty((rpoints+1, zpoints))
        flux[0,:] = 0
        flux[1:,:] = 1/2*(flux_m + flux_p - dr/dt*(var_p - var_m))
    if axis == 'z':
        flux = 1/2*(flux_m + flux_p - dz/dt*(var_p - var_m))
    return flux


############################
# Main Program Starts Here #
############################

# System Initial Conditions

R, dr, Z, dz, Rgrid, Zgrid = create_grids(rdomain, rpoints, zdomain, zpoints)
curvature_mod, backward, centered, forward =  radial_corrections(R)

N_e = np.ones_like(Rgrid) #max_den_0*vexp(-(Zgrid/radius_den_0)**2)
NVr_e = zeros_like(Rgrid)
NVt_e = zeros_like(Rgrid)
NVz_e = zeros_like(Rgrid)
En_e = T_0*N_e/(gas_gamma-1)

N_i = max_den_0*vexp(-(Rgrid/radius_den_0)**2)
NVr_i = zeros_like(Rgrid)
NVt_i = zeros_like(Rgrid)
NVz_i = zeros_like(Rgrid)
En_i = T_0*N_i/(gas_gamma - 1)


# Initial values for each time step
N_e_0 = empty_like(Rgrid)
NVr_e_0 = empty_like(Rgrid)
NVt_e_0 = empty_like(Rgrid)
NVz_e_0 = empty_like(Rgrid)
En_e_0 = empty_like(Rgrid)

N_i_0 = empty_like(Rgrid)
NVr_i_0 = empty_like(Rgrid)
NVt_i_0 = empty_like(Rgrid)
NVz_i_0 = empty_like(Rgrid)
En_i_0 = empty_like(Rgrid)

#Saving Paramaters
save_dim = (number_saves, rpoints, zpoints)
N_e_out = empty(save_dim)
Vr_e_out = empty(save_dim)
Vt_e_out = empty(save_dim)
Vz_e_out = empty(save_dim)
En_e_out = empty(save_dim)

N_i_out = empty(save_dim) 
Vr_i_out = empty(save_dim)
Vt_i_out = empty(save_dim)
Vz_i_out = empty(save_dim)
En_i_out = empty(save_dim)

Er_out = empty(save_dim)


#Time Loop
for t in range(nstep):
    time = t*dt
    
    ####################
    # Saving Variables #
    ####################
    if t%save_freq == 0:
        i = t//save_freq
        N_e_out[i] = N_e
        N_i_out[i] = N_i
        Vr_e_out[i] = NVr_e / N_e
        Vr_i_out[i] = NVr_i / N_i
        Vt_e_out[i] = NVt_e / N_e
        Vt_i_out[i] = NVt_i / N_i
        Vz_e_out[i] = NVz_e / N_e
        Vz_i_out[i] = NVz_i / N_i
        En_e_out[i] = En_e
        En_i_out[i] = En_i

    ###########################
    # Value at this time step #
    ###########################
    N_e_0[:,:] = N_e
    NVr_e_0[:,:] = NVr_e
    NVt_e_0[:,:] = NVt_e
    NVz_e_0[:,:] = NVz_e
    En_e_0[:,:] = En_e

    N_i_0[:,:] = N_i
    NVr_i_0[:,:] = NVr_i
    NVt_i_0[:,:] = NVt_i
    NVz_i_0[:,:] = NVz_i
    En_i_0[:,:] = En_i

    # Reconstruct conserved variables at the minus and plus side of interfaces with proper boudary conditions

    N_rm, N_rp, N_zm, N_zp = PM_states(N_e_0, z_bci = 'ref', z_bcf='ref')
    NVr_rm, NVr_rp, NVr_zm, NVr_zp = PM_states(NVr_e_0, z_bci = 'ref', z_bcf='ref')
    NVt_rm, NVt_rp, NVt_zm, NVt_zp = PM_states(NVt_e_0, z_bci = 'ref', z_bcf='ref')
    NVz_rm, NVz_rp, NVz_zm, NVz_zp = PM_states(NVz_e_0, z_bci = 'inv', z_bcf='inv')
    En_rm, En_rp, En_zm, En_zp = PM_states(En_e_0, z_bci = 'ref', z_bcf='ref')


    # Reconstruct velocities and pressures at interfaces
    Vr_rm = NVr_rm / N_rm
    Vt_rm = NVt_rm / N_rm
    Vz_rm = NVz_rm / N_rm

    Vr_rp = NVr_rp / N_rp
    Vt_rp = NVt_rp / N_rp
    Vz_rp = NVz_rp / N_rp
    
    Vr_zm = NVr_zm / N_zm
    Vt_zm = NVt_zm / N_zm
    Vz_zm = NVz_zm / N_zm

    Vr_zp = NVr_zp / N_zp
    Vt_zp = NVt_zp / N_zp
    Vz_zp = NVz_zp / N_zp

    P_rm = (En_rm - 1/2*N_rm*(Vr_rm**2 + Vt_rm**2 + Vz_rm**2))*(gas_gamma-1)
    P_rp = (En_rp - 1/2*N_rp*(Vr_rp**2 + Vt_rp**2 + Vz_rp**2))*(gas_gamma-1)

    P_zm = (En_zm - 1/2*N_zm*(Vr_zm**2 + Vt_zm**2 + Vz_zm**2))*(gas_gamma-1)
    P_zp = (En_zp - 1/2*N_zp*(Vr_zp**2 + Vt_zp**2 + Vz_zp**2))*(gas_gamma-1)

    # Source terms:
    # This has to be added eventually, for now none.
    N_source = 0
    NVr_source = 0
    NVt_source = 0
    NVz_source = 0
    En_source = 0

    # Compute fluxes at each side of the interface.
    N_rm_flux = NVr_rm
    N_rp_flux = NVr_rp
    N_zm_flux = NVz_zm
    N_zp_flux = NVz_zp

    NVr_rm_flux = NVr_rm*Vr_rm + P_rm
    NVr_rp_flux = NVr_rp*Vr_rp + P_rp
    NVr_zm_flux = NVr_zm*Vz_zm + P_zm
    NVr_zp_flux = NVr_zp*Vz_zp + P_zp

    NVt_rm_flux = NVt_rm*Vr_rm + P_rm
    NVt_rp_flux = NVt_rp*Vr_rp + P_rp
    NVt_zm_flux = NVt_zm*Vz_zm + P_zm
    NVt_zp_flux = NVt_zp*Vz_zp + P_zp

    NVz_rm_flux = NVz_rm*Vr_rm + P_rm
    NVz_rp_flux = NVz_rp*Vr_rp + P_rp
    NVz_zm_flux = NVz_zm*Vz_zm + P_zm
    NVz_zp_flux = NVz_zp*Vz_zp + P_zp

    En_rm_flux = (En_rm + P_rm)*Vr_rm
    En_rp_flux = (En_rp + P_rp)*Vr_rp
    En_zm_flux = (En_zm + P_zm)*Vz_zm
    En_zp_flux = (En_zp + P_zp)*Vz_zp


    N_r_flux = flux_function(N_rm_flux, N_rp_flux, N_rm, N_rp, 'r')
    N_z_flux = flux_function(N_zm_flux, N_zp_flux, N_zm, N_zp, 'z')
    NVr_r_flux = flux_function(NVr_rm_flux, NVr_rp_flux, NVr_rm, NVr_rp, 'r')
    NVr_z_flux = flux_function(NVr_zm_flux, NVr_zp_flux, NVr_zm, NVr_zp, 'z')
    NVt_r_flux = flux_function(NVt_rm_flux, NVt_rp_flux, NVt_rm, NVt_rp, 'r')
    NVt_z_flux = flux_function(NVt_zm_flux, NVt_zp_flux, NVt_zm, NVt_zp, 'z')
    NVz_r_flux = flux_function(NVz_rm_flux, NVz_rp_flux, NVz_rm, NVz_rp, 'r')
    NVz_z_flux = flux_function(NVz_zm_flux, NVz_zp_flux, NVz_zm, NVz_zp, 'z')
    En_r_flux = flux_function(En_rm_flux, En_rp_flux, En_rm, En_rp, 'r')
    En_z_flux = flux_function(En_zm_flux, En_zp_flux, En_zm, En_zp, 'z')
    

    # Evolution:
    N_e = compute_next_step(N_e_0, N_r_flux, N_z_flux, N_source)
    NVr_e = compute_next_step(NVr_e_0, NVr_r_flux, NVr_z_flux, NVr_source)
    NVt_e = compute_next_step(NVt_e_0, NVt_r_flux, NVt_z_flux, NVt_source)
    NVz_e = compute_next_step(NVz_e_0, NVz_r_flux, NVz_z_flux, NVz_source)
    En_e = compute_next_step(En_e_0, En_r_flux, En_z_flux, En_source)




###################
# Post-processing #
###################

def quick_plot(i, electrons = 1, ions=0, fields=0):
    import matplotlib.pyplot as plt
    if electrons:
        plt.figure(1)
        plt.subplot(221)
        plt.plot(den_e_out[i] * 10**19)
        
        plt.subplot(222)
        plt.plot(T_e_out[i])
        
        plt.subplot(223)
        plt.plot(vr_e_out[i]/c)

        plt.subplot(224)
        plt.plot(vth_e_out[i]/c)
        
        plt.show()
        

    if ions:
        plt.figure(2)
        plt.subplot(221)
        plt.plot(den_i_out[i] * 10**19)
        
        plt.subplot(222)
        plt.plot(T_i_out[i])
        
        plt.subplot(223)
        plt.plot(vr_i_out[i]/c)

        plt.subplot(224)
        plt.plot(vth_i_out[i]/c)
        
        plt.show()

    if fields:
        plt.figure(3)
        plt.subplot(221)
        plt.plot(Er_out[i])
        
        plt.subplot(222)
        plt.plot(Bz_int_out[i])
        
        plt.subplot(223)
        plt.plot(gradvr_e_out[i])

        plt.subplot(224)
        plt.plot(gradvth_e_out[i])
        
        plt.show()
    print('t= ' + str(dt*i*save_freq))


#Extra Stuff for now
#####################
# Fields and Energy #
#####################


def compute_electric_field(charge_sep, debye):
   dummy = charge_sep * debye * 6e10
   numerator = np.arange(1,rpoints+1)
   dummy *= numerator
   dummy2 = np.cumsum(dummy)
   denominator = 1/numerator
   dummy2 *= denominator
   return dummy2

def compute_magnetic_field(den_e, den_i, vth_e, vth_i, dr):
    dummy = 2*dr*(den_i*vth_i - den_e*vth_e) # Question: Factor of 2?
    dummy2 = np.flipud(dummy)
    dummy2[0] = 0
    dummy = np.cumsum(dummy2)
    dummy2 = np.flipud(dummy)

    return dummy2

def compute_thermalizations(R, time, radius_den_0, time_discharge, vol_discharge,
                            net_energy_in, discharge_on, den_e, den_i, T_e, T_i):

    if discharge_on:
        energy_point = net_energy_in*(1-np.exp(-time/time_discharge))*np.exp(-(R/radius_den_0)**2)
        discharge_factor = time_discharge * max_den_0 * vol_discharge
        energy_factor = energy_point / (2.4*discharge_factor) * np.exp(-time/time_discharge)
    else:
        energy_factor = 0

    condition_e = (den_e > 1e-20) * (T_e > 1)
    condition_i = (den_i > 1e-20) * (T_i > 1)
    
    # Change 
    coul_e = where(condition_e,
                   26-.5*np.log(den_e/T_e),
                   zero_vector)

    coul_i = where(condition_i,
                   26-.5*np.log(den_i/T_i),
                   zero_vector)

    collision_vel = where(condition_e,
                          3e13*coul_e*den_e/(T_e**1.5),
                          zero_vector)

    collision_ii = where(condition_i,
                         1.8e12*coul_i*den_i/(T_i**1.5),
                         zero_vector)

    collision_ei = where(condition_e,
                         3.2e10*coul_e*den_e/(T_e**1.5),
                         zero_vector)

    return energy_factor, collision_vel, collision_ii, collision_ei


def compute_debye(T_e, T_i, den_e, den_i):
    output = 2.5e-7*np.sqrt(1/(den_e/T_e+den_i/T_i))
    return output

def integration_scheme(scheme):
    if scheme == 'euler':
        time_vector = np.array([0])
        weight_vector = np.array([[1]])
    elif scheme == 'rk4':
        time_vector = np.array([0,0.5,0.5,1])
        weight_vector = np.array([[1/6,1/3,1/3,1/6]]) #note the shape of (1,4), it allows dot()
    elif scheme == 'heun':
        time_vector = np.array([0,1])
        weight_vector = np.array([[1/2, 1/2]])
    else:
        print("Sorry, we don't have that integration scheme")

    return time_vector, weight_vector
