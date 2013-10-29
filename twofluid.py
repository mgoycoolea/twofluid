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
from scipy import sparse as spr
empty = np.empty
vsum = np.sum
vexp = np.exp
dot = np.dot
where = np.where
vsqrt = np.sqrt
vabs = np.abs
zeros_like = np.zeros_like
empty_like = np.empty_like
maximum = np.maximum
minimum = np.minimum
fmax = np.fmax
fmin = np.fmin

##########################
# System Input Variables #
##########################
rdomain = 20
rpoints = 20
zdomain = 10
zpoints = 10
T_0 = 0.1
max_den_0 = 1e14
radius_den_0 = 2
Bz_ext = 0

dt = 1e-10
nstep = int(1e2)
save_freq = int(1e0)
scheme = 'heun' #either euler, heun, or rk4


####################
# Global Variables #
####################

c = 3e10 #speed of light
gas_gamma = 5/3
number_saves = nstep//save_freq

####################
# Helper Functions #
####################

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

def create_grids(rdomain, rpoints, zdomain, zpoints):
    ''' Return array where rdomain is the length of R
    and rpoints is the number of grid points. The first
    point is a distance dr/2 from the origin.'''


    R = np.linspace(0, rdomain, rpoints)
    dr = R[1] - R[0]
    R += dr/2
    Z = np.linspace(0, zdomain, zpoints)
    dz = Z[1] - Z[0]
    Rgrid, Zgrid = np.meshgrid(R, Z, indexing='ij')
    return R, dr, Z, dz, Rgrid, Zgrid


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

####################################
# Conservative Evolution Equations #
####################################

def radial_corrections(R):

    i_p = np.arange(1, rpoints + 1) + 1/2
    i = np.arange(0, rpoints) + 1/2
    i_m = np.arange(-1, rpoints - 1) + 1/2 
    i_m[0] = i_m[1] #This can't be negative

    curvature_mod = 1/(6*i)
    centered = 1 - 1/(12*i_p*i_m)
    forward =  1 - 1/(12*i*i_p)
    backward =  1 - 1/(12*i*i_m)
    
    ones = np.ones(zpoints)
    curvature_mod = np.meshgrid(curvature_mod, ones, indexing='ij')[0]
    backward = np.meshgrid(backward, ones, indexing='ij')[0]
    centered = np.meshgrid(centered, ones, indexing='ij')[0]
    forward = np.meshgrid(forward, ones, indexing='ij')[0]

    return curvature_mod, backward, centered, forward


def evolve_vars(N, NVr, NVt, NVz, En, B, species):

    # Reconstruct variables at the left and right of interfaces
    # Eg. o-L|R-o- ...- o-L|R-o where o are the middle values and | the interfaces


    # In the r-direction
    N_L_r, N_R_r = RL_states(N, 'r')
    NVr_L_r, NVr_R_r = RL_states(NVr, 'r')
    NVt_L_r, NVt_R_r = RL_states(NVt, 'r')
    NVz_L_r, NVz_R_r = RL_states(NVz, 'r')
    En_L_r, En_R_r = RL_states(En, 'r')

    # In the z-direction
    N_L_z, N_R_z = RL_states(N, 'z')
    NVr_L_z, NVr_R_z = RL_states(NVr, 'z')
    NVt_L_z, NVt_R_z = RL_states(NVt, 'z')
    NVz_L_z, NVz_R_z = RL_states(NVz, 'z')
    En_L_z, En_R_z = RL_states(En, 'z')


    # Reflective boundary conditions in z. Only z-momentum is opposite!
    N_L_z[:,0] =  N_R_z[:,0]
    N_R_z[:,-1] =  N_L_z[:,-1]
    NVr_L_z[:,0] =  NVr_R_z[:,0]
    NVr_R_z[:,-1] =  NVr_L_z[:,-1]
    NVt_L_z[:,0] =  NVt_R_z[:,0]
    NVt_R_z[:,-1] =  NVt_L_z[:,-1]
    NVz_L_z[:,0] =  -NVz_R_z[:,0]
    NVz_R_z[:,-1] =  -NVz_L_z[:,-1]
    En_L_z[:,0] =  En_R_z[:,0]
    En_R_z[:,-1] =  En_L_z[:,-1]

    # Reconstruct velocities and pressures at interfaces
    Vr_L_r = NVr_L_r / N_L_r
    Vt_L_r = NVt_L_r / N_L_r
    Vz_L_r = NVz_L_r / N_L_r

    Vr_R_r = NVr_R_r / N_R_r
    Vt_R_r = NVt_R_r / N_R_r
    Vz_R_r = NVz_R_r / N_R_r
    
    Vr_L_z = NVr_L_z / N_L_z
    Vt_L_z = NVt_L_z / N_L_z
    Vz_L_z = NVz_L_z / N_L_z

    Vr_R_z = NVr_R_z / N_R_z
    Vt_R_z = NVt_R_z / N_R_z
    Vz_R_z = NVz_R_z / N_R_z

    P_L_r = (En_L_r - 1/2*N_L_r*(Vr_L_r**2 + Vt_L_r**2 + Vz_L_r**2))*(gas_gamma-1)
    P_R_r = (En_R_r - 1/2*N_R_r*(Vr_R_r**2 + Vt_R_r**2 + Vz_R_r**2))*(gas_gamma-1)

    P_L_z = (En_L_z - 1/2*N_L_z*(Vr_L_z**2 + Vt_L_z**2 + Vz_L_z**2))*(gas_gamma-1)
    P_R_z = (En_R_z - 1/2*N_R_z*(Vr_R_z**2 + Vt_R_z**2 + Vz_R_z**2))*(gas_gamma-1)

    # Source terms:
    # This has to be added eventually, for now none.
    N_source = 0
    NVr_source = 0
    NVt_source = 0
    NVz_source = 0
    En_source = 0

    # Compute fluxes at each side of the interface.
    N_Lflux_r = NVr_L_r
    N_Rflux_r = NVr_R_r
    N_Lflux_z = NVz_L_z
    N_Rflux_z = NVz_R_z

    NVr_Lflux_r = NVr_L_r*Vr_L_r + P_L_r
    NVr_Rflux_r = NVr_R_r*Vr_R_r + P_R_r
    NVr_Lflux_z = NVr_L_z*Vz_L_z + P_L_z
    NVr_Rflux_z = NVr_R_z*Vz_R_z + P_R_z

    NVt_Lflux_r = NVt_L_r*Vr_L_r + P_L_r
    NVt_Rflux_r = NVt_R_r*Vr_R_r + P_R_r
    NVt_Lflux_z = NVt_L_z*Vz_L_z + P_L_z
    NVt_Rflux_z = NVt_R_z*Vz_R_z + P_R_z

    NVz_Lflux_r = NVz_L_r*Vr_L_r + P_L_r
    NVz_Rflux_r = NVz_R_r*Vr_R_r + P_R_r
    NVz_Lflux_z = NVz_L_z*Vz_L_z + P_L_z
    NVz_Rflux_z = NVz_R_z*Vz_R_z + P_R_z

    En_Lflux_r = (En_L_r - P_L_r)*Vr_L_r
    En_Rflux_r = (En_R_r - P_R_r)*Vr_R_r
    En_Lflux_z = (En_L_z - P_L_z)*Vz_L_z
    En_Rflux_z = (En_R_z - P_R_z)*Vz_R_z


    N_flux_r = flux_function(N_Lflux_r, N_Rflux_r, N_L_r, N_R_r, 'r')
    N_flux_z = flux_function(N_Lflux_z, N_Rflux_z, N_L_z, N_R_z, 'z')
    NVr_flux_r = flux_function(NVr_Lflux_r, NVr_Rflux_r, NVr_L_r, NVr_R_r, 'r')
    NVr_flux_z = flux_function(NVr_Lflux_z, NVr_Rflux_z, NVr_L_z, NVr_R_z, 'z')
    NVt_flux_r = flux_function(NVt_Lflux_r, NVt_Rflux_r, NVt_L_r, NVt_R_r, 'r')
    NVt_flux_z = flux_function(NVt_Lflux_z, NVt_Rflux_z, NVt_L_z, NVt_R_z, 'z')
    NVz_flux_r = flux_function(NVz_Lflux_r, NVz_Rflux_r, NVz_L_r, NVz_R_r, 'r')
    NVz_flux_z = flux_function(NVz_Lflux_z, NVz_Rflux_z, NVz_L_z, NVz_R_z, 'z')
    En_flux_r = flux_function(En_Lflux_r, En_Rflux_r, En_L_r, En_R_r, 'r')
    En_flux_z = flux_function(En_Lflux_z, En_Rflux_z, En_L_z, En_R_z, 'z')
    

    # Evolution:
    N_new = compute_next_step(N, N_flux_r, N_flux_z, N_source)
    NVr_new = compute_next_step(NVr, NVr_flux_r, NVr_flux_z, NVr_source)
    NVt_new = compute_next_step(NVt, NVt_flux_r, NVt_flux_z, NVt_source)
    NVz_new = compute_next_step(NVz, NVz_flux_r, NVz_flux_z, NVz_source)
    En_new = compute_next_step(En, En_flux_r, En_flux_z, En_source)

    return N_new, NVr_new, NVt_new, NVz_new, En_new, B


def RL_states(var, axis):
    
    limiter = slope_limiter(var, axis)

    if axis == 'r':
        U_L = empty((rpoints + 1, zpoints))
        U_R = empty((rpoints + 1, zpoints))
        U_L[1:,:] = var + 1/2*limiter*(1 - curvature_mod)
        U_R[1:-1,:] = var[1:,:] - 1/2*limiter[1:,:]*(1 + curvature_mod[1:,:])
        U_R[-1,:] = U_L[-1,:]

    if axis == 'z':
        U_L = empty((rpoints, zpoints + 1))
        U_R = empty((rpoints, zpoints + 1)) # One more interface than points in z-dir
        U_L[:,1:] = var + 1/2*limiter
        U_R[:,:-1] = var - 1/2*limiter
        # Must set the initial conditions for each variable at the first L and the last R state. 
    return U_L, U_R

def slope_limiter(var, axis):
    if axis == 'z':
        var_p = empty_like(var)
        var_p[:,:-1] = var[:,1:]
        var_p[:,-1] = var_p[:,-2]

        var_m = empty_like(var)
        var_m[:,1:] = var[:,:-1]
        var_m[:,0] = var_m[:,1]

        Up = var_p - var
        Um = var - var_m
        
        signUp = np.sign(Up) 

        lim_slope = fmax(0,fmin(2*vabs(Up), 
                                fmin(2*Um*signUp, 
                                     1/2*(Up+Um)*signUp) ) )


    if axis == 'r':

        var_p = empty_like(var)
        var_p[:-1,:] = var[1:,:]
        var_p[-1,:] = var_p[-2,:]

        var_m = empty_like(var)
        var_m[1:,:] = var[:-1,:]
        var_m[0,:] = var_m[1,:]

        Up = var_p - var
        Um = var - var_m
        
        signUp = np.sign(Up) 
        lim_slope = signUp*fmax(0, 
                                fmin(2*vabs(Up)/forward, 
                                     fmin(2*signUp*Um/backward,
                                          1/2*signUp*(Up+Um)/centered)))
    
    return lim_slope
        
    


def compute_next_step(var, flux_r, flux_z, sources):
    
    next_var = var - dt * ( ((Rgrid+1/2*dr)*flux_r[1:,:] - (Rgrid-1/2*dr)*flux_r[:-1,:])/(Rgrid*dr) +
                            (flux_z[:,1:] - flux_z[:,:-1])/dz + sources)
    
    next_var[0,:] =  var[0,:] - dt * ( 2*flux_r[1,:]/dr +
                            (flux_z[0,1:] - flux_z[0,:-1])/dz + sources)

    return next_var

def flux_function(Flux_L, Flux_R, Var_L, Var_R, axis):

    if axis == 'r':
        flux = empty((rpoints + 1, zpoints))
        flux[,:] = 1/2*(Flux_R + Flux_L - dr/dt*(Var_R - Var_L))
    if axis == 'z':
        flux = empty((rpoints, zpoints+1))
        flux = 1/2*(Flux_R + Flux_L - dz/dt*(Var_R - Var_L))
    return flux


############################
# Main Program Starts Here #
############################

# System Initial Conditions

R, dr, Z, dz, Rgrid, Zgrid = create_grids(rdomain, rpoints, zdomain, zpoints)
curvature_mod, backward, centered, forward =  radial_corrections(R)

N_e = max_den_0*vexp(-(Zgrid/radius_den_0)**2)
NVr_e = zeros_like(Rgrid)
NVt_e = zeros_like(Rgrid)
NVz_e = zeros_like(Rgrid)
En_e = T_0*N_e/(gas_gamma-1)

N_i = max_den_0*vexp(-(Rgrid/radius_den_0)**2)
NVr_i = zeros_like(Rgrid)
NVt_i = zeros_like(Rgrid)
NVz_i = zeros_like(Rgrid)
En_i = T_0*N_i/(gas_gamma - 1) 


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

N_new, NVr_new, NVt_new, NVz_new, En_new, B = evolve_vars(N_e, NVr_e, NVt_e, NVz_e, En_e, 0, 'e')
#Time Loop
for t in range(nstep):
    time = t*dt

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

        #Bz_int = compute_magnetic_field(den_e, den_i, vth_e, vth_i, dr)
        #Er_tot = compute_electric_field(charge_sep, debye)

        
        #masse = vsum(den_e)*dr
        #massi = vsum(den_i)*dr
        #Keenergy = vsum((vr_e**2 + vth_e**2)*den_e*10**19 * 9.1094e-28 )*0.5
        #Kienergy = vsum((vr_i**2 + vth_i**2)*den_i*10**19 * 1.6726e-24 )*0.5
        #Benergy = vsum(Bz_int**2) / (np.pi*8)
        #Eenergy = vsum(Er_tot**2) / (np.pi*8)
        #print(Keenergy,Kienergy, Benergy, Eenergy, Keenergy + Kienergy + Benergy + Eenergy)
        #v_e_max = vsqrt(vabs(-1e18*Er_tot*debye - 3.2e15*gradP_e*dr))
        #v_i_max = vsqrt(vabs(dr*(6e14*Er_tot - 2e12*gradP_i)))

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
