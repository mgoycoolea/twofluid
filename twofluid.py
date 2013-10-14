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
copy = np.copy
empty = np.empty
vsum = np.sum
vexp = np.exp
grad = np.gradient
dot = np.dot
clip = np.clip
where = np.where
vsqrt = np.sqrt
vabs = np.abs

##########################
# System Input Variables #
##########################
rdomain = 10
rpoints = 1000
T_0 = 0.1
max_energy_in = 5
max_den_0 = 1e-5
radius_den_0 = 2
time_discharge = 1e-8
vol_discharge = 1000
Bz_ext = 2e3

deltat = 1e-11
nstep = int(1e2)
save_freq = int(1e0)
scheme = 'heun' #either euler, heun, or rk4

discharge_on = False

electrons_on = True
ions_on = False

controls_on = False

####################
# Global Variables #
####################

c = 3e10
zero_vector = np.zeros(rpoints)
number_save_points = nstep//save_freq

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

def create_grids(rdomain, rpoints):
    ''' Return array where rdomain is the length of R
    and rpoints is the number of grid points. The first
    point is a distance dr/2 from the origin.'''

    dr = rdomain / rpoints
    n = arange(rpoints)
    R = dr*(n + 1/2)
    return R, dr


#####################
# Fields and Energy #
#####################

def calc_energy_in(max_den_0, vol_discharge, max_energy_in):
    ''' Compute if the max energy of the capacitor can
    excite the volume of electrons needed.'''

    energy_ionize = 1.6*13.6*max_den_0*vol_discharge

    if max_energy_in > energy_ionize:
        net_energy_in =  max_energy_in - energy_ionize
    else:
        net_energy_in = 0
        print('Not enough energy to ionize volume!')

    return net_energy_in, energy_ionize


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



def compute_gradients(N_e, T_e, vr_e, vth_e,
                       N_i, T_i, vr_i, vth_i, dr):
    
    gradlog_e = grad(N_e, dr)
    gradlog_i = grad(N_i, dr)
    gradT_e = grad(T_e, dr)
    gradT_i = grad(T_i, dr)
    gradvr_e = grad(vr_e, dr)
    gradvr_i = grad(vr_i, dr)
    gradvth_e = grad(vth_e, dr)
    gradvth_i = grad(vth_i, dr)

    gradP_e = gradT_e + T_e*gradlogn_e
    gradP_i = gradT_i + T_i*gradlogn_i

    return (gradlogn_e, gradT_e, gradvr_e, gradvth_e, gradP_e,
            gradlogn_i, gradT_i, gradvr_i, gradvth_i, gradP_i)

def compute_debye(T_e, T_i, den_e, den_i):
    output = clip(2.5e-7*np.sqrt((T_e+T_i)/(den_e+den_i)), 0, dr)
    return output

def computevar0(var):
    var0 = empty(rpoints)
    var0[1:-1] = (var[2:] + var[0:-2])/2
    var0[0] = (var[1] + var[0])/2
    var0[-1] = (var[-2] + var[-1])/2

    return var0
####################################
# Conservative Evolution Equations #
####################################

def flux_limiter(Up, Um):
    sign = where(Up > 0, 1, -1)
    lim_slope = sign*maximum(0, 
                             minimum(2*vabs(Up), 
                                      minimum(2*sign*Um,
                                              1/2*sign*(Up+Um))))
    return lim_slope


def dvar_dt(var, flux, source, dt, dr, dz, R, Vr):

    var_p = empty(rpoints)
    var_m = empty(rpoints)
    
    # Shifting the variable 1 entry up or down to do vector operations
    var_p[0:-1] =  var[1:]
    var_p[-1] = 0

    var_m[1:] = var[0:-1]
    var_m[0] = var_m[1]


    phi_diff = 1/2*((var_p - var) - (var - var_m)) #include Courant number to reduce viscocity

    limgrad = flux_limiter(var_p - var, var - var_m)

    limgrad_p = empty(rpoints)
    limgrad_m = empty(rpoints)
    # Shifting the variable 1 entry up or down to do vector operations
    
    limgrad_p[0:-1] =  limgrad[1:]
    limgrad_p[-1] =  0

    limgrad_m[1:] =  limgrad[0:-1]
    limgrad_m[0] =  limgrad_m[1]

    varL_p = var + 1/2 * limgrad
    varR_p = var_p - 1/2 * limgrad_p
    varL_m = var_m + 1/2 * limgrad_m
    varL_m = var - 1/2 * limgrad
    


############################
# Main Program Starts Here #
############################
#import pdb; pdb.set_trace()
time_vector, weight_vector = integration_scheme(scheme)
zero_vector = np.zeros(rpoints)

# System Initial Conditions
R, dr = create_grids(rdomain, rpoints)
net_energy_in, energy_ionize = calc_energy_in(max_den_0, vol_discharge, max_energy_in)

logn_e = -(R/radius_den_0)**2
logn_i = -(R/radius_den_0)**2
T_e = T_0*np.ones(rpoints)
T_i = T_0*np.ones(rpoints)
vr_e = 5e7*np.sqrt(T_e)
vr_i = np.zeros(rpoints)
vth_e = np.zeros(rpoints)
vth_i = np.zeros(rpoints)


#energy_0 = 0.5 * vsum(den_e * vr_e**2)*dr

#Saving Paramaters
order_integ = np.size(time_vector)

N_e_rk = empty((order_integ, rpoints)) 
N_i_rk = empty((order_integ, rpoints)) 
T_e_rk = empty((order_integ, rpoints))
T_i_rk = empty((order_integ, rpoints))
vr_e_rk = empty((order_integ, rpoints))
vr_i_rk = empty((order_integ, rpoints))
vth_e_rk = empty((order_integ, rpoints))
vth_i_rk = empty((order_integ, rpoints))

N_e_out = empty((number_save_points, rpoints)) 
N_i_out = empty((number_save_points, rpoints)) 
T_e_out = empty((number_save_points, rpoints))
T_i_out = empty((number_save_points, rpoints))
vr_e_out = empty((number_save_points, rpoints))
vr_i_out = empty((number_save_points, rpoints))
vth_e_out = empty((number_save_points, rpoints))
vth_i_out = empty((number_save_points, rpoints))
Er_out = empty((number_save_points, rpoints))
Bz_int_out = empty((number_save_points, rpoints))

gradN_e_out = empty((number_save_points, rpoints))
gradvr_e_out = empty((number_save_points, rpoints))
gradvth_e_out = empty((number_save_points, rpoints))
gradP_e_out = empty((number_save_points, rpoints))

 

#Time Loop
for t in range(nstep):
    time = t*deltat
    
    if (time > 5*time_discharge):
        discharge_on = False

    N_e_0 = copy(logn_e) 
    N_i_0 = copy(logn_i) 
    T_e_0 = copy(T_e)
    T_i_0 = copy(T_i)
    vr_e_0 = copy(vr_e)
    vr_i_0 = copy(vr_i)
    vth_e_0 = copy(vth_e)
    vth_i_0 = copy(vth_i)


    for n in range(order_integ):
        
        ############################
        # Fields, Energy and Grads #
        ############################
                
        time_rk = time + deltat*time_vector[n]
        den_i = max_den_0 * vexp(logn_i)
        den_e = max_den_0 * vexp(logn_e)
        #charge_sep = den_i - den_e
        charge_sep = 0
        #debye = compute_debye(T_e, T_i, den_e, den_i)
        debye = dr
        #Er_tot = compute_electric_field(charge_sep, debye)
        Er_tot = 0
        #Bz_int = compute_magnetic_field(den_e, den_i, vth_e, vth_i, dr)
        Bz_int = 0
        Bz_tot = 0 # Bz_int + Bz_ext
        
        (energy_factor, collision_vel,
         collision_ii, collision_ei) = 0,0,0,0 #compute_thermalizations(R, time_rk, radius_den_0, time_discharge, vol_discharge,
        #net_energy_in, discharge_on, den_e, den_i, T_e, T_i)

        (gradlogn_e, gradT_e, gradvr_e, gradvth_e, gradP_e,
         gradlogn_i, gradT_i, gradvr_i, gradvth_i, gradP_i) = compute_gradients(logn_e, T_e, vr_e, vth_e,
                                                                                logn_i, T_i, vr_i, vth_i, dr)

        if n == 0:
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
                den_e_out[i] = copy(max_den_0*vexp(logn_e)) 
                den_i_out[i] = copy(max_den_0*vexp(logn_i)) 
                T_e_out[i] = copy(T_e)
                T_i_out[i] = copy(T_i)
                vr_e_out[i] = copy(vr_e)
                vr_i_out[i] = copy(vr_i)
                vth_e_out[i] = copy(vth_e)
                vth_i_out[i] = copy(vth_i)
                Er_out[i] = copy(Er_tot)
                Bz_int_out[i] = copy(Bz_int)
                gradlogn_e_out[i] = copy(gradlogn_e)
                gradvr_e_out[i] = copy(gradvr_e)
                gradvth_e_out[i] = copy(gradvth_e)
                gradP_e_out[i] = copy(gradP_e)
                #print('mass =' + str(dr * vsum(den_e)) + ' energy =' + str(E - 0.5*dr*vsum(dene_e * (vr_e**2 vth_e**2))* + energy_0*r) )

        ######################
        #    Electron Eqs    #
        ######################

        if electrons_on:
            logn_e_rk[n] =  dlogne_dt(R, vr_e, gradvr_e, gradlogn_e)
            T_e_rk[n] = dTe_dt(R, vr_e, gradvr_e, T_e, T_i, gradT_e, collision_ei, den_e, energy_factor)
            vr_e_rk[n] = dVre_dt(vr_e, vth_e, gradvr_e, gradP_e, collision_vel, Er_tot, Bz_tot)
            vth_e_rk[n] = dVthe_dt(vr_e, vth_e, gradvth_e, collision_vel, Bz_tot)
        else:
            logn_e_rk[n] = zero_vector
            T_e_rk[n] = zero_vector
            vr_e_rk[n] = zero_vector
            vth_e_rk[n] = zero_vector

        ######################
        #       Ion Eqs      #
        ######################
            
        if ions_on:
            logn_i_rk[n] =  dlogni_dt(R, vr_i, gradvr_i, gradlogn_i)
            T_i_rk[n] = dTi_dt(T_e, T_i, collision_ei)
            vr_i_rk[n] = dVri_dt(vr_i, vth_i, gradvr_i, gradP_i, collision_ii, Er_tot, Bz_tot)
            vth_i_rk[n] = dVthi_dt(vr_i, vth_i, gradvth_i, collision_ii, Bz_tot)
        else:
            logn_i_rk[n] = zero_vector
            T_i_rk[n] = zero_vector
            vr_i_rk[n] = zero_vector
            vth_i_rk[n] = zero_vector

        ################################
        # RK4 Section, Adjusting Steps #
        ################################

        if (order_integ > 1) and (n < 3):
            rk_step = time_vector[n+1]
            logn_e = logn_e_0 + logn_e_rk[n] * (deltat*rk_step)
            logn_i = logn_i_0 + logn_i_rk[n] * (deltat*rk_step)
            T_e = T_e_0 + T_e_rk[n] * (deltat*rk_step)
            T_i = T_i_0 + T_i_rk[n] * (deltat*rk_step)
            vr_e = vr_e_0 + vr_e_rk[n] * (deltat*rk_step)
            vr_i = vr_i_0 + vr_i_rk[n] * (deltat*rk_step)
            vth_e = vth_e_0 + vth_e_rk[n] * (deltat*rk_step)
            vth_i = vth_i_0 + vth_i_rk[n] * (deltat*rk_step)

    #######################
    #   System Evolution  #    
    ####################### 
    
    # Apply weight to each RK element and sum them
    logn_e = logn_e_0 + deltat*dot(weight_vector, logn_e_rk)[0]
    logn_i = logn_i_0 + deltat*dot(weight_vector, logn_i_rk)[0]
    T_e = T_e_0 + deltat*dot(weight_vector, T_e_rk)[0]
    T_i = T_i_0 + deltat*dot(weight_vector, T_i_rk)[0]
    vr_e = vr_e_0 + deltat*dot(weight_vector, vr_e_rk)[0]
    vr_i = vr_i_0 + deltat*dot(weight_vector, vr_i_rk)[0]
    vth_e = vth_e_0 + deltat*dot(weight_vector, vth_e_rk)[0]
    vth_i = vth_i_0 + deltat*dot(weight_vector, vth_i_rk)[0]

    
    
    ###############
    #   Control   #
    ###############
    if controls_on:
        vr_e = clip(vr_e, -v_e_max, v_e_max)
        vr_i = clip(vr_i, -v_i_max, v_i_max)
        vth_e = clip(vth_e, -v_e_max, v_e_max)
        vth_i = clip(vth_i, -v_i_max, v_i_max)






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
    print('t= ' + str(deltat*i*save_freq))
