# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:42:56 2017

@author: ilys
"""
from shutil import copyfile
import matplotlib.pyplot as plt
import subprocess as sbprcs
import numpy as np
import csv
from scipy.signal import butter, lfilter, freqz, resample
import os
import pickle  
from scipy.interpolate import interp1d
import scipy.constants as C
import inspect
import cmath
import matplotlib.pyplot as plt


amu = C.atomic_mass
pi = np.pi
me = C.electron_mass
e = C.elementary_charge
epsilon_0 = C.epsilon_0
c = C.speed_of_light # m/s
mu0 = C.mu_0
eV2J = e   #  J/eV conversion
kb = C.k  # J/K Bolzman const
h = C.h


#def eV2J(T):   #
#    return e*T   
   

def Aki_to_fki(Aki,Jk,Ji,lambd):
    # Aki in units 1e8 s**-1 
    # lambd in nm
    gk = 2*Jk + 1 # upper level
    gi = 2*Ji + 1 # lower level
    lambd = 1e-9*lambd # nm -> m
    
    Coeff = (2*pi*e**2)/(me*c*epsilon_0)    
    
    fki = Aki*(lambd**2)*gk/(gi*Coeff)
    return  fki   
      

    
    

def lambda_ion_elastic_Ar(p):   #
    ng = n_g(p) #   m-3 gas density 
    sigma_ia = 1e-18      # m^2     total (elastic + charge exchange) cross section     ArgonCollisions090201-1.pdf
    result = 1/(ng*sigma_ia)
    return result   
   
def lambda_D(n_e,Te):   # Debye length
    l_D = np.sqrt((C.epsilon_0*C.elementary_charge*Te)/(C.elementary_charge**2*n_e))
    return l_D   

def magnetisation_factor_e(Te,p, B):   # electron mobility 
    # Te in eV, ng in m^-3, p in Pa
    omega_c =  omega_e(B)
    vju_e =  vju_e_atom_Lieberman(Te,p)
    factor = 1/(1+(omega_c/vju_e)**2)
    return factor


def magnetisation_factor_i(p, M, B):   # electron mobility 
    # Te in eV, ng in m^-3, p in Pa
    omega_c =  omega_i(B,M)
    vju_i =  vju_i_atom_Lieberman(p,M)
    factor = 1/(1+(omega_c/vju_i)**2)
    return factor

def mju_e_magn(Te,p,B):   # electron mobility 
    # Te in eV,  p in Pa, B in T
    result = mju_e(Te,p)*magnetisation_factor_e(Te,p, B)
    return result

def mju_i_magn(p,M,B):   # ion mobility 
    #  p in Pa, M in amu, B in T
    result = mju_i(p,M)*magnetisation_factor_i(p,M, B)
    return result

def D_e_magn(Te,p,B):   # electron mobility 
    # Te in eV, ng in m^-3, p in Pa
    De = D_e(Te,p)*magnetisation_factor_e(Te,p, B)
    return De


def D_i_magn(p,M,B):   # electron mobility 
    # Te in eV, ng in m^-3, p in Pa
    Di = D_i(p,M)*magnetisation_factor_i(p,M, B)
    return Di



def Da_magn(Te,p,B,M):   # ambipolar diffusion 
    
    D_e_mgn   =  D_e_magn(Te,p,B)
    D_i_mgn   =  D_i_magn(p,M,B)
    mju_e_mgn =  mju_e_magn(Te,p,B)
    mju_i_mgn = mju_i_magn(p,M,B)
    result = (mju_e_mgn*D_i_mgn + mju_i_mgn*D_e_mgn)/(mju_e_mgn + mju_i_mgn)
    return result


def Da(Te,p,M):   # ambipolar diffusion 
    
    D_e_ = D_e(Te,p)
    D_i_ = D_i(p,M)
    mju_e_ =  mju_e(Te,p)
    mju_i_ = mju_i(p,M)
    result = (mju_i_*D_e_ + mju_e_*D_i_)/(mju_e_ + mju_i_)
    return result




def Dturb(Te,B):   # turbulent diffusion
    result = eV2J*Te/(e*B)
    return result


def D_e(Te,p):   # electron mobility 
    # Te in eV, ng in m^-3, p in Pa
    vju_e =  vju_e_atom_Lieberman(Te,p)
    result = eV2J*Te/(me*vju_e)
    return result
    
    
def D_i(p,M):   # electron mobility 
    # Te in eV, ng in m^-3, p in Pa
    Ti = 300.0
    vju_i =  vju_i_atom_Lieberman(p,M)
    result = kb*Ti/(M*amu*vju_i)
    return result

def mju_e(Te,p):   # electron mobility 
    # Te in eV, ng in m^-3, p in Pa
    vju_e =  vju_e_atom_Lieberman(Te,p)
    result = e/(me*vju_e)
    return result
    
    
def mju_i(p,M):   # electron mobility 
    # Te in eV, ng in m^-3, p in Pa
    vju_i =  vju_i_atom_Lieberman(p,M)
    result = e/(M*amu*vju_i)
    return result

def vju_e_atom_Lieberman(Te,p):   # electron elastic collision frequency 
    # Te in eV, ng in m^-3, p in Pa
    ng = n_g(p)
    K_ea = Kel(Te) # elastic cross section only from Liebermann and Lichtenberg
    result = ng*K_ea
    return result


def vju_i_atom_Lieberman(p,M):   #  ion collision frequency 
    # ng in m^-3, M in amu, p in Pa
    ng = n_g(p)
    vi = v_i(M)
    sigma_ia = 1e-18      # m^2     total (elastic + charge exchange) cross section     ArgonCollisions090201-1.pdf
    result = ng*sigma_ia*vi         # i - a collision rate
    return result
    
    
def v_ion_sound(Te,M): # electron thermal velocity
    #  T in eV, M in amu
    result =  np.sqrt((eV2J*Te)/(M*amu) )  
    return result

def v_e(Te): # electron thermal velocity
    #  T in eV
    result =  1.6*np.sqrt((eV2J*Te)/me)  
    return result
    
def v_i(M): # electron thermal velocity
    #  M in  amu
    Ti = 300.0
    result =  1.6*np.sqrt((kb*Ti)/(M*amu))  
    return result    
    
def n_g(p): # neutral gas density
    #  M in  amu, p in Pa
    Tg = 300.0
    result =  float(p)/(Tg*kb) # m-3 gas densit  
    return result   
  
    
def omega_e(B): # electron gyrofrequency in rad/s
    # B in Tesla, M in amu
    result =  e*B/(me)
    return result


def rho_e(B,Te): # electron gyroradius in m
    # B in Tesla, M in amu
    ve = v_e(Te)
    omega_c = omega_e(B)
    result =  ve/omega_c
    return result


def rho_i(B, M): # electron gyrofrequency in rad/s
    # B in Tesla, M in amu
    vi = v_i(M)
    omega_c = omega_i(B, M)
    result =  vi/omega_c
    return result



def omega_i(B,M): # ion gyrofrequency in rad/s
    # B in Tesla, M in amu
    result =  e*B/(M*amu)
    return result


def prnt(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    var_name =  [k for k, v in callers_local_vars if v is var]
    var_name = str(var_name)
    var_name = var_name[2:-2]
    print( var_name + ': ' + str(var) )


def copy(src, dst):
    copyfile(src, dst)
    print( 'File is copied' )



def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]



def showvar(var):
    varname = retrieve_name(var)
    varname = varname[0]
    print( varname + ':  ' + str(var) )

def addlist(a,b):
    res =  [x + y for x, y in zip(a, b)]
    return res
    
def subtrlist(a,b):
    res =  [x - y for x, y in zip(a, b)]
    return res

def divlist(a,b):
    res =  [x/y for x, y in zip(a, b)]
    return res


#------------------------------- waves in plasma -------------------------------------
#------------------------------- waves in plasma -------------------------------------
#------------------------------- waves in plasma -------------------------------------
#------------------------------- waves in plasma -------------------------------------


def interp1(x,y,xnew):
    f = interp1d(x, y, kind='cubic')
    res = f(xnew)
    return res

def notexist(varname):
    res = True
    if varname in locals():
        res =  False
    if varname in globals():
        res =  False
    return res    

def omega_pe(n_e):  # electron plasma frequency
    res = np.sqrt(n_e*e**2/(me*epsilon_0))
    return res

    
def omega_pi(Mi,n_e):  # ion plasma frequency
    mi = Mi*amu
    res = np.sqrt(n_e*e**2/(mi*epsilon_0)) 
    return res

def omega_ci_to_omega_pi(Mi, n_e, B):    
    omga_pi =  omega_pi(Mi,n_e)
    omga_ci =  omega_i(B, Mi)
    res = omga_ci/omga_pi
    return res
    
    
    

def omega_ion(Mi,B):
    mi = Mi*amu
    omega_ci = e*B/(mi)
    freq = omega_ci/(2*pi)*1e-6 # MHz
    return  freq

def f_LH(Mi,B,n_e):
    
    
    mi = Mi*amu
    omega_ce = e*B/(me)
    omega_ci = e*B/(mi)
    omega_pe = np.sqrt(n_e*e**2/(me*epsilon_0))
    omega_pi = np.sqrt(n_e*e**2/(mi*epsilon_0)) 
    
    res = np.sqrt(omega_ce*omega_ci*(omega_pe**2 + omega_ce*omega_ci)/(omega_pe**2 + omega_ce**2))
    
    return res/(2*pi)
 

def f_UH(B,n_e):
    
    omega_ce = e*B/(me)
    omega_pe = np.sqrt(n_e*e**2/(me*epsilon_0))
    res = np.sqrt(omega_pe**2 + omega_ce**2 )
    return res/(2*pi)
 
    
def f_ci(Mi,B):
    
    mi = Mi*amu
    
    res = e*B/(mi)
    return res/(2*pi) 

def f_ce(B):
    
      
    res = e*B/(me)
    return res/(2*pi) 
   
def etha_tensor_coll(Mi,B,n_e,omega):
    
    nju_i = 20e3
    nju_e = 8e6
    
    mi = Mi*amu
    omega_ce = -e*B/(me)
    omega_ci = e*B/(mi)
    omega_pe = np.sqrt(n_e*e**2/(me*epsilon_0))
    omega_pi = np.sqrt(n_e*e**2/(mi*epsilon_0)) 
    
    nju_compl_i = nju_i*1j   
    nju_compl_e = nju_e*1j 
    #etha_perp_diag     = 1 -  (omega + nju_compl_e)*omega_pe**2/(omega*(omega*(omega + nju_compl_e) - omega_ce**2)) - (omega + nju_compl_i)*omega_pi**2/(omega*(omega*(omega + nju_compl_i) - omega_ci**2))
   
     #  from Golant Gilinskij Sacharov  Volni v anisotropnoj plasme
    etha_perp_diag     = 1 -  (omega + nju_compl_e)*omega_pe**2/(omega*((omega + nju_compl_e)**2 - omega_ce**2)) - (omega + nju_compl_i)*omega_pi**2/(omega*((omega + nju_compl_i)**2 - omega_ci**2))
   
    etha_parallel_diag = 1 -  omega_pe**2/(omega*(omega + nju_compl_e))  - omega_pi**2/(omega*(omega + nju_compl_i)) #  from Golant Fedorov Visokochastotnije metodi nagreva plasmi
    etha_T             = omega_ci*omega_pi**2/(omega*((omega + nju_compl_i)**2 - omega_ci**2)) +  omega_ce*omega_pe**2/(omega*((omega + nju_compl_e)**2 - omega_ce**2)) 

    return (etha_perp_diag,etha_T,etha_parallel_diag)
    
        

def etha_tensor(Mi,B,n_e,omega):
    
    
    mi = Mi*amu
    omega_ce = -e*B/(me)
    omega_ci = e*B/(mi)
    omega_pe = np.sqrt(n_e*e**2/(me*epsilon_0))
    omega_pi = np.sqrt(n_e*e**2/(mi*epsilon_0)) 
    
    etha_perp_diag     = 1 -  omega_pe**2/(omega**2 - omega_ce**2) - omega_pi**2/(omega**2 - omega_ci**2)
    etha_parallel_diag = 1 -  omega_pe**2/omega**2  - omega_pi**2/omega**2
    etha_T             = omega_ci*omega_pi**2/(omega*(omega**2 - omega_ci**2)) +  omega_ce*omega_pe**2/(omega*(omega**2 - omega_ce**2)) 

    return (etha_perp_diag,etha_T,etha_parallel_diag)
    
    
    
def etha_tensor_coll(Mi,B,n_e,omega):
    
    nju_i = 20e3
    nju_e = 8e6
    
    mi = Mi*amu
    omega_ce = -e*B/(me)
    omega_ci = e*B/(mi)
    omega_pe = np.sqrt(n_e*e**2/(me*epsilon_0))
    omega_pi = np.sqrt(n_e*e**2/(mi*epsilon_0)) 
    
    nju_compl_i = nju_i*1j   
    nju_compl_e = nju_e*1j 
    etha_perp_diag     = 1 -  (omega + nju_compl_e)*omega_pe**2/(omega*(omega*(omega + nju_compl_e) - omega_ce**2)) - (omega + nju_compl_i)*omega_pi**2/(omega*(omega*(omega + nju_compl_i) - omega_ci**2))
    
    
    
    
    etha_parallel_diag = 1 -  omega_pe**2/(omega*(omega + nju_compl_e))  - omega_pi**2/(omega*(omega + nju_compl_i)) #  from Golant Fedorov Visokochastotnije metodi nagreva plasmi
    etha_T             = omega_ci*omega_pi**2/(omega*((omega + nju_compl_i)**2 - omega_ci**2)) +  omega_ce*omega_pe**2/(omega*((omega + nju_compl_e)**2 - omega_ce**2)) 

    return (etha_perp_diag,etha_T,etha_parallel_diag)
    
        
    
    
def Stix_tensor(Mi,B,n_e,omega):
    
    mi = Mi*amu
    omega_ce = -e*B/(me)
    omega_ci = e*B/(mi)
    omega_pe = np.sqrt(n_e*e**2/(me*epsilon_0))
    omega_pi = np.sqrt(n_e*e**2/(mi*epsilon_0)) 
    
    R = 1 - omega_pe**2/(omega*(omega + omega_ce)) - omega_pi**2/(omega*(omega + omega_ci))
    L = 1 - omega_pe**2/(omega*(omega - omega_ce)) - omega_pi**2/(omega*(omega - omega_ci))
    P = 1 -  omega_pe**2/omega**2  - omega_pi**2/omega**2
    
    S = 0.5*(R+L)
    D = 0.5*(R-L)
  
  
    return (S,D,P)
    
    
def Ksq_x_Stix(Mi,B,n_e,omega,Nsq_z):

    S, D, P = Stix_tensor(Mi,B,n_e,omega)
    k0 = omega/c
    ksq_x = Nsq_z*(omega/c)**2
    Kfast =  ( (k0**2*S - ksq_x)**2 - (k0**2*D)**2 )/((k0**2*S - ksq_x))
    Kslow =    (P/S)*(k0**2*S - ksq_x)
    
    
    A = S
    B =  ksq_x*(P + S) - k0**2*(S**2-D**2+P*S)    
    C =  P*( k0**4*(S**2-D**2) +  ksq_x**2 - 2*k0**2*ksq_x*S ) 
    
    Ksqx_plus =   (-B + np.sqrt(B**2-4*A*C) )/(2*A)
    Ksqx_minus =   (-B - np.sqrt(B**2-4*A*C) )/(2*A)    
    
    return (Kfast,Kslow,Ksqx_plus,Ksqx_minus,A,B,C)
      
      
def Nsq_Stix_Uncorrected_SDP(Mi,B,n_e,omega,theta):
    # theta ist in degree
    theta_rad = theta*pi/180            # theta ist in radian
    
    
    S, D, P = Stix_tensor(Mi,B,n_e,omega)
    R = S + D
    L = S - D
    k0 = omega/c
   
    
    A = S*np.sin(theta_rad)**2 + P*np.cos(theta_rad)**2 
    B =  R*L*np.sin(theta_rad)**2   + P*S*(1 + np.cos(theta_rad)**2)
    C =  P*R*L 
    F = np.sqrt(B**2 - 4*A*C)
    
  
  
    Nsq_plus =   (B + F )/(2*A)
    Nsq_minus =   (B - F)/(2*A)    
    
    
  
      
 #####################################################################################  
    
 
    return ( Nsq_plus, Nsq_minus, A, B, F, S, D, P )
    
def Nsq_Stix_Uncorrected(Mi,B,n_e,omega,theta):
    # theta ist in degree
    theta_rad = theta*pi/180            # theta ist in radian
    
    
    S, D, P = Stix_tensor(Mi,B,n_e,omega)
    R = S + D
    L = S - D
    k0 = omega/c
   
    
    A = S*np.sin(theta_rad)**2 + P*np.cos(theta_rad)**2 
    B =  R*L*np.sin(theta_rad)**2   + P*S*(1 + np.cos(theta_rad)**2)
    C =  P*R*L 
    F = np.sqrt(B**2 - 4*A*C)
    
  
  
    Nsq_plus =   (B + F )/(2*A)
    Nsq_minus =   (B - F)/(2*A)    
    
    
  
      
 #####################################################################################  
    
 
    return ( Nsq_plus, Nsq_minus, A, B, F )
  
def Nsq_Stix(Mi,B,n_e,omega,theta):
    # theta ist in degree
    theta_rad = theta*pi/180            # theta ist in radian
    
    
    S, D, P = Stix_tensor(Mi,B,n_e,omega)
    R = S + D
    L = S - D
    k0 = omega/c
   
    
    A = S*np.sin(theta_rad)**2 + P*np.cos(theta_rad)**2 
    B =  R*L*np.sin(theta_rad)**2   + P*S*(1 + np.cos(theta_rad)**2)
    C =  P*R*L 
    F = np.sqrt(B**2 - 4*A*C)
    
    
    
    Nsq_plus =   (B + F )/(2*A)
    Nsq_minus =   (B - F)/(2*A)    
    
   
    
    
    return (Nsq_plus, Nsq_minus)
      
    
    
    
    
    
def Nsq_x(Mi,B,n_e,omega,Nsq_z):

    etha_perp_diag, etha_T, etha_parallel_diag = etha_tensor(Mi,B,n_e,omega)
    fast =   ((etha_perp_diag-Nsq_z)**2-etha_T**2)/(etha_perp_diag-Nsq_z)
    slow =    etha_parallel_diag*(etha_perp_diag-Nsq_z)/etha_perp_diag
    return (fast,slow)
    

def Nsq_x_full(Mi,B,n_e,omega,Nsq_z):

    etha_perp_diag, etha_T, etha_parallel_diag = etha_tensor(Mi,B,n_e,omega)
    A = etha_perp_diag
    B = (Nsq_z - etha_perp_diag)*(etha_perp_diag + etha_parallel_diag) + etha_T**2
    C = etha_parallel_diag*((etha_perp_diag - Nsq_z)**2 - etha_T**2)
  
    Nx_plus =   (-B + np.sqrt(B**2-4*A*C) )/(2*A)
    Nx_minus =   (-B - np.sqrt(B**2-4*A*C) )/(2*A)
    
    return (Nx_plus,Nx_minus,A,B,C)



    
def Nsq_z_full(Mi,B,n_e,omega,Nsq_x):

    etha_perp_diag, etha_T, etha_parallel_diag = etha_tensor(Mi,B,n_e,omega)
    A = etha_parallel_diag
    B = Nsq_x*(etha_parallel_diag + etha_perp_diag) - 2*etha_parallel_diag*etha_perp_diag
    C = etha_parallel_diag*(etha_perp_diag**2 -  etha_T**2) + Nsq_x*etha_T**2 + etha_perp_diag*Nsq_x**2 - Nsq_x*etha_perp_diag*(etha_parallel_diag + etha_perp_diag)
  
    Nz_plus =   (-B + cmath.sqrt(B**2-4*A*C) )/(2*A)
    Nz_minus =   (-B - cmath.sqrt(B**2-4*A*C) )/(2*A)
    
    return (Nz_plus, Nz_minus, A, B, C)





def Nsquare_extraord_ord(Mi,B,n_e,omega):
    etha_perp_diag, etha_T, etha_parallel_diag = etha_tensor(Mi,B,n_e,omega)
    extraord =   (etha_perp_diag**2 - etha_T**2)/etha_perp_diag
    ordinar = etha_parallel_diag
    return (extraord,ordinar)
      
def replace(a,val):
    for i in range (len(a)):
        if a[i] > val:
           a[i]=val
    return a       
    
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------


def current2B_small_coils(I):
    
    coeff = 63.5e-3/1000 #T/A    
    
    return coeff*I


def current2B_big_coils(I):
    
    coeff = 34e-3/1000 #T/A    
    
    return coeff*I



def slopes_finder(U,I,t,dt,sweeping_f,MaxCounts):
   
    T = 1/sweeping_f
    N_T = T/dt
    N_T_05 = int(0.5*T/dt)
    N_T_025 = 0.25*T/dt



#  *********  The first cicle  ************************
#  **********  falling SLOPE **********************



    indexes = np.arange(0,N_T,dtype = 'int')
 
    sub_V = U[indexes]
    ind_max_v_sm  = np.argmax(sub_V)
    ind_min_v_sm  = ind_max_v_sm + N_T_05
    
    
    
    ArrIndxsFalling = []
    ArrIndxsRising  = []
    
    
 #------------- First record about falling slope ----------------------------------------------------------
    FallingSlopeRecord = {'indxs': [ind_max_v_sm, ind_min_v_sm],'t': t[ind_max_v_sm], 'U':  U[ind_max_v_sm:ind_min_v_sm], 'I':  I[ind_max_v_sm:ind_min_v_sm]}   
    ArrIndxsFalling.append(FallingSlopeRecord)
 #---------------------------------------------------------------------   
 
    
   
    
    
    #------------------ cycle to analyse other slopes --------------------------------------
    for i in range(1,MaxCounts):
        
        indx_last_min  = ArrIndxsFalling[i-1]['indxs'][1]
        indx_sub_cicle = np.arange(indx_last_min + N_T_025, indx_last_min + N_T + N_T_025, dtype = 'int')
        sub_V_cicle = U[indx_sub_cicle]
        sub_t_cicle = t[indx_sub_cicle]
        ind_max_v_sm  = np.argmax(sub_V_cicle)
         
        t_max = sub_t_cicle[ind_max_v_sm]   
        
        ind_max = np.where(t==t_max)[0][0]
        ind_min = ind_max + N_T_05     
        ind_prev_min = ind_max - N_T_05  
       
        FallingSlopeRecord = {'indxs': [ind_max, ind_min], 't': t[ind_max],  'U':  U[ind_max:ind_min], 'I':  I[ind_max:ind_min]}   
        RisingSlopeRecord  = {'indxs': [ind_prev_min, ind_max], 't': t[ind_prev_min],  'U':  U[ind_prev_min:ind_max], 'I':  I[ind_prev_min:ind_max]}      

        ArrIndxsFalling.append(FallingSlopeRecord)
        ArrIndxsRising.append(RisingSlopeRecord)
        
      
        #print( i
    #--------------------------------  end   ----------------------------------------------------




    
    return (ArrIndxsFalling, ArrIndxsRising)

def fft(t,signal):
    dt = t[3] - t[2]
    signal = np.array(signal)
    Nsample = np.size(signal)
    S_VLoop = abs(np.fft.fft(signal))
    freq = np.fft.fftfreq(Nsample, d=dt)
    S_VLoop = S_VLoop[0:Nsample/2]
    freq = freq[0:Nsample/2] 
    return freq,S_VLoop

def interpol(x, y, xnew):
    f = interp1d(x, y)
    ynew = f(xnew)
    return ynew


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
#------------------------ filter example --------------------------------

# Filter requirements.
#order = 4
#fs = 500000.0       # sample rate, Hz
#cutoff = 50  # desired cutoff frequency of the filter, Hz
#
## Get the filter coefficients so we can check its frequency response.
#b, a = butter_lowpass(cutoff, fs, order)
#
## Plot the frequency response.
#w, h = freqz(b, a, worN=8000)
#plt.subplot(2, 1, 1)
#plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
#plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
#plt.axvline(cutoff, color='k')
#plt.xlim(0, 200)
#plt.ylim(0, 1.2)
#plt.title("Lowpass Filter Frequency Response")
#plt.xlabel('Frequency [Hz]')
#plt.grid()
#
## Filter the data, and plot both the original and filtered signals.
#y = butter_lowpass_filter(data, cutoff, fs, order)
#----------------------------------------------------------------

def remove_extension(filename):
    name,ext = os.path.splitext(filename)
    return name

def resample_signal(x,y,NewN):
    NewN = int(NewN)
    ynew = resample(y, NewN)
    xnew = np.linspace(x[0], x[-1], NewN, endpoint=False)
    return xnew, ynew




def Ies2Ne( Ies, Te, gas,probe_type):
    Ies = np.array(Ies)
    Te = np.array(Te)
    
    Ies = -Ies
    
    def cilindrical_tungsten_probe(): 
        mm2m = 1e-3
        diameter = 0.8; # mm
        l = 20; # mm
        r = mm2m*diameter/2 # m
        return 2*r*mm2m*l # probe surface  
        
    def cilindrical_probe_array(): 
        mm2m = 1e-3
        diameter = 2; # mm
        l = 10; # mm
        r = mm2m*diameter/2 # m
        return 2*r*mm2m*l # probe surface       
       
    def cilindrical_probe_melted(): 
        # probe has melted into ball
        mm2m = 1e-3
        diameter = 3; # mm
        r = mm2m*diameter/2 # m
        return 2*np.pi*r**2 # probe surface
     
        
        
    def planar_probe_w_semirigid_cable():    
        mm2m = 1e-3
        a = 10; # mm
        b = 15; # mm
        return a*b*mm2m**2 # probe surface
        
    def planar_probe_circular():    
        mm2m = 1e-3
        r = 5; # mm
        r = r*mm2m # mm -> m
        return pi*r**2 # probe surface    
    

    AreaFunc =  locals()[probe_type]
    As = AreaFunc()    
    #As = big_planar_circular_probe_flange()
    
    
    
    e = 1.6e-19 # electron charge
     
    
    if gas=='He':    
       RelativeMass = 2 # He
    if gas=='Ar':    
       RelativeMass = 40 # He
    
    
    me = 9.1e-31
    amu = 1.6e-27
    #kB = 8.6e-5 # ev/K
     
    AbsoluteMass = RelativeMass*amu
    ne = np.sqrt((3.14*me)/(e*8*Te))*4*Ies/(As*e)
    return ne
 



def Is2Ne( Is, Te, gas,probe_type):
    Is = np.array(Is)
    Te = np.array(Te)
    
    Is = -Is
    
    def big_planar_circular_probe_flange():    
        mm2m = 1e-3
        diameter = 20; # mm
        r = mm2m*diameter/2 # m
        return np.pi*r**2 # probe surface
        
    def cilindrical_probe(): 
        mm2m = 1e-3
        diameter = 2; # mm
        l = 15; # mm
        r = mm2m*diameter/2 # m
        return 2*r*mm2m*l # probe surface
        
        
    def cilindrical_probe_tungsten(): 
        mm2m = 1e-3
        diameter = 2; # mm
        l = 15; # mm
        r = mm2m*diameter/2 # m
        return 2*r*mm2m*l # probe surface    
        
    def cilindrical_probe_array(): 
        mm2m = 1e-3
        diameter = 2; # mm
        l = 10; # mm
        r = mm2m*diameter/2 # m
        return 2*r*mm2m*l # probe surface       
       
    def cilindrical_probe_melted(): 
        # probe has melted into ball
        mm2m = 1e-3
        diameter = 3; # mm
        r = mm2m*diameter/2 # m
        return 2*np.pi*r**2 # probe surface
     
        
        
    def planar_probe_w_semirigid_cable():    
        mm2m = 1e-3
        a = 10; # mm
        b = 15; # mm
        return a*b*mm2m**2 # probe surface
    

    AreaFunc =  locals()[probe_type]
    As = AreaFunc()    
    #As = big_planar_circular_probe_flange()
    
    
    
    e = 1.6e-19 # electron charge
     
    
    if gas=='He':    
       RelativeMass = 2 # He
    if gas=='Ar':    
       RelativeMass = 40 # He
    
    
    
    amu = 1.6e-27
    #kB = 8.6e-5 # ev/K
     
    AbsoluteMass = RelativeMass*amu
    ne = np.sqrt(AbsoluteMass/(e*Te))*Is/(As*e*0.6)
    return ne
 



def save_1D(x,y,filename):
    data = np.transpose([np.float64(x),np.float64(y)])
    np.savetxt(filename, data, fmt='%.4e')
    print( 'File ' + filename + ' saved')
    
    
def save_1D_w_error(x,y,ydelta,filename):
    data = np.transpose([np.float64(x), np.float64(y), np.float64(ydelta)])
    np.savetxt(filename, data, fmt='%.4e')
    print( 'File ' + filename + ' saved')



def Phase2N_LineIntegrated( phase ):
    pi = np.pi
    f = 47e9# %Hz
    omega = 2*pi*f;
    epsilon_0 = 8.85e-12
    me = 9.1e-31
    e = 1.6e-19
    nc = omega**2*epsilon_0*me/(e**2)
    print( 'Nc = ' + str(nc) )
    c = 3e8; # m/s
    N_LineIntegrated = phase*2*c*nc/omega
    return N_LineIntegrated
    
    
def PhaseDetektor2Phase(Signal_Ph2, Signal_Ph1):
    
    def Ph2_rising(DetektorSignal):
        a = 92 # V/degree
        return DetektorSignal*a
        
    def Ph2_falling(DetektorSignal):
        a = -96 # V/degree
        return DetektorSignal*a + 353   
    
    
    def Ph1(DetektorSignal):
        a = 92 # V/degree
        return 100 + DetektorSignal*a
        
    Signal_Ph2 = np.array(Signal_Ph2)
    Signal_Ph1 = np.array(Signal_Ph1)      
 
   
    PhArray = []
    for i in range(len(Signal_Ph2)):

        DetektorSignal_Ph2 = Signal_Ph2[i]
        DetektorSignal_Ph1 = Signal_Ph1[i]      
        
        if (DetektorSignal_Ph2 < 1.2) and (DetektorSignal_Ph1 < 1.0):
           Ph =  Ph2_rising(DetektorSignal_Ph2)   
        elif (DetektorSignal_Ph2 < 1.2) and (DetektorSignal_Ph1 > 1.0):   
           Ph =  Ph2_falling(DetektorSignal_Ph2) 
        elif (DetektorSignal_Ph2 > 1.2):   
           Ph =  Ph1(DetektorSignal_Ph1) 
        PhArray.append(Ph)   
        
    return PhArray


    
    
def store2d(fname,x,y,z):

    szx = len(x);
    szy = len(y);
    
    fileID = open(fname,'w')
    fileID.write('\n');

    for indx in range (0,szx):
        for indy in range (0,szy):
             Xv = x[indx]
             Yv = y[indy]
             Zv = z[indx,indy]
             Str_to_file = "{0:6.2f} {1:6.2f} {2:6.2f}\n".format(Xv,Yv,Zv) 
             fileID.write(Str_to_file)
        fileID.write('\n')
    print( 'File ' + fname + ' saved'    )
      
    
def plot_gnu_pdf(script, pdffile):
    plotfile = 'plot_file.plt'
    thefile = open(plotfile, 'w')
    for string in script:
        thefile.write("%s\n" % string)
    thefile.close()
    sbprcs.call('gnuplot '+  plotfile, shell=True) 
    sbprcs.call('ps2eps -rotate + -f results/tmp.ps', shell=True) 
    sbprcs.call('eps2eps results/tmp.eps  results/goodeps.eps', shell=True)
    sbprcs.call('epstopdf results/goodeps.eps --outfile=results/' + pdffile, shell=True)
    print( "PDF file is ready")
    
    
def plot_gnu_pdf_v2(script, pdffile):
    plotfile = 'plot_file.plt'
    thefile = open(plotfile, 'w')
    for string in script:
        thefile.write("%s\n" % string)
    thefile.close()
    sbprcs.call('gnuplot '+  plotfile, shell=True) 
    sbprcs.call('ps2eps -rotate + -f results/tmp.ps', shell=True) 
    sbprcs.call('epstopdf results/tmp.eps --outfile=results/' + pdffile, shell=True)
    print( "PDF file is ready" )
       
    
    
def plot_gnu(script, epsfile):
    plotfile = 'plot_file.plt'
    thefile = open(plotfile, 'w')
    for string in script:
        thefile.write("%s\n" % string)
    thefile.close()
    sbprcs.call('gnuplot '+  plotfile, shell=True) 
    sbprcs.call('ps2eps -rotate + -f results/tmp.ps', shell=True) 
    sbprcs.call('eps2eps results/tmp.eps results/' + epsfile, shell=True) 
    print( "EPS file is ready" )


def exec_linux(script):
    sbprcs.call(script, shell=True) 
    print( "EPS file is ready" )



def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
        
    loc = locals().copy()
    for var in loc:
        if var[0] == '_': continue
        
        del locals()[var]        
        

def cut_interval(x, y, x_start, x_end):

    
   
    idx1, tmp = find_nearest(x,x_start)    # 4.4 for falling slope
    idx2, tmp = find_nearest(x,x_end)
    
    x_cut = x[idx1:idx2]        
    Ne_cut = y[idx1:idx2]
    return (x_cut,Ne_cut)
 
    
def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]    
    
def find_nearest(array,value):
    array = np.array(array)
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])   
    
    
def read_lecroy_file(file_name,Comment):
    headers = []
    data_list = list()

    with open(file_name , 'rt') as csvfile:
       spamreader = csv.reader(csvfile,delimiter='	')
       for row in spamreader:
   
           rowasarray = np.asarray(row)
           try:
                Ncolumns = np.size(rowasarray)
                if Ncolumns == 0:
                   continue
                rowasarray_float = []
                

                if rowasarray[0] == 'Time':
                   for ind in xrange(0,Ncolumns):
                    headers.append(rowasarray[ind])


                for ind in xrange(0,Ncolumns):
                    rowasarray_float.append(float(rowasarray[ind]))
                
                #rowasarray_float = [float(rowasarray[0]), float(rowasarray[1]), float(rowasarray[2])]    
                #print(rowasarray_float)
                data_list.append(rowasarray_float)      
           except ValueError:
                print("Could not convert data to an integer.")
           except IndexError:
                print("Index error")
                
  
    data = np.asarray(data_list)  
    
    N_data_columns = np.size(headers)
    dic_data = {}
    
    for ind in xrange(0,N_data_columns):
        dic_data[headers[ind]] = data[:,ind]
    
    return dic_data
        
    
    
def read_csv_file(file_name,Comment):
    headers = []
    data_list = list()

    with open(file_name , 'rt') as csvfile:
       spamreader = csv.reader(csvfile)
       for row in spamreader:
   
           rowasarray = np.asarray(row)
           try:
                Ncolumns = np.size(rowasarray)
                if Ncolumns == 0:
                   continue
                rowasarray_float = []
                

                if rowasarray[0] == 'TIME':
                   for ind in xrange(0,Ncolumns):
                    headers.append(rowasarray[ind])


                for ind in xrange(0,Ncolumns):
                    rowasarray_float.append(float(rowasarray[ind]))
                
                #rowasarray_float = [float(rowasarray[0]), float(rowasarray[1]), float(rowasarray[2])]    
                #print(rowasarray_float)
                data_list.append(rowasarray_float)      
           except ValueError:
                print("Could not convert data to an integer.")
           except IndexError:
                print("Index error")
                
  
    data = np.asarray(data_list)  
    
    N_data_columns = np.size(headers)
    dic_data = {}
    
    for ind in xrange(0,N_data_columns):
        dic_data[headers[ind]] = data[:,ind]
    
    return dic_data
 
def read_dat_file(file_name,dlmtr):
    headers = []
    data_list = list()

    with open(file_name , 'rt') as csvfile:
       spamreader = csv.reader(csvfile, delimiter=dlmtr)
       for row in spamreader:
   
           rowasarray = np.asarray(row)
           try:
                Ncolumns = np.size(rowasarray)
                if Ncolumns == 0:
                   continue
                rowasarray_float = []
                

                if rowasarray[0] == 'TIME':
                   for ind in xrange(0,Ncolumns):
                    headers.append(rowasarray[ind])


                for ind in xrange(0,Ncolumns):
                    rowasarray_float.append(float(rowasarray[ind]))
                
                #rowasarray_float = [float(rowasarray[0]), float(rowasarray[1]), float(rowasarray[2])]    
                #print(rowasarray_float)
                data_list.append(rowasarray_float)      
           except ValueError:
                print("Could not convert data to an integer.")
           except IndexError:
                print("Index error")
                
  
    data = np.asarray(data_list)  
    
    N_data_columns = np.size(headers)
    dic_data = {}
    
    for ind in xrange(0,N_data_columns):
        dic_data[headers[ind]] = data[:,ind]
    
    return dic_data
    
   
    
def read_rigol_csv_file(file_name,Comment):
    headers_indicator = False
    data_list = list()
    headers = []
    
    with open(file_name , 'rt') as csvfile:
       spamreader = csv.reader(csvfile)
       for row in spamreader:
   
           rowasarray = np.asarray(row)
           try:
                Ncolumns = np.size(rowasarray)
                if Ncolumns == 0:
                   continue
                rowasarray_float = []
                

                if rowasarray[0] == 'Freq':
                   headers_indicator = True
                   headers.append(rowasarray[0])
                   headers.append(rowasarray[2])
                   continue
                
                if  headers_indicator: 
                    freq = float(rowasarray[0])
                    Amp  = float(rowasarray[2])
                    if rowasarray[3].strip() == 'mV':
                        Amp = Amp*1e3
                    rowasarray_float.append(freq)
                    rowasarray_float.append(Amp)
                    
                    
                    data_list.append(rowasarray_float)
                
                      
           except ValueError:
                print("Could not convert data to an integer.")
           except IndexError:
                print("Index error")
                
  
    data = np.asarray(data_list)  
    
    N_data_columns = np.size(headers)
    dic_data = {}
    
    for ind in xrange(0,N_data_columns):
        dic_data[headers[ind]] = data[:,ind]
    
    return dic_data
        
    
def read_RandS_csv_file(file_name,Comment):
    headers_indicator = False
    data_list = list()
    headers = []
    
    with open(file_name , 'rt') as csvfile:
       spamreader = csv.reader(csvfile, delimiter=';')
       for row in spamreader:
   
           rowasarray = np.asarray(row)
           try:
                Ncolumns = np.size(rowasarray)
                if Ncolumns == 0:
                   continue
                rowasarray_float = []
                

                if rowasarray[0] == 'freq[Hz]':
                   headers_indicator = True
                   for header_name in rowasarray:
                       if not header_name == '':
                          headers.append(header_name)
                   continue
                
                if  headers_indicator: 
                    for j in range(len(headers)):
                        value = float(rowasarray[j])
                        rowasarray_float.append(value)
                        
                    
                    
                    data_list.append(rowasarray_float)
                
                      
           except ValueError:
                print("Could not convert data to an integer.")
           except IndexError:
                print("Index error")
                
  
    data = np.asarray(data_list)  
    
    N_data_columns = np.size(headers)
    dic_data = {}
    
    for ind in xrange(0,N_data_columns):
        dic_data[headers[ind]] = data[:,ind]
    
    return dic_data    
    
def sumlists(A,B):
    return np.array([x + y for x, y in zip(A, B)])   
    
def find_Up_Iesat_from_intersect(U_array,I_array, sb_handler, plot_sign):   
    I_e_last_value = I_array[-1]
    I_2 = I_array[-1]
    U_2 = U_array[-1] #  last amplitude value of U
    idx_1, U_1 = find_nearest(U_array, U_2-20)
    I_1 = I_array[idx_1]
    m = (I_2 - I_1)/(U_2 - U_1)
    def y_Ie_sat(x):
        m = (I_2 - I_1)/(U_2 - U_1)
        return I_1 + m*(x-U_1)
    I_Ie_sat_linear = y_Ie_sat(U_array)    
    
    if plot_sign:
       sb_handler.plot(U_array, I_Ie_sat_linear)    
    
    
    idx_Ufl = np.argmin(abs(I_array))
    
    Ufl = U_array[idx_Ufl] # might be wrong by transformator measurement, does not correspond to real Ufl
    print( 'Ufl = ' + str(Ufl))
    
    U_1 = Ufl+0
    idx_1, U_1 = find_nearest(U_array, U_1)
    I_1 = I_array[idx_1]
    idx_2, U_2 = find_nearest(U_array, U_1+10)
    I_2 = I_array[idx_2]
    
    def y_Ie_rising(x):
        m = (I_2 - I_1)/(U_2 - U_1)
        return I_1 + m*(x-U_1)
        
    I_Ie_rising_linear = y_Ie_rising(U_array)  
    if plot_sign:
       sb_handler.plot(U_array, I_Ie_rising_linear)  
    
#    if plot_sign:
#       sb_handler.plot(U_array, abs(I_Ie_sat_linear - I_Ie_rising_linear ))  #     visualise intersection point
    
                          #--------  find intersection -------------------------
    
    idx_intersection  =  np.argmin(abs(I_Ie_sat_linear - I_Ie_rising_linear))
    Up = U_array[idx_intersection]
    print( 'Up = ' + str(Up))
    
    Ielectron_sat = I_array[idx_intersection] 
    
    
    return (Up, Ielectron_sat,Ufl, I_e_last_value)     
    
    
    
def find_Up_Iesat_for_fixed(U_array,I_array, Uref):   
  
   idx, tmp  = find_nearest(U_array,Uref)
   Ielectron_sat = I_array[idx] 
    
    
   return Ielectron_sat    
    
    
    
    
    
def get_data(Ibig,Ismall):
        
        
        W7A = {'L': 0.183,'h': 0.4,'r': 0.55,'Zpos':[0, 0.603],'Nturn':25,'I':Ibig} 
        WEGA = {'L': 0.05,'h': 0.11,'r': 0.285, 'Zpos':[ 1.286,  1.516,  1.746,  1.976,  2.126],'Nturn':13,'I':Ismall} 
        
        
        Coils = {'W7A': W7A,'WEGA':WEGA}
        
        dl_grid = 0.011
        
        
        
        x_arr =    np.arange(-3, 3 + dl_grid, dl_grid)   
        y_arr =    np.arange(-1.3,1.3 + dl_grid, dl_grid)
                
        
        file_to_store = 'PreAnalysis.pickle'
        with open(file_to_store) as f:  # Python 3: open(..., 'rb')
            dict_to_save = pickle.load(f)
            Atotal_W7A = dict_to_save['Atotal_W7A']
            Atotal_WEGA = dict_to_save['Atotal_WEGA']
            f.close()
             
        Atotal = np.zeros((len(y_arr),len(x_arr)))
        
        for dz_another_coil in W7A['Zpos']:
            dz_N_another_coil = int(dz_another_coil/dl_grid)        
            Atotal = Atotal +  np.roll(Atotal_W7A*float(W7A['I']), dz_N_another_coil, 1)   
        
        for dz_another_coil in WEGA['Zpos']:
            dz_N_another_coil = int(dz_another_coil/dl_grid)        
            Atotal = Atotal +  np.roll(Atotal_WEGA*float(WEGA['I']), dz_N_another_coil, 1)   
        
        B_vec = 2*np.array(np.gradient(Atotal))/dl_grid
        B = np.sqrt(B_vec[0]**2 + B_vec[1]**2)
        
        B = 1e3*B # T -> mT
        
        Atotal = Atotal*1e3 # for the convenience
        
        return (Atotal, B, x_arr, y_arr)
              
            

def Kiz(Te):  # ionisation rate coefficients
    res = 2.34e-14*Te**0.59*np.exp(-17.8/Te)
    return  res
    
    
    
def Kel(Te): # eleastic rate coefficients
    rhs = -31.38 + 1.609*np.log(Te) +  0.0618*(np.log(Te))**2 - 0.1171*(np.log(Te))**3
    res = np.exp(rhs)
    return  res    
    
    
def Kex(Te):  # excitation rate coefficients
    res = 5.02e-15*np.exp(-12.64/Te) + 1.91e-15*np.exp(-12.6/Te) + 1.35e-15*np.exp(-12.42/Te) + 2.72e-16*np.exp(-12.14/Te) +  2.12e-14*np.exp(-13.13/Te)
    return  res      
        
    
    
