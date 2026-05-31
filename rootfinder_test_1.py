import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy as sc
from scipy import optimize

# Constants
n = 2
m = 1.0
#zeta = 2.0
zeta = 0.4
omega_0 = 1.0
F_0 = 2.0
T_sys = 5.0

omega = (2*(np.pi))/T_sys
k = (omega_0**2)
c = (zeta)*(2)*(np.sqrt(k))

test_sim = 1
display_fig = 1

def inputfunc(t):
    F = F_0*(np.cos(omega*t))
    return F

def massspring_dyn(t,x):
    xdot = np.array([
        x[1], 
        -(2*zeta*omega_0)*(x[1]) - (omega_0**2)*(x[0]) + (inputfunc(t)/m)
        ])
    return xdot

def massspring_simulate(num_T, x0, sys_dyn):
    t_span = (0.0, T_sys)
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    x0_res = np.zeros((n,num_T))
    x0_res_t = np.zeros(num_T)
    x0_res[:,0] = x0

    for n1 in range(num_T):
        soln = solve_ivp(massspring_dyn, t_span, x0_res[:,n1], t_eval=t_eval, rtol=1e-9, atol=1e-12)
        #soln = solve_ivp(massspring_dyn, t_span, x0_res[:,n1], t_eval=t_eval, rtol=1e-12, atol=1e-15)
        if n1 == 0:
            x_res = soln.y
        else:
            x_res = np.concatenate((x_res[:,:-1], soln.y), axis=1)

        if n1 < num_T-1:
            x0_res[:,n1+1] = soln.y[:,-1]
            x0_res_t[n1+1] = x0_res_t[n1] + soln.t[-1]

    x_res_t = dt*np.arange(len(x_res[0,:]))

    return x_res_t, x_res, x0_res_t, x0_res

def massspring_ana_soln(x0, t_soln):
    x_ana_soln = np.zeros((2,len(t_soln)))
    if (zeta/omega_0) > 1.0:
        A_hom = (-(x0[0]/omega_0) - x0[1]*(zeta - np.sqrt((zeta**2) - 1)))/(2*np.sqrt((zeta**2) - 1))
        B_hom = x0[0] - A_hom
        A_par = (F_0/m)*((omega_0**2 - omega**2)/((omega_0**2 - omega**2)**2 + (2*zeta*omega_0*omega)**2))
        B_par = (F_0/m)*((2*zeta*omega_0*omega)/((omega_0**2 - omega**2)**2 + (2*zeta*omega_0*omega)**2))
        x_ana_soln[0,:] = A_hom*np.exp(-omega_0*(zeta + np.sqrt((zeta**2) - 1))*t_soln) + B_hom*np.exp(-omega_0*(zeta - np.sqrt((zeta**2) - 1))*t_soln) + A_par*np.cos(omega*t_soln) + B_par*np.sin(omega*t_soln)
        x_ana_soln[1,:] = A_hom*(-omega_0*(zeta + np.sqrt((zeta**2) - 1)))*np.exp(-omega_0*(zeta + np.sqrt((zeta**2) - 1))*t_soln) + B_hom*(-omega_0*(zeta - np.sqrt((zeta**2) - 1)))*np.exp(-omega_0*(zeta - np.sqrt((zeta**2) - 1))*t_soln) - A_par*omega*np.sin(omega*t_soln) + B_par*omega*np.cos(omega*t_soln)
    else:
        c = 2*zeta*omega_0
        k = omega_0**2
        A_par = (F_0/m)*((k - omega**2)/((k - omega**2)**2 + (c*omega)**2))
        B_par = (F_0/m)*((c*omega)/((k - omega**2)**2 + (c*omega)**2))
        A_hom = x0[0] - A_par
        B_hom = (2*(x0[1] - B_par*omega) + c*(x0[0] - A_par))/np.sqrt(4*k - c**2)
        x_ana_soln[0,:] = np.exp(-(c/2)*t_soln)*(A_hom*np.cos((1/2)*np.sqrt(4*k - c**2)*t_soln) + B_hom*np.sin((1/2)*np.sqrt(4*k - c**2)*t_soln)) + A_par*np.cos(omega*t_soln) + B_par*np.sin(omega*t_soln)
        x_ana_soln[1,:] = np.exp(-(c/2)*t_soln)*( (-c/2)*(A_hom*np.cos((1/2)*np.sqrt(4*k - c**2)*t_soln) + B_hom*np.sin((1/2)*np.sqrt(4*k - c**2)*t_soln)) + (np.sqrt(4*k - c**2)/2)*(-A_hom*np.sin((1/2)*np.sqrt(4*k - c**2)*t_soln) + B_hom*np.cos((1/2)*np.sqrt(4*k - c**2)*t_soln)) ) - A_par*omega*np.sin(omega*t_soln) + B_par*omega*np.cos(omega*t_soln)
    return x_ana_soln

def res_display(x_res_t, x_res, x0_res_t, x0_res, ana_soln = 1, error = 1):
    if ana_soln == 1:
        x_ana_soln = massspring_ana_soln(x0_res[:,0], x_res_t)
        x0_ana_soln = np.zeros(x0_res.shape)
        x0_ana_soln[:,0] = x0_res[:,0]
        for n1 in range(len(x0_ana_soln[0,:]) - 1):
            x0_ana_soln[:,n1+1] = massspring_ana_soln(x0_res[:,n1], np.array([0.0, T_sys]))[:,1]

    fig, axes = plt.subplots(2, sharex=True)
    axes[0].plot(x_res_t, x_res[0,:], label='Numerical Trajectory')
    axes[0].plot(x0_res_t, x0_res[0,:], 'o', label=('Numerical Trajectory at T (' + str(T_sys) + ' s)'))
    axes[0].set_ylabel('$ x $ (m)')
    axes[1].plot(x_res_t, x_res[1,:])
    axes[1].plot(x0_res_t, x0_res[1,:], 'o')
    if ana_soln == 1:
        axes[0].plot(x_res_t, x_ana_soln[0,:], ':', label='Analytical Trajectory')
        axes[0].plot(x0_res_t, x0_ana_soln[0,:], 'o', label=('Analytical Trajectory at T (' + str(T_sys) + ' s)'))
        axes[1].plot(x_res_t, x_ana_soln[1,:], ':')
        axes[1].plot(x0_res_t, x0_ana_soln[1,:], 'o')
    axes[1].set_ylabel('$ \dot{x} $ (m/s)')
    axes[1].set_xlabel('t (s)')
    fig.legend()
    plt.show()

    plt.figure()
    plt.plot(x_res[0,:], x_res[1,:], label='Numerical Trajectory')
    plt.plot(x0_res[0,:], x0_res[1,:], 'o', label=('Numerical Trajectory at T (' + str(T_sys) + ' s)'))
    plt.plot(x_res[0,0], x_res[1,0], '+', label='Start')
    plt.plot(x_res[0,-1], x_res[1,-1], '+', label='End')
    plt.xlabel('$ x $ (m)')
    plt.ylabel('$ \dot{x} $ (m)')
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(2, sharex=True)
    axes[0].plot(x0_res_t, x0_res[0,:], '-o', label=('Numerical Trajectory at T (' + str(T_sys) + ' s)'))
    axes[0].plot(x0_res_t, x0_ana_soln[0,:], '-o', label=('Analytical Trajectory at T (' + str(T_sys) + ' s)'))
    axes[0].set_ylabel('$ x $ (m)')
    axes[1].plot(x0_res_t, x0_res[1,:], '-o')
    axes[1].plot(x0_res_t, x0_ana_soln[1,:], '-o')
    axes[1].set_ylabel('$ \dot{x} $ (m/s)')
    axes[1].set_xlabel('t (s)')
    fig.legend()
    plt.show()

    if ana_soln == 1 and error == 1:
        plt.figure()
        plt.plot(x_res_t, np.linalg.norm(x_res - x_ana_soln, axis=0), label='Trajectory')
        plt.plot(x0_res_t, np.linalg.norm(x0_res - x0_ana_soln, axis=0), 'o', label=('Trajectory at T (' + str(T_sys) + ' s)'))
        plt.xlabel('t (s)')
        plt.ylabel('Error (m)')
        plt.legend()
        plt.show()

def masspring_twostep(x0):
    [x_res_t, x_res, x0_res_t, x0_res] = massspring_simulate(2, x0, massspring_dyn)
    x_res = x0_res[:,1]
    return x_res

def stability_analysis(x0, epsval = 1e-8):
    jacobian = np.zeros((n,n))

    for n1 in range(n):
        eps = np.zeros(n)
        eps[n1] = epsval
        x0_perturb = np.zeros((n,2))
        for n2 in range(2):
            if n2 == 0:
                x0_analysis = x0 + eps
            else:
                x0_analysis = x0 - eps
            x0_res = masspring_twostep(x0_analysis)
            x0_perturb[:,n2] = x0_res - x0
        jacobian[:,n1] = (1/2)*(1/epsval)*(x0_perturb[:,0] - x0_perturb[:,1])

    return jacobian

def jacobian_extrapolate(x0, epsval = 1e-8):
    jacobian_1 = stability_analysis(x0, epsval = (epsval/2))
    jacobian_2 = stability_analysis(x0, epsval = epsval)

    jacobian = (1/3)*((4*jacobian_1) - jacobian_2)

    return jacobian

def jacobian_analytical(x0):
    c = 2*zeta*omega_0
    k = omega_0**2
    jacobian = np.zeros((n,n))
    jacobian[0,0] = np.exp(-(c/2)*T_sys)*(np.cos((1/2)*np.sqrt(4*k - c**2)*T_sys) + (c/np.sqrt(4*k - c**2))*np.sin((1/2)*np.sqrt(4*k - c**2)*T_sys))
    jacobian[0,1] = np.exp(-(c/2)*T_sys)*((2/np.sqrt(4*k - c**2))*np.sin((1/2)*np.sqrt(4*k - c**2)*T_sys))
    jacobian[1,0] = np.exp(-(c/2)*T_sys)*( (-c/2)*(np.cos((1/2)*np.sqrt(4*k - c**2)*T_sys) + (c/np.sqrt(4*k - c**2))*np.sin((1/2)*np.sqrt(4*k - c**2)*T_sys)) + (np.sqrt(4*k - c**2)/2)*(-np.sin((1/2)*np.sqrt(4*k - c**2)*T_sys) + (c/np.sqrt(4*k - c**2))*np.cos((1/2)*np.sqrt(4*k - c**2)*T_sys)) )
    jacobian[1,1] = np.exp(-(c/2)*T_sys)*( (-c/2)*((2/np.sqrt(4*k - c**2))*np.sin((1/2)*np.sqrt(4*k - c**2)*T_sys)) + (np.sqrt(4*k - c**2)/2)*((2/np.sqrt(4*k - c**2))*np.cos((1/2)*np.sqrt(4*k - c**2)*T_sys)) )
    return jacobian

# Simulation Parameters
dt = 0.01
num_T = 20
#x0 = np.zeros(n)
x0 = np.array([5.0, 5.0])

if test_sim == 1:
    [x_test_t, x_test, x0_test_t, x0_test] = massspring_simulate(num_T, x0, massspring_dyn)
    res_display(x_test_t, x_test, x0_test_t, x0_test)

masspring_twostep_num_fp = lambda x0: x0 - masspring_twostep(x0)
#masspring_fp_num_full = optimize.root(masspring_twostep_num_fp, x0, jac=stability_analysis)
masspring_num_fp = optimize.fixed_point(masspring_twostep, x0)
#masspring_num_fp = masspring_fp_num_full.x
print('Numerical Solution:')
print(masspring_num_fp)

#masspring_twostep_ana_fp = lambda x0: x0 - massspring_ana_soln(x0, np.array([0.0, T_sys]))[:,1]
masspring_twostep_ana_fp = lambda x0: massspring_ana_soln(x0, np.array([0.0, T_sys]))[:,1]
masspring_jacobian_ana = lambda x0: sc.linalg.expm(np.array([[0, 1],[-(omega_0**2), -(2*zeta*omega_0)]])*T_sys)
#masspring_fp_ana_full = optimize.root(masspring_twostep_ana_fp, x0, jac=masspring_jacobian_ana)
masspring_ana_fp = optimize.fixed_point(masspring_twostep_ana_fp, x0)
print('Analytical Solution:')
print(masspring_ana_fp)

if display_fig == 1:
    [x_test_num_t, x_test_num, x0_test_num_t, x0_test_num] = massspring_simulate(num_T, masspring_num_fp, massspring_dyn)
    res_display(x_test_num_t, x_test_num, x0_test_num_t, x0_test_num)

    [x_test_ana_t, x_test_ana, x0_test_ana_t, x0_test_ana] = massspring_simulate(num_T, masspring_ana_fp, massspring_dyn)
    res_display(x_test_ana_t, x_test_ana, x0_test_ana_t, x0_test_ana)

jacobian_num = stability_analysis(masspring_num_fp)
jacobian_ana = jacobian_analytical(masspring_ana_fp)

[jacobian_num_eigval, jacobian_num_eigvec] = np.linalg.eig(jacobian_num)
[jacobian_ana_eigval, jacobian_ana_eigvec] = np.linalg.eig(jacobian_ana)

print('Numerical Jacobian:')
print(jacobian_num)
print('Numerical Jacobian Eigenvalues:')
print(jacobian_num_eigval)
print('Numerical Jacobian Eigenvectors:')
print(jacobian_num_eigvec)
print('Analytical Jacobian:')
print(jacobian_ana)
print('Analytical Jacobian Eigenvalues:')
print(jacobian_ana_eigval)
print('Analytical Jacobian Eigenvectors:')
print(jacobian_ana_eigvec)



