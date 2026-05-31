import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy as sc

# Constants
n = 2
m = 1.0
#zeta = 2.0
zeta = 0.5
omega_0 = 1.0
F_0 = 2.0
T_sys = 5.0

omega = (2*(np.pi))/T_sys
k = (omega_0**2)*(m)
c = (zeta)*(2)*(np.sqrt(m*k))

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
        soln = solve_ivp(massspring_dyn, t_span, x0_res[:,n1], t_eval=t_eval)
        if n1 == 0:
            x_res = soln.y
        else:
            x_res = np.concatenate((x_res[:,:-1], soln.y), axis=1)

        if n1 < num_T-1:
            x0_res[:,n1+1] = soln.y[:,-1]
            x0_res_t[n1+1] = x0_res_t[n1] + soln.t[-1]

    x_res_t = dt*np.arange(len(x_res[0,:]))

    return x_res_t, x_res, x0_res_t, x0_res

def res_display(x_res_t, x_res, x0_res_t, x0_res, ana_soln = 1):
    if ana_soln == 1:
        A_hom = (-(x0_res[0,0]/omega_0) - x0_res[1,0]*(zeta - np.sqrt((zeta**2) - 1)))/(2*np.sqrt((zeta**2) - 1))
        B_hom = x0_res[0,0] - A_hom
        A_par = (F_0/m)*((omega_0**2 - omega**2)/((omega_0**2 - omega**2)**2 + (2*zeta*omega_0*omega)**2))
        B_par = (F_0/m)*((2*zeta*omega_0*omega)/((omega_0**2 - omega**2)**2 + (2*zeta*omega_0*omega)**2))
        x_ana_soln = np.zeros((2,len(x_res_t)))
        x_ana_soln[0,:] = A_hom*np.exp(-omega_0*(zeta + np.sqrt((zeta**2) - 1))*x_res_t) + B_hom*np.exp(-omega_0*(zeta - np.sqrt((zeta**2) - 1))*x_res_t) + A_par*np.cos(omega*x_res_t) + B_par*np.sin(omega*x_res_t)
        x_ana_soln[1,:] = A_hom*(-omega_0*(zeta + np.sqrt((zeta**2) - 1)))*np.exp(-omega_0*(zeta + np.sqrt((zeta**2) - 1))*x_res_t) + B_hom*(-omega_0*(zeta - np.sqrt((zeta**2) - 1)))*np.exp(-omega_0*(zeta - np.sqrt((zeta**2) - 1))*x_res_t) - A_par*omega*np.sin(omega*x_res_t) + B_par*omega*np.cos(omega*x_res_t)

    fig, axes = plt.subplots(2, sharex=True)
    axes[0].plot(x_res_t, x_res[0,:], label='Trajectory')
    axes[0].plot(x0_res_t, x0_res[0,:], 'o', label=('Trajectory at T (' + str(T_sys) + ' s)'))
    axes[0].set_ylabel('$ x $ (m)')
    axes[1].plot(x_res_t, x_res[1,:])
    axes[1].plot(x0_res_t, x0_res[1,:], 'o')
    if ana_soln == 1:
        axes[0].plot(x_res_t, x_ana_soln[0,:], ':', label='Analytical Trajectory')
        axes[1].plot(x_res_t, x_ana_soln[1,:], ':')
    axes[1].set_ylabel('$ \dot{x} $ (m/s)')
    axes[1].set_xlabel('t (s)')
    fig.legend()
    plt.show()

    plt.figure()
    plt.plot(x_res[0,:], x_res[1,:], label='Trajectory')
    plt.plot(x0_res[0,:], x0_res[1,:], 'o', label=('Trajectory at T (' + str(T_sys) + ' s)'))
    plt.plot(x_res[0,0], x_res[1,0], '+', label='Start')
    plt.plot(x_res[0,-1], x_res[1,-1], '+', label='End')
    plt.xlabel('$ x $ (m)')
    plt.ylabel('$ \dot{x} $ (m)')
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(2, sharex=True)
    axes[0].plot(x0_res_t, x0_res[0,:], '-o')
    axes[0].set_ylabel('$ x $ (m)')
    axes[1].plot(x0_res_t, x0_res[1,:], '-o')
    axes[1].set_ylabel('$ \dot{x} $ (m/s)')
    axes[1].set_xlabel('t (s)')
    plt.show()

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
            [x_res_t, x_res, x0_res_t, x0_res] = massspring_simulate(2, x0_analysis, massspring_dyn)
            x0_perturb[:,n2] = x0_res[:,1] - x0
        jacobian[:,n1] = (1/2)*(1/epsval)*(x0_perturb[:,0] - x0_perturb[:,1])

    return jacobian

def jacobian_extrapolate(x0, epsval = 1e-8):
    jacobian_1 = stability_analysis(x0, epsval = (epsval/2))
    jacobian_2 = stability_analysis(x0, epsval = epsval)

    jacobian = (1/3)*((4*jacobian_1) - jacobian_2)

    return jacobian

# Simulation Parameters
dt = 0.01
num_T = 20
#x0 = np.zeros(n)
x0 = np.array([1.0, 1.0])

[x_res_t, x_res, x0_res_t, x0_res] = massspring_simulate(num_T, x0, massspring_dyn)
res_display(x_res_t, x_res, x0_res_t, x0_res)

[x_test_t, x_test, x0_test_t, x0_test] = massspring_simulate(2, x0_res[:,-1], massspring_dyn)
res_display(x_test_t, x_test, x0_test_t, x0_test)

#jacobian_num = stability_analysis(x_res[:,-1])
jacobian_num = jacobian_extrapolate(x0_res[:,-1])

jacobian_ana = sc.linalg.expm(np.array([[0, 1],[-(omega_0**2), -(2*zeta*omega_0)]])*T_sys)

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

testnum = 200
poincare_perturb_num = np.zeros(testnum)
poincare_perturb_ana = np.zeros(testnum)

for n1 in range(testnum):
    perturbation = (-1e-3) + (np.random.rand(n))*(2e-3)

    [x_perturb_t, x_perturb, x0_perturb_t, x0_perturb] = massspring_simulate(2, x0_res[:,-1] + perturbation, massspring_dyn)

    poincare_perturb_num[n1] = np.linalg.norm((x0_perturb[:,-1]) - (x0_res[:,-1] + (jacobian_num@perturbation)))
    poincare_perturb_ana[n1] = np.linalg.norm((x0_perturb[:,-1]) - (x0_res[:,-1] + (jacobian_ana@perturbation)))

print('Numerical Jacobian Perturbation Error:')
print('Average: ' + str(np.mean(poincare_perturb_num)))
print('Standard Deviation: ' + str(np.std(poincare_perturb_num)))
print('Analytical Jacobian Perturbation Error:')
print('Average: ' + str(np.mean(poincare_perturb_ana)))
print('Standard Deviation: ' + str(np.std(poincare_perturb_ana)))



