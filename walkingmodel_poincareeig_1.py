import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.utils import Bunch

from walkingmodel_function import model_dyn_def, model_linear_dyn_def, model_num_linear_dyn_def, step_dyn, jacobian_num, plotfunc_state_full, U0_array_func

from walkingmodel_function import ddU_x_0
from walkingmodel_function import ddU_z_0
from walkingmodel_function import steploc_constoffset as steploc

# Variable T

epsval_analysis_plot = 0

linear_dyn = 1
propcont = 0
copquad = 1

# Constants
n = 4
g = 9.81
l = 1.09

omega_0 = np.sqrt(g/l)

def stability_analysis(x0, T, U0_analysis, dot_U = False, copquad=0, epsval = 1e-8):
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
            if copquad == 1:
                [x_stability_analysis, x0_stability_analysis, U0_stability_analysis, T_stability_analysis] = step_dyn(model_dyn, x0_analysis, dt, U0_analysis, dot_U=dot_U, cop_params=cop_params, const_T = T, propcont=propcont, stepprint=0)
            else:
                [x_stability_analysis, x0_stability_analysis, U0_stability_analysis, T_stability_analysis] = step_dyn(model_dyn, x0_analysis, dt, U0_analysis, dot_U=dot_U, const_T = T, propcont=propcont, stepprint=0)
            x0_perturb[:,n2] = x0_stability_analysis.step[:,1] - x0
        jacobian[:,n1] = (1/2)*(1/epsval)*(x0_perturb[:,0] - x0_perturb[:,1])

    return jacobian

def jacobian_extrapolate(x0, T, U0_analysis, dot_U = False, copquad=0, epsval = 1e-8):
    jacobian_1 = stability_analysis(x0, T, U0_analysis, dot_U = dot_U, copquad=copquad, epsval = (epsval/2))
    jacobian_2 = stability_analysis(x0, T, U0_analysis, dot_U = dot_U, copquad=copquad, epsval = epsval)

    jacobian = (1/3)*((4*jacobian_1) - jacobian_2)

    return jacobian

# Simulation parameters
dt = 0.01
n_steps = 80
s_c = 0.75
w_c = 0.1
z_c = 0.0
x0 = np.array([0.25, 0.02, 0.0, 0.0])

#const_T_array = np.arange(0.05, 0.3 + dt, dt)
const_T_array = np.arange(0.4, 0.6 + dt, dt)
eigval_array = np.zeros((2, len(const_T_array)))
fixedpoint_array = np.zeros((n, len(const_T_array)))

for m1 in range(len(const_T_array)):
    const_T = np.round(const_T_array[m1], 2)
    print('Period (s): ' + str(const_T))

    if copquad == 1:
        #alpha = 0.04
        #alpha = 0.1
        alpha = 0.4
        #alpha = 0.5
        #alpha = 0.6

        #beta = 0.1
        #beta = 0.2
        beta = 0.4
        #beta = 1.0
        #beta = 2.0
        #beta = 8.0

        # Define variables
        b1 = (alpha*s_c)/const_T
        c1 = -alpha*beta*b1*s_c
        c2 = beta*(b1**2)

        cop_params = {'b1': b1, 'c1': c1, 'c2': c2}

        U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=cop_params['b1'], dU_z_0=cop_params['c1'])
        U0_analysis = U0_array[:,:2]

        def ddU_z_quad(t,x):
            return 2*cop_params['c2']
        
        ddU_x = ddU_x_0
        ddU_z = ddU_z_quad

        def dot_U_func(t,x):
            dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z(t,x)])
            return dot_U
    else:
        U0_array = U0_array_func(n_steps, s_c, w_c, z_c)
        U0_analysis = U0_array_func(2, s_c, w_c, z_c)
        ddU_x = ddU_x_0
        ddU_z = ddU_z_0
        dot_U_func = False

    # Define model
    if linear_dyn == 1:
        nonlin_model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)
        xinit_lin_test = np.zeros(n)
        model_dyn = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn)
    else:
        model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)

    # Simulation results
    if copquad == 1:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, const_T=const_T, propcont=propcont, stepprint=0)
    else:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, dot_U=dot_U_func, const_T=const_T, propcont=propcont, stepprint=0)

    # Results Analysis
    T_analysis = T_array[-1]
    x0_analysis = x0_full.step[:,-1]
    if copquad == 1:
        [x_full_analysis, x0_full_analysis, U0_full_analysis, T_array_analysis] = step_dyn(model_dyn, x0_analysis, dt, U0_analysis, dot_U=dot_U_func, cop_params=cop_params, const_T = np.round(np.mean(T_analysis), 2), propcont=propcont, stepprint=0)
    else:
        [x_full_analysis, x0_full_analysis, U0_full_analysis, T_array_analysis] = step_dyn(model_dyn, x0_analysis, dt, U0_analysis, dot_U=dot_U_func, const_T = np.round(np.mean(T_analysis), 2), propcont=propcont, stepprint=0)

    #jacobian = stability_analysis(x0_analysis, np.mean(np.round(np.mean(T_analysis), 2)), U0_analysis, dot_U=dot_U_func, copquad=copquad)
    jacobian = jacobian_extrapolate(x0_analysis, np.mean(np.round(np.mean(T_analysis), 2)), U0_analysis, dot_U=dot_U_func, copquad=copquad)

    [jacobian_eigval, jacobian_eigvec] = np.linalg.eig(jacobian)

    fixedpoint_array[:,m1] = x0_analysis

    eigval_array[0,m1] = np.real(jacobian_eigval[1])
    eigval_array[1,m1] = np.real(jacobian_eigval[3])

# Plot results

plt.figure()
plt.plot(omega_0*const_T_array, eigval_array[0,:],'bo', label='Numerical $ \lambda_{1} $')
#plt.plot(omega_0*np.linspace(0.0, 1/omega_0), -np.exp(-omega_0*np.linspace(0.0, 1/omega_0)),'b-', label='Analytical $ \lambda_{1} $')
plt.plot(omega_0*np.linspace(const_T_array[0], const_T_array[-1]), -np.exp(-omega_0*np.linspace(const_T_array[0], const_T_array[-1])),'b-', label='Analytical $ \lambda_{1} $')
plt.plot(omega_0*const_T_array, eigval_array[1,:],'ro', label='Numerical $ \lambda_{2} $')
#plt.plot(omega_0*np.array([0.0, 1/omega_0]), np.array([0.0, 0.0]),'r-', label='Analytical $ \lambda_{2} $')
plt.plot(omega_0*np.array([const_T_array[0], const_T_array[-1]]), np.array([0.0, 0.0]),'r-', label='Analytical $ \lambda_{2} $')
plt.xlabel('$ \omega_{0}T $')
plt.ylabel('Eigenvalue ($ \lambda $)')
plt.legend()
plt.show()

fig, axes = plt.subplots(4, sharex=True)
step_label = ['$ x $ (m)', '$ z $ (m)', '$ \dot{x} $ (m/s)', '$ \dot{z} $ (m/s)']
for n1 in range(4):
    axes[n1].plot(omega_0*const_T_array, fixedpoint_array[n1,:],'-o')
    axes[n1].set_ylabel(step_label[n1])
    if n1 == 3:
        axes[n1].set_xlabel('$ \omega_{0}T $')
plt.show()



