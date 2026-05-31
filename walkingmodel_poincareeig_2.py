import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.utils import Bunch
from scipy import optimize

from walkingmodel_function import model_dyn_def, model_num_linear_dyn_def, step_dyn, jacobian_num, U0_array_func, jacobian_extrapolate

from walkingmodel_function import ddU_x_0
from walkingmodel_function import ddU_z_0
from walkingmodel_function import steploc_constoffset as steploc

epsval_analysis_plot = 0

linear_dyn = 1
constoff = 1
propcont = 0
fpfunc = 0
copquad = 1

# Constants
n = 4
g = 9.81
l = 1.09

omega_0 = np.sqrt(g/l)

def x0_fp_func(x0, model_dyn, U0_array, dot_U_func):
    def simulate_twostep_fp(x1):
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, const_T=const_T, propcont=propcont, constoff=constoff, stepprint=0)
        x2 = x0_full.step[:,1]
        return x2
    x0 = optimize.fixed_point(simulate_twostep_fp, x0)
    return x0

# Simulation parameters
dt = 0.01
const_T = 0.5
n_steps = 2
s_c = 0.75
w_c = 0.1
z_c = 0.0
x0 = np.array([0.25, 0.02, 0.0, 0.0])

#const_T_array = np.arange(0.05, 0.3 + dt, dt)
const_T_array = np.arange(0.4, 0.6 + dt, dt)
#const_T_array = np.arange(0.6, 1.0 + dt, dt)
eigval_array = np.zeros((n, len(const_T_array)))
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

        def ddU_z_quad(t,x):
            return 2*cop_params['c2']
        
        ddU_x = ddU_x_0
        ddU_z = ddU_z_quad

        def dot_U_func(t,x):
            dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z(t,x)])
            return dot_U
    else:
        U0_array = U0_array_func(n_steps, s_c, w_c, z_c)
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

    def simulate_twostep(x1):
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, const_T=const_T, propcont=propcont, constoff=constoff, stepprint=0)
        x2 = x0_full.step[:,1]
        return x2

    # Find Fixed Point
    if fpfunc == 1:
        constoff_fp = optimize.fixed_point(simulate_twostep, x0)
        #constoff_fp = x0_fp_func(x0, model_dyn, U0_array, dot_U_func)

        # Stability Analysis
        jacobian_analysis = jacobian_num(constoff_fp, n, simulate_twostep)
        [jacobian_analysis_eigval, jacobian_analysis_eigvec] = np.linalg.eig(jacobian_analysis)

        #print(jacobian_analysis_eigval)

        fixedpoint_array[:,m1] = constoff_fp
        x0 = constoff_fp

        eigval_array[:,m1] = np.real(jacobian_analysis_eigval)
    else:
        constoff_fp_func = lambda x: x - simulate_twostep(x)
        #constoff_fp_jac = lambda x: jacobian_num(x, n, simulate_twostep, epsval=1e-4)
        constoff_fp_jac = lambda x: jacobian_extrapolate(x, n, simulate_twostep, epsval=1e-4)

        #constoff_fp_full = optimize.root(constoff_fp_func, x0, jac=constoff_fp_jac)
        #constoff_fp_full = optimize.root(constoff_fp_func, x0)
        constoff_fp_full = optimize.root(constoff_fp_func, x0, method='lm')
        constoff_fp = constoff_fp_full.x

        if constoff_fp_full.success:
            # Stability Analysis
            jacobian_analysis = jacobian_num(constoff_fp, n, simulate_twostep)
            [jacobian_analysis_eigval, jacobian_analysis_eigvec] = np.linalg.eig(jacobian_analysis)

            fixedpoint_array[:,m1] = constoff_fp
            x0 = constoff_fp

            eigval_array[:,m1] = np.real(jacobian_analysis_eigval)
        else:
            print('Numerical Fixed Point Not Found')
            print(constoff_fp_full.message)

# Plot results

plt.figure()
#plt.plot(omega_0*const_T_array, np.abs(eigval_array[0,:]),'bo', label='Numerical $ |\lambda_{x}| $')
#plt.plot(omega_0*const_T_array, np.abs(eigval_array[1,:]),'ro', label='Numerical $ |\lambda_{z}| $')
#plt.plot(omega_0*const_T_array, np.abs(eigval_array[2,:]),'cx', label='Numerical $ |\lambda_{\dot{x}}| $')
#plt.plot(omega_0*const_T_array, np.abs(eigval_array[3,:]),'mx', label='Numerical $ |\lambda_{\dot{z}}| $')
if constoff == 1:
    plt.plot(omega_0*const_T_array, np.abs(eigval_array[1,:]),'bo', label='Numerical $ |\lambda_{1}| $')
    plt.plot(omega_0*const_T_array, np.abs(eigval_array[3,:]),'ro', label='Numerical $ |\lambda_{2}| $')
    #plt.plot(omega_0*np.linspace(0.0, 1/omega_0), np.abs(-np.exp(-omega_0*np.linspace(0.0, 1/omega_0))),'b-', label='Analytical $ |\lambda_{1}| $')
    plt.plot(omega_0*np.linspace(const_T_array[0], const_T_array[-1]), np.abs(-np.exp(-omega_0*np.linspace(const_T_array[0], const_T_array[-1]))),'b-', label='Analytical $ |\lambda_{1}| $')
    #plt.plot(omega_0*np.array([0.0, 1/omega_0]), np.abs(np.array([0.0, 0.0])),'r-', label='Analytical $ |\lambda_{2}| $')
    plt.plot(omega_0*np.array([const_T_array[0], const_T_array[-1]]), np.array([0.0, 0.0]),'r-', label='Analytical $ |\lambda_{2}| $')
else:
    plt.plot(omega_0*const_T_array, np.abs(eigval_array[0,:]),'bo', label='Numerical $ |\lambda_{1}| $')
    plt.plot(omega_0*const_T_array, np.abs(eigval_array[1,:]),'ro', label='Numerical $ |\lambda_{2}| $')
    #plt.plot(omega_0*np.linspace(0.0, 1/omega_0), np.abs(-np.exp(omega_0*np.linspace(0.0, 1/omega_0))),'b-', label='Analytical $ |\lambda_{1}| $')
    plt.plot(omega_0*np.linspace(const_T_array[0], const_T_array[-1]), np.abs(-np.exp(omega_0*np.linspace(const_T_array[0], const_T_array[-1]))),'b-', label='Analytical $ |\lambda_{1}| $')
    #plt.plot(omega_0*np.linspace(0.0, 1/omega_0), np.abs(-np.exp(-omega_0*np.linspace(0.0, 1/omega_0))),'r-', label='Analytical $ |\lambda_{2}| $')
    plt.plot(omega_0*np.linspace(const_T_array[0], const_T_array[-1]), np.abs(-np.exp(-omega_0*np.linspace(const_T_array[0], const_T_array[-1]))),'r-', label='Analytical $ |\lambda_{2}| $')
plt.xlabel('$ \omega_{0}T $')
plt.ylabel('Absolute Eigenvalue ($ |\lambda| $)')
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



