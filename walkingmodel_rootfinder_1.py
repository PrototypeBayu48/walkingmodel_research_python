import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.utils import Bunch
from scipy import optimize

from walkingmodel_function import model_dyn_def, model_linear_dyn_def, model_num_linear_dyn_def, step_dyn, jacobian_num, plotfunc_state_full

from walkingmodel_function import ddU_x_0 as ddU_x
from walkingmodel_function import ddU_z_0 as ddU_z
from walkingmodel_function import steploc_constoffset as steploc

epsval_analysis_plot = 0

linear_dyn = 1
constoff = 1
propcont = 0

display_fig = 1

# Constants
n = 4
g = 9.81
l = 1.09

omega_0 = np.sqrt(g/l)

def simulate_twostep(x1):
    n_steps = 2

    U0_array = np.zeros((n,n_steps))
    U0_array[0,:] = np.linspace(0.0, (n_steps-1)*(s_c), n_steps)
    U0_array[1,:] = z_c + (w_c/2)*((-1)**(np.arange(n_steps) + 1))

    [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, const_T=const_T, constoff=constoff, propcont=propcont, stepprint=0)
    x2 = x0_full.step[:,1]

    return x2

# Simulation parameters
dt = 0.01
const_T = 0.5
n_steps = 10
s_c = 0.75
w_c = 0.1
z_c = 0.0
x0 = np.array([0.25, 0.02, 0.0, 0.0])

# Define model
if linear_dyn == 1:
    nonlin_model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)
    xinit_lin_test = np.zeros(n)
    model_dyn = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn)
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)

# Find Fixed Point
constoff_fp_func = lambda x: x - simulate_twostep(x)
constoff_fp_jac = lambda x: jacobian_num(x, n, simulate_twostep)
#constoff_fp_full = optimize.root(constoff_fp_func, x0, jac=constoff_fp_jac)
constoff_fp = optimize.fixed_point(simulate_twostep, x0)
#constoff_fp = constoff_fp_full.x
'''
if constoff_fp_full.success:
    print('Numerical Fixed Point:')
    print(constoff_fp)
else:
    print('Numerical Fixed Point Not Found')
'''

if display_fig == 1:
    U0_array = np.zeros((n,n_steps))
    U0_array[0,:] = np.linspace(0.0, (n_steps-1)*(s_c), n_steps)
    U0_array[1,:] = z_c + (w_c/2)*((-1)**(np.arange(n_steps) + 1))

    [x_constoff_fp_full, x0_constoff_fp_full, U0_constoff_fp_full, T_constoff_fp_array] = step_dyn(model_dyn, constoff_fp, dt, U0_array, const_T=const_T, propcont=propcont, stepprint=0)

    # Plot results
    plotfunc_state_full(dt, omega_0, x_constoff_fp_full, x0_constoff_fp_full, U0_constoff_fp_full, U0_array)

# Stability Analysis
jacobian_analysis = jacobian_num(constoff_fp, n, simulate_twostep)

[jacobian_analysis_eigval, jacobian_analysis_eigvec] = np.linalg.eig(jacobian_analysis)

print('Jacobian:')
print(jacobian_analysis)
print('Jacobian Eigenvalues:')
print(jacobian_analysis_eigval)
print('Jacobian Eigenvectors:')
print(jacobian_analysis_eigvec)



