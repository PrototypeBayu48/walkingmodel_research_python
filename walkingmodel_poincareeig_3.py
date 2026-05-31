import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.utils import Bunch
from scipy import optimize

from walkingmodel_function import model_dyn_def, model_num_linear_dyn_def, step_dyn, jacobian_num, U0_array_func, model_linear_dyn_def

from walkingmodel_function import ddU_x_0 as ddU_x
from walkingmodel_function import ddU_z_0 as ddU_z
from walkingmodel_function import steploc_constoffset as steploc

# Quadratic CoP Test, Vary Alpha and Beta

linear_dyn = 1
constoff = 1
propcont = 0

# Constants
n = 4
g = 9.81
l = 1.09

omega_0 = np.sqrt(g/l)

def simulate_twostep(x1):
    b1 = (alpha*s_c)/const_T
    c1 = -alpha*beta*b1*s_c
    c2 = beta*(b1**2)
    cop_params = {'b1': b1, 'c1': c1, 'c2': c2}

    def ddU_x(t,x):
        return 0.0

    def ddU_z_quad(t,x):
        return 2*cop_params['c2']

    def dot_U_func(t,x):
        dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z_quad(t,x)])
        return dot_U

    # Define model
    if linear_dyn == 1:
        model_dyn = model_linear_dyn_def([g, l], ddU_x, ddU_z_quad)

        '''
        nonlin_model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)
        xinit_lin_test = np.zeros(n)
        model_dyn_num = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn)
        '''
    else:
        model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)

    U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=cop_params['b1'], dU_z_0=cop_params['c1'])

    [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, const_T=const_T, constoff=constoff, propcont=propcont, stepprint=0)
    x2 = x0_full.step[:,1]

    return x2

# Simulation parameters
dt = 0.01
n_steps = 2

const_T_def = 0.5

s_c = 0.75
w_c = 0.1
z_c = 0.0
x0 = np.array([0.25, 0.02, 0.0, 0.0])

alpha_array = np.arange(0.0, 0.4 + 0.05, 0.05)
#beta_array = np.arange(0.0, 0.4 + 0.05, 0.05)
beta_array = np.arange(0.0, 4.0 + 0.5, 0.5)

eigval_array = np.zeros((len(alpha_array), len(beta_array), n))

for m1 in range(len(alpha_array)):
    alpha = np.round(alpha_array[m1], 2)
    print('Alpha: ' + str(alpha))
    for m2 in range(len(beta_array  )):
        const_T = const_T_def
        beta = np.round(beta_array[m2], 2)
        
        print('Beta: ' + str(beta))

        # Find Fixed Point
        constoff_fp_func = lambda x: x - simulate_twostep(x)
        constoff_fp_jac = lambda x: jacobian_num(x, n, simulate_twostep)
        constoff_fp_full = optimize.root(constoff_fp_func, x0)
        constoff_fp = constoff_fp_full.x

        if constoff_fp_full.success:
            # Stability Analysis
            jacobian_analysis = jacobian_num(constoff_fp, n, simulate_twostep)
            [jacobian_analysis_eigval, jacobian_analysis_eigvec] = np.linalg.eig(jacobian_analysis)

            x0 = constoff_fp

            eigval_array[m1,m2,:] = np.flip(np.sort(np.abs(np.real(jacobian_analysis_eigval))))
        else:
            print('Numerical Fixed Point Not Found')
            print(constoff_fp_full.message)


max_eigval_array = np.max(eigval_array, axis=2)
print('Max Absolute Eigenvalue: ' + str(np.max(max_eigval_array)))
print('Min Absolute Eigenvalue: ' + str(np.min(max_eigval_array)))

# Plot results

'''
plt.matshow(max_eigval_array)
plt.xlabel('Beta')
plt.ylabel('Alpha')
plt.show()
'''

fig, axes = plt.subplots(ncols=2, nrows=2)
m1 = 0
for n1 in range(2):
    for n2 in range(2):
        pos = axes[n1,n2].imshow(eigval_array[:,:,m1])
        fig.colorbar(pos, ax=axes[n1,n2])
        axes[n1,n2].set_xticks(range(len(beta_array)), labels=np.round(beta_array,2), rotation=45, ha='right', rotation_mode='anchor')
        axes[n1,n2].set_yticks(range(len(alpha_array)), labels=np.round(alpha_array,2))
        if n1 == 1:
            axes[n1,n2].set_xlabel('Beta')
        if n2 == 0:
            axes[n1,n2].set_ylabel('Alpha')
        m1 += 1
plt.show()



