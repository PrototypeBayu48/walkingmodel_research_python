import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.utils import Bunch
from scipy import optimize

from walkingmodel_function import model_dyn_def, model_num_linear_dyn_def, step_dyn, jacobian_num, U0_array_func

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

    def ddU_x(t,x):
        return 0.0

    def ddU_z(t,x):
        return 2*c2

    def dot_U_func(t,x):
        dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z(t,x)])
        return dot_U

    U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=b1, dU_z_0=c1)

    [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, const_T=const_T, constoff=constoff, propcont=propcont, stepprint=0)
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

#alpha_array = np.arange(0.0, 0.4 + 0.05, 0.05)
alpha_array = np.arange(0.0, 0.8 + 0.1, 0.1)
#alpha_array = np.arange(0.0, 0.6 + 0.05, 0.05)
#alpha_array = np.arange(0.4, 0.6 + 0.01, 0.01)

#beta_array = np.arange(0.0, 4.0 + 0.5, 0.5)
beta_array = np.arange(0.0, 8.0 + 1.0, 1.0)
#beta_array = np.arange(0.0, 12.0 + 2.0, 2.0)
#beta_array = np.arange(0.0, 20.0 + 4.0, 4.0)

viable_array = np.zeros((len(alpha_array), len(beta_array)))

# Define model
if linear_dyn == 1:
    nonlin_model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)
    xinit_lin_test = np.zeros(n)
    model_dyn = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn)
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)

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

            eigval = np.flip(np.sort(np.abs(np.real(jacobian_analysis_eigval))))
            if eigval[0] > 1.0:
                viable_array[m1,m2] = -1.0
            else:
                viable_array[m1,m2] = 1.0
        else:
            print('Numerical Fixed Point Not Found')
            print(constoff_fp_full.message)
            viable_array[m1,m2] = -1.0

# Plot results

fig, axes = plt.subplots()
pos = axes.imshow(viable_array, cmap='RdBu')
fig.colorbar(pos, ax=axes)
axes.set_xticks(range(len(beta_array)), labels=np.round(beta_array,2), rotation=45, ha='right', rotation_mode='anchor')
axes.set_yticks(range(len(alpha_array)), labels=np.round(alpha_array,2))
axes.set_xlabel('Beta')
axes.set_ylabel('Alpha')
plt.show()



