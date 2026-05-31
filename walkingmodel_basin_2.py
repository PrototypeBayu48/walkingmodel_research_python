import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import optimize
from joblib import Parallel, delayed

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def, step_analysis_func

# Noise Recovery Basin

linear_dyn = 1
propcont = 0
copquad = 1
x0_fp = 1
footgeom = 0
copadj = 1

def x0_fp_func(x0, model_dyn, U0_array, dot_U_func, cop_params):
    def simulate_twostep(x1):
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
        x2 = x0_full.step[:,1]
        return x2
    #x0 = optimize.fixed_point(simulate_twostep, x0)
    constoff_fp_func = lambda x: x - simulate_twostep(x)
    constoff_fp_full = optimize.root(constoff_fp_func, x0)
    x0 = constoff_fp_full.x
    return x0

# Constants
n = 4
g = 9.81
l = 1.09
omega_0 = np.sqrt(g/l)

const_T = 0.5
#n_steps = 10
n_steps = 20
s_c = 0.75
w_c = 0.1
z_c = 0.0

# Define CoP Variables
footl = 0.30
footw = 0.12

#copparlim = np.array([[(footl/s_c), 0.01], [(4*footw)/(footl**2), 0.5]])
copparlim = np.array([[(footl/s_c), 0.0], [(4*footw)/(footl**2), 0.0]])

if footgeom == 1:
    alpha = copparlim[0,0]
    beta = copparlim[1,0]
else:
    #alpha = 0.4
    alpha = 0.35
    #alpha = 0.2
    #beta = 0.2
    beta = copparlim[1,0]/3

b1 = (alpha*s_c)/const_T
c1 = -alpha*beta*b1*s_c
c2 = beta*(b1**2)

cop_params = {'b1': b1, 'c1': c1, 'c2': c2}
cop_params_test = cop_params.copy()

if copquad == 1:
    print('U_x Coefficient:')
    print(np.array([cop_params['b1']]))
    print('U_z Coefficients:')
    print(np.array([cop_params['c1'], cop_params['c2']]))

# Define CoP functions
def cop_func_x(t):
    return cop_params_test['b1']*(t)

def cop_func_z(t):
    return cop_params_test['c2']*(t**2) + cop_params_test['c1']*(t)

def ddU_x(t,x):
    return 0.0

def ddU_z_quad(t,x):
    return 2*cop_params_test['c2']

def ddU_z_const(t,x):
    return 0.0

def dot_U_func(t,x):
    dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z_quad(t,x)])
    return dot_U

# Define model
if linear_dyn == 1:
    model_dyn = model_linear_dyn_def([g, l], ddU_x, ddU_z_quad)
    model_dyn_const = model_linear_dyn_def([g, l], ddU_x, ddU_z_const)

    '''
    nonlin_model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)
    nonlin_model_dyn_const = model_dyn_def([g, l], ddU_x, ddU_z_const)
    xinit_lin_test = np.zeros(n)
    model_dyn_num = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn)
    model_dyn_num_const = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn_const)
    '''
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)
    model_dyn_const = model_dyn_def([g, l], ddU_x, ddU_z_const)

# Simulation
dt = 0.01
t_span = (0.0, const_T)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

if copquad == 1:
    U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=cop_params['b1'], dU_z_0=cop_params['c1'])
else:
    U0_array = U0_array_func(n_steps, s_c, w_c, z_c)

# Define x0
U0_array_fp = U0_array[:,:2]
if copquad == 1:
    x_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn, U0_array_fp, dot_U_func, cop_params)
    print('Quadratic CoP:')
    print(x_fp)
else:
    x_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn_const, U0_array_fp, False, None)
    print('Constant CoP:')
    print(x_fp)

if x0_fp == 1:
    x0 = x_fp
else:
    x0 = np.array([0.25, 0.02, 0.0, 0.0])

# Define simulation parameters
#x_test = np.linspace(0.0, 0.002, 20)
#x_test = np.linspace(0.0, 0.08, 20)
x_test = np.linspace(0.0, 0.1, 20)
#x_test = np.linspace(0.0, 0.2, 20)

#z_test = np.linspace(0.0, 0.005, 20)
#z_test = np.linspace(0.0, 0.02, 20)
#z_test = np.linspace(0.0, 0.04, 20)
#z_test = np.linspace(0.0, 0.08, 20)
z_test = np.linspace(0.0, 0.1, 20)

testlen = 100

# Simulation
#crit = 'viability'
crit = 'step_lw'

def simulate_grid_point(n1, n2, x_val, z_val):
    np.random.seed(int( (n1*1000) + n2 ))
    point_score = 0.0
    
    for p1 in range(testlen):
        #cop_params_test = cop_params.copy()
        cop_params_test.update(cop_params)
        U0_noise_lim = np.array([x_val, z_val])

        if copquad == 1:
            [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params_test, U0_noise=U0_noise_lim, copparlim = copparlim, const_T=const_T, propcont=propcont, constoff=1, copadj=copadj, stepprint=0)
        else:
            [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_const, x0, dt, U0_array, U0_noise=U0_noise_lim, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

        U0_array_res = U0_full.res

        leftright = 1
        trial_result = 1.0
        
        if crit == 'viability':
            for m1 in range(n_steps):
                if m1 > 0:
                    if (np.abs(U0_array_res[1,m1] - U0_array_res[1,m1-1]) > (0.8*l)) or (np.abs(U0_array_res[0,m1] - U0_array_res[0,m1-1]) > (0.8*l)):
                        trial_result = -1.0
                        break
                    
                    if leftright == 1:
                        if (U0_array_res[1,m1] >= U0_array_res[1,m1-1]) or (U0_array_res[0,m1] <= U0_array_res[0,m1-1]):
                            trial_result = -1.0
                            break
                    else:
                        if (U0_array_res[1,m1] <= U0_array_res[1,m1-1]) or (U0_array_res[0,m1] <= U0_array_res[0,m1-1]):
                            trial_result = -1.0
                            break

                leftright = 0 if leftright == 1 else 1

            point_score += trial_result
        elif crit == 'step_lw':
            step_ana_res_mean = np.mean(step_analysis_func(U0_full.res), axis=1)
            trial_result = np.linalg.norm(np.array([s_c, w_c]) - step_ana_res_mean)
            point_score += trial_result

    return n1, n2, (point_score / testlen)

# 2. Execute the parallel grid search
print("Starting parallel simulation...")
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(simulate_grid_point)(n1, n2, x_test[n1], z_test[n2])
    for n1 in range(len(x_test))
    for n2 in range(len(z_test))
)

# 3. Reconstruct the 2D average basin
basin_noise_avg = np.zeros((len(x_test), len(z_test)))
for n1, n2, avg_val in results:
    basin_noise_avg[n1, n2] = avg_val
                    
#basin_noise_avg = np.mean(basin_noise, axis=0)
print('Average Value: ' + str(np.mean(basin_noise_avg)))

# Display results
if crit == 'viability':
    fig_type = 'Recovery Basin'
    fig_cmap = 'RdBu'
elif crit == 'step_lw':
    fig_type = 'Step Deviation Basin'
    fig_cmap = 'bwr'

fig, axes = plt.subplots()
if copquad == 1:
    if copadj == 1:
        fig.suptitle('Quadratic CoP + Adjustment Noise ' + fig_type + ' (' + str(testlen) + ' Trials)')
    else:
        fig.suptitle('Quadratic CoP Noise ' + fig_type + ' (' + str(testlen) + ' Trials)')
else:
    fig.suptitle('Constant CoP Noise ' + fig_type + ' (' + str(testlen) + ' Trials)')

pos = axes.imshow(basin_noise_avg.T, cmap=fig_cmap, origin='lower', extent=[x_test[0], x_test[-1], z_test[0], z_test[-1]], aspect='auto')
axes.set_xlabel(r'$ |\delta U_{x}(0)| $')
axes.set_ylabel(r'$ |\delta U_{z}(0)| $')

fig.colorbar(pos, ax=axes)
plt.show()



