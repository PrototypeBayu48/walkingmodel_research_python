import numpy as np
import scipy as sc
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy import optimize
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
import pickle
import os

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def, jacobian_num, x0_fp_func

# Generate Walking Dynamics Basin

linear_dyn = 1
propcont = 1
copquad = 1
x0_fp = 1
footgeom = 0

#copadj = False
copadj = 'basin'
noise = 0

# Constants
n = 4
g = 9.81
l = 1.09
omega_0 = np.sqrt(g/l)

k = 0.1

const_T = 0.5
#n_steps = 10
n_steps = 14
#n_steps = 20
s_c = 0.75
w_c = 0.1
z_c = 0.0

# Define CoP Variables
footl = 0.30
footw = 0.12

copparlim = np.array([[footl, 0.0], [(4*footw)/(footl**2), 0.0]])
print(copparlim)

if copquad == 1:
    if footgeom == 1:
        alpha = copparlim[0,0]
        beta = copparlim[1,0]
    else:
        alpha = copparlim[0,0]
        beta = copparlim[1,0]/2
else:
    alpha = 0
    beta = 0

b1 = (alpha)/const_T
c1 = -alpha*beta*b1
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
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z_quad)
    model_dyn_const = model_dyn_def([g, l], ddU_x, ddU_z_const)

# Simulation
dt = 0.01
t_span = (0.0, const_T)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

if copquad == 1:
    U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=cop_params_test['b1'], dU_z_0=cop_params_test['c1'])
else:
    U0_array = U0_array_func(n_steps, s_c, w_c, z_c)

# Define x0
def simulate_twostep(x1):
    if copquad == 1:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array_fp, dot_U=dot_U_func, cop_params=cop_params_test, copparlim = copparlim, const_T=const_T, propcont=propcont, copadj=copadj, constoff=1, stepprint=0)
    else:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_const, x1, dt, U0_array_fp, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
    x2 = x0_full.step[:,1]
    return x2

U0_array_fp = U0_array[:,:2]
x_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), simulate_twostep)

if x0_fp == 1:
    x0 = x_fp
else:
    x0 = np.array([0.25, 0.02, 0.0, 0.0])

# Define simulation parameters
#deltaz = 40
deltaz = 100
#deltaz = 200
#deltaz = 240
#deltaz = 260

#z_test = np.linspace(-0.1, 0.2, deltaz)
z_test = np.linspace(-0.2, 0.2, deltaz)
#z_test = np.linspace(-0.35, 0.45, deltaz)
#z_test = np.linspace(x0[1] - 0.002, x0[1] + 0.002, deltaz)
#z_test = np.linspace(x0[1] - 0.2, x0[1] + 0.2, deltaz)
#z_test = np.linspace(x0[1] - 0.4, x0[1] + 0.4, deltaz)

if propcont == 1:
    dz_test = np.linspace(-0.15, 0.15, 8) - U0_array[3,0]
    #dz_test = np.linspace(-0.65, 0.25, 8) - U0_array[3,0]

    z_c_err_test = np.linspace(-1.0, 1.0, deltaz)

    basin_x0_z = np.zeros((len(dz_test), len(z_test), len(z_c_err_test), n, n_steps))
    basin_U0_z = np.zeros((len(dz_test), len(z_test), len(z_c_err_test), n, n_steps))
else:
    #dz_test = np.linspace(-0.2, 0.8, deltaz)
    dz_test = np.linspace(-0.65, 0.25, deltaz) - U0_array[3,0]
    #dz_test = np.linspace(x0[3] - 0.002, x0[3] + 0.002, deltaz)
    #dz_test = np.linspace(x0[3] - 0.2, x0[3] + 0.2, deltaz)
    #dz_test = np.linspace(x0[3] - 0.4, x0[3] + 0.4, deltaz)

    z_c_err_test = None

    basin_x0_z = np.zeros((len(dz_test), len(z_test), n, n_steps))
    basin_U0_z = np.zeros((len(dz_test), len(z_test), n, n_steps))

testlen = 1

# Simulation
z_only = 1

def simulate_grid_point(n1, n2, x0_test):
    np.random.seed(int( (n1*1000) + n2 ))
    cop_params_test.update(cop_params)

    if copquad == 1:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0_test, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params_test, copparlim = copparlim, const_T=const_T, propcont=propcont, copadj=copadj, constoff=1, stepprint=0)
    else:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_const, x0_test, dt, U0_array, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

    x0_step = x0_full.step
    x0_inertial = x0_full.inertial
    U0_array_res = U0_full.res
    x_inertial = x_full.inertial
    U_inertial = U0_full.inertial
    
    return n1, n2, x0_step, U0_array_res

def simulate_grid_point_propcont(n1, n2, n3, x, z, xdot, zdot, z_c_err):
    np.random.seed(int( (n1*1000) + n2 ))
    cop_params_test.update(cop_params)

    x0_test = np.array([x, z, xdot, zdot])

    sys_par_test = (g, l, k, (x0_test[1] + U0_array[1,0] + ((x0_test[3] + U0_array[3,0])/omega_0)) - z_c_err, 0.8*l)

    if copquad == 1:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0_test, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params_test, sys_par=sys_par_test, copparlim = copparlim, const_T=const_T, propcont=propcont, copadj=copadj, constoff=1, stepprint=0)
    else:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_const, x0_test, dt, U0_array, sys_par=sys_par_test, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

    x0_step = x0_full.step
    x0_inertial = x0_full.inertial
    U0_array_res = U0_full.res
    x_inertial = x_full.inertial
    U_inertial = U0_full.inertial
    
    return n1, n2, n3, x0_step, U0_array_res

print('Simulating Lateral Motion')
if propcont == 1:
    results_z = Parallel(n_jobs=-1, batch_size="auto", verbose=1)(
        delayed(simulate_grid_point_propcont)(n1, n2, n3, x0[0], z_test[n2], x0[2], dz_test[n1], z_c_err_test[n3])
        for n1 in range(len(dz_test))
        for n2 in range(len(z_test))
        for n3 in range(len(z_c_err_test))
    )
    for n1, n2, n3, x0_step, U0_array_res in results_z:
        basin_x0_z[n1, n2, n3, :, :] = x0_step
        basin_U0_z[n1, n2, n3, :, :] = U0_array_res
else:
    results_z = Parallel(n_jobs=-1, verbose=5)(
        delayed(simulate_grid_point)(n1, n2, np.array([x0[0], z_test[n2], x0[2], dz_test[n1]]))
        for n1 in range(len(dz_test))
        for n2 in range(len(z_test))
    )
    for n1, n2, x0_step, U0_array_res in results_z:
        basin_x0_z[n1, n2, :, :] = x0_step
        basin_U0_z[n1, n2, :, :] = U0_array_res

if z_only != 1:
    deltax = deltaz
    #x_test = np.linspace(-0.5, 0.1, deltax)
    x_test = np.linspace(x0[0] - 0.2, x0[0] + 0.2, deltax)
    #x_test = np.linspace(np.min(fp_array[0,:]) - 0.2, np.max(fp_array[0,:]) + 0.2, deltax)

    #dx_test = np.linspace(-1.0, 2.0, deltax)
    dx_test = np.linspace(x0[2] - 1.0, x0[2] + 1.0, deltax)
    #dx_test = np.linspace(np.min(fp_array[2,:]) - 1.0, np.max(fp_array[2,:]) + 1.0, deltax) - U0_array[2,0]

    basin_x0_x = np.zeros((len(dz_test), len(z_test), n, n_steps))
    basin_U0_x = np.zeros((len(dz_test), len(z_test), n, n_steps))

    print('Simulating Forward Motion')
    if propcont == 1:
        results_x = Parallel(n_jobs=-1, batch_size="auto", verbose=1)(
            delayed(simulate_grid_point_propcont)(n1, n2, n3, x_test[n2], x0[1], dx_test[n1], x0[3], z_c_err_test[n3])
            for n1 in range(len(dx_test))
            for n2 in range(len(x_test))
            for n3 in range(len(z_c_err_test))
        )
        for n1, n2, n3, x0_step, U0_array_res in results_x:
            basin_x0_x[n1, n2, n3, :, :] = x0_step
            basin_U0_x[n1, n2, n3, :, :] = U0_array_res
    else:
        results_x = Parallel(n_jobs=-1, verbose=5)(
            delayed(simulate_grid_point)(n1, n2, np.array([x_test[n2], x0[1], dx_test[n1], x0[3]]))
            for n1 in range(len(dx_test))
            for n2 in range(len(x_test))
        )
        for n1, n2, x0_step, U0_array_res in results_x:
            basin_x0_x[n1, n2, :, :] = x0_step
            basin_U0_x[n1, n2, :, :] = U0_array_res
else:
    x_test = None
    dx_test = None
    basin_x0_x = None
    basin_U0_x = None

if linear_dyn == 1:
    sys_lin = True
else:
    sys_lin = False

if propcont == 1:
    sys_propcont = True
else:
    sys_propcont = False

sys_const_dict = {'g': g, 'l': l, 'footl': footl, 'footw': footw, 'k': k}
sys_setting_dict = {'linear': sys_lin, 'propcont': sys_propcont, 'copadj': copadj}

if propcont == 1:
    basin_res = {'dt': dt, 'const_T': const_T, 'U0_array': U0_array, 'x0': x0, 
                'sys_const': sys_const_dict, 'sys_setting': sys_setting_dict, 'cop_params': cop_params, 
                'x_test': x_test, 'dx_test': dx_test, 'basin_x0_x': basin_x0_x, 'basin_U0_x': basin_U0_x, 
                'z_test': z_test, 'dz_test': dz_test, 'basin_x0_z': basin_x0_z, 'basin_U0_z': basin_U0_z, 
                'z_c_err_test': z_c_err_test}
    if copquad == 1:
        if copadj == False:
            file_name = "basin_walkdyn_propcont_1.pkl"
        else:
            file_name = "basin_walkdyn_propcont_copadj_1.pkl"
    else:
        file_name = "basin_walkdyn_const_propcont_1.pkl"
else:
    basin_res = {'dt': dt, 'const_T': const_T, 'U0_array': U0_array, 'x0': x0, 
                'sys_const': sys_const_dict, 'sys_setting': sys_setting_dict, 'cop_params': cop_params, 
                'x_test': x_test, 'dx_test': dx_test, 'basin_x0_x': basin_x0_x, 'basin_U0_x': basin_U0_x, 
                'z_test': z_test, 'dz_test': dz_test, 'basin_x0_z': basin_x0_z, 'basin_U0_z': basin_U0_z}
    if copquad == 1:
        if copadj == False:
            file_name = "basin_walkdyn_1.pkl"
        else:
            file_name = "basin_walkdyn_copadj_1.pkl"
    else:
        file_name = "basin_walkdyn_const_1.pkl"

with open(os.path.join(os.getcwd(), "simulation_data", file_name), "wb") as file:
    pickle.dump(basin_res, file)



