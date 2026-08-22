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

# Process Basin Data

# Load Data
data_propcont = True
data_quadcop = True

if data_propcont:
    if data_quadcop:
        file_name = "basin_walkdyn_propcont_1.pkl"
    else:
        file_name = "basin_walkdyn_const_propcont_1.pkl"
else:
    if data_quadcop:
        file_name = "basin_walkdyn_1.pkl"
    else:
        file_name = "basin_walkdyn_const_1.pkl"

with open(os.path.join(os.getcwd(), "simulation_data", file_name), "rb") as file:
    basin_res = pickle.load(file)

U0_array = basin_res['U0_array']

z_test = basin_res['z_test']
dz_test = basin_res['dz_test']
basin_x0_z = basin_res['basin_x0_z']
basin_U0_z = basin_res['basin_U0_z']

if data_propcont:
    z_c_err_test = basin_res['z_c_err_test']
    propcont_dz_idx = 7

    basin_x0_z = np.swapaxes(basin_x0_z[propcont_dz_idx, :, :, :, :], 0, 1)
    basin_U0_z = np.swapaxes(basin_U0_z[propcont_dz_idx, :, :, :, :], 0, 1)

if basin_res['x_test'] != None:
    x_test = basin_res['x_test']
    dx_test = basin_res['dx_test']
    basin_x0_x = basin_res['basin_x0_x']
    basin_U0_x = basin_res['basin_U0_x']

# Settings
if basin_res['cop_params']['c2'] > 0.0:
    copquad = 1
else:
    copquad = 0

if basin_res['sys_setting']['propcont']:
    propcont = 1
else:
    propcont = 0

if basin_res['sys_setting']['copadj']:
    copadj = 1
else:
    copadj = 0

# Constants
n = 4
g = basin_res['sys_const']['g']
l = basin_res['sys_const']['l']
omega_0 = np.sqrt(g/l)

k = 0.1

const_T = basin_res['const_T']
n_steps = len(U0_array[0,:])
s_c = U0_array[0,1] - U0_array[0,0]
w_c = np.abs(U0_array[1,1] - U0_array[1,0])
z_c = (U0_array[1,1] + U0_array[1,0])/2

footl = basin_res['sys_const']['footl']
footw = basin_res['sys_const']['footw']

dt = basin_res['dt']

if data_quadcop:
    alpha = (basin_res['cop_params']['b1'])*(basin_res['const_T'])
    beta = (basin_res['cop_params']['c2'])/(basin_res['cop_params']['b1']**2)
else:
    alpha = 0.0
    beta = 0.0

deltaz = len(z_test)

if basin_res['x_test'] != None:
    deltax = len(x_test)

# Variables
x0 = basin_res['x0']
copparlim = np.array([[footl, 0.0], [(4*footw)/(footl**2), 0.0]])
cop_params = basin_res['cop_params'].copy()
cop_params_test = basin_res['cop_params'].copy()

U0_array_fp = U0_array[:,:2]

# Functions
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
if basin_res['sys_setting']['linear']:
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

def simulate_twostep(x1):
    if copquad == 1:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x1, dt, U0_array_fp, dot_U=dot_U_func, cop_params=cop_params_test, copparlim = copparlim, const_T=const_T, propcont=propcont, copadj=copadj, constoff=1, stepprint=0)
    else:
        [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_const, x1, dt, U0_array_fp, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
    x2 = x0_full.step[:,1]
    return x2

# Fixed point simulation
deltafp = 0
#deltafp = 10

if deltafp > 0:
    alphabeta_array_fp = np.zeros((2,deltafp))
    alphabeta_array_fp[0,:] = np.linspace(copparlim[0,1], copparlim[0,0], deltafp)
    alphabeta_array_fp[1,:] = np.linspace(copparlim[1,1], copparlim[1,0], deltafp)
    fp_array = np.zeros((deltafp,deltafp,n))
    alphabeta_array_res = np.zeros((deltafp,deltafp,2))

    def grid_fixedpoint(n1, n2, alphabeta):
        b1 = alphabeta[0]/const_T
        c1 = -alphabeta[0]*alphabeta[1]*b1
        c2 = alphabeta[1]*(b1**2)
        cop_params_fptest = {'b1': b1, 'c1': c1, 'c2': c2}
        def ddU_z_quad_fptest(t,x):
            return 2*cop_params_fptest['c2']
        def dot_U_func_fptest(t,x):
            dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z_quad_fptest(t,x)])
            return dot_U
        
        model_dyn_fptest = model_linear_dyn_def([g, l], ddU_x, ddU_z_quad_fptest)
        U0_array_fptest = U0_array_func(2, s_c, w_c, z_c, dU_x_0=cop_params_fptest['b1'], dU_z_0=cop_params_fptest['c1'])
        def simulate_twostep_fptest(x1):
            [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_fptest, x1, dt, U0_array_fptest, dot_U=dot_U_func_fptest, cop_params=cop_params_fptest, copparlim = copparlim, const_T=const_T, propcont=propcont, copadj=copadj, constoff=1, stepprint=0)
            x2 = x0_full.step[:,1]
            return x2

        x0_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), simulate_twostep_fptest)
        x0_fp[2:] = x0_fp[2:] + U0_array_fptest[2:,0]

        return n1, n2, x0_fp

    results_fp = Parallel(n_jobs=-1, verbose=5)(
        delayed(grid_fixedpoint)(n1, n2, np.array([alphabeta_array_fp[0,n1], alphabeta_array_fp[1,n2]]))
        for n1 in range(deltafp)
        for n2 in range(deltafp)
    )
    for n1, n2, x0_fp in results_fp:
        fp_array[n1, n2, :] = x0_fp
        alphabeta_array_res[n1, n2, :] = np.array([alphabeta_array_fp[0,n1], alphabeta_array_fp[1,n2]])

    fp_array = fp_array.reshape((-1,n)).T
    alphabeta_array_res = alphabeta_array_res.reshape((-1,2)).T
    c2_array_res = alphabeta_array_res[1,:]*(( (alphabeta_array_res[0,:])/const_T )**2)

    fp_ana_array = np.zeros((2,int(deltafp*deltafp)))
    fp_ana_array[0,:] = (w_c/2)*np.ones(int(deltafp*deltafp))
    fp_ana_array[1,:] = -(1/(2*omega_0*( np.exp(omega_0*const_T) + 1 )))*( (w_c*(omega_0**2)*(np.exp(omega_0*const_T) - 1)) + c2_array_res*( np.exp(omega_0*const_T)*(2*const_T*omega_0 - 4) + (2*const_T*omega_0) + 4 ) )

# Data Processing
crit = 'viability'
#crit = 'convergence'
#crit = 'stepwidth'

testlen = 1
x_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), simulate_twostep)

z_only = 1
w_max = 0.8*l
tol_conv = 1e-4

def dataprocess_func(n1, n2, x0_res, U0_res, crit):
    point_score = 0.0

    for p1 in range(testlen):
        if crit == 'viability':
            leftright = 1
            trial_result = 1.0

            for m1 in range(n_steps):
                if m1 > 0:
                    if (np.abs(U0_res[1,m1] - U0_res[1,m1-1]) > (w_max)) or (np.abs(U0_res[0,m1] - U0_res[0,m1-1]) > (w_max)):
                        trial_result = -1.0
                        break
                    
                    if leftright == 1:
                        if (U0_res[1,m1] >= U0_res[1,m1-1]) or (U0_res[0,m1] <= U0_res[0,m1-1]):
                            trial_result = -1.0
                            break
                    else:
                        if (U0_res[1,m1] <= U0_res[1,m1-1]) or (U0_res[0,m1] <= U0_res[0,m1-1]):
                            trial_result = -1.0
                            break

                leftright = 0 if leftright == 1 else 1
        elif crit == 'convergence':
            for m1 in range(n_steps):
                stepnorm = np.linalg.norm(np.array([x_fp[1], x_fp[3]]) - np.array([x0_res[1,m1], x0_res[3,m1]]))
                if stepnorm < tol_conv:
                    trial_result = m1
                    break
                else:
                    trial_result = m1
        elif crit == 'stepwidth':
            trial_result = 0.0
            for m1 in range(n_steps):
                if m1 > 0:
                    stepwidth = np.abs(U0_res[1,m1] - U0_res[1,m1-1])
                    if stepwidth > trial_result:
                        trial_result = stepwidth
            trial_result = trial_result - w_max

        point_score += trial_result
    
    return n1, n2, (point_score / testlen)

print('Simulating Lateral Motion')
if data_propcont:
    basin_z = np.zeros((len(z_c_err_test), len(z_test)))
    results_z = Parallel(n_jobs=-1, verbose=5)(
        delayed(dataprocess_func)(n1, n2, basin_x0_z[n1,n2,:,:], basin_U0_z[n1,n2,:,:], crit)
        for n1 in range(len(z_c_err_test))
        for n2 in range(len(z_test))
    )
    for n1, n2, avg_val in results_z:
        basin_z[n1, n2] = avg_val
else:
    basin_z = np.zeros((len(dz_test), len(z_test)))
    results_z = Parallel(n_jobs=-1, verbose=5)(
        delayed(dataprocess_func)(n1, n2, basin_x0_z[n1,n2,:,:], basin_U0_z[n1,n2,:,:], crit)
        for n1 in range(len(dz_test))
        for n2 in range(len(z_test))
    )
    for n1, n2, avg_val in results_z:
        basin_z[n1, n2] = avg_val

print('Average Basin Value (z): ' + str(np.mean(basin_z)))

if z_only != 1:
    basin_x = np.zeros((len(dx_test), len(x_test)))
    print('Simulating Forward Motion')
    results_x = Parallel(n_jobs=-1, verbose=5)(
        delayed(dataprocess_func)(n1, n2, basin_x0_x[n1,n2,:,:], basin_U0_x[n1,n2,:,:], crit)
        for n1 in range(len(dx_test))
        for n2 in range(len(x_test))
    )
    for n1, n2, avg_val in results_x:
        basin_x[n1, n2] = avg_val

    print('Average Basin Value (x): ' + str(np.mean(basin_x)))

# Define recovery basin boundaries
b_z = ( (1)/(1 + np.exp(omega_0*const_T)) )*( (w_c) - (cop_params_test['c1'])*( (const_T) + ((1 - np.exp(omega_0*const_T))/omega_0) ) - cop_params_test['c2']*( (const_T**2) + ((2*const_T)/(omega_0)) + ((2*(1 - np.exp(omega_0*const_T)))/(omega_0**2)) ) )
if data_propcont:
    deltaz_dense = 10000
    z_test_dense = np.linspace(z_test[0], z_test[-1], deltaz_dense)
    zeta_test_dense = z_test_dense + (dz_test[propcont_dz_idx] + U0_array[3,0])/omega_0

    quadcop_term = ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) )

    basin_boundary_ana = np.zeros((4,deltaz_dense))
    propcont_boundary_coeff = np.zeros((4,4))
    basin_top_check = np.inf
    basin_bottom_check = -np.inf
    propcont_boundary_coeff[0,0] = np.exp(-omega_0*const_T)*( (k)/(k + 1) )
    propcont_boundary_coeff[1,0] = np.exp(-omega_0*const_T)*( (-b_z)/(k + 1) ) + np.exp(-omega_0*const_T)*quadcop_term
    propcont_boundary_coeff[2,0] = np.exp(-omega_0*const_T)*( (w_max - b_z)/(k + 1) ) + np.exp(-omega_0*const_T)*quadcop_term
    propcont_boundary_coeff[3,0] = np.exp(omega_0*const_T)*(k - propcont_boundary_coeff[0,0]*(k + 1))
    for n1 in np.arange(1,4):
        propcont_boundary_coeff[0,n1] = np.exp(-omega_0*const_T)
        propcont_boundary_coeff[1,n1] = (b_z*(propcont_boundary_coeff[0,n1-1] - 1) + propcont_boundary_coeff[1,n1-1] + quadcop_term*(-propcont_boundary_coeff[0,n1-1]*(k+1) + k))/propcont_boundary_coeff[3,n1-1]
        propcont_boundary_coeff[2,n1] = (b_z*(propcont_boundary_coeff[0,n1-1] - 1) + propcont_boundary_coeff[2,n1-1] + quadcop_term*(-propcont_boundary_coeff[0,n1-1]*(k+1) + k))/propcont_boundary_coeff[3,n1-1]
        propcont_boundary_coeff[3,n1] = np.exp(omega_0*const_T)*(k - propcont_boundary_coeff[0,n1]*(k + 1))

        bound_1 = -( ( zeta_test_dense - propcont_boundary_coeff[1,n1] )/propcont_boundary_coeff[0,n1] ) + zeta_test_dense
        bound_2 = -( ( zeta_test_dense - propcont_boundary_coeff[2,n1] )/propcont_boundary_coeff[0,n1] ) + zeta_test_dense

        basin_top = np.maximum(bound_1, bound_2)
        basin_bottom = np.minimum(bound_1, bound_2)

        if basin_top[0] < basin_top_check:
            basin_boundary_ana[2,:] = basin_top
            basin_top_check = basin_top[0].copy()
        if basin_bottom[0] > basin_bottom_check:
            basin_boundary_ana[3,:] = basin_bottom
            basin_bottom_check = basin_bottom[0].copy()

    bound_1_0 = -( ( zeta_test_dense - propcont_boundary_coeff[1,0] )/propcont_boundary_coeff[0,0] ) + zeta_test_dense
    bound_2_0 = -( ( zeta_test_dense - propcont_boundary_coeff[2,0] )/propcont_boundary_coeff[0,0] ) + zeta_test_dense
    basin_top_0 = np.maximum(bound_1_0, bound_2_0)
    basin_bottom_0 = np.minimum(bound_1_0, bound_2_0)
    basin_boundary_ana[0,:] = basin_top_0
    basin_boundary_ana[1,:] = basin_bottom_0

    basin_top_total = np.minimum(basin_boundary_ana[0,:], basin_boundary_ana[2,:])
    basin_bottom_total = np.maximum(basin_boundary_ana[1,:], basin_boundary_ana[3,:])
    basin_region = basin_top_total >= basin_bottom_total
    #basin_boundary_ana[:, ~basin_region] = np.nan
    for n1 in range(4):
        if n1 == 0 or n1 == 2:
            basin_boundary_ana[n1, ~(basin_region & (np.abs(basin_boundary_ana[n1,:] - basin_top_total) < 1e-9))] = np.nan
        else:
            basin_boundary_ana[n1, ~(basin_region & (np.abs(basin_boundary_ana[n1,:] - basin_bottom_total) < 1e-9))] = np.nan
else:
    def dz_lim_low_func(z):
        return omega_0*((np.exp(-omega_0*const_T)*(-b_z + ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)
    def dz_lim_high_func(z):
        return omega_0*((np.exp(-omega_0*const_T)*(w_max - b_z + ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)
    basin_boundary_ana = np.zeros((2,deltaz))
    #basin_boundary_ana[0,:] = -omega_0*(z_test + (b_z*np.exp(-omega_0*const_T)))
    basin_boundary_ana[0,:] = dz_lim_low_func(z_test)
    #basin_boundary_ana[1,:] = omega_0*((np.exp(-omega_0*const_T)*(w_max - b_z)) - z_test)
    basin_boundary_ana[1,:] = dz_lim_high_func(z_test)

# Obtain Eigenvalues and Eigenvectors
def simulate_twostep_fp_vectorized(x1):
    return np.apply_along_axis(simulate_twostep, axis=0, arr=x1)

#jacobian_fp = jacobian_num(x_fp, n, simulate_twostep_fp)
jacobian_fp_res = sc.differentiate.jacobian(simulate_twostep_fp_vectorized, x_fp)
jacobian_fp = jacobian_fp_res.df
[jacobian_fp_eigval, jacobian_fp_eigvec] = np.linalg.eig(jacobian_fp)
eigvec_x_count = 0
eigvecmag_x_array = np.zeros(n)
eigvecmag_z_array = np.zeros(n)
for n1 in range(n):
    eigvecmag_x = np.linalg.norm(np.array([jacobian_fp_eigvec[0,n1], jacobian_fp_eigvec[2,n1]]))
    eigvecmag_z = np.linalg.norm(np.array([jacobian_fp_eigvec[1,n1], jacobian_fp_eigvec[3,n1]]))
    if eigvecmag_z >= eigvecmag_x or eigvec_x_count == 2:
        eigvecmag_z_array[n1] = 1
    else:
        eigvecmag_x_array[n1] = 1
        eigvec_x_count += 1

eigval_z = jacobian_fp_eigval[eigvecmag_z_array == 1]
eigvec_z = jacobian_fp_eigvec[:, eigvecmag_z_array == 1]

eigvec_slope_ana_1 = -omega_0/1
eigvec_slope_ana_2 = -omega_0*( (np.exp(const_T*omega_0) - np.exp(-const_T*omega_0))/(np.exp(const_T*omega_0) + np.exp(-const_T*omega_0)) )/1

print('Jacobian:')
print(jacobian_fp)
print('Jacobian Eigenvalues:')
print(jacobian_fp_eigval)
print('Jacobian Eigenvectors:')
print(jacobian_fp_eigvec)

print('Jacobian Eigenvector Slope 1:')
print('Analytical: ' + str(eigvec_slope_ana_1))
print('Numerical: ' + str(eigvec_z[3, np.abs(eigval_z) == np.max(np.abs(eigval_z))]/eigvec_z[1, np.abs(eigval_z) == np.max(np.abs(eigval_z))]))
print('Jacobian Eigenvector Slope 2:')
print('Analytical: ' + str(eigvec_slope_ana_2))
print('Numerical: ' + str(eigvec_z[3, np.abs(eigval_z) == np.min(np.abs(eigval_z))]/eigvec_z[1, np.abs(eigval_z) == np.min(np.abs(eigval_z))]))

# Simulate sample trajectories
sample_traj = False
x0_sample_rand = False

if sample_traj:
    if x0_sample_rand:
        n_sample_traj = 4
    else:
        #x0_sample_array_z = np.array([-0.1, -0.1, 0.1, 0.1])
        x0_sample_array_z = np.array([-0.1, 0.1, 0.1, 0.1])
        #x0_sample_array_zdot = np.array([-0.2, 0.2, -0.1, 0.1]) - U0_array[3,0]
        #x0_sample_array_zdot = np.array([0.0, dz_lim_low_func(x0_sample_array_z[1]), dz_lim_high_func(x0_sample_array_z[2]), ( dz_lim_high_func(x0_sample_array_z[3]) + dz_lim_low_func(x0_sample_array_z[3]) )/2]) - U0_array[3,0]
        x0_sample_array_zdot = np.array([0.0, dz_lim_high_func(x0_sample_array_z[1]), dz_lim_high_func(x0_sample_array_z[2]) + 0.01, dz_lim_high_func(x0_sample_array_z[3]) - 0.01]) - U0_array[3,0]
        x0_sample_array = np.repeat(x_fp.reshape((-1,1)), len(x0_sample_array_z), axis=1)
        x0_sample_array[1,:] = x0_sample_array_z
        x0_sample_array[3,:] = x0_sample_array_zdot
        n_sample_traj = len(x0_sample_array[0,:])

if sample_traj:
    x0_sample_res = np.zeros((n_sample_traj,n,n_steps))
    t_sample_res = dt*np.arange(int( n_steps*(const_T/dt) + 1 ))
    x_step_sample_res = np.zeros((n_sample_traj,n,int( n_steps*(const_T/dt) + 1 )))
    x_inertial_sample_res = np.zeros((n_sample_traj,n,int( n_steps*(const_T/dt) + 1 )))
    U_inertial_sample_res = np.zeros((n_sample_traj,n,int( n_steps*(const_T/dt) + 1 )))

    wmax_sample_res = np.zeros(n_sample_traj)
    for n1 in range(n_sample_traj):
        if x0_sample_rand:
            x0_sample = x0.copy()
            if n1 <= 1:
                #eigvec_c = np.random.uniform(-0.002, 0.002)
                eigvec_c = np.random.uniform(-0.1, 0.1)
                x0_sample[1] = x_fp[1] + eigvec_c*eigvec_z[1, np.abs(eigval_z) == np.min(np.abs(eigval_z))]
                x0_sample[3] = x_fp[3] + eigvec_c*eigvec_z[3, np.abs(eigval_z) == np.min(np.abs(eigval_z))]
            else:
                #x0_sample[1] = np.random.uniform(z_test[0], z_test[-1])
                #x0_sample[1] = np.random.uniform(np.min(fp_array[1,:]) - 0.02, np.max(fp_array[1,:]) + 0.02)
                #x0_sample[1] = np.random.uniform(x_fp[1] - 0.001, x_fp[1] + 0.001)
                x0_sample[1] = np.random.uniform(x_fp[1] - 0.02, x_fp[1] + 0.02)
                x0_sample[3] = np.random.uniform(dz_lim_low_func(x0_sample[1]), dz_lim_high_func(x0_sample[1])) - U0_array[3,0]
                #x0_sample[3] = np.random.uniform(x_fp[3] - 0.001, x_fp[3] + 0.001)
        else:
            x0_sample = x0_sample_array[:,n1]
        if copquad == 1:
            [x_sample_full, x0_sample_full, U0_sample_full, T_sample_array] = step_dyn(model_dyn, x0_sample, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, copparlim = copparlim, const_T=const_T, propcont=propcont, copadj=copadj, constoff=1, ana_soln=0, stepprint=0)
        else:
            [x_sample_full, x0_sample_full, U0_sample_full, T_sample_array] = step_dyn(model_dyn_const, x0_sample, dt, U0_array, const_T=const_T, propcont=propcont, constoff=1, ana_soln=0, stepprint=0)
        x0_sample_res[n1,:,:] = x0_sample_full.step
        x0_sample_res[n1,2:,:] = x0_sample_res[n1,2:,:] + U0_array[2:,:]
        x_step_sample_res[n1,:,:] = x_sample_full.step
        x_step_sample_res[n1,2:,:] = x_step_sample_res[n1,2:,:] + U0_sample_full.step[2:,:]
        x_inertial_sample_res[n1,:,:] = x_sample_full.inertial
        U_inertial_sample_res[n1,:,:] = U0_sample_full.inertial

        wmax_sample = 0.0
        for m1 in range(n_steps):
            if m1 > 0:
                stepwidth = np.abs(U0_sample_full.res[1,m1] - U0_sample_full.res[1,m1-1])
                if stepwidth > wmax_sample:
                    wmax_sample = stepwidth
        wmax_sample_res[n1] = wmax_sample - w_max

# Display results
eigvec_color = ['C1','C2','C3','C4']
eig_color_count = 0
if crit == 'viability':
    fig_type = 'Recovery Basin'
    fig_cmap = 'RdBu'
    fig_norm = 'linear'
elif crit == 'convergence':
    fig_type = 'Convergence Basin'
    #fig_cmap = 'bwr'
    #fig_cmap_bounds = np.arange(np.min(basin_z), np.max(basin_z) + 1)
    #fig_cmap_bounds = np.arange(np.min(basin_z), np.max(basin_z) + 2)
    fig_cmap_bounds = np.arange(0.0, 13.0)
    fig_cmap = plt.get_cmap('bwr', len(fig_cmap_bounds) - 1)
    fig_norm = matplotlib.colors.BoundaryNorm(boundaries=fig_cmap_bounds, ncolors=fig_cmap.N)
elif crit == 'stepwidth':
    fig_type = 'Step Width Basin'
    fig_cmap = 'coolwarm'
    fig_norm = 'linear'

if data_propcont:
    xlim = (z_test[0] + (dz_test[propcont_dz_idx] + U0_array[3,0])/omega_0, z_test[-1] + (dz_test[propcont_dz_idx] + U0_array[3,0])/omega_0)
    ylim = (z_c_err_test[0], z_c_err_test[-1])

    xlabel = r'$\zeta$ (m)'
    #ylabel = r'$(z - z_{c})$ (m)'
    ylabel = r'$\zeta - z_{c}$ (m)'
else:
    xlim = (z_test[0], z_test[-1])
    ylim = (dz_test[0] + U0_array[3,0], dz_test[-1] + U0_array[3,0])

    xlabel = r'$z$ (m)'
    ylabel = r'$\dot{z}$ (m/s)'

if z_only == 1:
    fig, axes = plt.subplots(figsize=(8,8))
    pos_z = axes.imshow(basin_z, cmap=fig_cmap, norm=fig_norm, 
                        origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], aspect='auto')
    axes.tick_params(axis='both', which='major', labelsize=12)
    if not data_propcont:
        #axes.plot(x_fp[1], x_fp[3] + U0_array[3,0], 'o', mfc='w', mec='k', mew=1.5, markersize=10, label='Reference Fixed Point', zorder=10)
        axes.plot(x_fp[1], x_fp[3] + U0_array[3,0], 'o', mfc='w', mec='k', mew=1.5, markersize=10, label='Fixed Point', zorder=10)
        axes.plot(z_test, basin_boundary_ana[0,:], 
                color='yellow', path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()], lw=3.0, label='Analytical Boundary')
        axes.plot(z_test, basin_boundary_ana[1,:], 
                color='yellow', path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()], lw=3.0)
        if deltafp > 0:
            axes.plot(fp_array[1,:], fp_array[3,:], 'go', label='Quadratic CoP Numerical Fixed Points')
            axes.plot(fp_ana_array[0,:], fp_ana_array[1,:], 'm.', label='Quadratic CoP Analytical Fixed Points')
        if crit == 'viability' or crit == 'convergence':
            for n1 in range(4):
                if eigvecmag_z_array[n1] == 1:
                    axes.axline((x_fp[1], x_fp[3] + U0_array[3,0]), slope=(jacobian_fp_eigvec[3,n1]/jacobian_fp_eigvec[1,n1]),
                                color=eigvec_color[eig_color_count], path_effects=[pe.Stroke(linewidth=4.0, foreground='black'), pe.Normal()], lw=2.0, label=rf"Eigenvector, $\lambda = {jacobian_fp_eigval[n1]:.4g}$")
                    eig_color_count += 1
    else:
        for n1 in range(4):
            if n1 == 0:
                axes.plot(zeta_test_dense, basin_boundary_ana[n1,:], 
                        color='yellow', path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()], lw=3.0, label='Analytical Boundary')
            else:
                axes.plot(zeta_test_dense, basin_boundary_ana[n1,:], 
                        color='yellow', path_effects=[pe.Stroke(linewidth=5.0, foreground='black'), pe.Normal()], lw=3.0)
    if sample_traj:
        for n1 in range(n_sample_traj):
            if n1 == 0:
                axes.plot(x0_sample_res[n1,1,:], x0_sample_res[n1,3,:], 'kx:', label='Sample Trajectory')
                axes.plot(x0_sample_res[n1,1,0], x0_sample_res[n1,3,0], 'C2x', label='Sample Trajectory Start')
            else:
                axes.plot(x0_sample_res[n1,1,:], x0_sample_res[n1,3,:], 'kx:')
                axes.plot(x0_sample_res[n1,1,0], x0_sample_res[n1,3,0], 'C2x')
    axes.set_xlim(xlim[0], xlim[1])
    axes.set_ylim(ylim[0], ylim[1])
    axes.set_xlabel(xlabel, fontsize=14)
    axes.set_ylabel(ylabel, fontsize=14)
else:
    fig, axes = plt.subplots(1, 2, figsize=(10,8))
    pos_x = axes[0].imshow(basin_x, cmap='RdBu', origin='lower', extent=[x_test[0], x_test[-1], dx_test[0] + U0_array[2,0], dx_test[-1] + U0_array[2,0]], aspect='auto')
    axes[0].plot(fp_array[0,:], fp_array[2,:], 'go')
    axes[0].plot(x_fp[0], x_fp[2] + U0_array[2,0], 'ko', markersize=6)
    axes[0].set_xlabel(r'$x$ (m)')
    axes[0].set_ylabel(r'$\dot{x}$ (m/s)')
    pos_z = axes[1].imshow(basin_z, cmap='RdBu', origin='lower', extent=[z_test[0], z_test[-1], dz_test[0] + U0_array[3,0], dz_test[-1] + U0_array[3,0]], aspect='auto')
    axes[1].plot(fp_array[1,:], fp_array[3,:], 'go', label='Quadratic CoP Numerical Fixed Points')
    axes[1].plot(fp_ana_array[0,:], fp_ana_array[1,:], 'm.', label='Quadratic CoP Analytical Fixed Points')
    axes[1].plot(x_fp[1], x_fp[3] + U0_array[3,0], 'ko', markersize=6, label='Reference Fixed Point')
    axes[1].set_xlabel(r'$z$ (m)')
    axes[1].set_ylabel(r'$\dot{z}$ (m/s)')

if data_propcont:
    if copquad == 1:
        fig_title = rf"{fig_type} Results ($\alpha = {alpha:.2f}$, $\beta = {beta:.2f}$) ($\dot{{z}} = {(dz_test[propcont_dz_idx] + U0_array[3,0]):.2f}$ m/s)"
    else:
        fig_title = rf"{fig_type} Results ($\alpha = 0.0$, $\beta = 0.0$) ($\dot{{z}} = {(dz_test[propcont_dz_idx] + U0_array[3,0]):.2f}$ m/s)"
else:
    if copquad == 1:
        #fig_title = rf"{fig_type} Results (Reference: $\alpha$ = {alpha:.2f}, $\beta$ = {beta:.2f})"
        fig_title = rf"{fig_type} Results ($\alpha = {alpha:.2f}$, $\beta = {beta:.2f}$)"
    else:
        #fig_title = rf"{fig_type} Results (Reference: $\alpha = 0.0$, $\beta = 0.0$)"
        fig_title = rf"{fig_type} Results ($\alpha = 0.0$, $\beta = 0.0$)"
fig.suptitle(fig_title, fontsize=18)
fig.legend(loc='upper right')
if crit == 'convergence':
    cbar = fig.colorbar(pos_z, ax=axes, ticks=fig_cmap_bounds)
    cbar.set_label(rf'Steps to Convergence ($\epsilon = {tol_conv:.2g}$)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
elif crit == 'stepwidth':
    cbar = fig.colorbar(pos_z, ax=axes)
    cbar.set_label(rf'Relative Maximum Step Width (Maximum Width = ${w_max:.2g}$)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
plt.show()

if sample_traj:
    print('Sample Relative Maximum Step Width:')
    print(wmax_sample_res)
    if z_only != 1:
        fig, axes = plt.subplots(ncols=1, nrows=4, sharex=True)
        for n1 in range(n_sample_traj):
            for n2 in range(n):
                if n2 == 0:
                    axes[n2].plot(t_sample_res, x_step_sample_res[n1,n2,:], label=f'Trajectory {n1 + 1}')
                else:
                    axes[n2].plot(t_sample_res, x_step_sample_res[n1,n2,:])
        axes[0].set_ylabel(r'$x$ (m)')
        axes[1].set_ylabel(r'$z$ (m)')
        axes[2].set_ylabel(r'$\dot{x}$ (m/s)')
        axes[3].set_ylabel(r'$\dot{z}$ (m/s)')
        axes[3].set_xlabel(r'$t$ (s)')
    else:
        fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True)
        for n1 in range(n_sample_traj):
            if n1 == 0:
                axes[0].axhline(x_fp[1], ls=':', label=f'Fixed Point')
                axes[1].axhline(x_fp[3] + U0_array[3,0], ls=':')
            axes[0].plot(t_sample_res, x_step_sample_res[n1,1,:], label=f'Trajectory {n1 + 1}')
            axes[1].plot(t_sample_res, x_step_sample_res[n1,3,:])
        axes[0].set_ylabel(r'$z$ (m)')
        axes[1].set_ylabel(r'$\dot{z}$ (m/s)')
        axes[1].set_xlabel(r'$t$ (s)')
    fig.legend()
    plt.show()

    plt.figure()
    for n1 in range(n_sample_traj):
        plt.plot(x_inertial_sample_res[n1,1,:], x_inertial_sample_res[n1,0,:], zorder=int(10+n1), label=f'Trajectory {n1 + 1}')
        plt.plot(U_inertial_sample_res[n1,1,:], U_inertial_sample_res[n1,0,:], 'x:', label=f'CoP {n1 + 1}')
    plt.xlabel(r'$Z$ (m)')
    plt.ylabel(r'$X$ (m)')
    plt.legend(loc='upper right')
    plt.show()



