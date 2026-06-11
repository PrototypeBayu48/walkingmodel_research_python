import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy import optimize
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def

# Quadratic CoP Test

linear_dyn = 1
propcont = 0
copquad = 1
x0_fp = 1
footgeom = 0
copadj = 0
noise = 0

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
n_steps = 10
s_c = 0.75
w_c = 0.1
z_c = 0.0

# Define CoP Variables
footl = 0.30
footw = 0.12

#copparlim = np.array([[(footl/s_c), 0.01], [(4*footw)/(footl**2), 0.5]])
copparlim = np.array([[(footl/s_c), 0.0], [(4*footw)/(footl**2), 0.0]])
print(copparlim)

if copquad == 1:
    if footgeom == 1:
        alpha = copparlim[0,0]
        beta = copparlim[1,0]
    else:
        #alpha = 0.4
        alpha = 0.35
        #alpha = 0.2
        #beta = 0.2
        beta = copparlim[1,0]/3
else:
    alpha = 0
    beta = 0

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
    U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=cop_params_test['b1'], dU_z_0=cop_params_test['c1'])
else:
    U0_array = U0_array_func(n_steps, s_c, w_c, z_c)

# Define x0
U0_array_fp = U0_array[:,:2]
if copquad == 1:
    x_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn, U0_array_fp, dot_U_func, cop_params)
else:
    x_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn_const, U0_array_fp, False, None)

if x0_fp == 1:
    x0 = x_fp
else:
    x0 = np.array([0.25, 0.02, 0.0, 0.0])

# Fixed point simulation
deltafp = 20
alphabeta_array_fp = np.zeros((2,deltafp))
alphabeta_array_fp[0,:] = np.linspace(copparlim[0,1], copparlim[0,0], deltafp)
alphabeta_array_fp[1,:] = np.linspace(copparlim[1,1], copparlim[1,0], deltafp)
fp_array = np.zeros((deltafp,deltafp,n))
alphabeta_array_res = np.zeros((deltafp,deltafp,2))

def grid_fixedpoint(n1, n2, alphabeta):
    b1 = (alphabeta[0]*s_c)/const_T
    c1 = -alphabeta[0]*alphabeta[1]*b1*s_c
    c2 = alphabeta[1]*(b1**2)
    cop_params_fptest = {'b1': b1, 'c1': c1, 'c2': c2}
    def ddU_z_quad_fptest(t,x):
        return 2*cop_params_fptest['c2']
    def dot_U_func_fptest(t,x):
        dot_U = np.array([x[2], x[3], ddU_x(t,x), ddU_z_quad_fptest(t,x)])
        return dot_U
    
    model_dyn_fptest = model_linear_dyn_def([g, l], ddU_x, ddU_z_quad_fptest)
    U0_array_fptest = U0_array_func(2, s_c, w_c, z_c, dU_x_0=cop_params_fptest['b1'], dU_z_0=cop_params_fptest['c1'])
    x0_fp = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn_fptest, U0_array_fptest, dot_U_func_fptest, cop_params_fptest)
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
c2_array_res = alphabeta_array_res[1,:]*(( (alphabeta_array_res[0,:]*s_c)/const_T )**2)

fp_ana_array = np.zeros((2,int(deltafp*deltafp)))
fp_ana_array[0,:] = (w_c/2)*np.ones(int(deltafp*deltafp))
fp_ana_array[1,:] = -(1/(2*omega_0*( np.exp(omega_0*const_T) + 1 )))*( (w_c*(omega_0**2)*(np.exp(omega_0*const_T) - 1)) + c2_array_res*( np.exp(omega_0*const_T)*(2*const_T*omega_0 - 4) + (2*const_T*omega_0) + 4 ) )

# Define simulation parameters
deltaz = 40

#z_test = np.linspace(-0.1, 0.2, deltaz)
#z_test = np.linspace(x0[1] - 0.2, x0[1] + 0.2, deltaz)
#z_test = np.linspace(x0[1] - 0.4, x0[1] + 0.4, deltaz)
z_test = np.linspace(np.min(fp_array[1,:]) - 0.2, np.max(fp_array[1,:]) + 0.2, deltaz)

#dz_test = np.linspace(-0.2, 0.8, deltaz)
#dz_test = np.linspace(x0[3] - 0.2, x0[3] + 0.2, deltaz)
#dz_test = np.linspace(x0[3] - 0.4, x0[3] + 0.4, deltaz)
dz_test = np.linspace(np.min(fp_array[3,:]) - 0.4, np.max(fp_array[3,:]) + 0.4, deltaz) - U0_array[3,0]

basin_z = np.ones((len(dz_test), len(z_test)))

deltax = deltaz
#x_test = np.linspace(-0.5, 0.1, deltax)
#x_test = np.linspace(x0[0] - 0.2, x0[0] + 0.2, deltax)
x_test = np.linspace(np.min(fp_array[0,:]) - 0.2, np.max(fp_array[0,:]) + 0.2, deltax)

#dx_test = np.linspace(-1.0, 2.0, deltax)
#dx_test = np.linspace(x0[2] - 1.0, x0[2] + 1.0, deltax)
dx_test = np.linspace(np.min(fp_array[2,:]) - 1.0, np.max(fp_array[2,:]) + 1.0, deltax) - U0_array[2,0]
basin_x = np.ones((len(dx_test), len(x_test)))

if noise == 1:
    testlen = 100
    U0_noise_lim = np.array([0.002, 0.005])
    #U0_noise_lim = np.array([0.08, 0.08])
else:
    testlen = 1
    U0_noise_lim = np.array([0.0, 0.0])

# Simulation
z_only = 1
w_max = 0.8*l

def simulate_grid_point(n1, n2, x0_test):
    np.random.seed(int( (n1*1000) + n2 ))
    point_score = 0.0

    for p1 in range(testlen):
        cop_params_test.update(cop_params)

        if copquad == 1:
            [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0_test, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, U0_noise=U0_noise_lim, copparlim = copparlim, const_T=const_T, propcont=propcont, copadj=copadj, constoff=1, stepprint=0)
        else:
            [x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn_const, x0_test, dt, U0_array, U0_noise=U0_noise_lim, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)

        x0_inertial = x0_full.inertial
        U0_array_res = U0_full.res

        leftright = 1
        trial_result = 1.0
        
        for m1 in range(n_steps):
            if m1 > 0:
                if (np.abs(U0_array_res[1,m1] - U0_array_res[1,m1-1]) > (w_max)) or (np.abs(U0_array_res[0,m1] - U0_array_res[0,m1-1]) > (w_max)):
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
    
    return n1, n2, (point_score / testlen)


print('Simulating Lateral Motion')
results_z = Parallel(n_jobs=-1, verbose=5)(
    delayed(simulate_grid_point)(n1, n2, np.array([x0[0], z_test[n2], x0[2], dz_test[n1]]))
    for n1 in range(len(x_test))
    for n2 in range(len(z_test))
)
for n1, n2, avg_val in results_z:
    basin_z[n1, n2] = avg_val

print('Average Recovery Basin Value (z): ' + str(np.mean(basin_z)))

if z_only != 1:
    print('Simulating Forward Motion')
    results_x = Parallel(n_jobs=-1, verbose=5)(
        delayed(simulate_grid_point)(n1, n2, np.array([x_test[n2], x0[1], dx_test[n1], x0[3]]))
        for n1 in range(len(x_test))
        for n2 in range(len(z_test))
    )
    for n1, n2, avg_val in results_x:
        basin_x[n1, n2] = avg_val

    print('Average Recovery Basin Value (x): ' + str(np.mean(basin_x)))

# Define recovery basin boundaries
b_z = ( (1)/(1 + np.exp(omega_0*const_T)) )*( (w_c) - (cop_params_test['c1'])*( (const_T) + ((1 - np.exp(omega_0*const_T))/omega_0) ) - cop_params_test['c2']*( (const_T**2) + ((2*const_T)/(omega_0)) + ((2*(1 - np.exp(omega_0*const_T)))/(omega_0**2)) ) )
def dz_lim_low_func(z):
    return omega_0*((np.exp(-omega_0*const_T)*(-b_z + ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)
def dz_lim_high_func(z):
    return omega_0*((np.exp(-omega_0*const_T)*(w_max - b_z + ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)
basin_boundary_ana = np.zeros((2,deltaz))
#basin_boundary_ana[0,:] = -omega_0*(z_test + (b_z*np.exp(-omega_0*const_T)))
#basin_boundary_ana[0,:] = omega_0*((np.exp(-omega_0*const_T)*(-b_z + ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z_test)
basin_boundary_ana[0,:] = dz_lim_low_func(z_test)
#basin_boundary_ana[1,:] = omega_0*((np.exp(-omega_0*const_T)*(w_max - b_z)) - z_test)
#basin_boundary_ana[1,:] = omega_0*((np.exp(-omega_0*const_T)*(w_max - b_z + ( (cop_params_test['c2']/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (cop_params_test['c1']/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z_test)
basin_boundary_ana[1,:] = dz_lim_high_func(z_test)

# Simulate sample trajectories
sample_traj = 0

if sample_traj == 1:
    n_sample_traj = 4
    x0_sample_res = np.zeros((n_sample_traj,n,n_steps))
    for n1 in range(n_sample_traj):
        x0_sample = x0.copy()
        x0_sample[1] = np.random.uniform(z_test[0], z_test[-1])
        x0_sample[3] = np.random.uniform(dz_lim_low_func(x0_sample[1]), dz_lim_high_func(x0_sample[1]))
        if copquad == 1:
            [x_sample_full, x0_sample_full, U0_sample_full, T_sample_array] = step_dyn(model_dyn, x0_sample, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, U0_noise=U0_noise_lim, copparlim = copparlim, const_T=const_T, propcont=propcont, copadj=copadj, constoff=1, stepprint=0)
        else:
            [x_sample_full, x0_sample_full, U0_sample_full, T_sample_array] = step_dyn(model_dyn_const, x0_sample, dt, U0_array, U0_noise=U0_noise_lim, const_T=const_T, propcont=propcont, constoff=1, stepprint=0)
        x0_sample_res[n1,:,:] = x0_sample_full.step

# Display results
if z_only == 1:
    fig, axes = plt.subplots()
    pos_z = axes.imshow(basin_z, cmap='RdBu', origin='lower', extent=[z_test[0], z_test[-1], dz_test[0] + U0_array[3,0], dz_test[-1] + U0_array[3,0]], aspect='auto')
    axes.plot(z_test, basin_boundary_ana[0,:], 'C1', label='Analytical Boundary')
    axes.plot(z_test, basin_boundary_ana[1,:], 'C1')
    axes.plot(fp_array[1,:], fp_array[3,:], 'go', label='Quadratic CoP Numerical Fixed Points')
    axes.plot(fp_ana_array[0,:], fp_ana_array[1,:], 'm.', label='Quadratic CoP Analytical Fixed Points')
    axes.plot(x_fp[1], x_fp[3] + U0_array[3,0], 'ko', markersize=6, label='Reference Fixed Point')
    if sample_traj == 1:
        for n1 in range(n_sample_traj):
            axes.plot(x0_sample_res[n1,1,:], x0_sample_res[n1,3,:], 'kx:')
            axes.plot(x0_sample_res[n1,1,0], x0_sample_res[n1,3,0], 'C2x')
    axes.set_xlim(z_test[0], z_test[-1])
    axes.set_ylim(dz_test[0] + U0_array[3,0], dz_test[-1] + U0_array[3,0])
    axes.set_xlabel(r'$z$ (m)')
    axes.set_ylabel(r'$\dot{z}$ (m/s)')
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

if copquad == 1:
    fig_title = 'Recovery Basin Results (Reference: alpha = ' + f"{alpha:.2f}" + ', beta = ' + f"{beta:.2f}" +')'
else:
    fig_title = 'Recovery Basin Results (Reference: alpha = ' + str(0.0) + ', beta = ' + str(0.0) +')'
fig.suptitle(fig_title)
fig.legend(loc='upper right')
fig.colorbar(pos_z, ax=axes)
plt.show()



