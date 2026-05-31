import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import optimize
import os

from walkingmodel_function import model_dyn_def, U0_array_func, step_dyn, plotfunc_state_full, model_num_linear_dyn_def, model_linear_dyn_def

# Quadratic CoP Parameter Adjustment Test, Comparison to Hof (2008)

linear_dyn = 1
propcont = 0

x0_fp = 0
footgeom = 0

U0_exp = 1

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

if U0_exp == 1:
    # Load Data
    cop_full = np.loadtxt(os.getcwd() + "/experimental_data/hof_2008/hof_2008_fig_3b_cop.csv", delimiter=",", dtype=float)
    cop_full = cop_full.T
    traj_full = np.loadtxt(os.getcwd() + "/experimental_data/hof_2008/hof_2008_fig_3b_traj.csv", delimiter=",", dtype=float)
    traj_full = traj_full.T
    xcom_full = np.loadtxt(os.getcwd() + "/experimental_data/hof_2008/hof_2008_fig_3b_xcom.csv", delimiter=",", dtype=float)
    xcom_full = xcom_full.T

    footpoint = np.array([7, 5, 6, 6, 7, 6])
    t_step_full = 0.1*np.arange(len(cop_full[0,:]))

    traj_exp = np.zeros((n,len(traj_full[0,:])))
    #t_traj_exp = np.linspace(t_step_full[0], t_step_full[-1], len(traj_full[0,:]))
    t_traj_exp = np.linspace(t_step_full[0], t_step_full[-1]/2, len(traj_full[0,:]))
    traj_exp[0,:] = traj_full[1,:]
    traj_exp[1,:] = traj_full[0,:]
    traj_exp[2:,:] = np.gradient(traj_exp[:2,:], t_traj_exp[1] - t_traj_exp[0], axis=1)

    #l = g*(( (xcom_full[1,0] - traj_exp[0,0])/traj_exp[2,0] )**2)
    l = g*(( (xcom_full[0,0] - traj_exp[1,0])/traj_exp[3,0] )**2)
    omega_0 = np.sqrt(g/l)
else:
    l = 1.09
    omega_0 = np.sqrt(g/l)

const_T = 0.5
n_steps = 4
z_c = 0.0
if U0_exp == 1:
    s_c = cop_full[1,footpoint[0]] - cop_full[1,0]
    w_c = cop_full[0,footpoint[0]] - cop_full[0,0]
else:
    s_c = 0.75
    w_c = 0.1

# Define CoP Variables
footl = 0.30
footw = 0.12

#copparlim = np.array([[(footl/s_c), 0.01], [(4*footw)/(footl**2), 0.5]])
#copparlim = np.array([[2*(footl/s_c), 0.1], [(4*footw)/(footl**2), 0.5]])
copparlim = np.array([[0.6, 0.2], [3.0, 0.2]])

if footgeom == 1:
    alpha = copparlim[0,0]
    beta = copparlim[1,0]
elif U0_exp == 1:
    datalen = len(cop_full[0,:])

    U_step_1 = np.zeros((2,footpoint[0]))
    U_step_1[0,:] = cop_full[1,:footpoint[0]]
    U_step_1[1,:] = cop_full[0,:footpoint[0]]
    U_step_1 = U_step_1 - np.repeat(U_step_1[:,0].reshape((-1,1)), footpoint[0], 1)

    t_step_1 = 0.1*np.arange(len(U_step_1[0,:]))
    reg_step_1_x = LinearRegression(fit_intercept=False).fit(t_step_1.reshape((-1,1)), U_step_1[0,:].reshape((-1,1)))
    reg_step_1_z = LinearRegression(fit_intercept=False).fit(np.hstack((t_step_1.reshape((-1,1)), (t_step_1.reshape((-1,1)))**2)), U_step_1[1,:].reshape((-1,1)))
else:
    #alpha = 0.4
    alpha = 0.2
    beta = 0.2

if U0_exp == 1:
    b1 = reg_step_1_x.coef_[0,0]
    c1 = reg_step_1_z.coef_[0,0]
    c2 = reg_step_1_z.coef_[0,1]
else:
    b1 = (alpha*s_c)/const_T
    c1 = -alpha*beta*b1*s_c
    c2 = beta*(b1**2)

cop_params = {'b1': b1, 'c1': c1, 'c2': c2}

print('U_x Coefficient:')
print(np.array([cop_params['b1']]))
print('U_z Coefficients:')
print(np.array([cop_params['c1'], cop_params['c2']]))

# Define CoP functions
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

# Simulation
dt = 0.01
t_span = (0.0, const_T)
t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

if U0_exp == 1:
    start = 0

    U0_array = np.zeros((4,n_steps))
    for n1 in range(n_steps):
        U0_array[0,n1] = cop_full[1,start]
        U0_array[1,n1] = cop_full[0,start]
        start += footpoint[n1]

    U0_array[2,:] = cop_params['b1']
    U0_array[3,:] = cop_params['c1']
else:
    U0_array = U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0=cop_params['b1'], dU_z_0=cop_params['c1'])

# Define x0
if x0_fp == 1:
    U0_array_fp = U0_array[:,:2]
    x0 = x0_fp_func(np.array([0.25, 0.02, 0.0, 0.0]), model_dyn, U0_array_fp, dot_U_func, cop_params)
    print('Quadratic CoP x0:')
else:
    if U0_exp == 1:
        x0 = traj_exp[:,0] - U0_array[:,0]
        print('Experimental x0:')

    else:
        x0 = np.array([0.25, 0.02, 0.0, 0.0])
print(x0)

const_T_array = 0.1*(footpoint - 1)

[x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, dot_U=dot_U_func, cop_params=cop_params, copparlim = copparlim, const_T=const_T_array, propcont=propcont, constoff=0, copadj=1, stepprint=0)

t_step = x_full.t
x_step_3D = x_full.inertial_3D
x0_step_3D = x0_full.inertial_3D
U_array = U0_full.inertial
cop_params_res = U0_full.params_res

xcom = x_full.inertial[:2,:] + (x_full.inertial[2:,:]/omega_0)
xcom_x0 = x0_full.inertial[:2,:] + (x0_full.inertial[2:,:]/omega_0)

alphabeta_res = np.zeros((2,n_steps))
alphabeta_res[0,:] = (cop_params_res[0,:]*(T_array))/s_c
alphabeta_res[1,:] = cop_params_res[2,:]/(cop_params_res[0,:]**2)

# Display results
plt.figure()
plt.plot(x_step_3D[2,:], x_step_3D[0,:], 'C0-o', label='Quadratic CoP Trajectory')
plt.plot(x_step_3D[2,0], x_step_3D[0,0], 'go', label='Trajectory Start')
plt.plot(x_step_3D[2,-1], x_step_3D[0,-1], 'ro', label='Trajectory End')
plt.plot(xcom[1,:], xcom[0,:], 'C0--', label='Quadratic CoP XcoM')
plt.plot(x0_step_3D[2,:], x0_step_3D[0,:], 'm+', label='Step Transition')
plt.plot(U_array[1,:], U_array[0,:], 'c:x', label='Quadratic CoP')
plt.plot(U_array[1,0], U_array[0,0], 'gX', label='CoP Start')
plt.plot(U_array[1,-1], U_array[0,-1], 'rX', label='CoP End')
if U0_exp == 1:
    plt.plot(traj_exp[1,:], traj_exp[0,:], 'C1-', label='Experimental Trajectory')
    plt.plot(xcom_full[0,:], xcom_full[1,:], 'C1--', label='Experimental XcoM')
    plt.plot(cop_full[0,:], cop_full[1,:], 'b:x', label='Experimental CoP')
plt.xlabel('Z (m)')
plt.ylabel('X (m)')
plt.legend()
plt.show()

fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True)
axes[0].plot(np.arange(n_steps) + 1, alphabeta_res[0,:], 'C0-o', label='Results')
axes[0].plot(np.arange(n_steps) + 1, copparlim[0,0]*np.ones(n_steps), 'C1:', label='Limit')
axes[0].plot(np.arange(n_steps) + 1, copparlim[0,1]*np.ones(n_steps), 'C1:')
axes[1].plot(np.arange(n_steps) + 1, alphabeta_res[1,:], 'C0-o')
axes[1].plot(np.arange(n_steps) + 1, copparlim[1,0]*np.ones(n_steps), 'C1:')
axes[1].plot(np.arange(n_steps) + 1, copparlim[1,1]*np.ones(n_steps), 'C1:')
axes[0].set_ylabel(r'$ \alpha $')
axes[1].set_ylabel(r'$ \beta $')
axes[1].set_xlabel('Steps')
fig.legend()
plt.show()



