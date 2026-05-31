import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.utils import Bunch

from walkingmodel_function import model_dyn_def, model_linear_dyn_def, poincare, plotfunc_state_3D, plotfunc_state_full

from walkingmodel_function import ddU_x_0 as ddU_x
from walkingmodel_function import ddU_z_0 as ddU_z
from walkingmodel_function import steploc_constoffset as steploc

# Variable T

#const_T = 0.5
const_T = 0.2
propcont = 0

# Constants
n = 4
g = 9.81
l = 1.09

omega_0 = np.sqrt(g/l)

# Define model
model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)

def step_dyn(model_dyn, x0, dt, U0_array, const_T=0, propcont=0, stepprint=1):
    steps_len = U0_array.shape[1]
    leftright = 1

    T_c = 0.5
    s_c = U0_array[0,1] - U0_array[0,0]
    w_c = U0_array[1,1] - U0_array[1,0]
    omega_0 = np.sqrt(g/l)
    b_x = (s_c)/(np.exp(omega_0*T_c) - 1)
    b_z = (w_c)/(np.exp(omega_0*T_c) + 1)

    U0_res = np.zeros(np.shape(U0_array))
    U0_res[:,0] = U0_array[:,0]
    U0_viable = np.zeros((steps_len, 2, 5001))

    x0_array = np.zeros((n, steps_len))
    x0_array_t = np.zeros(steps_len)
    x0_array[:,0] = x0

    T_array = np.zeros(steps_len)

    for n1 in range(steps_len):
        if n1 == 0:
            s = s_c
            xi_n = ( ((x0_array[0, n1] + U0_res[0, n1]) + (x0_array[2, n1]/omega_0)) )
        else:
            s = U0_res[0, n1] - U0_res[0, n1-1]
        if const_T == 0:
            T = (1/omega_0)*(np.log((s/(xi_n - U0_res[0, n1])) + 1))
        else:
            T = const_T

        T_array[n1] = T
        t_span = (0.0, T)
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

        if stepprint == 1:
            print('Step: ' + str(n1 + 1))
            print('Period (s): ' + str(T))

        soln = solve_ivp(model_dyn, t_span, x0_array[:, n1], t_eval=t_eval, rtol=1e-9, atol=1e-12)
        x_step = soln.y
        t_step = soln.t

        xi_n = ( ((x_step[0, -1] + U0_res[0, n1]) + (x_step[2, -1]/omega_0)) )

        if n1 != steps_len-1:
            [U0_res[:2, n1+1], U0_viable[n1, :, :]] = steploc(x_step[:,-1], U0_res[:2, n1], np.array([omega_0, b_x, b_z]), leftright=leftright, propcont=propcont)
            x0_array[:, n1+1] = poincare(x_step[:, -1], U0_res[:, n1:n1+2], leftright=leftright)
            x0_array_t[n1+1] = x0_array_t[n1] + t_step[-1]

        if leftright == 1:
            if n1 == 0:
                x_res_inertial = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + x_step
                x_res = x_step
                x_step_full = x_step
            else:
                x_step_inertial = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + x_step
                x_res_inertial = np.concatenate((x_res_inertial[:,:-1], x_step_inertial), axis=1)
                x_res = np.concatenate((x_res[:,:-1], x_step), axis=1)
                x_step_full = np.concatenate((x_step_full[:,:-1], x_step), axis=1)
            leftright = 0
        else:
            if n1 == 0:
                x_res_inertial = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)
                x_res = (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)
                x_step_full = x_step
            else:
                x_step_inertial = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1) + (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)
                x_res_inertial = np.concatenate((x_res_inertial[:,:-1], x_step_inertial), axis=1)
                x_res = np.concatenate((x_res[:,:-1], (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)), axis=1)
                x_step_full = np.concatenate((x_step_full[:,:-1], x_step), axis=1)
            leftright = 1
    
    x_res_3D = np.zeros((3, x_res_inertial.shape[1]))
    x_res_3D[0,:] = x_res_inertial[0,:]
    x_res_3D[1,:] = np.sqrt((l**2) - ((x_res[0,:]**2) + (x_res[1,:]**2)))
    x_res_3D[2,:] = x_res_inertial[1,:]

    x0_array_inertial = np.zeros(x0_array.shape)

    even = 1
    for n1 in range(steps_len):
        if even == 1:
            x0_array_inertial[:,n1] = U0_res[:,n1] + x0_array[:,n1]
            even = 0
        else:
            x0_array_inertial[:,n1] = U0_res[:,n1] + (np.diag([1.0, -1.0, 1.0, -1.0])@x0_array[:,n1])
            even = 1

    x0_inertial_3D = np.zeros((3, steps_len))
    x0_inertial_3D[0,:] = x0_array_inertial[0,:]
    x0_inertial_3D[1,:] = np.sqrt((l**2) - (((x0_array_inertial[0,:] - U0_res[0,:])**2) + ((x0_array_inertial[1,:] - U0_res[1,:])**2)))
    x0_inertial_3D[2,:] = x0_array_inertial[1,:]

    x_full = Bunch(t = np.round(dt*np.arange(x_res_inertial.shape[1]), 2), step = x_step_full, inertial = x_res_inertial, inertial_3D = x_res_3D)
    x0_full = Bunch(t = x0_array_t, step = x0_array, inertial = x0_array_inertial, inertial_3D = x0_inertial_3D)
    U0_full = Bunch(res = U0_res, viable = U0_viable)

    return x_full, x0_full, U0_full, T_array

def model_num_linear_dyn_def(xinit, nonlin_model_dyn, epsval = 1e-8):
    A_mat = np.zeros((n,n))

    for n1 in range(n):
        eps = np.zeros(n)
        eps[n1] = epsval
        xdot_perturb = np.zeros((n,2))
        for n2 in range(2):
            if n2 == 0:
                xinit_analysis = xinit + eps
            else:
                xinit_analysis = xinit - eps
            xdot_perturb[:,n2] = nonlin_model_dyn(0,xinit_analysis)
        A_mat[:,n1] = (1/2)*(1/epsval)*(xdot_perturb[:,0] - xdot_perturb[:,1])

    def lin_model_dyn(t, x):
        xdot = (A_mat@(x - xinit)) + nonlin_model_dyn(0,xinit)
        return xdot
    
    return lin_model_dyn

def U0_array_def(n_steps, s_c, w_c, z_c):
    U0_array = np.zeros((n,n_steps))
    U0_array[0,:] = np.linspace(0.0, (n_steps-1)*(s_c), n_steps)
    U0_array[1,:] = z_c + (w_c/2)*((-1)**(np.arange(n_steps) + 1))

    return U0_array

def plotfunc_state_compare_step(dt, x_full_1, x0_full_1, x_full_2, x0_full_2):
    x_array_t = dt*np.arange(x_full_1.inertial.shape[1])
    x0_array_t = x0_full_1.t

    fig, axes = plt.subplots(4, sharex=True)
    for n1 in range(4):
        if n1 == 0:
            axes[n1].plot(x_array_t, x_full_1.step[n1,:], label='Trajectory 1')
            axes[n1].plot(x0_array_t, x0_full_1.step[n1,:], 'co', label='Step Transition 1')
            axes[n1].plot(x_array_t, x_full_2.step[n1,:], label='Trajectory 2')
            axes[n1].plot(x0_array_t, x0_full_2.step[n1,:], 'mo', label='Step Transition 2')
        else:
            axes[n1].plot(x_array_t, x_full_1.step[n1,:])
            axes[n1].plot(x0_array_t, x0_full_1.step[n1,:], 'co')
            axes[n1].plot(x_array_t, x_full_2.step[n1,:])
            axes[n1].plot(x0_array_t, x0_full_2.step[n1,:], 'mo')
    axes[0].set_ylabel('$ x $ (m)')
    axes[1].set_ylabel('$ z $ (m)')
    axes[2].set_ylabel('$ \dot{x} $ (m/s)')
    axes[3].set_xlabel('t (s)')
    axes[3].set_ylabel('$ \dot{z} $ (m/s)')
    fig.legend()
    plt.show()

# Simulation parameters
dt = 0.01
n_steps = 40
s_c = 0.75
w_c = 0.1
z_c = 0.0
x0 = np.array([0.25, 0.02, 0.0, 0.0])

U0_array = U0_array_def(n_steps, s_c, w_c, z_c)

# Simulation results
[x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, const_T=const_T, propcont=propcont)

# Linearized model
n_steps_lin_test = 10
xinit_lin_test = np.zeros(n)
#xinit_lin_test = x0_full.step[:,-1]
#xinit_lin_test = (x_full.step[:, x_full.t == 19.85]).squeeze()
#xinit_lin_test = (x_full.step[:, x_full.t == 19.5]).squeeze()

lin_model_dyn = model_num_linear_dyn_def(xinit_lin_test, model_dyn)
U0_array_lin_test = U0_array_def(n_steps_lin_test, s_c, w_c, z_c)

#x0_lin_test = x0
x0_lin_test = x0_full.step[:,-2]

[x_lin_full, x0_lin_full, U0_lin_full, T_lin_array] = step_dyn(lin_model_dyn, x0_lin_test, dt, U0_array_lin_test, const_T=const_T, propcont=propcont, stepprint=0)
[x_test_full, x0_test_full, U0_test_full, T_test_array] = step_dyn(model_dyn, x0_lin_test, dt, U0_array_lin_test, const_T=const_T, propcont=propcont, stepprint=0)

print('Error: ')
print(np.linalg.norm(np.mean(x_lin_full.step - x_test_full.step, axis=1)))

# Plot results
xzval_inertial = 0
xzval_step = 1

plotfunc_state_full(dt, omega_0, x_full, x0_full, U0_full, U0_array, xzval_inertial=xzval_inertial, xzval_step=xzval_step)

plt.figure()
plt.plot(T_array, '-o')
plt.xlabel('Steps')
plt.ylabel('T (s)')
plt.show()

plotfunc_state_compare_step(dt, x_lin_full, x0_lin_full, x_test_full, x0_test_full)



