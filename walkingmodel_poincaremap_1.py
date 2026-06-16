import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.utils import Bunch

from walkingmodel_function import model_dyn_def, model_linear_dyn_def, model_num_linear_dyn_def, poincare_step, plotfunc_state_3D, plotfunc_state_full

from walkingmodel_function import ddU_x_0 as ddU_x
from walkingmodel_function import ddU_z_0 as ddU_z
from walkingmodel_function import steploc_constoffset as steploc

# Variable T

epsval_analysis_plot = 0

linear_dyn = 0
const_T = 0.5
propcont = 0

# Constants
n = 4
g = 9.81
l = 1.09

omega_0 = np.sqrt(g/l)

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
            x0_array[:, n1+1] = poincare_step(x_step[:, -1], U0_res[:, n1:n1+2], leftright=leftright)
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

def stability_analysis(x0, T, epsval = 1e-8):
    jacobian = np.zeros((n,n))

    U0_analysis = np.zeros((n,2))
    U0_analysis[0,:] = np.linspace(0.0, (2-1)*(s_c), 2)
    U0_analysis[1,:] = z_c + (w_c/2)*((-1)**(np.arange(2) + 1))

    for n1 in range(n):
        eps = np.zeros(n)
        eps[n1] = epsval
        x0_perturb = np.zeros((n,2))
        for n2 in range(2):
            if n2 == 0:
                x0_analysis = x0 + eps
            else:
                x0_analysis = x0 - eps
            [x_stability_analysis, x0_stability_analysis, U0_stability_analysis, T_stability_analysis] = step_dyn(model_dyn, x0_analysis, dt, U0_analysis, const_T = T, propcont=propcont, stepprint=0)
            x0_perturb[:,n2] = x0_stability_analysis.step[:,1] - x0
        jacobian[:,n1] = (1/2)*(1/epsval)*(x0_perturb[:,0] - x0_perturb[:,1])

    return jacobian

def epsval_analysis(x0, T):
    epsval_array = np.array([1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    eigval_array = np.zeros((n, len(epsval_array)))

    print('Eigenvalues:')
    for n1 in range(len(epsval_array)):
        jacobian = stability_analysis(x0, T, epsval = epsval_array[n1])
        [jacobian_eigval, jacobian_eigvec] = np.linalg.eig(jacobian)
        print('Epsilon = ' + str(epsval_array[n1]))
        print(jacobian_eigval)
        eigval_array[:,n1] = np.abs(np.real(jacobian_eigval))

    fig, axes = plt.subplots(4, sharex=True)
    for n1 in range(n):
        axes[n1].loglog(epsval_array, eigval_array[n1,:],'-o')
        axes[n1].set_ylabel('$|real(\lambda)|$ ' + str(n1+1))
    axes[n-1].set_xlabel('$\epsilon$')
    plt.show()

def jacobian_extrapolate(x0, T, epsval = 1e-8):
    jacobian_1 = stability_analysis(x0, T, epsval = (epsval/2))
    jacobian_2 = stability_analysis(x0, T, epsval = epsval)

    jacobian = (1/3)*((4*jacobian_1) - jacobian_2)

    return jacobian

# Simulation parameters
dt = 0.01
n_steps = 40
s_c = 0.75
w_c = 0.1
z_c = 0.0
x0 = np.array([0.25, 0.02, 0.0, 0.0])

U0_array = np.zeros((n,n_steps))
U0_array[0,:] = np.linspace(0.0, (n_steps-1)*(s_c), n_steps)
U0_array[1,:] = z_c + (w_c/2)*((-1)**(np.arange(n_steps) + 1))

# Define model
if linear_dyn == 1:
    #model_dyn = model_linear_dyn_def([g, l], ddU_x, ddU_z)
    nonlin_model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)
    [x_full_init, x0_full_init, U0_full_init, T_array_init] = step_dyn(nonlin_model_dyn, x0, dt, U0_array, const_T=const_T, propcont=propcont)
    #xinit_lin_test = (x_full_init.step[:, x_full_init.t == 19.85]).squeeze()
    xinit_lin_test = np.zeros(n)
    model_dyn = model_num_linear_dyn_def(xinit_lin_test, nonlin_model_dyn)
else:
    model_dyn = model_dyn_def([g, l], ddU_x, ddU_z)

# Simulation results
[x_full, x0_full, U0_full, T_array] = step_dyn(model_dyn, x0, dt, U0_array, const_T=const_T, propcont=propcont)

x_res_inertial = x_full.inertial
x_res_3D = x_full.inertial_3D
x0_array_t = x0_full.t
x0_array = x0_full.inertial
U0_res = U0_full.res

xcom = np.zeros((2, x_res_inertial.shape[1]))
xcom[0,:] = x_res_inertial[0,:] + (x_res_inertial[2,:]/omega_0)
xcom[1,:] = x_res_inertial[1,:] + (x_res_inertial[3,:]/omega_0)

U0_res_lin_t = np.zeros(2*len(x0_array_t))
U0_res_lin = np.zeros((2, 2*len(x0_array_t)))
U0_array_lin = np.zeros((2, 2*len(x0_array_t)))
for n1 in np.arange(len(x0_array_t)):
    if n1 == (len(x0_array_t) - 1):
        U0_res_lin_t[2*n1] = x0_array_t[n1]
        U0_res_lin_t[(2*n1)+1] = dt*(np.arange(x_res_3D.shape[1])[-1])
    else:
        U0_res_lin_t[(2*n1):((2*n1)+2)] = x0_array_t[n1:n1+2]
    U0_res_lin[:, (2*n1):((2*n1)+2)] = np.repeat(U0_res[:2, n1].reshape(2,1), 2, axis=1)
    U0_array_lin[:, (2*n1):((2*n1)+2)] = np.repeat(U0_array[:2, n1].reshape(2,1), 2, axis=1)

U0_res_lin_interp = np.zeros((2, x_res_inertial.shape[1]))
U0_res_lin_interp[0,:] = np.interp(dt*np.arange(x_res_3D.shape[1]), U0_res_lin_t, U0_res_lin[0,:])
U0_res_lin_interp[1,:] = np.interp(dt*np.arange(x_res_3D.shape[1]), U0_res_lin_t, U0_res_lin[1,:])

U0_array_lin_interp = np.zeros((2, x_res_inertial.shape[1]))
U0_array_lin_interp[0,:] = np.interp(dt*np.arange(x_res_3D.shape[1]), U0_res_lin_t, U0_array_lin[0,:])
U0_array_lin_interp[1,:] = np.interp(dt*np.arange(x_res_3D.shape[1]), U0_res_lin_t, U0_array_lin[1,:])

# Results Analysis
analysis_sample_len = 20
x0_steadystate = x0_array[:,-analysis_sample_len:]
U0_res_analysis = U0_res[:,-analysis_sample_len:]
T_analysis = T_array[-analysis_sample_len:]

A_mat = np.ones((analysis_sample_len, 2))
A_mat[:,0] = np.arange(analysis_sample_len) + 1

[x0_steadystate_x_m, x0_steadystate_x_c] = np.squeeze( np.linalg.lstsq(A_mat, np.reshape(x0_steadystate[0,:], (-1,1)))[0] )

print('x:')
print('m = ' + str(x0_steadystate_x_m))
print('c = ' + str(x0_steadystate_x_c))
print('z:')
print('Peak 1 = ' + str(x0_steadystate[1,-1]))
print('Peak 2 = ' + str(x0_steadystate[1,-2]))
print('xdot:')
print('Converged Value = ' + str(np.mean(x0_steadystate[2,:])))
print('zdot:')
print('Peak 1 = ' + str(x0_steadystate[3,-1]))
print('Peak 2 = ' + str(x0_steadystate[3,-2]))
print('Period:')
print('Mean = ' + str(np.round(np.mean(T_analysis), 2)))

x0_analysis = np.array([x0_steadystate[0,-4] - U0_res_analysis[0,-4], x0_steadystate[1,-4] - U0_res_analysis[1,-4], np.mean(x0_steadystate[2,:]), x0_steadystate[3,-4]])
U0_analysis = np.zeros((n,2))
U0_analysis[0,:] = np.linspace(0.0, (2-1)*(s_c), 2)
U0_analysis[1,:] = z_c + (w_c/2)*((-1)**(np.arange(2) + 1))

[x_full_analysis, x0_full_analysis, U0_full_analysis, T_array_analysis] = step_dyn(model_dyn, x0_analysis, dt, U0_analysis, const_T = np.round(np.mean(T_analysis), 2), propcont=propcont, stepprint=0)

if epsval_analysis_plot == 1:
    epsval_analysis(x0_analysis, np.mean(T_analysis))

#jacobian = stability_analysis(x0_analysis, np.mean(np.round(np.mean(T_analysis), 2)))
jacobian = jacobian_extrapolate(x0_analysis, np.mean(np.round(np.mean(T_analysis), 2)))

[jacobian_eigval, jacobian_eigvec] = np.linalg.eig(jacobian)

print('Jacobian:')
print(jacobian)
print('Jacobian Eigenvalues:')
print(jacobian_eigval)
print('Jacobian Eigenvectors:')
print(jacobian_eigvec)

# Plot results
xzval_inertial = 0
xzval_step = 1

plotfunc_state_full(dt, omega_0, x_full, x0_full, U0_full, U0_array, xzval_inertial=xzval_inertial, xzval_step=xzval_step)

plt.figure()
plt.plot(T_array, '-o')
plt.xlabel('Steps')
plt.ylabel('T (s)')
plt.show()

fig, axes = plt.subplots(4, sharex=True)
step_label = ['$ x $ (m)', '$ z $ (m)', '$ \dot{x} $ (m/s)', '$ \dot{z} $ (m/s)']
local_label = ['$ X - u_{x} $ (m)', '$ Z - u_{z} $ (m)', '$ \dot{X} $ (m/s)', '$ \dot{Z} $ (m/s)']
inertial_label = ['$ X $ (m)', '$ Z $ (m)', '$ \dot{X} $ (m/s)', '$ \dot{Z} $ (m/s)']
for n1 in range(4):
    if xzval_inertial == 0:
            if xzval_step == 1:
                axes[n1].plot(x0_full.step[n1,:],'-o')
                axes[n1].set_ylabel(step_label[n1])
            else:
                axes[n1].plot(x0_array[n1,:] - U0_res[n1,:],'-o')
                axes[n1].set_ylabel(local_label[n1])
    else:
        axes[n1].plot(x0_array[n1,:],'-o')
        axes[n1].set_ylabel(inertial_label[n1])
    if n1 == 3:
        axes[n1].set_xlabel('Steps')
plt.show()

ax = plt.figure().add_subplot(projection='3d')
if xzval_inertial == 0:
    if xzval_step == 1:
        ax.plot(x0_full.step[1,:], x0_full.step[3,:], x0_full.step[0,:], '-o', label='Step Transition')
        ax.plot(x0_full.step[1,0], x0_full.step[3,0], x0_full.step[0,0], 'o', label='Start')
        ax.plot(x0_full.step[1,-1], x0_full.step[3,-1], x0_full.step[0,-1], 'o', label='End')
        ax.set_xlabel('$ z $ (m)')
        ax.set_ylabel('$ \dot{z} $ (m/s)')
        ax.set_zlabel('$ x $ (m)')
        ax.set_title('$ \dot{x} $ Converged Value = ' + str(np.mean(x0_steadystate[2,:])))
    else:
        ax.plot(x0_array[1,:] - U0_res[1,:], x0_array[3,:], x0_array[0,:] - U0_res[0,:], '-o', label='Step Transition')
        ax.plot(x0_array[1,0] - U0_res[1,0], x0_array[3,0], x0_array[0,0] - U0_res[0,0], 'o', label='Start')
        ax.plot(x0_array[1,-1] - U0_res[1,-1], x0_array[3,-1], x0_array[0,-1] - U0_res[0,-1], 'o', label='End')
        ax.set_xlabel('$ Z - u_{z} $ (m)')
        ax.set_ylabel('$ \dot{Z} $ (m/s)')
        ax.set_zlabel('$ X - u_{x} $ (m)')
        ax.set_title('$ \dot{X} $ Converged Value = ' + str(np.mean(x0_steadystate[2,:])))
else:
    ax.plot(x0_array[1,:], x0_array[3,:], x0_array[0,:], '-o', label='Step Transition')
    ax.plot(x0_array[1,0], x0_array[3,0], x0_array[0,0], 'o', label='Start')
    ax.plot(x0_array[1,-1], x0_array[3,-1], x0_array[0,-1], 'o', label='End')
    ax.set_xlabel('$ Z $ (m)')
    ax.set_ylabel('$ \dot{Z} $ (m/s)')
    ax.set_zlabel('$ X $ (m)')
    ax.set_title('$ \dot{X} $ Converged Value = ' + str(np.mean(x0_steadystate[2,:])))
ax.legend()
plt.show()

plotfunc_state_full(dt, omega_0, x_full_analysis, x0_full_analysis, U0_full_analysis, U0_analysis, xzval_inertial=xzval_inertial, xzval_step=xzval_step)

fig, axes = plt.subplots(4, sharex=True)
step_label = ['$ x $ (m)', '$ z $ (m)', '$ \dot{x} $ (m/s)', '$ \dot{z} $ (m/s)']
local_label = ['$ X - u_{x} $ (m)', '$ Z - u_{z} $ (m)', '$ \dot{X} $ (m/s)', '$ \dot{Z} $ (m/s)']
inertial_label = ['$ X $ (m)', '$ Z $ (m)', '$ \dot{X} $ (m/s)', '$ \dot{Z} $ (m/s)']
for n1 in range(4):
    if xzval_inertial == 0:
            if xzval_step == 1:
                axes[n1].plot(x0_full_analysis.step[n1,:],'-o')
                axes[n1].set_ylabel(step_label[n1])
            else:
                axes[n1].plot(x0_full_analysis.inertial[n1,:] - U0_full_analysis.res[n1,:],'-o')
                axes[n1].set_ylabel(local_label[n1])
    else:
        axes[n1].plot(x0_full_analysis.inertial[n1,:],'-o')
        axes[n1].set_ylabel(inertial_label[n1])
    if n1 == 3:
        axes[n1].set_xlabel('Steps')
plt.show()

fig, axes = plt.subplots(1, 2)
step_label = ['$ x $ (m)', '$ z $ (m)', '$ \dot{x} $ (m/s)', '$ \dot{z} $ (m/s)']
eigvec_lines = ['b-', 'g-', 'r-', 'c-']
for n1 in range(2):
    if n1 == 0:
        axes[n1].plot(x0_full_analysis.step[n1,0], x0_full_analysis.step[n1+2,0], 'ko', label='Periodic Step Transition')
        axes[n1].plot(x_full.step[n1,:], x_full.step[n1+2,:], 'C1:', lw=0.5, label='Trajectory')
        axes[n1].plot(x0_full.step[n1,:], x0_full.step[n1+2,:], 'C2+', label='Step Transition')
        axes[n1].plot(x0_full.step[n1,0], x0_full.step[n1+2,0], 'C3+', label='Start')
        axes[n1].plot(x0_full.step[n1,-1], x0_full.step[n1+2,-1], 'C4+', label='End')
    axes[n1].plot(x0_full_analysis.step[n1,0], x0_full_analysis.step[n1+2, 0], 'ko')
    axes[n1].plot(x_full.step[n1,:], x_full.step[n1+2,:], 'C1:', lw=0.5)
    axes[n1].plot(x0_full.step[n1,:], x0_full.step[n1+2,:], 'C2+')
    axes[n1].plot(x0_full.step[n1,0], x0_full.step[n1+2,0], 'C3+')
    axes[n1].plot(x0_full.step[n1,-1], x0_full.step[n1+2,-1], 'C4+')
    axes[n1].set_xlabel(step_label[n1])
    axes[n1].set_ylabel(step_label[n1+2])
    for n2 in range(4):
        if n1 == 0:
            axes[n1].plot(np.array([x0_full_analysis.step[n1,0] - ((1e10)*(jacobian_eigvec[n1,n2])), x0_full_analysis.step[n1,0] + ((1e10)*(jacobian_eigvec[n1,n2]))]), np.array([x0_full_analysis.step[n1+2,0] - ((1e10)*(jacobian_eigvec[n1+2,n2])), x0_full_analysis.step[n1+2,0] + ((1e10)*(jacobian_eigvec[n1+2,n2]))]), (eigvec_lines[n2] + '-'), lw=0.8, label=('Eigenvector, lambda = ' + str(jacobian_eigval[n2])))
        axes[n1].plot(np.array([x0_full_analysis.step[n1,0] - ((1e10)*(jacobian_eigvec[n1,n2])), x0_full_analysis.step[n1,0] + ((1e10)*(jacobian_eigvec[n1,n2]))]), np.array([x0_full_analysis.step[n1+2,0] - ((1e10)*(jacobian_eigvec[n1+2,n2])), x0_full_analysis.step[n1+2,0] + ((1e10)*(jacobian_eigvec[n1+2,n2]))]), (eigvec_lines[n2] + '-'), lw=0.8)
axes[0].set_xlim(x0_full_analysis.step[0,0] - ((np.max(np.abs(x_full.step[0,:] - x0_full_analysis.step[0,0]))) + 0.1), x0_full_analysis.step[0,0] + ((np.max(np.abs(x_full.step[0,:] - x0_full_analysis.step[0,0]))) + 0.1))
axes[0].set_ylim(x0_full_analysis.step[2,0] - ((np.max(np.abs(x_full.step[2,:] - x0_full_analysis.step[2,0]))) + 0.1), x0_full_analysis.step[2,0] + ((np.max(np.abs(x_full.step[2,:] - x0_full_analysis.step[2,0]))) + 0.1))
axes[1].set_xlim(x0_full_analysis.step[1,0] - ((np.max(np.abs(x_full.step[1,:] - x0_full_analysis.step[1,0]))) + 0.01), x0_full_analysis.step[1,0] + ((np.max(np.abs(x_full.step[1,:] - x0_full_analysis.step[1,0]))) + 0.01))
axes[1].set_ylim(x0_full_analysis.step[3,0] - ((np.max(np.abs(x_full.step[3,:] - x0_full_analysis.step[3,0]))) + 0.01), x0_full_analysis.step[3,0] + ((np.max(np.abs(x_full.step[3,:] - x0_full_analysis.step[3,0]))) + 0.01))
fig.legend()
plt.show()



