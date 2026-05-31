import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

# Load Data
cop_full = np.loadtxt(os.getcwd() + "/experimental_data/hof_2008/hof_2008_fig_3b_cop.csv", delimiter=",", dtype=float)
cop_full = cop_full.T

datalen = len(cop_full[0,:])

# Display Data
plt.figure()
plt.plot(cop_full[0,:], cop_full[1,:], '-o')
plt.xlabel('Z (m)')
plt.ylabel('X (m)')
plt.show()

# Data Analysis
dt = 0.1
n_steps = 6

footpoint = np.array([7, 5, 6, 6, 7, 6])
start = 0
startleft = 0
startright = 0
leftfoot = 1

l_local_array = np.zeros((3, footpoint[0] + footpoint[2] + footpoint[4]))
r_local_array = np.zeros((3, footpoint[1] + footpoint[3] + footpoint[5]))

step_coef_x = np.zeros(n_steps)
step_coef_z = np.zeros((2,n_steps))
alphabeta_res = np.zeros((2,n_steps))
stepparam = np.zeros((2,n_steps))

showfig = 0
printstepreg = 0

for n1 in range(n_steps):
    cop_local = cop_full[:, start:(start + footpoint[n1])]

    if n1 != n_steps-1:
        stepparam[0,n1] = cop_full[1, start + footpoint[n1]] - cop_local[1,0]
        if leftfoot == 1:
            stepparam[1,n1] = cop_full[0, start + footpoint[n1]] - cop_local[0,0]
        else:
            stepparam[1,n1] = -(cop_full[0, start + footpoint[n1]] - cop_local[0,0])
    else:
        stepparam[0,n1] = stepparam[0,n1-1]

    cop_local[0,:] -= cop_local[0,0]
    cop_local[1,:] -= cop_local[1,0]
    t_local = dt*np.arange(len(cop_local[0,:]))

    if leftfoot == 1:
        l_local_array[0, startleft:(startleft + footpoint[n1])] = t_local
        l_local_array[1, startleft:(startleft + footpoint[n1])] = cop_local[1,:]
        l_local_array[2, startleft:(startleft + footpoint[n1])] = cop_local[0,:]
        startleft += footpoint[n1]
        leftfoot = 0
    else:
        cop_local[0,:] = -cop_local[0,:]
        r_local_array[0, startright:(startright + footpoint[n1])] = t_local
        r_local_array[1, startright:(startright + footpoint[n1])] = cop_local[1,:]
        r_local_array[2, startright:(startright + footpoint[n1])] = cop_local[0,:]
        startright += footpoint[n1]
        leftfoot = 1

    reg_step_x = LinearRegression(fit_intercept=False).fit(t_local.reshape((-1,1)), cop_local[1,:].reshape((-1,1)))
    step_coef_x[n1] = reg_step_x.coef_.squeeze()

    reg_step_z = LinearRegression(fit_intercept=False).fit(np.hstack((t_local.reshape((-1,1)), (t_local.reshape((-1,1)))**2)), cop_local[0,:].reshape((-1,1)))
    step_coef_z[:,n1] = reg_step_z.coef_.squeeze()

    alphabeta_res[0,n1] = (step_coef_x[n1]*(t_local[-1]))/stepparam[0,n1]
    alphabeta_res[1,n1] = step_coef_z[1,n1]/(step_coef_x[n1]**2)

    if printstepreg == 1:
        print('Step ' + str(n1+1) + ':')
        print('x coefficients: ' + str(reg_step_x.coef_))
        print('x R^2: ' + str(reg_step_x.score(t_local.reshape((-1,1)), cop_local[1,:].reshape((-1,1)))))
        print('z coefficients: ' + str(reg_step_z.coef_))
        print('z R^2: ' + str(reg_step_z.score(np.hstack((t_local.reshape((-1,1)), (t_local.reshape((-1,1)))**2)), cop_local[0,:].reshape((-1,1)))))

    start += footpoint[n1]

    if showfig == 1:
        t_reg_step = np.linspace(t_local[0], t_local[-1])
        reg_step_x = (t_reg_step)*(reg_step_x.coef_[0,0])
        reg_step_z = (t_reg_step)*(reg_step_z.coef_[0,0]) + (t_reg_step**2)*(reg_step_z.coef_[0,1])

        plt.figure()
        plt.plot(cop_local[0,:], cop_local[1,:], '-o')
        plt.plot(reg_step_z, reg_step_x, ':')
        plt.xlabel('z (m)')
        plt.ylabel('x (m)')
        plt.show()

        fig, axes = plt.subplots(1,2, sharex=True)
        axes[0].plot(dt*np.arange(len(cop_local[0,:])), cop_local[1,:],'-o')
        axes[0].plot(t_reg_step, reg_step_x,':')
        axes[0].set_ylabel('x (m)')
        axes[0].set_xlabel('Time (s)')
        axes[1].plot(dt*np.arange(len(cop_local[0,:])), cop_local[0,:],'-o')
        axes[1].plot(t_reg_step, reg_step_z,':')
        axes[1].set_ylabel('z (m)')
        axes[1].set_xlabel('Time (s)')
        plt.show()

print('Alpha Mean Value: ' + str(np.mean(alphabeta_res[0,:])))
print('Alpha Max Value: ' + str(max(alphabeta_res[0,:])))
print('Alpha Min Value: ' + str(min(alphabeta_res[0,:])))

print('Beta Mean Value: ' + str(np.mean(alphabeta_res[1,:])))
print('Beta Max Value: ' + str(max(alphabeta_res[1,:])))
print('Beta Min Value: ' + str(min(alphabeta_res[1,:])))

print('s_c Mean Value: ' + str(np.mean(stepparam[0,:-1])))
print('s_c Max Value: ' + str(max(stepparam[0,:-1])))
print('s_c Min Value: ' + str(min(stepparam[0,:-1])))

print('w_c Mean Value: ' + str(np.mean(stepparam[1,:-1])))
print('w_c Max Value: ' + str(max(stepparam[1,:-1])))
print('w_c Min Value: ' + str(min(stepparam[1,:-1])))

# Linear Regression
t_reg_test = np.linspace(0.0, 0.6)
reg_l_test = np.zeros((2,len(t_reg_test)))
reg_r_test = np.zeros((2,len(t_reg_test)))
reg_gen_test = np.zeros((2,len(t_reg_test)))

# Left Foot
reg_l_x = LinearRegression(fit_intercept=False).fit(l_local_array[0,:].reshape((-1,1)), l_local_array[1,:].reshape((-1,1)))
print('Left Foot x:')
print('Coefficients: ' + str(reg_l_x.coef_))
print('R^2: ' + str(reg_l_x.score(l_local_array[0,:].reshape((-1,1)), l_local_array[1,:].reshape((-1,1)))))

reg_l_test[0,:] = (t_reg_test)*(reg_l_x.coef_[0,0])

reg_l_z = LinearRegression(fit_intercept=False).fit(np.hstack((l_local_array[0,:].reshape((-1,1)), (l_local_array[0,:].reshape((-1,1)))**2)), l_local_array[2,:].reshape((-1,1)))
print('Left Foot z:')
print('Coefficients: ' + str(reg_l_z.coef_))
print('R^2: ' + str(reg_l_z.score(np.hstack((l_local_array[0,:].reshape((-1,1)), (l_local_array[0,:].reshape((-1,1)))**2)), l_local_array[2,:].reshape((-1,1)))))

reg_l_test[1,:] = (t_reg_test)*(reg_l_z.coef_[0,0]) + (t_reg_test**2)*(reg_l_z.coef_[0,1])

# Right Foot
reg_r_x = LinearRegression(fit_intercept=False).fit(r_local_array[0,:].reshape((-1,1)), r_local_array[1,:].reshape((-1,1)))
print('Right Foot x:')
print('Coefficients: ' + str(reg_r_x.coef_))
print('R^2: ' + str(reg_r_x.score(r_local_array[0,:].reshape((-1,1)), r_local_array[1,:].reshape((-1,1)))))

reg_r_test[0,:] = (t_reg_test)*(reg_r_x.coef_[0,0])

reg_r_z = LinearRegression(fit_intercept=False).fit(np.hstack((r_local_array[0,:].reshape((-1,1)), (r_local_array[0,:].reshape((-1,1)))**2)), r_local_array[2,:].reshape((-1,1)))
print('Right Foot z:')
print('Coefficients: ' + str(reg_r_z.coef_))
print('R^2: ' + str(reg_r_z.score(np.hstack((r_local_array[0,:].reshape((-1,1)), (r_local_array[0,:].reshape((-1,1)))**2)), r_local_array[2,:].reshape((-1,1)))))

reg_r_test[1,:] = (t_reg_test)*(reg_r_z.coef_[0,0]) + (t_reg_test**2)*(reg_r_z.coef_[0,1])

# Both Feet
foot_local_array = np.hstack((l_local_array, r_local_array))

reg_gen_x = LinearRegression(fit_intercept=False).fit(foot_local_array[0,:].reshape((-1,1)), foot_local_array[1,:].reshape((-1,1)))
print('Both Feet x:')
print('Coefficients: ' + str(reg_gen_x.coef_))
print('R^2: ' + str(reg_gen_x.score(foot_local_array[0,:].reshape((-1,1)), foot_local_array[1,:].reshape((-1,1)))))

reg_gen_test[0,:] = (t_reg_test)*(reg_gen_x.coef_[0,0])

reg_gen_z = LinearRegression(fit_intercept=False).fit(np.hstack((foot_local_array[0,:].reshape((-1,1)), (foot_local_array[0,:].reshape((-1,1)))**2)), foot_local_array[2,:].reshape((-1,1)))
print('Both Feet z:')
print('Coefficients: ' + str(reg_gen_z.coef_))
print('R^2: ' + str(reg_gen_z.score(np.hstack((foot_local_array[0,:].reshape((-1,1)), (foot_local_array[0,:].reshape((-1,1)))**2)), foot_local_array[2,:].reshape((-1,1)))))

reg_gen_test[1,:] = (t_reg_test)*(reg_gen_z.coef_[0,0]) + (t_reg_test**2)*(reg_gen_z.coef_[0,1])

# Display Results
fig, axes = plt.subplots(1,2, sharex=True)
axes[0].plot(l_local_array[0,:], l_local_array[1,:],'C0o', label='Left Foot')
axes[0].plot(r_local_array[0,:], r_local_array[1,:],'C1o', label='Right Foot')
axes[0].plot(t_reg_test, reg_l_test[0,:],'C0:', label='Left Foot Regression')
axes[0].plot(t_reg_test, reg_r_test[0,:],'C1:', label='Right Foot Regression')
axes[0].plot(t_reg_test, reg_gen_test[0,:],'C2:', label='Both Feet Regression')
axes[0].set_ylabel('x (m)')
axes[0].set_xlabel('Time (s)')
axes[1].plot(l_local_array[0,:], l_local_array[2,:],'C0o')
axes[1].plot(r_local_array[0,:], r_local_array[2,:],'C1o')
axes[1].plot(t_reg_test, reg_l_test[1,:],'C0:')
axes[1].plot(t_reg_test, reg_r_test[1,:],'C1:')
axes[1].plot(t_reg_test, reg_gen_test[1,:],'C2:')
axes[1].set_ylabel('z (m)')
axes[1].set_xlabel('Time (s)')
fig.legend()
plt.show()



