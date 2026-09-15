import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
import time

# General Functions

def model_num_linear_dyn_def(xinit, nonlin_model_dyn, n = 4, epsval = 1e-8):
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

def jacobian_num(x0, n, func, epsval = 1e-8):
    jacobian = np.zeros((n,n))

    for n1 in range(n):
        eps = np.zeros(n)
        eps[n1] = epsval
        x0_perturb = np.zeros((n,2))
        for n2 in range(2):
            if n2 == 0:
                x0_analysis = x0 + eps
            else:
                x0_analysis = x0 - eps
            x0_res = func(x0_analysis)
            x0_perturb[:,n2] = x0_res - x0
        jacobian[:,n1] = (1/2)*(1/epsval)*(x0_perturb[:,0] - x0_perturb[:,1])

    return jacobian

def jacobian_extrapolate(x0, n, func, epsval = 1e-8):
    jacobian_1 = jacobian_num(x0, n, func, epsval = (epsval/2))
    jacobian_2 = jacobian_num(x0, n, func, epsval = epsval)

    jacobian = (1/3)*((4*jacobian_1) - jacobian_2)

    return jacobian

# Walking Model Functions

def ddU_x_0(t,x):
    return 0.0

def ddU_z_0(t,x):
    return 0.0

def model_dyn_def(param, ddU_x, ddU_z):
    g = param[0]
    L = param[1]
    def model_dyn(t, x):
        xdot = np.array([
            x[2], x[3],
            (x[0]/(L**2))*( (-1/( (L**2) - ((x[0]**2) + (x[1]**2)) ))*( (x[2]**2)*((L**2) - (x[1]**2)) + 2*x[0]*x[2]*x[1]*x[3] + (x[3]**2)*((L**2) - (x[0]**2)) ) + x[1]*ddU_z(t,x) + g*np.sqrt(np.abs((L**2) - ((x[0]**2) + (x[1]**2))) )) + ddU_x(t,x)*(((x[0]**2) - (L**2))/(L**2)),
            (x[1]/(L**2))*( (-1/( (L**2) - ((x[0]**2) + (x[1]**2)) ))*( (x[2]**2)*((L**2) - (x[1]**2)) + 2*x[0]*x[2]*x[1]*x[3] + (x[3]**2)*((L**2) - (x[0]**2)) ) + x[0]*ddU_x(t,x) + g*np.sqrt(np.abs((L**2) - ((x[0]**2) + (x[1]**2))) )) + ddU_z(t,x)*(((x[1]**2) - (L**2))/(L**2))
        ])
        return xdot
    return model_dyn

def model_linear_dyn_def(param, ddU_x, ddU_z):
    g = param[0]
    L = param[1]
    def model_dyn(t, x):
        xdot = np.array([
            x[2], x[3],
            ((g/L)*(x[0]) - ddU_x(t,np.zeros(4))),
            ((g/L)*(x[1]) - ddU_z(t,np.zeros(4)))
        ])
        return xdot
    return model_dyn

def U0_array_func(n_steps, s_c, w_c, z_c, dU_x_0 = 0, dU_z_0 = 0, n = 4):
    U0_array = np.zeros((n,n_steps))
    U0_array[0,:] = np.linspace(0.0, (n_steps-1)*(s_c), n_steps)
    U0_array[1,:] = z_c + (w_c/2)*((-1)**(np.arange(n_steps) + 1))
    U0_array[2,:] = dU_x_0
    U0_array[3,:] = dU_z_0
    return U0_array

def poincare_step(x1, U_step, leftright=1):
    L = np.abs(U_step[0,-1] - U_step[0,0])
    W = np.abs(U_step[1,0] - U_step[1,-1])
    #dUx = U_step[2,-1] - U_step[2,0]
    dUx = U_step[2,0] - U_step[2,-1]
    dUz = U_step[3,-1] + U_step[3,0]
    if ((leftright == 1) and (U_step[1,-1] <= U_step[1,0])) or ((leftright == 0) and (U_step[1,-1] >= U_step[1,0])):
        x2 = np.array([-L, -W, dUx, -dUz]) + np.squeeze(np.diag([1.0, -1.0, 1.0, -1.0])@x1)
        #x2 = np.array([-L, W, dUx, -dUz]) + np.squeeze(np.diag([1.0, -1.0, 1.0, -1.0])@x1)
    else:
        #x2 = np.array([-L, W, 0.0, 0.0]) + np.squeeze(np.diag([1.0, -1.0, 1.0, -1.0])@x1)
        x2 = np.array([-L, W, dUx, -dUz]) + np.squeeze(np.diag([1.0, -1.0, 1.0, -1.0])@x1)
    return x2

def steploc_constoffset(xloc, U_init, const, leftright=1, propcont=1):
    r = np.sqrt((xloc[0]**2) + (xloc[1]**2))
    angle = np.linspace(0.0, 2*np.pi, 5001)

    omega_0 = const[0]
    b_x = const[1]
    b_z = const[2]
    k1 = const[3]
    z_c = const[4]

    if leftright == 1:
        zeta_n = (xloc[1] + U_init[1]) + ((xloc[3] + U_init[3])/omega_0)
    else:
        zeta_n = ((-xloc[1]) + U_init[1]) + (-(xloc[3] + U_init[3])/omega_0)
    xi_n = (xloc[0] + U_init[0]) + ((xloc[2] + U_init[2])/omega_0)

    U_viable = np.zeros((2,angle.shape[0]))
    if leftright == 1:
        U_viable[0,:] = np.repeat(xloc[0], angle.shape[0]) + (r*np.sin(angle)) + np.repeat(U_init[0], angle.shape[0])
        U_viable[1,:] = np.repeat(xloc[1], angle.shape[0]) + (r*np.cos(angle)) + np.repeat(U_init[1], angle.shape[0])
    else:
        U_viable[0,:] = np.repeat(xloc[0], angle.shape[0]) + (r*np.sin(angle)) + np.repeat(U_init[0], angle.shape[0])
        U_viable[1,:] = -np.repeat(xloc[1], angle.shape[0]) + (r*np.cos(angle)) + np.repeat(U_init[1], angle.shape[0])

    U_step = np.zeros(2)
    U_step[0] = xi_n - b_x
    '''
    if propcont == 1:
        k1 = 0.1
        z_c = 0.0
        if leftright == 1:
            U_step[1] = zeta_n + b_z + k1*(zeta_n - z_c)
        else:
            U_step[1] = zeta_n - b_z + k1*(zeta_n - z_c)
    else:
        if leftright == 1:
            U_step[1] = zeta_n + b_z
        else:
            U_step[1] = zeta_n - b_z
    '''
    if propcont == 1:
        U_step[1] = zeta_n + b_z + k1*(zeta_n - z_c)
    else:
        U_step[1] = zeta_n + b_z

    return U_step, U_viable

def copadj_steploc_func(xloc, U_init, U_step, T, const, copparlim, leftright=1, propcont=0):
    omega_0 = const[0]
    s_c = const[1]
    w_c = const[2]
    T_c = const[3]
    k1 = const[4]
    z_c = const[5]

    if leftright == 1:
        zeta_n = (xloc[1] + U_init[1]) + ((xloc[3] + U_init[3])/omega_0)
    else:
        zeta_n = ((-xloc[1]) + U_init[1]) + (-(xloc[3] + U_init[3])/omega_0)
    xi_n = (xloc[0] + U_init[0]) + ((xloc[2] + U_init[2])/omega_0)

    alpha = (T)*( (omega_0*(1 - np.exp(omega_0*T_c)))/((omega_0*T_c) + 1 - np.exp(omega_0*T_c)) )*( xi_n - U_step[0] + (s_c/(1 - np.exp(omega_0*T_c))) )
    if propcont == 1:
        if leftright == 1:
            #beta = (((omega_0*T_c)/(alpha*s_c))**2)*( ( w_c - (U_step[1] - zeta_n - k1*(zeta_n - z_c))*(1 + np.exp(omega_0*T_c)) )/( (omega_0*T_c) + 2 + (np.exp(omega_0*T_c)*((omega_0*T_c) - 2)) ) )
            beta = (((omega_0*T)/(alpha))**2)*( ( w_c - (U_step[1] - zeta_n - k1*(zeta_n - z_c))*(1 + np.exp(omega_0*T_c)) )/( ((omega_0*T_c)**2) + 2*omega_0*T_c - T*T_c*(omega_0**2) + (np.exp(omega_0*T_c) - 1)*((omega_0*T) - 2) ) )
        else:
            #beta = (((omega_0*T_c)/(alpha*s_c))**2)*( ( w_c + (U_step[1] - zeta_n - k1*(zeta_n - z_c))*(1 + np.exp(omega_0*T_c)) )/( (omega_0*T_c) + 2 + (np.exp(omega_0*T_c)*((omega_0*T_c) - 2)) ) )
            beta = (((omega_0*T)/(alpha))**2)*( ( w_c + (U_step[1] - zeta_n - k1*(zeta_n - z_c))*(1 + np.exp(omega_0*T_c)) )/( ((omega_0*T_c)**2) + 2*omega_0*T_c - T*T_c*(omega_0**2) + (np.exp(omega_0*T_c) - 1)*((omega_0*T) - 2) ) )
    else:
        if leftright == 1:
            #beta = (((omega_0*T_c)/(alpha*s_c))**2)*( ( w_c - (U_step[1] - zeta_n)*(1 + np.exp(omega_0*T_c)) )/( (omega_0*T_c) + 2 + (np.exp(omega_0*T_c)*((omega_0*T_c) - 2)) ) )
            beta = (((omega_0*T)/(alpha))**2)*( ( w_c - (U_step[1] - zeta_n)*(1 + np.exp(omega_0*T_c)) )/( ((omega_0*T_c)**2) + 2*omega_0*T_c - T*T_c*(omega_0**2) + (np.exp(omega_0*T_c) - 1)*((omega_0*T) - 2) ) )
        else:
            #beta = (((omega_0*T_c)/(alpha*s_c))**2)*( ( w_c + (U_step[1] - zeta_n)*(1 + np.exp(omega_0*T_c)) )/( (omega_0*T_c) + 2 + (np.exp(omega_0*T_c)*((omega_0*T_c) - 2)) ) )
            beta = (((omega_0*T)/(alpha))**2)*( ( w_c + (U_step[1] - zeta_n)*(1 + np.exp(omega_0*T_c)) )/( ((omega_0*T_c)**2) + 2*omega_0*T_c - T*T_c*(omega_0**2) + (np.exp(omega_0*T_c) - 1)*((omega_0*T) - 2) ) )
    
    if alpha > copparlim[0,0]:
        alpha = copparlim[0,0]
    elif alpha < copparlim[0,1]:
        alpha = copparlim[0,1]
    if beta > copparlim[1,0]:
        beta = copparlim[1,0]
    elif beta < copparlim[1,1]:
        beta = copparlim[1,1]

    return alpha, beta

def copadj_basin_func(x0, U0, const, copparlim, cop_params_test):
    omega_0 = const[0]
    const_T = const[1]
    b_z_init = const[2]
    w_max = const[3]
    alpha = const[4]
    beta_init = const[5]

    b1 = alpha/const_T
    c1 = -alpha*beta_init*b1
    c2 = beta_init*(b1**2)

    def dz_lim_low_func(z):
        return omega_0*((np.exp(-omega_0*const_T)*(-b_z_init + ( (c2/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (c1/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)
    def dz_lim_high_func(z):
        return omega_0*((np.exp(-omega_0*const_T)*(w_max - b_z_init + ( (c2/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (c1/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) ))) - z)

    if x0[3] + U0[3] < dz_lim_low_func(x0[1]):
        beta = copparlim[1,0]
    elif x0[3] + U0[3] > dz_lim_high_func(x0[1]):
        beta = copparlim[1,1]
    else:
        beta = beta_init
    return beta

def copadj_basin_propcont_func(x0, U0, const, copparlim):
    omega_0 = const[0]
    const_T = const[1]
    k = const[2]
    w_c = const[3]
    w_max = const[4]
    alpha = const[5]
    beta_init = const[6]
    z_c = const[7]

    def basin_propcont_check(x0, U0, beta):
        zeta_test = x0[1] + (x0[3] + U0[3])/omega_0

        b1 = alpha/const_T
        c1 = -alpha*beta*b1
        c2 = beta*(b1**2)

        b_z = ( (1)/(1 + np.exp(omega_0*const_T)) )*( (w_c) - (c1)*( (const_T) + ((1 - np.exp(omega_0*const_T))/omega_0) ) - (c2)*( (const_T**2) + ((2*const_T)/(omega_0)) + ((2*(1 - np.exp(omega_0*const_T)))/(omega_0**2)) ) )
        quadcop_term = ( (c2/(omega_0**2))*(2*np.exp(omega_0*const_T) - (omega_0*const_T)**2 - (2*omega_0*const_T) - 2) + (c1/omega_0)*(np.exp(omega_0*const_T) - (omega_0*const_T) - 1) )
    
        basin_boundary_ana = np.zeros(4)
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
    
            bound_1 = -( ( zeta_test - propcont_boundary_coeff[1,n1] )/propcont_boundary_coeff[0,n1] ) + zeta_test
            bound_2 = -( ( zeta_test - propcont_boundary_coeff[2,n1] )/propcont_boundary_coeff[0,n1] ) + zeta_test
    
            basin_top = np.maximum(bound_1, bound_2)
            basin_bottom = np.minimum(bound_1, bound_2)
    
            if basin_top < basin_top_check:
                basin_boundary_ana[2] = basin_top
                basin_top_check = basin_top.copy()
            if basin_bottom > basin_bottom_check:
                basin_boundary_ana[3] = basin_bottom
                basin_bottom_check = basin_bottom.copy()
    
        bound_1_0 = -( ( zeta_test - propcont_boundary_coeff[1,0] )/propcont_boundary_coeff[0,0] ) + zeta_test
        bound_2_0 = -( ( zeta_test - propcont_boundary_coeff[2,0] )/propcont_boundary_coeff[0,0] ) + zeta_test
        basin_top_0 = np.maximum(bound_1_0, bound_2_0)
        basin_bottom_0 = np.minimum(bound_1_0, bound_2_0)
        basin_boundary_ana[0] = basin_top_0
        basin_boundary_ana[1] = basin_bottom_0

        z_c_err = zeta_test - z_c

        if (z_c_err < basin_boundary_ana[0] and z_c_err < basin_boundary_ana[2]) and (z_c_err > basin_boundary_ana[1] and z_c_err > basin_boundary_ana[3]):
            basin_check = True
        else:
            basin_check = False

        return basin_check

    if basin_propcont_check(x0, U0, beta_init):
        beta = beta_init
    elif basin_propcont_check(x0, U0, copparlim[1,1]):
        beta = copparlim[1,1]
    elif basin_propcont_check(x0, U0, copparlim[1,0]):
        beta = copparlim[1,0]
    else:
        beta = copparlim[1,0]
    return beta

def walkingmodel_ana_soln(t_eval, xinit, omega_0, dot_U = False, cop_params = None):
    xsoln = np.zeros((4,len(t_eval)))
    if dot_U == False or cop_params == None:
        xsoln[0,:] = xinit[0]*np.cosh(omega_0*t_eval) + xinit[2]*(np.sinh(omega_0*t_eval)/omega_0)
        xsoln[1,:] = xinit[1]*np.cosh(omega_0*t_eval) + xinit[3]*(np.sinh(omega_0*t_eval)/omega_0)
        xsoln[2,:] = xinit[0]*omega_0*np.sinh(omega_0*t_eval) + xinit[2]*np.cosh(omega_0*t_eval)
        xsoln[3,:] = xinit[1]*omega_0*np.sinh(omega_0*t_eval) + xinit[3]*np.cosh(omega_0*t_eval)
    else:
        xsoln[0,:] = xinit[0]*np.cosh(omega_0*t_eval) + xinit[2]*(np.sinh(omega_0*t_eval)/omega_0)
        xsoln[1,:] = xinit[1]*np.cosh(omega_0*t_eval) + xinit[3]*(np.sinh(omega_0*t_eval)/omega_0) - cop_params['c2']*(t_eval**2) - cop_params['c1']*( (np.sinh(omega_0*t_eval) - omega_0*t_eval)/omega_0 )
        xsoln[2,:] = xinit[0]*omega_0*np.sinh(omega_0*t_eval) + xinit[2]*np.cosh(omega_0*t_eval)
        xsoln[3,:] = xinit[1]*omega_0*np.sinh(omega_0*t_eval) + xinit[3]*np.cosh(omega_0*t_eval) - cop_params['c2']*( (2*np.sinh(omega_0*t_eval) - 2*omega_0*t_eval)/omega_0 ) - cop_params['c1']*(np.cosh(omega_0*t_eval) - 1)
    return xsoln

def step_dyn(model_dyn, x0, dt, U0_array, dot_U = False, copadj = False, cop_params = None, T_c = 0.5, U0_noise = 0.0, sys_par = (9.81, 1.09, 0.1, 0.0, 0.8*1.09), copparlim = np.array([[np.inf, -np.inf],[np.inf, -np.inf]]), const_T=0.0, leftright = 1, constoff=1, propcont=0, ana_soln=0, stepprint=1):
    n = 4
    g = sys_par[0]
    l = sys_par[1]

    k1 = sys_par[2]
    z_c = sys_par[3]

    w_max = sys_par[4]

    if dot_U == False:
        def dot_U_func(t,x):
            dot_U = np.array([x[2], x[3], 0.0, 0.0])
            return dot_U
    elif dot_U == 'quad':
        def dot_U_func(t,x):
            dot_U = np.array([x[2], x[3], 0.0, 2*cop_params['c2']])
            return dot_U
    else:
        dot_U_func = dot_U

    steps_len = U0_array.shape[1]

    U0_res = np.zeros(np.shape(U0_array))
    U0_res[:,0] = U0_array[:,0]
    U0_res[2:,:] = U0_array[2:,:]
    U0_viable = np.zeros((steps_len, 2, 5001))
    U0_noise_res = np.zeros((2,len(U0_array[0,:])))

    x0_array = np.zeros((n, steps_len))
    x0_array_t = np.zeros(steps_len)
    x0_array[:,0] = x0

    deltat_cont_res = np.zeros(len(U0_array[0,:]) - 1)

    if isinstance(const_T, float):
        const_T_array = const_T*np.ones(steps_len)
    else:
        const_T_array = const_T[:steps_len]

    T_array = np.zeros(steps_len)
    cop_params_res = np.zeros((3,steps_len))

    s_c = U0_array[0,1] - U0_array[0,0]
    w_c = np.abs(U0_array[1,1] - U0_array[1,0])
    omega_0 = np.sqrt(g/l)
    if dot_U == False:
        b_x = (s_c)/(np.exp(omega_0*T_c) - 1)
        b_z_1 = (w_c)/(np.exp(omega_0*T_c) + 1)
        b_z_2 = -b_z_1
    else:
        alpha = cop_params['b1']*const_T
        beta = cop_params['c2']/(cop_params['b1']**2)

        b1 = cop_params['b1']
        c1 = cop_params['c1']
        c2 = cop_params['c2']
        b_x = -( (s_c)/(1 - np.exp(omega_0*T_c)) - (b1)*( (1/omega_0) + ((T_c)/(1 - np.exp(omega_0*T_c))) ) )
        b_z_1 = ( (1)/(1 + np.exp(omega_0*T_c)) )*( (w_c) - (c1)*( (T_c) + ((1 - np.exp(omega_0*T_c))/omega_0) ) - c2*( (T_c**2) + ((2*T_c)/(omega_0)) + ((2*(1 - np.exp(omega_0*T_c)))/(omega_0**2)) ) )
        b_z_2 = -b_z_1
        cop_params_res[0,:] = b1
        cop_params_res[1,:] = c1
        cop_params_res[2,:] = c2

        b_z_init = b_z_1.copy()
        beta_init = beta.copy()

    for n1 in range(steps_len):
        if np.linalg.norm(x0_array[:2, n1]) > l:
            if stepprint == 1:
                print('x0 exceeds leg length')
            #break

        if n1 == 0:
            s = s_c
            xi_n = ( ((x0_array[0, n1] + U0_res[0, n1]) + ((x0_array[2, n1] + U0_res[2, n1])/omega_0)) )
        else:
            s = U0_res[0, n1] - U0_res[0, n1-1]
            xi_n = ( ((x_step[0, -1] + U0_res[0, n1]) + ((x_step[2, -1] + U0_res[2, n1])/omega_0)) )

        if isinstance(const_T, float) and const_T == 0:
            T = (1/omega_0)*(np.log((s/(xi_n - U0_res[0, n1])) + 1))
        else:
            #T = const_T
            T = const_T_array[n1]

        T_array[n1] = T
        t_span = (0.0, T)
        t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

        if t_eval[-1] > T:
            t_eval = t_eval[0:-1]

        if stepprint == 1:
            print('Step: ' + str(n1 + 1))
            print('Period (s): ' + str(T))

        if copadj == 'basin':
            if propcont == 1:
                if leftright == 1:
                    const_array_step = np.array([omega_0, const_T, k1, w_c, w_max, alpha, beta_init, z_c - U0_res[1, n1]])
                else:
                    const_array_step = np.array([omega_0, const_T, k1, w_c, w_max, alpha, beta_init, -(z_c - U0_res[1, n1])])
                beta = copadj_basin_propcont_func(x0_array[:, n1], U0_res[:, n1], const_array_step, copparlim)
            else:
                const_array_step = np.array([omega_0, const_T, b_z_init, w_max, alpha, beta_init])
                beta = copadj_basin_func(x0_array[:, n1], U0_res[:, n1], const_array_step, copparlim, cop_params)
            cop_params['c1'] = -alpha*beta*cop_params['b1']
            cop_params['c2'] = beta*(cop_params['b1']**2)
            x0_array[3, n1] = (x0_array[3, n1] + U0_res[3, n1]) - cop_params['c1']
            U0_res[3, n1] = cop_params['c1']
            cop_params_res[1, n1] = cop_params['c1']
            cop_params_res[2, n1] = cop_params['c2']
            b_z_1 = ( (1)/(1 + np.exp(omega_0*T_c)) )*( (w_c) - (cop_params['c1'])*( (T_c) + ((1 - np.exp(omega_0*T_c))/omega_0) ) - cop_params['c2']*( (T_c**2) + ((2*T_c)/(omega_0)) + ((2*(1 - np.exp(omega_0*T_c)))/(omega_0**2)) ) )
            b_z_2 = -b_z_1          

        if ana_soln == 1:
            x_step = walkingmodel_ana_soln(t_eval, x0_array[:, n1], omega_0, dot_U = dot_U_func, cop_params = cop_params)
            t_step = t_eval
        else:
            soln_x = solve_ivp(model_dyn, t_span, x0_array[:, n1], t_eval=t_eval, rtol=1e-9, atol=1e-12)
            x_step = soln_x.y
            t_step = soln_x.t

        soln_U = solve_ivp(dot_U_func, (t_step[0], t_step[-1]), np.array([0.0, 0.0, U0_res[2, n1], U0_res[3, n1]]), t_eval=t_step)
        U_step = soln_U.y

        if n1 != steps_len-1:
            t_con_1 = time.time()

            if leftright == 1:
                const_array = np.array([omega_0, b_x, b_z_1, k1, z_c])
            else:
                const_array = np.array([omega_0, b_x, b_z_2, k1, z_c])
            
            if constoff == 1:
                [U0_res[:2, n1+1], U0_viable[n1, :, :]] = steploc_constoffset(x_step[:,-1], np.array([U0_res[0, n1], U0_res[1, n1], 0.0, 0.0]) + U_step[:,-1], const_array, leftright=leftright, propcont=propcont)
            else:
                U0_res[:2, n1+1] = U0_array[:2, n1+1]

            U0_noise_res[:, n1+1] = U0_noise*(2*np.random.rand(2) - np.ones(2))
            U0_res[:2, n1+1] = U0_res[:2, n1+1] + U0_noise_res[:, n1+1]

            if copadj == 'steploc':
                const_array_step = np.array([omega_0, s_c, w_c, T_c, k1, z_c])
                #const_array_step = np.array([omega_0, U0_res[0, n1+1] - U0_res[0, n1], np.abs(U0_res[1, n1+1] - U0_res[1, n1]), T_c, k1, z_c])
                #[alpha, beta] = copadj_steploc_func(x_step[:,-1], np.array([U0_res[0, n1], U0_res[1, n1], 0.0, 0.0]) + U_step[:,-1], U0_array[:, n1+1], T, np.array([omega_0, s_c, w_c, T_c]), copparlim, leftright=leftright, propcont=propcont)
                #[alpha, beta] = copadj_steploc_func(x_step[:,-1], np.array([U0_res[0, n1], U0_res[1, n1], 0.0, 0.0]) + U_step[:,-1], U0_res[:, n1+1], T, np.array([omega_0, s_c, w_c, T_c]), copparlim, leftright=leftright, propcont=propcont)
                #[alpha, beta] = copadj_steploc_func(x_step[:,-1], np.array([U0_res[0, n1], U0_res[1, n1], 0.0, 0.0]) + U_step[:,-1], U0_res[:, n1+1], T, np.array([omega_0, U0_res[0, n1+1] - U0_res[0, n1], np.abs(U0_res[1, n1+1] - U0_res[1, n1]), T_c]), copparlim, leftright=leftright, propcont=propcont)
                [alpha, beta] = copadj_steploc_func(x_step[:,-1], np.array([U0_res[0, n1], U0_res[1, n1], 0.0, 0.0]) + U_step[:,-1], U0_res[:, n1+1], T, const_array_step, copparlim, leftright=leftright, propcont=propcont)
                cop_params['b1'] = (alpha)/T
                cop_params['c1'] = -alpha*beta*cop_params['b1']
                cop_params['c2'] = beta*(cop_params['b1']**2)
                #U0_res[:, n1+1] = U0_array[:, n1+1]
                U0_res[2, n1+1] = cop_params['b1']
                U0_res[3, n1+1] = cop_params['c1']
                cop_params_res[0, n1+1] = cop_params['b1']
                cop_params_res[1, n1+1] = cop_params['c1']
                cop_params_res[2, n1+1] = cop_params['c2']

            x0_array[:, n1+1] = poincare_step(x_step[:, -1], np.concatenate(((np.array([U0_res[0, n1], U0_res[1, n1], 0.0, 0.0]) + U_step[:,-1]).reshape(-1,1), U0_res[:, n1+1].reshape(-1,1)), axis=1), leftright=leftright)
            x0_array_t[n1+1] = x0_array_t[n1] + t_step[-1]

            t_con_2 = time.time()
            deltat_cont_res[n1] = t_con_2 - t_con_1

        U0_res_array = np.repeat(np.reshape(U0_res[:, n1], (U0_res[:, n1].shape[0],1)), t_step.shape[0], axis=1)
        U0_res_array[2:,:] = 0.0

        if leftright == 1:
            if n1 == 0:
                x_res_inertial = U0_res_array + x_step + U_step
                x_res = x_step
                x_step_full = x_step

                U_inertial = U0_res_array + U_step
                U_step_full = U_step
            else:
                x_step_inertial = U0_res_array + x_step + U_step
                x_res_inertial = np.concatenate((x_res_inertial[:,:-1], x_step_inertial), axis=1)
                x_res = np.concatenate((x_res[:,:-1], x_step), axis=1)
                x_step_full = np.concatenate((x_step_full[:,:-1], x_step), axis=1)

                U_step_inertial = U0_res_array + U_step
                U_inertial = np.concatenate((U_inertial[:,:-1], U_step_inertial), axis=1)
                U_step_full = np.concatenate((U_step_full[:,:-1], U_step), axis=1)
            leftright = 0
        else:
            if n1 == 0:
                x_res_inertial = U0_res_array + (np.diag([1.0, -1.0, 1.0, -1.0])@(x_step + U_step))
                x_res = (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)
                x_step_full = x_step

                U_inertial = U0_res_array + (np.diag([1.0, -1.0, 1.0, -1.0])@U_step)
                U_step_full = U_step
            else:
                x_step_inertial = U0_res_array + (np.diag([1.0, -1.0, 1.0, -1.0])@(x_step + U_step))
                x_res_inertial = np.concatenate((x_res_inertial[:,:-1], x_step_inertial), axis=1)
                x_res = np.concatenate((x_res[:,:-1], (np.diag([1.0, -1.0, 1.0, -1.0])@x_step)), axis=1)
                x_step_full = np.concatenate((x_step_full[:,:-1], (x_step)), axis=1)

                U_step_inertial = U0_res_array + (np.diag([1.0, -1.0, 1.0, -1.0])@U_step)
                U_inertial = np.concatenate((U_inertial[:,:-1], U_step_inertial), axis=1)
                U_step_full = np.concatenate((U_step_full[:,:-1], U_step), axis=1)
            leftright = 1
    
    x_res_3D = np.zeros((6, x_res_inertial.shape[1]))
    x_res_3D[0,:] = x_res_inertial[0,:]
    x_res_3D[1,:] = np.sqrt(np.abs((l**2) - ((x_res[0,:]**2) + (x_res[1,:]**2))))
    x_res_3D[2,:] = x_res_inertial[1,:]
    x_res_3D[3,:] = x_res_inertial[2,:]
    x_res_3D[4,:] = -( (x_res[0,:]*x_res[2,:]) + (x_res[1,:]*x_res[3,:]) )/x_res_3D[1,:]
    x_res_3D[5,:] = x_res_inertial[3,:]

    x0_array_inertial = np.zeros(x0_array.shape)

    even = 1
    for n1 in range(steps_len):
        if even == 1:
            x0_array_inertial[:,n1] = U0_res[:,n1] + x0_array[:,n1]
            even = 0
        else:
            x0_array_inertial[:,n1] = (np.diag([1.0, 1.0, 1.0, -1.0])@U0_res[:,n1]) + (np.diag([1.0, -1.0, 1.0, -1.0])@x0_array[:,n1])
            even = 1

    x0_inertial_3D = np.zeros((6, steps_len))
    x0_inertial_3D[0,:] = x0_array_inertial[0,:]
    x0_inertial_3D[1,:] = np.sqrt(np.abs((l**2) - (((x0_array_inertial[0,:] - U0_res[0,:])**2) + ((x0_array_inertial[1,:] - U0_res[1,:])**2))))
    x0_inertial_3D[2,:] = x0_array_inertial[1,:]
    x0_inertial_3D[3,:] = x0_array_inertial[2,:]
    x0_inertial_3D[4,:] = -( (x0_array[0,:]*x0_array[2,:]) + (x0_array[1,:]*x0_array[3,:]) )/x0_inertial_3D[1,:]
    x0_inertial_3D[5,:] = x0_array_inertial[3,:]

    x_full = Bunch(t = np.round(dt*np.arange(x_res_inertial.shape[1]), 2), step = x_step_full, inertial = x_res_inertial, inertial_3D = x_res_3D)
    x0_full = Bunch(t = x0_array_t, step = x0_array, inertial = x0_array_inertial, inertial_3D = x0_inertial_3D)
    U0_full = Bunch(res = U0_res, viable = U0_viable, step = U_step_full, inertial = U_inertial, params_res = cop_params_res, noise_res = U0_noise_res, deltat = deltat_cont_res)

    return x_full, x0_full, U0_full, T_array

def step_analysis_func(U0_array):
    steps_len = U0_array.shape[1]
    step_ana = np.zeros((2,steps_len-1))

    for n1 in range(steps_len-1):
        step_ana[0,n1] = np.abs(U0_array[0,n1+1] - U0_array[0,n1])
        step_ana[1,n1] = np.abs(U0_array[1,n1+1] - U0_array[1,n1])

    return step_ana

def x0_fp_func(x0, simulate_twostep):
    constoff_fp_func = lambda x: x - simulate_twostep(x)
    constoff_fp_full = optimize.root(constoff_fp_func, x0)
    x0 = constoff_fp_full.x
    return x0

# Plotting Functions

def plotfunc_state_3D(dt, omega_0, x_res_inertial, x_res_3D, x0_array_t, x0_array_3D, U0_array, U0_res):
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

    fig, axes = plt.subplots(3, sharex=True)
    axes[0].plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[0,:])
    axes[0].plot(dt*np.arange(x_res_3D.shape[1]), xcom[0,:], ':')
    axes[0].plot(x0_array_t, x0_array_3D[0,:], 'co')
    axes[0].plot(U0_res_lin_t, U0_array_lin[0,:], 'g-')
    axes[0].plot(U0_res_lin_t, U0_res_lin[0,:], 'r-')
    axes[0].set_ylabel('x (m)')
    axes[1].plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[1,:])
    axes[1].plot(x0_array_t, x0_array_3D[1,:], 'co')
    axes[1].set_ylabel('y (m)')
    axes[2].plot(dt*np.arange(x_res_3D.shape[1]), x_res_3D[2,:], label='Trajectory')
    axes[2].plot(dt*np.arange(x_res_3D.shape[1]), xcom[1,:], ':', label='XcoM')
    axes[2].plot(x0_array_t, x0_array_3D[2,:], 'co', label='Step Transition')
    axes[2].plot(U0_res_lin_t, U0_array_lin[1,:], 'g-', label='U_0 Target')
    axes[2].plot(U0_res_lin_t, U0_res_lin[1,:], 'r-', label='U_0 Result')
    axes[2].set_xlabel('t (s)')
    axes[2].set_ylabel('z (m)')
    fig.legend()
    plt.show()

def plotfunc_state_full(dt, omega_0, x_full, x0_full, U0_full, U0_array, xzval_inertial=0, xzval_step=1):
    x_res_inertial = x_full.inertial
    x_res_3D = x_full.inertial_3D
    x0_array_t = x0_full.t
    x0_array = x0_full.inertial
    U0_res = U0_full.res
    U_inertial = U0_full.inertial

    xcom = np.zeros((2, x_res_inertial.shape[1]))
    xcom[0,:] = x_res_inertial[0,:] + (x_res_inertial[2,:]/omega_0)
    xcom[1,:] = x_res_inertial[1,:] + (x_res_inertial[3,:]/omega_0)

    U0_res_lin_t = np.zeros(2*len(x0_array_t))
    U0_res_lin = np.zeros((2, 2*len(x0_array_t)))
    U0_array_lin = np.zeros((2, 2*len(x0_array_t)))
    for n1 in np.arange(len(x0_array_t)):
        if n1 == (len(x0_array_t) - 1):
            U0_res_lin_t[2*n1] = x0_array_t[n1]
            U0_res_lin_t[(2*n1)+1] = dt*(np.arange(x_res_inertial.shape[1])[-1])
        else:
            U0_res_lin_t[(2*n1):((2*n1)+2)] = x0_array_t[n1:n1+2]
        U0_res_lin[:, (2*n1):((2*n1)+2)] = np.repeat(U0_res[:2, n1].reshape(2,1), 2, axis=1)
        U0_array_lin[:, (2*n1):((2*n1)+2)] = np.repeat(U0_array[:2, n1].reshape(2,1), 2, axis=1)

    U0_res_lin_interp = np.zeros((2, x_res_inertial.shape[1]))
    U0_res_lin_interp[0,:] = np.interp(dt*np.arange(x_res_inertial.shape[1]), U0_res_lin_t, U0_res_lin[0,:])
    U0_res_lin_interp[1,:] = np.interp(dt*np.arange(x_res_inertial.shape[1]), U0_res_lin_t, U0_res_lin[1,:])

    U0_array_lin_interp = np.zeros((2, x_res_inertial.shape[1]))
    U0_array_lin_interp[0,:] = np.interp(dt*np.arange(x_res_inertial.shape[1]), U0_res_lin_t, U0_array_lin[0,:])
    U0_array_lin_interp[1,:] = np.interp(dt*np.arange(x_res_inertial.shape[1]), U0_res_lin_t, U0_array_lin[1,:])

    fig, axes = plt.subplots(4, sharex=True)
    if xzval_inertial == 1:
        axes[0].plot(dt*np.arange(x_res_inertial.shape[1]), x_res_inertial[0,:])
        axes[0].plot(dt*np.arange(x_res_inertial.shape[1]), xcom[0,:], ':')
        axes[0].plot(x0_array_t, x0_array[0,:], 'co')
        #axes[0].plot(U0_res_lin_t, U0_array_lin[0,:], 'g-')
        #axes[0].plot(U0_res_lin_t, U0_res_lin[0,:], 'r-')
        axes[0].plot(dt*np.arange(x_res_inertial.shape[1]), U_inertial[0,:], 'r-')
        axes[1].plot(dt*np.arange(x_res_inertial.shape[1]), x_res_inertial[1,:], label='Trajectory')
        axes[1].plot(dt*np.arange(x_res_inertial.shape[1]), xcom[1,:], ':', label='XcoM')
        axes[1].plot(x0_array_t, x0_array[1,:], 'co', label='Step Transition')
        #axes[1].plot(U0_res_lin_t, U0_array_lin[1,:], 'g-', label='U_0 Target')
        #axes[1].plot(U0_res_lin_t, U0_res_lin[1,:], 'r-', label='U_0 Result')
        axes[1].plot(dt*np.arange(x_res_inertial.shape[1]), U_inertial[1,:], 'r-', label='U Result')
        axes[2].plot(dt*np.arange(x_res_inertial.shape[1]), x_res_inertial[2,:])
        axes[2].plot(x0_array_t, x0_array[2,:], 'co')
        axes[3].plot(dt*np.arange(x_res_inertial.shape[1]), x_res_inertial[3,:])
        axes[3].plot(x0_array_t, x0_array[3,:], 'co')
        axes[0].set_ylabel('$ X $ (m)')
        axes[1].set_ylabel('$ Z $ (m)')
        axes[2].set_ylabel('$ \dot{X} $ (m/s)')
        axes[3].set_xlabel('t (s)')
        axes[3].set_ylabel('$ \dot{Z} $ (m/s)')
    else:
        if xzval_step == 1:
            axes[0].plot(dt*np.arange(x_res_inertial.shape[1]), x_full.step[0,:])
            axes[0].plot(x0_array_t, x0_full.step[0,:], 'co')
            axes[1].plot(dt*np.arange(x_res_inertial.shape[1]), x_full.step[1,:], label='Trajectory')
            axes[1].plot(x0_array_t, x0_full.step[1,:], 'co', label='Step Transition')
            axes[2].plot(dt*np.arange(x_res_inertial.shape[1]), x_full.step[2,:])
            axes[2].plot(x0_array_t, x0_full.step[2,:], 'co')
            axes[3].plot(dt*np.arange(x_res_inertial.shape[1]), x_full.step[3,:])
            axes[3].plot(x0_array_t, x0_full.step[3,:], 'co')
            axes[0].set_ylabel('$ x $ (m)')
            axes[1].set_ylabel('$ z $ (m)')
            axes[2].set_ylabel('$ \dot{x} $ (m/s)')
            axes[3].set_xlabel('t (s)')
            axes[3].set_ylabel('$ \dot{z} $ (m/s)')
        else:
            axes[0].plot(dt*np.arange(x_res_inertial.shape[1]), x_res_inertial[0,:] - U0_res_lin_interp[0,:])
            axes[0].plot(dt*np.arange(x_res_inertial.shape[1]), xcom[0,:] - U0_res_lin_interp[0,:], ':')
            axes[0].plot(x0_array_t, x0_array[0,:] - U0_res[0,:], 'co')
            axes[1].plot(dt*np.arange(x_res_inertial.shape[1]), x_res_inertial[1,:] - U0_res_lin_interp[1,:], label='Trajectory')
            axes[1].plot(dt*np.arange(x_res_inertial.shape[1]), xcom[1,:] - U0_res_lin_interp[1,:], ':', label='XcoM')
            axes[1].plot(x0_array_t, x0_array[1,:] - U0_res[1,:], 'co', label='Step Transition')
            axes[2].plot(dt*np.arange(x_res_inertial.shape[1]), x_res_inertial[2,:])
            axes[2].plot(x0_array_t, x0_array[2,:], 'co')
            axes[3].plot(dt*np.arange(x_res_inertial.shape[1]), x_res_inertial[3,:])
            axes[3].plot(x0_array_t, x0_array[3,:], 'co')
            axes[0].set_ylabel('$ X - u_{x} $ (m)')
            axes[1].set_ylabel('$ Z - u_{z} $ (m)')
            axes[2].set_ylabel('$ \dot{X} $ (m/s)')
            axes[3].set_xlabel('t (s)')
            axes[3].set_ylabel('$ \dot{Z} $ (m/s)')
    fig.legend()
    plt.show()



