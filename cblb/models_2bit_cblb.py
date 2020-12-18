import numpy as np
from models import *

def MUX_2_1_model(state, T, params):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X = params
    params_yes = gamma_x, n_y, theta_x, delta_x, rho_x
    params_not = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x

    # print(state)
    I0, I1, S0 = state[:3]
    I0_out, I1_out = state[3:5]
    L_I0_I0, L_I1_S0, L_I1_I1, L_I0, L_I1 = state[5:10]
    N_I0_S0, N_I0_I0, N_I1_S0, N_I1_I1, N_I0, N_I1 = state[10:16]
    out = state[16]

    """
    I0
    """
    dI0_out = 0

    # yes S0: I0_S0
    state_yes_I0_S0 = I0_out, S0, N_I0_S0
    dI0_out += yes_cell_wrapper(state_yes_I0_S0, params_yes)
    dN_I0_S0 = population(N_I0_S0, r_X) 

    # not I0: I0_I0
    state_not_I0_I0 = L_I0_I0, I0_out, I0, N_I0_I0
    dL_I0_I0, dd = not_cell_wrapper(state_not_I0_I0, params_not)
    dI0_out += dd
    dN_I0_I0 = population(N_I0_I0, r_X)

    """
    I1
    """
    dI1_out = 0

    # not S0: I1_S0
    state_not_I1_S0 = L_I1_S0, I1_out, S0, N_I1_S0
    dL_I1_S0, dd = not_cell_wrapper(state_not_I1_S0, params_not)    
    dI1_out += dd
    dN_I1_S0 = population(N_I1_S0, r_X)

    # not I1: I1_I1
    state_not_I1_I1 = L_I1_I1, I1_out, I1, N_I1_I1
    dL_I1_I1, dd = not_cell_wrapper(state_not_I1_I1, params_not)
    dI1_out += dd
    dN_I1_I1 = population(N_I1_I1, r_X)

    """
    out
    """
    dout = 0

    # not I0: I0
    state_not_I0 = L_I0, out, I0_out, N_I0
    dL_I0, dd = not_cell_wrapper(state_not_I0, params_not)
    dout += dd
    dN_I0 = population(N_I0, r_X)

    # not I1: I1
    state_not_I1 = L_I1, out, I1_out, N_I1
    dL_I1, dd = not_cell_wrapper(state_not_I1, params_not)
    dout += dd
    dN_I1 = population(N_I1, r_X)

    dI0, dI1, dS0 = 0, 0, 0

    dstate = np.array([dI0, dI1, dS0, dI0_out, dI1_out,
              dL_I0_I0, dL_I1_S0, dL_I1_I1, dL_I0, dL_I1,
              dN_I0_S0, dN_I0_I0, dN_I1_S0, dN_I1_I1,
              dN_I0, dN_I1, dout])

    return dstate

def CLB_model_MUX_2_1(state, T, params):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b = params

    """
    latches
    """
    #########
    # params
    # set params for symmetric toggle switch topology
    gamma_L_Y, theta_L_Y = gamma_L_X, theta_L_X
    n_x, m_y = n_y, m_x
    eta_y, omega_y = eta_x, omega_x
 
    params_toggle =  [delta_L, gamma_L_X, gamma_L_Y, n_x, n_y, theta_L_X, theta_L_Y, eta_x, eta_y, omega_x, omega_y, m_x, m_y, delta_x, delta_y, rho_x, rho_y, r_X, r_Y]

    # degradation rates for induction of switches are specific for each toggle switch    
    params_toggle_I0 =  params_toggle.copy()
    params_toggle_I0[-4:-2] = rho_I0_a, rho_I0_b
    params_toggle_I1 =  params_toggle.copy()
    params_toggle_I1[-4:-2] = rho_I1_a, rho_I1_b 

    #########
    # states
    
    # latch I0
    I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b = state[:6]
    state_toggle_IO = I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b

    # latch I1
    I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b = state[6:12]
    state_toggle_I1 = I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b

    #########
    # models
    dstate_toggle_IO = toggle_model(state_toggle_IO, T, params_toggle_I0)
    dstate_toggle_I1 = toggle_model(state_toggle_I1, T, params_toggle_I1)

    dstate_toggles = np.append(dstate_toggle_IO, dstate_toggle_I1, axis=0)
    """
    mux 2/1
    """
    #########
    # params
    params_mux = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X

    #########
    # state
    I0, I1 = I0_a, I1_a
    state_mux = np.append([I0, I1], state[12:], axis=0)

    ########
    # model
    dstate_mux = MUX_2_1_model(state_mux, T, params_mux)
    dstate_mux = dstate_mux[2:] # ignore dI0, dI1

    """
    return
    """
    dstate = np.append(dstate_toggles, dstate_mux, axis=0)

    return dstate

def CLB_model_MUX_2_1_ODE(T, state, params):
    return CLB_model_MUX_2_1(state, T, params)

def CLB_2_generate_stoichiometry():
    N_toggle_IO = toggle_generate_stoichiometry()
    N_toggle_I1 = toggle_generate_stoichiometry()


    N_mux = MUX_2_1_generate_stoichiometry()
    # skip first four rows (I0, I1)
    N_mux = N_mux[2:,:]

    return merge_N(merge_N(N_toggle_IO, N_toggle_I1), N_mux)

def MUX_2_1_generate_stoichiometry():

    """
    I0, I1, I2, I3, S0, S1 = state[:6]
    I0_out, I1_out, I2_out, I3_out = state[6:10]
    L_I0_I0, L_I1_S0, L_I1_I1, L_I2_S1, L_I2_I2, L_I3_S0, L_I3_S1, L_I3_I3, L_I0, L_I1, L_I2, L_I3 = state[10:22]
    N_I0_S0, N_I0_S1, N_I0_I0, N_I1_S0, N_I1_S1, N_I1_I1, N_I2_S0, N_I2_S1, N_I2_I2, N_I3_S0, N_I3_S1, N_I3_I3, N_I0, N_I1, N_I2, N_I3 = state[22:38]
    out = state[38]
    """
    
    I0_out, I1_out = range(3, 5)
    L_I0_I0, L_I1_S0, L_I1_I1, L_I0, L_I1 = range(5, 10)
    out = 16

    #
    # x axis ... species 
    # y axis ... reactions
    #
    N = np.zeros((17, 28))

    """
    # yes S0: I0_S0
    """

    r = 0
    # reaction 0
    # 0 --> I0_out     
    N[I0_out, r] = 1
    
    
    r += 1        
    # reaction 1
    # I0_out --> 0
    N[I0_out, r] = -1
    
    r += 1
    # reaction 2
    # I0_out --> 0
    N[I0_out, r] = -1

    """
    # not I0: I0_I0 
    """

    r += 1
    # reaction 3
    # 0 --> L_I0_I0
    N[L_I0_I0, r] = 1

    r += 1
    # reaction 4
    # L_I0_I0 --> 0
    N[L_I0_I0, r] = -1
    
    r += 1
    # reaction 5
    # 0 --> I0_out     
    N[I0_out, r] = 1

    r += 1
    # reaction 6
    # I0_out --> 0
    N[I0_out, r] = -1

    r += 1
    # reaction 7
    # I0_out --> 0
    N[I0_out, r] = -1

    """
    # not S0: I1_S0
    """

    r += 1
    # reaction 8
    # 0 --> L_I1_S0
    N[L_I1_S0, r] = 1

    r += 1
    # reaction 9
    # L_I1_S0 --> 0
    N[L_I1_S0, r] = -1

    r += 1
    # reaction 10
    # 0 --> I1_out
    N[I1_out, r] = 1
    
    r += 1
    # reaction 11
    # I1_out --> 0
    N[I1_out, r] = -1

    r += 1
    # reaction 12
    # I1_out --> 0
    N[I1_out, r] = -1

    """
    # not I1: I1_I1
    """

    r += 1
    # reaction 13
    # 0 --> L_I1_I1
    N[L_I1_I1, r] = 1

    r += 1
    # reaction 14
    # L_I1_I1 --> 0
    N[L_I1_I1, r] = -1

    r += 1
    # reaction 15
    # 0 --> I1_out
    N[I1_out, r] = 1

    r += 1
    # reaction 16
    # I1_out --> 0
    N[I1_out, r] = -1

    r += 1
    # reaction 17
    # I1_out --> 0
    N[I1_out, r] = -1

    """
    # not I0: I0
    """

    r += 1
    # reaction 18
    # 0 --> L_I0
    N[L_I0, r] = 1

    r += 1
    # reaction 19
    # L_I0 --> 0
    N[L_I0, r] = -1

    r += 1
    # reaction 20
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 21
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 22
    # out --> 0
    N[out, r] = -1

    """
    # not I1: I1
    """

    r += 1
    # reaction 23
    # 0 --> L_I1
    N[L_I1, r] = 1

    r += 1
    # reaction 24
    # L_I1 --> 0
    N[L_I1, r] = -1

    r += 1
    # reaction 25
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 26
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 27
    # out --> 0
    N[out, r] = -1

    return N

def MUX_2_1_model_stochastic(state, params, Omega):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X = params
    params_yes = gamma_x, n_y, theta_x, delta_x, rho_x
    params_not = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x


    I0, I1, S0 = state[:3]
    I0_out, I1_out = state[3:5]
    L_I0_I0, L_I1_S0, L_I1_I1, L_I0, L_I1 = state[5:10]
    N_I0_S0, N_I0_I0, N_I1_S0, N_I1_I1, N_I0, N_I1 = state[10:16]
    out = state[16]
        
    """
     I0
    """
    # yes S0: I0_S0
    state_yes_I0_S0 = I0_out, S0, N_I0_S0, N_I0_S0
    p_I0_S0 = yes_cell_stochastic(state_yes_I0_S0, params_yes, Omega)

    # not I0: I0_I0
    state_not_I0_I0 = L_I0_I0, I0_out, I0, N_I0_I0, N_I0_I0
    p_I0_I0 = not_cell_stochastic(state_not_I0_I0, params_not, Omega)    
    
    """
     I1
    """
    # not S0: I1_S0
    state_not_I1_S0 = L_I1_S0, I1_out, S0, N_I1_S0, N_I1_S0
    p_I1_S0 = not_cell_stochastic(state_not_I1_S0, params_not, Omega)
    
    # not I1: I1_I1
    state_not_I1_I1 = L_I1_I1, I1_out, I1, N_I1_I1, N_I1_I1
    p_I1_I1 = not_cell_stochastic(state_not_I1_I1, params_not, Omega)

    """
    out
    """
    # not I0: I0
    state_not_I0 = L_I0, out, I0_out, N_I0, N_I0
    p_I0 = not_cell_stochastic(state_not_I0, params_not, Omega)

    # not I1: I1
    state_not_I1 = L_I1, out, I1_out, N_I1, N_I1
    p_I1 = not_cell_stochastic(state_not_I1, params_not, Omega)   
    
    return (p_I0_S0 + p_I0_I0 + p_I1_S0 + p_I1_I1 + p_I0 + p_I1)

def CLB_2_model_stohastic(state, params, Omega):
    
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b = params
 
    """
    latches
    """
    #########
    # params

    # set params for symmetric toggle switch topology
    gamma_L_Y, theta_L_Y = gamma_L_X, theta_L_X
    n_x, m_y = n_y, m_x
    eta_y, omega_y = eta_x, omega_x
 
    params_toggle =  [delta_L, gamma_L_X, gamma_L_Y, n_x, n_y, theta_L_X, theta_L_Y, eta_x, eta_y, omega_x, omega_y, m_x, m_y, delta_x, delta_y, rho_x, rho_y, r_X, r_Y]

    # degradation rates for induction of switches are specific for each toggle switch    
    params_toggle_I0 =  params_toggle.copy()
    params_toggle_I0[-4:-2] = rho_I0_a, rho_I0_b
    params_toggle_I1 =  params_toggle.copy()
    params_toggle_I1[-4:-2] = rho_I1_a, rho_I1_b

    #########
    # states
    
    # latch I0
    I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b = state[:6]
    state_toggle_IO = I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b

    # latch I1
    I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b = state[6:12]
    state_toggle_I1 = I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b

    #########
    # models
    p_toggle_IO = toggle_model_stochastic(state_toggle_IO, params_toggle_I0, Omega)
    p_toggle_I1 = toggle_model_stochastic(state_toggle_I1, params_toggle_I1, Omega)

    """
    mux
    """
    #########
    # params
    params_mux = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X

    #########
    # state
    I0, I1 = I0_a, I1_a
    state_mux = np.append([I0, I1], state[12:], axis=0)

    ########
    # model
    p_mux = MUX_2_1_model_stochastic(state_mux, params_mux, Omega)

    """
    return
    """
    return p_toggle_IO + p_toggle_I1 + p_mux
