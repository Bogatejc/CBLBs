import numpy as np
from models import *

def MUX_8_1_model_ode(state, T, params):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X = params
    params_yes = gamma_x, n_y, theta_x, delta_x, rho_x
    params_not = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x

    I0, I1, I2, I3, I4, I5, I6, I7, S0, S1, S2 = state[:11]
    I0_out, I1_out, I2_out, I3_out, I4_out, I5_out, I6_out, I7_out = state[11:19]
    L_I0_I0, L_I1_S2, L_I1_I1, L_I2_S1, L_I2_I2, L_I3_S1, L_I3_S2, L_I3_I3, L_I4_S0, L_I4_I4, L_I5_S0, L_I5_S2, L_I5_I5, L_I6_S0, L_I6_S1, L_I6_I6, L_I7_S0, L_I7_S1, L_I7_S2, L_I7_I7, L_I0, L_I1, L_I2, L_I3, L_I4, L_I5, L_I6, L_I7 = state[19:47]
    N_I0_S0, N_I0_S1, N_I0_S2, N_I0_I0, N_I1_S0, N_I1_S1, N_I1_S2, N_I1_I1, N_I2_S0, N_I2_S1, N_I2_S2, N_I2_I2, N_I3_S0, N_I3_S1, N_I3_S2, N_I3_I3, N_I4_S0, N_I4_S1, N_I4_S2, N_I4_I4, N_I5_S0, N_I5_S1, N_I5_S2, N_I5_I5, N_I6_S0, N_I6_S1, N_I6_S2, N_I6_I6, N_I7_S0, N_I7_S1, N_I7_S2, N_I7_I7, N_I0, N_I1, N_I2, N_I3, N_I4, N_I5, N_I6, N_I7 = state[47:87]
    out = state[87]

    """
    I0
    """
    dI0_out = 0

    # yes S0: I0_S0
    state_yes_I0_S0 = I0_out, S0, N_I0_S0
    dI0_out += yes_cell_wrapper(state_yes_I0_S0, params_yes)
    dN_I0_S0 = population(N_I0_S0, r_X)

    # yes S1: I0_S1
    state_yes_I0_S1 = I0_out, S1, N_I0_S1
    dI0_out += yes_cell_wrapper(state_yes_I0_S1, params_yes)
    dN_I0_S1 = population(N_I0_S1, r_X)

    # yes S2: I0_S2
    state_yes_I0_S2 = I0_out, S2, N_I0_S2
    dI0_out += yes_cell_wrapper(state_yes_I0_S2, params_yes)
    dN_I0_S2 = population(N_I0_S2, r_X)

    # not I0: I0_I0
    state_not_I0_I0 = L_I0_I0, I0_out, I0, N_I0_I0
    dL_I0_I0, dd = not_cell_wrapper(state_not_I0_I0, params_not)
    dI0_out += dd
    dN_I0_I0 = population(N_I0_I0, r_X)

    """
    I1
    """
    dI1_out = 0

    # yes S0: I1_S0
    state_yes_I1_S0 = I1_out, S0, N_I1_S0
    dI1_out += yes_cell_wrapper(state_yes_I1_S0, params_yes)
    dN_I1_S0 = population(N_I1_S0, r_X)

    # yes S1: I1_S1
    state_yes_I1_S1 = I1_out, S1, N_I1_S1
    dI1_out += yes_cell_wrapper(state_yes_I1_S1, params_yes)
    dN_I1_S1 = population(N_I1_S1, r_X)

    # not S2: I1_S2
    state_not_I1_S2 = L_I1_S2, I1_out, S2, N_I1_S2
    dL_I1_S2, dd = not_cell_wrapper(state_not_I1_S2, params_not)
    dI1_out += dd
    dN_I1_S2 = population(N_I1_S2, r_X)
    
    # not I1: I1_I1
    state_not_I1_I1 = L_I1_I1, I1_out, I1, N_I1_I1
    dL_I1_I1, dd = not_cell_wrapper(state_not_I1_I1, params_not)
    dI1_out += dd
    dN_I1_I1 = population(N_I1_I1, r_X)

    """
    I2
    """
    dI2_out = 0

    # yes S0: I2_S0
    state_yes_I2_S0 = I2_out, S0, N_I2_S0
    dI2_out += yes_cell_wrapper(state_yes_I2_S0, params_yes)
    dN_I2_S0 = population(N_I2_S0, r_X)

    # not S1: I2_S1
    state_not_I2_S1 = L_I2_S1, I2_out, S1, N_I2_S1
    dL_I2_S1, dd = not_cell_wrapper(state_not_I2_S1, params_not)
    dI2_out += dd
    dN_I2_S1 = population(N_I2_S1, r_X)

    # yes S2: I2_S2
    state_yes_I2_S2 = I2_out, S2, N_I2_S2
    dI2_out += yes_cell_wrapper(state_yes_I2_S2, params_yes)
    dN_I2_S2 = population(N_I2_S2, r_X)
    
    # not I2: I2_I2
    state_not_I2_I2 = L_I2_I2, I2_out, I2, N_I2_I2
    dL_I2_I2, dd = not_cell_wrapper(state_not_I2_I2, params_not)
    dI2_out += dd
    dN_I2_I2 = population(N_I2_I2, r_X)

    """
    I3
    """
    dI3_out = 0

    # yes S0: I3_S0
    state_yes_I3_S0 = I3_out, S0, N_I3_S0
    dI3_out += yes_cell_wrapper(state_yes_I3_S0, params_yes)
    dN_I3_S0 = population(N_I3_S0, r_X)

    # not S1: I3_S1
    state_not_I3_S1 = L_I3_S1, I3_out, S1, N_I3_S1
    dL_I3_S1, dd = not_cell_wrapper(state_not_I3_S1, params_not)
    dI3_out += dd
    dN_I3_S1 = population(N_I3_S1, r_X)

    # not S2: I3_S2
    state_not_I3_S2 = L_I3_S2, I3_out, S2, N_I3_S2
    dL_I3_S2, dd = not_cell_wrapper(state_not_I3_S2, params_not)
    dI3_out += dd
    dN_I3_S2 = population(N_I3_S2, r_X)
    
    # not I3: I3_I3
    state_not_I3_I3 = L_I3_I3, I3_out, I3, N_I3_I3
    dL_I3_I3, dd = not_cell_wrapper(state_not_I3_I3, params_not)
    dI3_out += dd
    dN_I3_I3 = population(N_I3_I3, r_X)

    """
    I4
    """
    dI4_out = 0

    # not S0: I4_S0
    state_not_I4_S0 = L_I4_S0, I4_out, S0, N_I4_S0
    dL_I4_S0, dd = not_cell_wrapper(state_not_I4_S0, params_not)
    dI4_out += dd
    dN_I4_S0 = population(N_I4_S0, r_X)

    # yes S1: I4_S1
    state_yes_I4_S1 = I4_out, S1, N_I4_S1
    dI4_out += yes_cell_wrapper(state_yes_I4_S1, params_yes)
    dN_I4_S1 = population(N_I4_S1, r_X)

    # yes S2: I4_S2
    state_yes_I4_S2 = I4_out, S2, N_I4_S2
    dI4_out += yes_cell_wrapper(state_yes_I4_S2, params_yes)
    dN_I4_S2 = population(N_I4_S2, r_X)

    # not I4: I4_I4
    state_not_I4_I4 = L_I4_I4, I4_out, I4, N_I4_I4
    dL_I4_I4, dd = not_cell_wrapper(state_not_I4_I4, params_not)
    dI4_out += dd
    dN_I4_I4 = population(N_I4_I4, r_X)

    """
    I5
    """
    dI5_out = 0

    # not S0: I5_S0
    state_not_I5_S0 = L_I5_S0, I5_out, S0, N_I5_S0
    dL_I5_S0, dd = not_cell_wrapper(state_not_I5_S0, params_not)
    dI5_out += dd
    dN_I5_S0 = population(N_I5_S0, r_X)

    # yes S1: I5_S1
    state_yes_I5_S1 = I5_out, S1, N_I5_S1
    dI5_out += yes_cell_wrapper(state_yes_I5_S1, params_yes)
    dN_I5_S1 = population(N_I5_S1, r_X)

    # not S2: I5_S2
    state_not_I5_S2 = L_I5_S2, I5_out, S2, N_I5_S2
    dL_I5_S2, dd = not_cell_wrapper(state_not_I5_S2, params_not)
    dI5_out += dd
    dN_I5_S2 = population(N_I5_S2, r_X)

    # not I5: I5_I5
    state_not_I5_I5 = L_I5_I5, I5_out, I5, N_I5_I5
    dL_I5_I5, dd = not_cell_wrapper(state_not_I5_I5, params_not)
    dI5_out += dd
    dN_I5_I5 = population(N_I5_I5, r_X)

    """
    I6
    """
    dI6_out = 0

    # not S0: I6_S0
    state_not_I6_S0 = L_I6_S0, I6_out, S0, N_I6_S0
    dL_I6_S0, dd = not_cell_wrapper(state_not_I6_S0, params_not)
    dI6_out += dd
    dN_I6_S0 = population(N_I6_S0, r_X)

    # not S1: I6_S1
    state_not_I6_S1 = L_I6_S1, I6_out, S1, N_I6_S1
    dL_I6_S1, dd = not_cell_wrapper(state_not_I6_S1, params_not)
    dI6_out += dd
    dN_I6_S1 = population(N_I6_S1, r_X)

    # yes S2: I6_S2
    state_yes_I6_S2 = I6_out, S2, N_I6_S2
    dI6_out += yes_cell_wrapper(state_yes_I6_S2, params_yes)
    dN_I6_S2 = population(N_I6_S2, r_X)

    # not I6: I6_I6
    state_not_I6_I6 = L_I6_I6, I6_out, I6, N_I6_I6
    dL_I6_I6, dd = not_cell_wrapper(state_not_I6_I6, params_not)
    dI6_out += dd
    dN_I6_I6 = population(N_I6_I6, r_X)

    """
    I7
    """
    dI7_out = 0

    # not S0: I7_S0
    state_not_I7_S0 = L_I7_S0, I7_out, S0, N_I7_S0
    dL_I7_S0, dd = not_cell_wrapper(state_not_I7_S0, params_not)
    dI7_out += dd
    dN_I7_S0 = population(N_I7_S0, r_X)

    # not S1: I7_S1
    state_not_I7_S1 = L_I7_S1, I7_out, S1, N_I7_S1
    dL_I7_S1, dd = not_cell_wrapper(state_not_I7_S1, params_not)
    dI7_out += dd
    dN_I7_S1 = population(N_I7_S1, r_X)

    # not S2: I7_S2
    state_not_I7_S2 = L_I7_S2, I7_out, S2, N_I7_S2
    dL_I7_S2, dd = not_cell_wrapper(state_not_I7_S2, params_not)
    dI7_out += dd
    dN_I7_S2 = population(N_I7_S2, r_X)

    # not I7: I7_I7
    state_not_I7_I7 = L_I7_I7, I7_out, I7, N_I7_I7
    dL_I7_I7, dd = not_cell_wrapper(state_not_I7_I7, params_not)
    dI7_out += dd
    dN_I7_I7 = population(N_I7_I7, r_X)

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

    # not I2: I2
    state_not_I2 = L_I2, out, I2_out, N_I2
    dL_I2, dd = not_cell_wrapper(state_not_I2, params_not)    
    dout += dd
    dN_I2 = population(N_I2, r_X)

    # not I3: I3
    state_not_I3 = L_I3, out, I3_out, N_I3
    dL_I3, dd = not_cell_wrapper(state_not_I3, params_not)    
    dout += dd
    dN_I3 = population(N_I3, r_X)

    # not I4: I4
    state_not_I4 = L_I4, out, I4_out, N_I4
    dL_I4, dd = not_cell_wrapper(state_not_I4, params_not)    
    dout += dd
    dN_I4 = population(N_I4, r_X)

    # not I5: I5
    state_not_I5 = L_I5, out, I5_out, N_I5
    dL_I5, dd = not_cell_wrapper(state_not_I5, params_not)    
    dout += dd
    dN_I5 = population(N_I5, r_X)

    # not I6: I6
    state_not_I6 = L_I6, out, I6_out, N_I6
    dL_I6, dd = not_cell_wrapper(state_not_I6, params_not)    
    dout += dd
    dN_I6 = population(N_I6, r_X)

    # not I7: I7
    state_not_I7 = L_I7, out, I7_out, N_I7
    dL_I7, dd = not_cell_wrapper(state_not_I7, params_not)    
    dout += dd
    dN_I7 = population(N_I7, r_X)

    dI0, dI1, dI2, dI3, dI4, dI5, dI6, dI7, dS0, dS1, dS2 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    dstate = np.array([dI0, dI1, dI2, dI3, dI4, dI5, dI6, dI7, dS0, dS1, dS2,
              dI0_out, dI1_out, dI2_out, dI3_out, dI4_out, dI5_out, dI6_out, dI7_out,
              dL_I0_I0, dL_I1_S2, dL_I1_I1, dL_I2_S1, dL_I2_I2, dL_I3_S1, dL_I3_S2, dL_I3_I3, dL_I4_S0, dL_I4_I4,
              dL_I5_S0, dL_I5_S2, dL_I5_I5, dL_I6_S0, dL_I6_S1, dL_I6_I6, dL_I7_S0, dL_I7_S1, dL_I7_S2, dL_I7_I7,
              dL_I0, dL_I1, dL_I2, dL_I3, dL_I4, dL_I5, dL_I6, dL_I7,
              dN_I0_S0, dN_I0_S1, dN_I0_S2, dN_I0_I0, dN_I1_S0, dN_I1_S2, dN_I1_S1, dN_I1_I1, dN_I2_S0, dN_I2_S1, dN_I2_S2, dN_I2_I2, dN_I3_S0, dN_I3_S1, dN_I3_S2, dN_I3_I3,
              dN_I4_S0, dN_I4_S1, dN_I4_S2, dN_I4_I4, dN_I5_S0, dN_I5_S1, dN_I5_S2, dN_I5_I5, dN_I6_S0, dN_I6_S1, dN_I6_S2, dN_I6_I6, dN_I7_S0, dN_I7_S1, dN_I7_S2, dN_I7_I7,
              dN_I0, dN_I1, dN_I2, dN_I3, dN_I4, dN_I5, dN_I6, dN_I7,
              dout])
    
    return dstate

def MUX_8_1_generate_stoichiometry():
    I0_out, I1_out, I2_out, I3_out, I4_out, I5_out, I6_out, I7_out = range(11, 19)
    L_I0_I0, L_I1_S2, L_I1_I1, L_I2_S1, L_I2_I2, L_I3_S1, L_I3_S2, L_I3_I3, L_I4_S0, L_I4_I4, L_I5_S0, L_I5_S2, L_I5_I5, L_I6_S0, L_I6_S1, L_I6_I6, L_I7_S0, L_I7_S1, L_I7_S2, L_I7_I7, L_I0, L_I1, L_I2, L_I3, L_I4, L_I5, L_I6, L_I7 = range(19, 47)
    out = 87

    #
    # x axis ... species
    # y axis ... reactions
    #
    N = np.zeros((88, 176))

    ##################### I0
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
    # yes S1: I0_S1
    """
    r += 1
    # reaction 3
    # 0 --> I0_out
    N[I0_out, r] = 1

    r += 1
    # reaction 4
    # I0_out --> 0
    N[I0_out, r] = -1

    r += 1
    # reaction 5
    # I0_out --> 0
    N[I0_out, r] = -1

    """
    # yes S2: I0_S2
    """
    r += 1
    # reaction 6
    # 0 --> I0_out
    N[I0_out, r] = 1

    r += 1
    # reaction 7
    # I0_out --> 0
    N[I0_out, r] = -1

    r += 1
    # reaction 8
    # I0_out --> 0
    N[I0_out, r] = -1

    """
    # not I0: I0_I0 
    """
    r += 1
    # reaction 9
    # 0 --> L_I0_I0
    N[L_I0_I0, r] = 1

    r += 1
    # reaction 10
    # L_I0_I0 --> 0
    N[L_I0_I0, r] = -1
    
    r += 1
    # reaction 11
    # 0 --> I0_out
    N[I0_out, r] = 1

    r += 1
    # reaction 12
    # I0_out --> 0
    N[I0_out, r] = -1

    r += 1
    # reaction 13
    # I0_out --> 0
    N[I0_out, r] = -1

    ##################### I1
    """
    # yes S0: I1_S0
    """
    r += 1
    # reaction 14
    # 0 --> I1_out
    N[I1_out, r] = 1
    
    r += 1
    # reaction 15
    # I1_out --> 0
    N[I1_out, r] = -1
    
    r += 1
    # reaction 16
    # I1_out --> 0
    N[I1_out, r] = -1
    
    """
    # yes S1: I1_S1
    """
    r += 1
    # reaction 17
    # 0 --> I1_out
    N[I1_out, r] = 1

    r += 1
    # reaction 18
    # I1_out --> 0
    N[I1_out, r] = -1

    r += 1
    # reaction 19
    # I1_out --> 0
    N[I1_out, r] = -1

    """
    # not S2: I1_S2
    """
    r += 1
    # reaction 20
    # 0 --> L_I1_S2
    N[L_I1_S2, r] = 1

    r += 1
    # reaction 21
    # L_I1_S2 --> 0
    N[L_I1_S2, r] = -1

    r += 1
    # reaction 22
    # 0 --> I1_out
    N[I1_out, r] = 1
    
    r += 1
    # reaction 23
    # I1_out --> 0
    N[I1_out, r] = -1

    r += 1
    # reaction 24
    # I1_out --> 0
    N[I1_out, r] = -1

    """
    # not I1: I1_I1
    """
    r += 1
    # reaction 25
    # 0 --> L_I1_I1
    N[L_I1_I1, r] = 1

    r += 1
    # reaction 26
    # L_I1_I1 --> 0
    N[L_I1_I1, r] = -1

    r += 1
    # reaction 27
    # 0 --> I1_out
    N[I1_out, r] = 1

    r += 1
    # reaction 28
    # I1_out --> 0
    N[I1_out, r] = -1

    r += 1
    # reaction 29
    # I1_out --> 0
    N[I1_out, r] = -1

    ##################### I2
    """
    # yes S0: I2_S0
    """
    r += 1
    # reaction 30
    # 0 --> I2_out
    N[I2_out, r] = 1

    r += 1
    # reaction 31
    # I2_out --> 0
    N[I2_out, r] = -1

    r += 1
    # reaction 32
    # I2_out --> 0
    N[I2_out, r] = -1

    """
    # not S1: I2_S1
    """
    r += 1
    # reaction 33
    # 0 --> L_I2_S1
    N[L_I2_S1, r] = 1

    r += 1
    # reaction 34
    # L_I2_S1 --> 0
    N[L_I2_S1, r] = -1

    r += 1
    # reaction 35
    # 0 --> I2_out
    N[I2_out, r] = 1
    
    r += 1
    # reaction 36
    # I2_out --> 0
    N[I2_out, r] = -1

    r += 1
    # reaction 37
    # I2_out --> 0
    N[I2_out, r] = -1

    """
    # yes S2: I2_S2
    """
    r += 1
    # reaction 38
    # 0 --> I2_out
    N[I2_out, r] = 1

    r += 1
    # reaction 39
    # I2_out --> 0
    N[I2_out, r] = -1

    r += 1
    # reaction 40
    # I2_out --> 0
    N[I2_out, r] = -1

    """
    # not I2: I2_I2
    """
    r += 1
    # reaction 41
    # 0 --> L_I2_I2
    N[L_I2_I2, r] = 1

    r += 1
    # reaction 42
    # L_I2_I2 --> 0
    N[L_I2_I2, r] = -1

    r += 1
    # reaction 43
    # 0 --> I2_out
    N[I2_out, r] = 1

    r += 1
    # reaction 44
    # I2_out --> 0
    N[I2_out, r] = -1

    r += 1
    # reaction 45
    # I2_out --> 0
    N[I2_out, r] = -1

    ##################### I3
    """
    # yes S0: I3_S0
    """
    r += 1
    # reaction 46
    # 0 --> I3_out
    N[I3_out, r] = 1

    r += 1
    # reaction 47
    # I3_out --> 0
    N[I3_out, r] = -1

    r += 1
    # reaction 48
    # I3_out --> 0
    N[I3_out, r] = -1

    """
    # not S1: I3_S1
    """
    r += 1
    # reaction 49
    # 0 --> L_I3_S1
    N[L_I3_S1, r] = 1

    r += 1
    # reaction 50
    # L_I3_S1 --> 0
    N[L_I3_S1, r] = -1

    r += 1
    # reaction 51
    # 0 --> I3_out
    N[L_I3_S1, r] = 1

    r += 1
    # reaction 52
    # I3_out --> 0
    N[I3_out, r] = -1

    r += 1
    # reaction 53
    # I3_out --> 0
    N[I3_out, r] = -1

    """
    # not S2: I3_S2
    """
    r += 1
    # reaction 54
    # 0 --> L_I3_S2
    N[L_I3_S2, r] = 1

    r += 1
    # reaction 55
    # L_I3_S2 --> 0
    N[L_I3_S2, r] = -1

    r += 1
    # reaction 56
    # 0 --> I3_out
    N[I3_out, r] = 1
    
    r += 1
    # reaction 57
    # I3_out --> 0
    N[I3_out, r] = -1

    r += 1
    # reaction 58
    # I3_out --> 0
    N[I3_out, r] = -1

    """
    # not I3: I3_I3
    """
    r += 1
    # reaction 59
    # 0 --> L_I3_I3
    N[L_I3_I3, r] = 1

    r += 1
    # reaction 60
    # L_I3_I3 --> 0
    N[L_I3_I3, r] = -1

    r += 1
    # reaction 61
    # 0 --> I3_out
    N[I3_out, r] = 1

    r += 1
    # reaction 62
    # I3_out --> 0
    N[I3_out, r] = -1

    r += 1
    # reaction 63
    # I3_out --> 0
    N[I3_out, r] = -1

    ##################### I4
    """
    # not S0: I4_S0
    """
    r += 1
    # reaction 64
    # 0 --> L_I4_S0
    N[L_I4_S0, r] = 1

    r += 1
    # reaction 65
    # 0 --> L_I4_S0
    N[L_I4_S0, r] = -1

    r += 1
    # reaction 66
    # 0 --> I4_out
    N[I4_out, r] = 1

    r += 1
    # reaction 67
    # I4_out --> 0
    N[I4_out, r] = -1

    r += 1
    # reaction 68
    # I4_out --> 0
    N[I4_out, r] = -1

    """
    # yes S1: I4_S1
    """
    r += 1
    # reaction 69
    # 0 --> I4_out
    N[I4_out, r] = 1

    r += 1
    # reaction 70
    # I4_out --> 0
    N[I4_out, r] = -1

    r += 1
    # reaction 71
    # I4_out --> 0
    N[I4_out, r] = -1

    """
    # yes S2: I4_S2
    """
    r += 1
    # reaction 72
    # 0 --> I4_out
    N[I4_out, r] = 1

    r += 1
    # reaction 73
    # I4_out --> 0
    N[I4_out, r] = -1

    r += 1
    # reaction 74
    # I4_out --> 0
    N[I4_out, r] = -1

    """
    # not I4: I4_I4
    """
    r += 1
    # reaction 75
    # 0 --> L_I4_I4
    N[L_I4_I4, r] = 1

    r += 1
    # reaction 76
    # L_I4_I4 --> 0
    N[L_I4_I4, r] = -1
    
    r += 1
    # reaction 77
    # 0 --> I4_out
    N[I4_out, r] = 1

    r += 1
    # reaction 78
    # I4_out --> 0
    N[I4_out, r] = -1

    r += 1
    # reaction 79
    # I4_out --> 0
    N[I4_out, r] = -1

    ##################### I5
    """
    # not S0: I5_S0
    """
    r += 1
    # reaction 80
    # 0 --> L_I5_S0
    N[L_I5_S0, r] = 1

    r += 1
    # reaction 81
    # 0 --> L_I5_S0
    N[L_I5_S0, r] = -1

    r += 1
    # reaction 82
    # 0 --> I5_out
    N[I5_out, r] = 1

    r += 1
    # reaction 83
    # I5_out --> 0
    N[I5_out, r] = -1

    r += 1
    # reaction 84
    # I5_out --> 0
    N[I5_out, r] = -1

    """
    # yes S1: I5_S1
    """
    r += 1
    # reaction 85
    # 0 --> I5_out
    N[I5_out, r] = 1

    r += 1
    # reaction 86
    # I5_out --> 0
    N[I5_out, r] = -1

    r += 1
    # reaction 87
    # I5_out --> 0
    N[I5_out, r] = -1

    """
    # not S2: I5_S2
    """
    r += 1
    # reaction 88
    # 0 --> L_I5_S2
    N[L_I5_S2, r] = 1

    r += 1
    # reaction 89
    # L_I5_S2 --> 0
    N[L_I5_S2, r] = -1

    r += 1
    # reaction 90
    # 0 --> I5_out
    N[I5_out, r] = 1

    r += 1
    # reaction 91
    # I5_out --> 0
    N[I5_out, r] = -1

    r += 1
    # reaction 92
    # I5_out --> 0
    N[I5_out, r] = -1

    """
    # not I5: I5_I5 
    """
    r += 1
    # reaction 93
    # 0 --> L_I5_I5
    N[L_I5_I5, r] = 1

    r += 1
    # reaction 94
    # L_I5_I5 --> 0
    N[L_I5_I5, r] = -1
  
    r += 1
    # reaction 95
    # 0 --> I5_out
    N[I5_out, r] = 1

    r += 1
    # reaction 96
    # I5_out --> 0
    N[I5_out, r] = -1

    r += 1
    # reaction 97
    # I5_out --> 0
    N[I5_out, r] = -1

    ##################### I6
    """
    # not S0: I6_S0
    """
    r += 1
    # reaction 98
    # 0 --> L_I6_S0
    N[L_I6_S0, r] = 1

    r += 1
    # reaction 99
    # 0 --> L_I6_S0
    N[L_I6_S0, r] = -1

    r += 1
    # reaction 100
    # 0 --> I6_out
    N[I6_out, r] = 1

    r += 1
    # reaction 101
    # I6_out --> 0
    N[I6_out, r] = -1

    r += 1
    # reaction 102
    # I6_out --> 0
    N[I6_out, r] = -1

    """
    # not S1: I6_S1
    """
    r += 1
    # reaction 103
    # 0 --> L_I6_S1
    N[L_I6_S1, r] = 1

    r += 1
    # reaction 104
    # L_I6_S1 --> 0
    N[L_I6_S1, r] = -1

    r += 1
    # reaction 105
    # 0 --> I6_out
    N[L_I6_S1, r] = 1

    r += 1
    # reaction 106
    # I6_out --> 0
    N[I6_out, r] = -1

    r += 1
    # reaction 107
    # I6_out --> 0
    N[I6_out, r] = -1

    """
    # yes S2: I6_S2
    """
    r += 1
    # reaction 108
    # 0 --> I6_out
    N[I6_out, r] = 1

    r += 1
    # reaction 109
    # I6_out --> 0
    N[I6_out, r] = -1

    r += 1
    # reaction 110
    # I6_out --> 0
    N[I6_out, r] = -1

    """
    # not I6: I6_I6
    """
    r += 1
    # reaction 111
    # 0 --> L_I6_I6
    N[L_I6_I6, r] = 1

    r += 1
    # reaction 112
    # L_I6_I6 --> 0
    N[L_I6_I6, r] = -1
    
    r += 1
    # reaction 113
    # 0 --> I6_out
    N[I6_out, r] = 1

    r += 1
    # reaction 114
    # I6_out --> 0
    N[I6_out, r] = -1

    r += 1
    # reaction 115
    # I6_out --> 0
    N[I6_out, r] = -1

    ##################### I7
    """
    # not S0: I7_S0
    """
    r += 1
    # reaction 116
    # 0 --> L_I7_S0
    N[L_I7_S0, r] = 1

    r += 1
    # reaction 117
    # 0 --> L_I7_S0
    N[L_I7_S0, r] = -1

    r += 1
    # reaction 118
    # 0 --> I7_out
    N[I7_out, r] = 1

    r += 1
    # reaction 119
    # I7_out --> 0
    N[I7_out, r] = -1

    r += 1
    # reaction 120
    # I7_out --> 0
    N[I7_out, r] = -1

    """
    # not S1: I7_S1
    """
    r += 1
    # reaction 121
    # 0 --> L_I7_S1
    N[L_I7_S1, r] = 1

    r += 1
    # reaction 122
    # L_I7_S1 --> 0
    N[L_I7_S1, r] = -1

    r += 1
    # reaction 123
    # 0 --> I7_out
    N[L_I7_S1, r] = 1

    r += 1
    # reaction 124
    # I7_out --> 0
    N[I7_out, r] = -1

    r += 1
    # reaction 125
    # I7_out --> 0
    N[I7_out, r] = -1

    """
    # not S2: I7_S2
    """
    r += 1
    # reaction 126
    # 0 --> L_I7_S2
    N[L_I7_S2, r] = 1

    r += 1
    # reaction 127
    # L_I7_S2 --> 0
    N[L_I7_S2, r] = -1

    r += 1
    # reaction 128
    # 0 --> I7_out
    N[I7_out, r] = 1
    
    r += 1
    # reaction 129
    # I7_out --> 0
    N[I7_out, r] = -1

    r += 1
    # reaction 130
    # I7_out --> 0
    N[I7_out, r] = -1

    """
    # not I7: I7_I7
    """
    r += 1
    # reaction 131
    # 0 --> L_I7_I7
    N[L_I7_I7, r] = 1

    r += 1
    # reaction 132
    # L_I7_I7 --> 0
    N[L_I7_I7, r] = -1

    r += 1
    # reaction 133
    # 0 --> I7_out
    N[I7_out, r] = 1

    r += 1
    # reaction 134
    # I7_out --> 0
    N[I7_out, r] = -1

    r += 1
    # reaction 135
    # I7_out --> 0
    N[I7_out, r] = -1

    ##################### out
    """
    # not I0: I0
    """
    r += 1
    # reaction 136
    # 0 --> L_I0
    N[L_I0, r] = 1

    r += 1
    # reaction 137
    # L_I0 --> 0
    N[L_I0, r] = -1

    r += 1
    # reaction 138
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 139
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 140
    # out --> 0
    N[out, r] = -1

    """
    # not I1: I1
    """
    r += 1
    # reaction 141
    # 0 --> L_I1
    N[L_I1, r] = 1

    r += 1
    # reaction 142
    # L_I1 --> 0
    N[L_I1, r] = -1

    r += 1
    # reaction 143
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 144
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 145
    # out --> 0
    N[out, r] = -1

    """
    # not I2: I2
    """
    r += 1
    # reaction 146
    # 0 --> L_I2
    N[L_I2, r] = 1

    r += 1
    # reaction 147
    # L_I2 --> 0
    N[L_I2, r] = -1

    r += 1
    # reaction 148
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 149
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 150
    # out --> 0
    N[out, r] = -1

    """
    # not I3: I3
    """
    r += 1
    # reaction 151
    # 0 --> L_I3
    N[L_I3, r] = 1

    r += 1
    # reaction 152
    # L_I3 --> 0
    N[L_I3, r] = -1

    r += 1
    # reaction 153
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 154
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 155
    # out --> 0
    N[out, r] = -1

    """
    # not I4: I4
    """
    r += 1
    # reaction 156
    # 0 --> L_I4
    N[L_I4, r] = 1

    r += 1
    # reaction 157
    # L_I4 --> 0
    N[L_I4, r] = -1

    r += 1
    # reaction 158
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 159
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 160
    # out --> 0
    N[out, r] = -1

    """
    # not I5: I5
    """
    r += 1
    # reaction 161
    # 0 --> L_I5
    N[L_I5, r] = 1

    r += 1
    # reaction 162
    # L_I5 --> 0
    N[L_I5, r] = -1

    r += 1
    # reaction 163
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 164
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 165
    # out --> 0
    N[out, r] = -1

    """
    # not I6: I6
    """
    r += 1
    # reaction 166
    # 0 --> L_I6
    N[L_I6, r] = 1

    r += 1
    # reaction 167
    # L_I6 --> 0
    N[L_I6, r] = -1

    r += 1
    # reaction 168
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 169
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 170
    # out --> 0
    N[out, r] = -1

    """
    # not I7: I7
    """
    r += 1
    # reaction 171
    # 0 --> L_I7
    N[L_I7, r] = 1

    r += 1
    # reaction 172
    # L_I7 --> 0
    N[L_I7, r] = -1

    r += 1
    # reaction 173
    # 0 --> out
    N[out, r] = 1

    r += 1
    # reaction 174
    # out --> 0
    N[out, r] = -1

    r += 1
    # reaction 175
    # out --> 0
    N[out, r] = -1

    return N

def CLB_8_generate_stoichiometry():
    N_toggle_I0 = toggle_generate_stoichiometry()
    N_toggle_I1 = toggle_generate_stoichiometry()
    N_toggle_I2 = toggle_generate_stoichiometry()
    N_toggle_I3 = toggle_generate_stoichiometry()
    N_toggle_I4 = toggle_generate_stoichiometry()
    N_toggle_I5 = toggle_generate_stoichiometry()
    N_toggle_I6 = toggle_generate_stoichiometry()
    N_toggle_I7 = toggle_generate_stoichiometry()


    N_mux = MUX_8_1_generate_stoichiometry()
    # skip first eight rows (I0, I1, I2, I3, I4, I5, I6, I7)
    N_mux = N_mux[8:, :]

    return merge_N(merge_N(merge_N(merge_N(merge_N(merge_N(merge_N(merge_N(N_toggle_I0, N_toggle_I1), N_toggle_I2), N_toggle_I3), N_toggle_I4), N_toggle_I5), N_toggle_I6), N_toggle_I7), N_mux)

def MUX_8_1_model_ssa(state, params, Omega):
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X = params
    params_yes = gamma_x, n_y, theta_x, delta_x, rho_x
    params_not = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x

    I0, I1, I2, I3, I4, I5, I6, I7, S0, S1, S2 = state[:11]
    I0_out, I1_out, I2_out, I3_out, I4_out, I5_out, I6_out, I7_out = state[11:19]
    L_I0_I0, L_I1_S2, L_I1_I1, L_I2_S1, L_I2_I2, L_I3_S1, L_I3_S2, L_I3_I3, L_I4_S0, L_I4_I4, L_I5_S0, L_I5_S2, L_I5_I5, L_I6_S0, L_I6_S1, L_I6_I6, L_I7_S0, L_I7_S1, L_I7_S2, L_I7_I7, L_I0, L_I1, L_I2, L_I3, L_I4, L_I5, L_I6, L_I7 = state[19:47]
    N_I0_S0, N_I0_S1, N_I0_S2, N_I0_I0, N_I1_S0, N_I1_S1, N_I1_S2, N_I1_I1, N_I2_S0, N_I2_S1, N_I2_S2, N_I2_I2, N_I3_S0, N_I3_S1, N_I3_S2, N_I3_I3, N_I4_S0, N_I4_S1, N_I4_S2, N_I4_I4, N_I5_S0, N_I5_S1, N_I5_S2, N_I5_I5, N_I6_S0, N_I6_S1, N_I6_S2, N_I6_I6, N_I7_S0, N_I7_S1, N_I7_S2, N_I7_I7, N_I0, N_I1, N_I2, N_I3, N_I4, N_I5, N_I6, N_I7 = state[47:87]
    out = state[87]

    """
    I0
    """
    # yes S0: I0_S0
    state_yes_I0_S0 = I0_out, S0, N_I0_S0, N_I0_S0
    p_I0_S0 = yes_cell_stochastic(state_yes_I0_S0, params_yes, Omega)

    # yes S1: I0_S1
    state_yes_I0_S1 = I0_out, S1, N_I0_S1, N_I0_S1
    p_I0_S1 = yes_cell_stochastic(state_yes_I0_S1, params_yes, Omega)

    # yes S2: I0_S2
    state_yes_I0_S2 = I0_out, S2, N_I0_S2, N_I0_S2
    p_I0_S2 = yes_cell_stochastic(state_yes_I0_S2, params_yes, Omega)

    # not I0: I0_I0
    state_not_I0_I0 = L_I0_I0, I0_out, I0, N_I0_I0, N_I0_I0
    p_I0_I0 = not_cell_stochastic(state_not_I0_I0, params_not, Omega)

    """
    I1
    """
    # yes S0: I1_S0
    state_yes_I1_S0 = I1_out, S0, N_I1_S0, N_I1_S0
    p_I1_S0 = yes_cell_stochastic(state_yes_I1_S0, params_yes, Omega)

    # yes S1: I1_S1
    state_yes_I1_S1 = I1_out, S1, N_I1_S1, N_I1_S1
    p_I1_S1 = yes_cell_stochastic(state_yes_I1_S1, params_yes, Omega)

    # not S2: I1_S2
    state_not_I1_S2 = L_I1_S2, I1_out, S2, N_I1_S2, N_I1_S2
    p_I1_S2 = not_cell_stochastic(state_not_I1_S2, params_not, Omega)
    
    # not I1: I1_I1
    state_not_I1_I1 = L_I1_I1, I1_out, I1, N_I1_I1, N_I1_I1
    p_I1_I1 = not_cell_stochastic(state_not_I1_I1, params_not, Omega)

    """
    I2
    """
    # yes S0: I2_S0
    state_yes_I2_S0 = I2_out, S0, N_I2_S0, N_I2_S0
    p_I2_S0 = yes_cell_stochastic(state_yes_I2_S0, params_yes, Omega)

    # not S1: I2_S1
    state_not_I2_S1 = L_I2_S1, I2_out, S1, N_I2_S1, N_I2_S1
    p_I2_S1 = not_cell_stochastic(state_not_I2_S1, params_not, Omega)

    # yes S2: I2_S2
    state_yes_I2_S2 = I2_out, S2, N_I2_S2, N_I2_S2
    p_I2_S2 = yes_cell_stochastic(state_yes_I2_S2, params_yes, Omega)
    
    # not I2: I2_I2
    state_not_I2_I2 = L_I2_I2, I2_out, I2, N_I2_I2, N_I2_I2
    p_I2_I2 = not_cell_stochastic(state_not_I2_I2, params_not, Omega)

    """
    I3
    """
    # yes S0: I3_S0
    state_yes_I3_S0 = I3_out, S0, N_I3_S0, N_I3_S0
    p_I3_S0 = yes_cell_stochastic(state_yes_I3_S0, params_yes, Omega)

    # not S1: I3_S1
    state_not_I3_S1 = L_I3_S1, I3_out, S1, N_I3_S1, N_I3_S1
    p_I3_S1 = not_cell_stochastic(state_not_I3_S1, params_not, Omega)

    # not S2: I3_S2
    state_not_I3_S2 = L_I3_S2, I3_out, S2, N_I3_S2, N_I3_S2
    p_I3_S2 = not_cell_stochastic(state_not_I3_S2, params_not, Omega)
    
    # not I3: I3_I3
    state_not_I3_I3 = L_I3_I3, I3_out, I3, N_I3_I3, N_I3_I3
    p_I3_I3 = not_cell_stochastic(state_not_I3_I3, params_not, Omega)

    """
    I4
    """
    # not S0: I4_S0
    state_not_I4_S0 = L_I4_S0, I4_out, S0, N_I4_S0, N_I4_S0
    p_I4_S0 = not_cell_stochastic(state_not_I4_S0, params_not, Omega)

    # yes S1: I4_S1
    state_yes_I4_S1 = I4_out, S1, N_I4_S1, N_I4_S1
    p_I4_S1 = yes_cell_stochastic(state_yes_I4_S1, params_yes, Omega)

    # yes S2: I4_S2
    state_yes_I4_S2 = I4_out, S2, N_I4_S2, N_I4_S2
    p_I4_S2 = yes_cell_stochastic(state_yes_I4_S2, params_yes, Omega)

    # not I4: I4_I4
    state_not_I4_I4 = L_I4_I4, I4_out, I4, N_I4_I4, N_I4_I4
    p_I4_I4 = not_cell_stochastic(state_not_I4_I4, params_not, Omega)

    """
    I5
    """
    # not S0: I5_S0
    state_not_I5_S0 = L_I5_S0, I5_out, S0, N_I5_S0, N_I5_S0
    p_I5_S0 = not_cell_stochastic(state_not_I5_S0, params_not, Omega)

    # yes S1: I5_S1
    state_yes_I5_S1 = I5_out, S1, N_I5_S1, N_I5_S1
    p_I5_S1 = yes_cell_stochastic(state_yes_I5_S1, params_yes, Omega)

    # not S2: I5_S2
    state_not_I5_S2 = L_I5_S2, I5_out, S2, N_I5_S2, N_I5_S2
    p_I5_S2 = not_cell_stochastic(state_not_I5_S2, params_not, Omega)

    # not I5: I5_I5
    state_not_I5_I5 = L_I5_I5, I5_out, I5, N_I5_I5, N_I5_I5
    p_I5_I5 = not_cell_stochastic(state_not_I5_I5, params_not, Omega)

    """
    I6
    """
    # not S0: I6_S0
    state_not_I6_S0 = L_I6_S0, I6_out, S0, N_I6_S0, N_I6_S0
    p_I6_S0 = not_cell_stochastic(state_not_I6_S0, params_not, Omega)

    # not S1: I6_S1
    state_not_I6_S1 = L_I6_S1, I6_out, S1, N_I6_S1, N_I6_S1
    p_I6_S1 = not_cell_stochastic(state_not_I6_S1, params_not, Omega)

    # yes S2: I6_S2
    state_yes_I6_S2 = I6_out, S2, N_I6_S2, N_I6_S2
    p_I6_S2 = yes_cell_stochastic(state_yes_I6_S2, params_yes, Omega)

    # not I6: I6_I6
    state_not_I6_I6 = L_I6_I6, I6_out, I6, N_I6_I6, N_I6_I6
    p_I6_I6 = not_cell_stochastic(state_not_I6_I6, params_not, Omega)

    """
    I7
    """
    # not S0: I7_S0
    state_not_I7_S0 = L_I7_S0, I7_out, S0, N_I7_S0, N_I7_S0
    p_I7_S0 = not_cell_stochastic(state_not_I7_S0, params_not, Omega)

    # not S1: I7_S1
    state_not_I7_S1 = L_I7_S1, I7_out, S1, N_I7_S1, N_I7_S1
    p_I7_S1 = not_cell_stochastic(state_not_I7_S1, params_not, Omega)

    # not S2: I7_S2
    state_not_I7_S2 = L_I7_S2, I7_out, S2, N_I7_S2, N_I7_S2
    p_I7_S2 = not_cell_stochastic(state_not_I7_S2, params_not, Omega)

    # not I7: I7_I7
    state_not_I7_I7 = L_I7_I7, I7_out, I7, N_I7_I7, N_I7_I7
    p_I7_I7 = not_cell_stochastic(state_not_I7_I7, params_not, Omega)

    """
    out
    """
    # not I0: I0
    state_not_I0 = L_I0, out, I0_out, N_I0, N_I0
    p_I0 = not_cell_stochastic(state_not_I0, params_not, Omega)

    # not I1: I1
    state_not_I1 = L_I1, out, I1_out, N_I1, N_I1
    p_I1 = not_cell_stochastic(state_not_I1, params_not, Omega)

    # not I2: I2
    state_not_I2 = L_I2, out, I2_out, N_I2, N_I2
    p_I2 = not_cell_stochastic(state_not_I2, params_not, Omega)

    # not I3: I3
    state_not_I3 = L_I3, out, I3_out, N_I3, N_I3
    p_I3 = not_cell_stochastic(state_not_I3, params_not, Omega)

    # not I4: I4
    state_not_I4 = L_I4, out, I4_out, N_I4, N_I4
    p_I4 = not_cell_stochastic(state_not_I4, params_not, Omega)

    # not I5: I5
    state_not_I5 = L_I5, out, I5_out, N_I5, N_I5
    p_I5 = not_cell_stochastic(state_not_I5, params_not, Omega)

    # not I6: I6
    state_not_I6 = L_I6, out, I6_out, N_I6, N_I6
    p_I6 = not_cell_stochastic(state_not_I6, params_not, Omega)

    # not I7: I7
    state_not_I7 = L_I7, out, I7_out, N_I7, N_I7
    p_I7 = not_cell_stochastic(state_not_I7, params_not, Omega)

    return (p_I0_S0 + p_I0_S1 + p_I0_S2 + p_I0_I0 +
            p_I1_S0 + p_I1_S1 + p_I1_S2 + p_I1_I1 +
            p_I2_S0 + p_I2_S1 + p_I2_S2 + p_I2_I2 +
            p_I3_S0 + p_I3_S1 + p_I3_S2 + p_I3_I3 +
            p_I4_S0 + p_I4_S1 + p_I4_S2 + p_I4_I4 +
            p_I5_S0 + p_I5_S1 + p_I5_S2 + p_I5_I5 +
            p_I6_S0 + p_I6_S1 + p_I6_S2 + p_I6_I6 +
            p_I7_S0 + p_I7_S1 + p_I7_S2 + p_I7_I7 +
            p_I0 + p_I1 + p_I2 + p_I3 + p_I4 + p_I5 + p_I6 + p_I7)


def CLB_model_MUX_8_1_ode(state, T, params):
    
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b, rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b = params
  
    """
    latches
    """
    #########
    # params

    # set params for symmetric toggle switch topology
    gamma_L_Y, theta_L_Y = gamma_L_X, theta_L_X
    n_x, m_y = n_y, m_x
    eta_y, omega_y = eta_x, omega_x

    params_toggle = [delta_L, gamma_L_X, gamma_L_Y, n_x, n_y, theta_L_X, theta_L_Y, eta_x, eta_y, omega_x, omega_y, m_x, m_y, delta_x, delta_y, rho_x, rho_y, r_X, r_Y]

    # degradation rates for induction of switches are specific for each toggle switch    
    params_toggle_I0 = params_toggle.copy()
    params_toggle_I0[-4:-2] = rho_I0_a, rho_I0_b
    params_toggle_I1 = params_toggle.copy()
    params_toggle_I1[-4:-2] = rho_I1_a, rho_I1_b
    params_toggle_I2 = params_toggle.copy()
    params_toggle_I2[-4:-2] = rho_I2_a, rho_I2_b
    params_toggle_I3 = params_toggle.copy()
    params_toggle_I3[-4:-2] = rho_I3_a, rho_I3_b
    params_toggle_I4 = params_toggle.copy()
    params_toggle_I4[-4:-2] = rho_I4_a, rho_I4_b
    params_toggle_I5 = params_toggle.copy()
    params_toggle_I5[-4:-2] = rho_I5_a, rho_I5_b
    params_toggle_I6 = params_toggle.copy()
    params_toggle_I6[-4:-2] = rho_I6_a, rho_I6_b
    params_toggle_I7 = params_toggle.copy()
    params_toggle_I7[-4:-2] = rho_I7_a, rho_I7_b

    #########
    # states
    
    # latch I0
    I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b = state[:6]
    state_toggle_IO = I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b

    # latch I1
    I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b = state[6:12]
    state_toggle_I1 = I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b

    # latch I2
    I2_L_A, I2_L_B, I2_a, I2_b, I2_N_a, I2_N_b = state[12:18]
    state_toggle_I2 = I2_L_A, I2_L_B, I2_a, I2_b, I2_N_a, I2_N_b

    # latch I3
    I3_L_A, I3_L_B, I3_a, I3_b, I3_N_a, I3_N_b = state[18:24]
    state_toggle_I3 = I3_L_A, I3_L_B, I3_a, I3_b, I3_N_a, I3_N_b

    # latch I4
    I4_L_A, I4_L_B, I4_a, I4_b, I4_N_a, I4_N_b = state[24:30]
    state_toggle_I4 = I4_L_A, I4_L_B, I4_a, I4_b, I4_N_a, I4_N_b

    # latch I5
    I5_L_A, I5_L_B, I5_a, I5_b, I5_N_a, I5_N_b = state[30:36]
    state_toggle_I5 = I5_L_A, I5_L_B, I5_a, I5_b, I5_N_a, I5_N_b

    # latch I6
    I6_L_A, I6_L_B, I6_a, I6_b, I6_N_a, I6_N_b = state[36:42]
    state_toggle_I6 = I6_L_A, I6_L_B, I6_a, I6_b, I6_N_a, I6_N_b

    # latch I7
    I7_L_A, I7_L_B, I7_a, I7_b, I7_N_a, I7_N_b = state[42:48]
    state_toggle_I7 = I7_L_A, I7_L_B, I7_a, I7_b, I7_N_a, I7_N_b

    #########
    # models
    dstate_toggle_IO = toggle_model(state_toggle_IO, T, params_toggle_I0)
    dstate_toggle_I1 = toggle_model(state_toggle_I1, T, params_toggle_I1)
    dstate_toggle_I2 = toggle_model(state_toggle_I2, T, params_toggle_I2)
    dstate_toggle_I3 = toggle_model(state_toggle_I3, T, params_toggle_I3)
    dstate_toggle_I4 = toggle_model(state_toggle_I4, T, params_toggle_I4)
    dstate_toggle_I5 = toggle_model(state_toggle_I5, T, params_toggle_I5)
    dstate_toggle_I6 = toggle_model(state_toggle_I6, T, params_toggle_I6)
    dstate_toggle_I7 = toggle_model(state_toggle_I7, T, params_toggle_I7)

    dstate_toggles = np.append(np.append(np.append(np.append(np.append(np.append(np.append(dstate_toggle_IO, dstate_toggle_I1, axis=0), dstate_toggle_I2, axis = 0), dstate_toggle_I3, axis = 0), dstate_toggle_I4, axis = 0), dstate_toggle_I5, axis = 0), dstate_toggle_I6, axis = 0), dstate_toggle_I7, axis = 0)

    """
    mux 8/1
    """
    #########
    # params
    params_mux = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X

    #########
    # state
    I0, I1, I2, I3, I4, I5, I6, I7 = I0_a, I1_a, I2_a, I3_a, I4_a, I5_a, I6_a, I7_a
    state_mux = np.append([I0, I1, I2, I3, I4, I5, I6, I7], state[48:], axis=0)

    ########
    # model
    dstate_mux = MUX_8_1_model_ode(state_mux, T, params_mux)
    dstate_mux = dstate_mux[8:] # ignore dI0, dI1, dI2, dI3, dI4, dI5, dI6, dI7

    """
    return
    """
    dstate = np.append(dstate_toggles, dstate_mux, axis = 0)
    return dstate

def CLB_8_model_stohastic(state, params, Omega):
    
    delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b, rho_I4_a, rho_I4_b, rho_I5_a, rho_I5_b, rho_I6_a, rho_I6_b, rho_I7_a, rho_I7_b = params
  
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
    params_toggle_I2 =  params_toggle.copy()
    params_toggle_I2[-4:-2] = rho_I2_a, rho_I2_b
    params_toggle_I3 =  params_toggle.copy()
    params_toggle_I3[-4:-2] = rho_I3_a, rho_I3_b
    params_toggle_I4 =  params_toggle.copy()
    params_toggle_I4[-4:-2] = rho_I4_a, rho_I4_b
    params_toggle_I5 =  params_toggle.copy()
    params_toggle_I5[-4:-2] = rho_I5_a, rho_I5_b
    params_toggle_I6 =  params_toggle.copy()
    params_toggle_I6[-4:-2] = rho_I6_a, rho_I6_b
    params_toggle_I7 =  params_toggle.copy()
    params_toggle_I7[-4:-2] = rho_I7_a, rho_I7_b

    #########
    # states
    
    # latch I0
    I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b = state[:6]
    state_toggle_I0 = I0_L_A, I0_L_B, I0_a, I0_b, I0_N_a, I0_N_b

    # latch I1
    I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b = state[6:12]
    state_toggle_I1 = I1_L_A, I1_L_B, I1_a, I1_b, I1_N_a, I1_N_b

    # latch I2
    I2_L_A, I2_L_B, I2_a, I2_b, I2_N_a, I2_N_b = state[12:18]
    state_toggle_I2 = I2_L_A, I2_L_B, I2_a, I2_b, I2_N_a, I2_N_b

    # latch I3
    I3_L_A, I3_L_B, I3_a, I3_b, I3_N_a, I3_N_b = state[18:24]
    state_toggle_I3 = I3_L_A, I3_L_B, I3_a, I3_b, I3_N_a, I3_N_b

    # latch I4
    I4_L_A, I4_L_B, I4_a, I4_b, I4_N_a, I4_N_b = state[24:30]
    state_toggle_I4 = I4_L_A, I4_L_B, I4_a, I4_b, I4_N_a, I4_N_b

    # latch I5
    I5_L_A, I5_L_B, I5_a, I5_b, I5_N_a, I5_N_b = state[30:36]
    state_toggle_I5 = I5_L_A, I5_L_B, I5_a, I5_b, I5_N_a, I5_N_b

    # latch I6
    I6_L_A, I6_L_B, I6_a, I6_b, I6_N_a, I6_N_b = state[36:42]
    state_toggle_I6 = I6_L_A, I6_L_B, I6_a, I6_b, I6_N_a, I6_N_b

    # latch I7
    I7_L_A, I7_L_B, I7_a, I7_b, I7_N_a, I7_N_b = state[42:48]
    state_toggle_I7 = I7_L_A, I7_L_B, I7_a, I7_b, I7_N_a, I7_N_b

    #########
    # models
    p_toggle_I0 = toggle_model_stochastic(state_toggle_I0, params_toggle_I0, Omega)
    p_toggle_I1 = toggle_model_stochastic(state_toggle_I1, params_toggle_I1, Omega)
    p_toggle_I2 = toggle_model_stochastic(state_toggle_I2, params_toggle_I2, Omega)
    p_toggle_I3 = toggle_model_stochastic(state_toggle_I3, params_toggle_I3, Omega)
    p_toggle_I4 = toggle_model_stochastic(state_toggle_I4, params_toggle_I4, Omega)
    p_toggle_I5 = toggle_model_stochastic(state_toggle_I5, params_toggle_I5, Omega)
    p_toggle_I6 = toggle_model_stochastic(state_toggle_I6, params_toggle_I6, Omega)
    p_toggle_I7 = toggle_model_stochastic(state_toggle_I7, params_toggle_I7, Omega)

    
    """
    mux 8/1
    """
    #########
    # params
    params_mux = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X

    #########
    # state
    I0, I1, I2, I3, I4, I5, I6, I7 = I0_a, I1_a, I2_a, I3_a, I4_a, I5_a, I6_a, I7_a
    state_mux = np.append([I0, I1, I2, I3, I4, I5, I6, I7], state[48:], axis=0)

    ########
    # model
    p_mux = MUX_8_1_model_ssa(state_mux, params_mux, Omega)

    # print("P MUX = ", p_mux)
    """
    return
    """
    return (p_toggle_I0 + p_toggle_I1 + p_toggle_I2 + p_toggle_I3 + p_toggle_I4 + p_toggle_I5 + p_toggle_I6 + p_toggle_I7 + p_mux)

def CLB_model_MUX_8_1_ODE(T, state, params):
    return CLB_model_MUX_8_1_ode(state, T, params)