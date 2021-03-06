from scipy.integrate import ode
import matplotlib.pyplot as plt

from models_population import *
from parameters import *
from parameters_population import *

rho_x = 0
rho_y = 0



N_cells = N_I0_S0, N_I0_S1, N_I0_I0, N_I1_S0, N_I1_S1, N_I1_I1, N_I2_S0, N_I2_S1, N_I2_I2, N_I3_S0, N_I3_S1, N_I3_I3, N_I0, N_I1, N_I2, N_I3


params = delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x,
params += tuple(N_cells)


# I0, I1, I2, I3
I = np.array([0, 0, 0, 1])
# S0, S1
S = np.array([1, 1])



# simulation parameters
t_end = 1500
N = t_end


Y0 = np.zeros(23)
#Y0[22:38] = 1 # number of cells

Y0[:4] = I
Y0[4:6] = S


T = np.linspace(0, t_end, N)

t1 = t_end
dt = t_end/N
T = np.arange(0,t1+dt,dt)
Y = np.zeros([1+N,len(Y0)])
Y[0,:] = Y0

r = ode(MUX_4_1_model_ODE).set_integrator('zvode', method='bdf')
r.set_initial_value(Y0, T[0]).set_f_params(params)

i = 1
while r.successful() and r.t < t1:
    Y[i,:] = r.integrate(r.t+dt)
    i += 1

out = Y[:,-1]
plt.plot(T,out)
plt.show()


"""
# Y = a, b, N_A
a = Y[:,0]
b = Y[:,1]
N_A = Y[:,2]

ax1 = plt.subplot(211)
ax1.plot(T,a)
ax1.plot(T,b)
ax1.legend(["a", "b"])

ax2 = plt.subplot(212)
ax2.plot(T,N_A)
ax2.legend(["N_A"])


plt.show()
"""