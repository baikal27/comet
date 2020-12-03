import numpy as np

a = 0.3 # au
e = 0.5 # unitless
#tau = #perihelian passagetime
#G =
#M =
#m =
#mu = G*(M+m)
mu = 1
n = np.sqrt(mu) / (a**(3/2))
P = 2*np.pi/n
t = np.linspace(0, P, 1000)
M = n*t

#M = E - e*np.sin(E)