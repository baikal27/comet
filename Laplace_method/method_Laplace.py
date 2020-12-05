import numpy as np
import sympy
from scipy import optimize
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.coordinates import Angle, Latitude, Longitude  # Angles
import astropy.units as u



# ---------------------------------------------------------------------------------
# ------------- determining the line of sight Unit vectors -------------------------
# ---------------------------------------------------------------------------------
def radec_to_Lvector(in_ra, in_dec):
	L_v = np.array([np.cos(in_dec) * np.cos(in_ra), np.cos(in_dec) * np.sin(in_ra),
	                np.sin(in_dec)]).reshape(3,1)
	return L_v

def L_1dot(L1, L2, L3, t1, t2, t3):
	L_dot = (t2 - t3) / ((t1-t2)*(t1-t3)) * L1 + (2*t2 - t1 -t3) / ((t2-t1)*(t2-t3)) * L2 \
	      + (t2 - t1) / ((t3-t1)*(t3-t2)) * L3
	return L_dot

def L_2dot(L1, L2, L3, t1, t2, t3):
	L_ddot = 2 / ((t1-t2)*(t1-t3)) * L1 + 2 / ((t2-t1)*(t2-t3)) * L2 + \
	         2 /((t3-t1)*(t3-t2)) * L3
	return L_ddot
# ---------------------------------------------------------------------------------
# ------------- Solving for the  vector r -----------------------------------------
# ---------------------------------------------------------------------------------
def determ(L, Ldot, Lddot):
	deter = np.concatenate((L, Ldot, Lddot), axis=1)
	D = np.linalg.det(deter)
	return D

def determ1(L, Ldot, Rddot):
	deter1 = np.concatenate((L, Ldot, Rddot), axis=1)
	D1 = np.linalg.det(deter1)
	return D1

def determ2(L, Ldot, R):
	deter2 = np.concatenate((L, Ldot, R), axis=1)
	D2 = np.linalg.det(deter2)
	return D2

def determ3(L, Rddot, Lddot):
	deter3 = np.concatenate((L, Rddot, Lddot), axis=1)
	D3 = np.linalg.det(deter3)
	return D3

def determ4(L, R, Lddot):
	deter4 = np.concatenate((L, R, Lddot), axis=1)
	D4 = np.linalg.det(deter4)
	return D4
#------------------------------------------------------------

mu = 1.
in_ra = [1, 2, 3]
in_dec = [11, 12, 13]
in_R_ra = [4, 5, 6] #site of observation
in_R_dec = [14, 15, 16] #site of observation
t_obs = [111, 112, 113]

L_v_list = []
R_v_list = []

for cm_ra, cm_dec, site_ra, site_dec in zip(in_ra, in_dec, in_R_ra, in_R_dec):
	L_v_list.append(radec_to_Lvector(cm_ra, cm_dec))
	R_v_list.append(radec_to_Lvector(site_ra, site_dec))

print(f'L_v_list[0]: {L_v_list[0]}')

L_dot = L_1dot(L_v_list[0], L_v_list[1], L_v_list[2], t_obs[0], t_obs[1], t_obs[2])
L_ddot = L_2dot(L_v_list[0], L_v_list[1], L_v_list[2], t_obs[0], t_obs[1], t_obs[2])
R_dot = np.zeros_like(L_dot)
R_ddot = np.zeros_like(L_ddot) #need to check

D = determ(L_v_list[1], L_dot, L_ddot)
D1 = determ1(L_v_list[1], L_dot, R_ddot)
D2 = determ2(L_v_list[1], L_dot, R_v_list[1])
D3 = determ1(L_v_list[1], R_ddot, L_ddot)
D4 = determ1(L_v_list[1], R_v_list[1], L_ddot)

'''
print(f'L_dot: {L_dot}')
print(f'L_ddot: {L_ddot}')
print(f'R_ddot: {R_ddot}')
print(f'D: {D}')
print(f'D1: {D1}')
print(f'D2: {D2}')
'''

rho_scalar = sympy.Symbol('rho_scalar')
r_scalar = sympy.Symbol('r_scalar', positive=True)

L_v = L_v_list[1]
R_v = R_v_list[1]

#r_v = rho_scalar*L_v + R_v
rho_scalar = -2*D1/D - 2*mu/(r_scalar**3)*D2/D
r_square = rho_scalar**2 + 2*rho_scalar*(np.dot(L_v.T, R_v)) + np.dot(R_v.T, R_v)
r_scalar2 = sympy.sqrt(r_square)

print(f'rho_scalar: {rho_scalar}')
print(f'r_scalar2: {r_scalar2}')

# --------------------- plot for approximation of r_scalar -------------------------
sude_r = np.linspace(-100, 100, 1000)
y1 = sude_r
y2_list = []
for i in sude_r:
	y2_ele = r_scalar2.subs(r_scalar, i)
	y2_list.append(y2_ele)
y2 = np.array(y2_list)
#print(f'y2: {y2}')

fig, ax = plt.subplots(num=1)
ax.plot(sude_r, y1, 'r-')
ax.plot(sude_r, y2, 'b--')
plt.show()

rlim = input('input integer rmin, rmax \n r1, r2 = ').split(',')
r1, r2 = rlim
r1 = int(r1.strip())
r2 = int(r2.strip())

# --------------------- bisection method for exact r_scalar ---------------------
def funct(r, D1, D, mu, D2, L_v, R_v):
	rho = -2 * D1 / D - 2 * mu / (r ** 3) * D2 / D
	return rho**2 + 2*rho*(np.dot(L_v.T, R_v)) + np.dot(R_v.T, R_v) - r**2

r = optimize.bisect(funct, r1, r2, args=(D1, D, mu, D2, L_v, R_v))
rho = rho_scalar.subs(r_scalar, r)
r_vector = rho * L_v + R_v
r_vector = np.array(r_vector).astype(np.float64)
print(f'r = {r:.6f}')
print(f'rho = {rho:.6f}')
print(f'r_vector: {r_vector}')

rhodot_scalar = -D3/D - mu/(r**3)*D4/D
rdot_vector = rhodot_scalar*L_v + rho*L_dot + R_dot
rdot_vector = np.array(rdot_vector).astype(np.float64)
print(f'rdot_vector: {rdot_vector}')

# -- from GCRS(Geocentric Celestial Reference System) to ICRS(International Celestial Reference System) ---
r_icrs = SkyCoord(r_vector, )

def rv_vec_to_6elems(r, v):
	todeg = 180 / np.pi
	torad = np.pi / 180
	mu = 1

	#r = np.array([3 * np.sqrt(3) / 4, 3 / 4, 0])
	#v = np.array([-1 / np.sqrt(8), np.sqrt(3 / 8), 1 / np.sqrt(2)])
	print(type(r[0,1]), type(v))
	print(np.linalg.norm(r))

	k = np.array([0, 0, 1])

	h = np.cross(r, v)
	e = 1 / mu * ((np.dot(v, v.T) - mu / np.linalg.norm(r)) * r - (np.dot(r, v.T) * r))
	n = np.cross(k, h)

	print('angular momentum: {}'.format(h))
	print('eccentricity: {}'.format(e))
	print('ascending node: {} \n'.format(n))

	i = np.arccos(h[2] / np.linalg.norm(h))
	omega = np.arccos(n[0] / np.linalg.norm(n))
	smallomega = np.arccos(np.dot(n, e) / (np.linalg.norm(n) * np.linalg.norm(e)))
	nuzero = np.arccos(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r)))
	uzero = np.arccos(np.dot(n, r) / (np.linalg.norm(n) * np.linalg.norm(r)))

	i = i * todeg
	omega = omega * todeg
	smallomega = smallomega * todeg
	nuzero = nuzero * todeg
	uzero = uzero * todeg

	if n[1] < 0:
		omega = 360 - omega

	if e[2] < 0:
		smallomega = 360 - smallomega

	if np.dot(r, v) < 0:
		nuzero = 360 - nuzero

	if r[2] < 0:
		uzero = 360 - uzero

	print('eccentricity: {}'.format(np.linalg.norm(e)))
	print('semi-major axis: {}'.format(np.dot(h, h) / (mu * (1 - np.dot(e, e)))))
	print('inclination: {}'.format(i))
	print('omega: {}'.format(omega))
	print('small omega: {}'.format(smallomega))
	print('nu0: {}'.format(nuzero))
	print('u0: {}'.format(uzero))

rv_vec_to_6elems(np.array(r_vector).T, np.array(rdot_vector).T)


R_list = [array([3489.8, 3430.2, 4078.5]), array([3460.1, 3460.1, 4078.5]), array([3429.9, 3490.1, 4078.5])]
rho: [ 0.71643  0.68074 -0.1527 ], [ 0.56897  0.79531 -0.20917], [ 0.4184   0.87008 -0.26059]
tau: -118.1, 119.48, 237.58
p_list = [array([-0.025255,  0.060751,  0.162292]), array([-0.044533,  0.122805,  0.33853 ]), array([-0.020947,  0.062974,  0.182463])]
D_list: [-0.0015198, 782.16, 1646.53, 887.09, 784.73, 1651.52, 889.59, 787.31, 1656.55, 892.11]
r1_r2_r3_vect
[6091.63785083 5902.42352299 3523.94388172] [5659.29230471 6534.14543624 3270.01263797] [5176.22568618 7121.65605408 2990.84462103]
v2_vect
[-3.85567746  5.14026247 -2.24539987]
i, h, e, Omega, omega, theta
2.6747036137846094 58311.66993185606 0.17121234628445342 4.455464041223287 0.35025820088546444 0.49646987174893026
i, h, e, Omega, omega, theta
0.5255292339007297 62751.595679016835 0.10212352018766661 4.709235920381623 1.528711903266394 0.8300620608087065


R1 = [3489.83840195 3430.17309296 4078.53953565]
R2 = [3460.13435573 3460.13435573 4078.53953565]
R3 = [3429.86853405 3490.13772773 4078.53953565]
rho
[ 0.71643  0.68074 -0.1527 ] [ 0.56897  0.79531 -0.20917] [ 0.4184   0.87008 -0.26059]
tau
-118.1 119.48 237.58
p_list
[array([-0.025255,  0.060751,  0.162292]), array([-0.044533,  0.122805,  0.33853 ]), array([-0.020947,  0.062974,  0.182463])]
[-0.0015198, 782.16, 1646.54, 887.09, 784.74, 1651.54, 889.6, 787.32, 1656.57, 892.13]
r1_r2_r3_vect
[6088.09905634 5898.99762274 3524.74586156] [5658.39806152 6532.8817903  3270.39355372] [5176.92085058 7123.20483952 2990.43159309]
v2_vect
[-3.83783667  5.16122959 -2.25052064]
i, h, e, Omega, omega, theta
(0.5269866846211149, 62796.20717094289, 0.10604075122494959, 4.706591205387281, 1.5185976617528034, 0.8424108482678727)