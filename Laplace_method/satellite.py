import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation, SkyCoord, GCRS
import astropy.units as u
from astropy.time import Time
import sympy
from scipy import optimize
import os
from astropy.io import fits

''' geocentric from jpl-horizons 2020-04-22 to 2020-07-21, asteroid 5 Astraea
# ra with HMS, dec with DMS
---Time: UT-----  ---R.A.__(ICRF)__DEC---  ---R.A.__(a-apparent)__DEC--(ref. frame defined by the Earth equator)
2020-04-22 00:00  08 27 26.03 +20 24 07.9  08 28 35.32 +20 20 07.6
2020-05-22 00:00  09 15 47.25 +17 58 11.6  09 16 54.14 +17 53 10.4
2020-06-22 00:00  10 11 56.52 +13 48 53.4  10 13 00.76 +13 42 57.6
'''
'''seoul from jpl-horizons 2020-04-22 to 2020-07-21, asteroid 5 Astraea
---Time: UT-----  ---R.A.__(ICRF)__DEC---  ---R.A.__(a-apparent)__DEC--
2020-04-22 00:00  08 27 26.17 +20 24 04.0 08 28 35.45 +20 20 03.6
2020-05-22 00:00  09 15 47.42 +17 58 08.6 09 16 54.30 +17 53 07.2
2020-06-22 00:00  10 11 56.70 +13 48 51.0 10 13 00.92 +13 42 55.2
'''
'''(DOAO)topo-site from jpl-horizons 2020-04-22 to 2020-07-21, asteroid 5 Astraea
---Time: UT-----  ---R.A.__(ICRF)__DEC---  ---R.A.__(a-apparent)__DEC--
2020-04-22 00:00  08 27 26.18 +20 24 04.1 08 28 35.46 +20 20 03.7
2020-05-22 00:00  09 15 47.43 +17 58 08.7 09 16 54.31 +17 53 07.3
2020-06-22 00:00  10 11 56.71 +13 48 51.1 10 13 00.93 +13 42 55.3
'''

DOAO_site = {'lon_dms':'127d26m49s', 'lat_dms':'34d41m34s', 'height_m':81}
DOAO_site2 = {'lon_deg':127.44694444444447, 'lat_deg':34.692777777777785, 'height_m':81}
# radec_obs is the apparent RA-DEC of 5 Astraea at DOAO topo-site.
time_obs = ['2020-04-22 00:00', '2020-05-22 00:00', '2020-06-22 00:00']
radec_obs = ['08 27 26.03 +20 24 07.9', '09 15 47.25 +17 58 11.6', '10 11 56.52 +13 48 53.4']

coo = SkyCoord(radec_obs, unit=(u.hourangle, u.deg))

t = Time(time_obs, format='iso', scale='utc')
#print(f't: {t}')
loc = EarthLocation(DOAO_site['lon_dms'], DOAO_site['lat_dms'], DOAO_site['height_m']*u.m)
#print(loc.geocentric)

lapc_t = t[1]
lapc_R = loc.get_gcrs_posvel(lapc_t)[0] #(cartesian, m)
lapc_Rdot = loc.get_gcrs_posvel(lapc_t)[1] #(cartesian, m/s)
print(f'laplace time: {lapc_t}')
print(f'laplace_R: {lapc_R}')
print(f'laplace_Rdot: {lapc_Rdot}')

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

L_v_list = []
for ra, dec in zip(coo.ra.deg, coo.dec.deg):
    L_v_list.append(radec_to_Lvector(ra, dec))
print(L_v_list)

# at t=t2, we have to get L_dot, L_ddot needed to all L_v_list elements
L_dot = L_1dot(L_v_list[0], L_v_list[1], L_v_list[2], t[0].jd, t[1].jd, t[2].jd)
L_ddot = L_2dot(L_v_list[0], L_v_list[1], L_v_list[2], t[0].jd, t[1].jd, t[2].jd)
print(f'Ldot_vector: {L_dot}')
print(f'Lddot_vector: {L_ddot}')

#R_dot = np.zeros_like(L_dot)
#R_ddot = np.zeros_like(L_ddot) #need to check

R_v = np.array([lapc_R.x/u.m, lapc_R.y/u.m, lapc_R.z/u.m]).reshape(3,1)
R_dot = np.array([lapc_Rdot.x/u.m*u.s, lapc_Rdot.y/u.m*u.s, lapc_Rdot.z/u.m*u.s]).reshape(3,1)
R_ddot = np.zeros_like(R_v)
print(R_v) # unit: m
print(R_dot) # unit: m/s
print(R_ddot)

D = determ(L_v_list[1], L_dot, L_ddot)
D1 = determ1(L_v_list[1], L_dot, R_ddot)
D2 = determ2(L_v_list[1], L_dot, R_v)
D3 = determ1(L_v_list[1], R_ddot, L_ddot)
D4 = determ1(L_v_list[1], R_v, L_ddot)
print(D, D1, D2, D3, D4,)

L_v = L_v_list[1]

mu = 1.
rho_scalar = sympy.Symbol('rho_scalar')
r_scalar = sympy.Symbol('r_scalar', positive=True)

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








