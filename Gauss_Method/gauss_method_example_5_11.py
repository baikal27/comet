import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3
from matplotlib import animation

def deg2rad(deg):
	deg = np.array(deg)
	return deg*np.pi/180.
def rad2deg(rad):
	rad = np.array(rad)
	return rad*180./np.pi

# given the para
time = [0, 118.10, 237.58] # sec
ra = [43.537, 54.420, 64.318] # deg
ra = deg2rad(ra)
dec = [-8.7833, -12.074, -15.105] # deg
dec = deg2rad(dec)
lst = [44.506, 45.0, 45.499] # deg
lst = deg2rad(lst)
lat = 40.0 # deg
lat = deg2rad(lat)
height = 1. # km

global mu, tau1, tau3, tau
# recalling the para
mu = 398600 # km3/sec2
Re = 6378 # km ; the equatorial radius of the earth
f = 0.003353 # unitless; the flattening factor

def cal_R(Re, f, height, lat, lst): # calculation for position vector of observer R at each observation time
	RI = (Re / np.sqrt(1 - (2 * f - f ** 2) * np.sin(lat) ** 2) + height) * np.cos(lat) * (np.cos(lst))
	RJ = (Re / np.sqrt(1 - (2 * f - f ** 2) * np.sin(lat) ** 2) + height) * np.cos(lat) * (np.sin(lst))
	RK = (Re * (1-f)**2 / np.sqrt(1 - (2 * f - f ** 2) * np.sin(lat) ** 2) + height) * np.sin(lat)
	return np.round_(RI, 1), np.round_(RJ, 1), np.round_(np.array(RK), 1)

RI, RJ, RK = cal_R(Re, f, height, lat, lst)

R1 = np.array([RI[0], RJ[0], RK])
R2 = np.array([RI[1], RJ[1], RK])
R3 = np.array([RI[2], RJ[2], RK])
global R_list
R_list = [R1, R2, R3]

print(f'R_list = {R_list}')

def cal_direction_cosine(ra, dec): # calculation for direction cosine of target at each observation time
	rhoI = np.cos(dec)*np.cos(ra)
	rhoJ = np.cos(dec)*np.sin(ra)
	rhoK = np.sin(dec)
	return rhoI, rhoJ, rhoK

rhoI, rhoJ, rhoK = cal_direction_cosine(ra, dec)
rho = ['rho1', 'rho2', 'rho3']

for i in range(len(rhoI)):
	globals()['rho{}'.format(i+1)] = np.round_(np.array([rhoI[i], rhoJ[i], rhoK[i]]), 5)
rho1 = rho1
rho2 = rho2
rho3 = rho3

global rho_list
rho_list = [rho1, rho2, rho3]
#print(rho1, rho2, rho3)
print(f'rho: {rho1}, {rho2}, {rho3}')

#1 calculate the time intervals tau1, tau3, and tau
tau1 = np.round_(time[0] - time[1], 2)
tau3 = np.round_(time[2] - time[1], 2)
tau =  np.round_(time[2] - time[0], 2)
print(f'tau: {tau1}, {tau3}, {tau}')

#2 calculate the cross products p1 = pho2^ X pho3^, p2 = pho1^ X pho3^ and p3 = pho1^ X pho2^
p1 = np.round_(np.cross(rho2, rho3), 6)
p2 = np.round_(np.cross(rho1, rho3), 6)
p3 = np.round_(np.cross(rho1, rho2), 6)
p_list = [p1, p2, p3]
print(f'p_list = {p_list}')

#3 calculate D0 = rho1^ dot p1
D0 = np.round_(np.dot(rho1, p1), 7)

#4 calculate D11, D12, D13, D21, D22, D23, D31, D32, D33; D11 = R1 dot p1, D23 = R2 dot p3, D31 = R1 dot p1
for i in range(3):
	for j in range(3):
		globals()['D{}{}'.format(i+1,j+1)] = np.round_(np.dot(R_list[i], p_list[j]), 2)
D11 = D11
D12 = D12
D13 = D13
D21 = D21
D22 = D22
D23 = D23
D31 = D31
D32 = D32
D33 = D33
D_list = [D0, D11, D12, D13, D21, D22, D23, D31, D32, D33]
print(f'D_list: {D_list}')

#5 calculate A, B
A = 1 / D0 * (-D12*tau3/tau + D22 + D32*tau1/tau)
AA = (-1646.5*119.47/237.58 + 1651.5 + 1656.6*(-118.10)/237.58) / -0.0015198
print(-D12, tau3, tau, D22, D32, tau1)

#print(D0, -D12, tau3, tau, D22, D32, tau1, tau)
B = 1 / (6*D0) * (D12*(tau3**2 - tau**2)*tau3/tau + D32*(tau**2 - tau1**2)*tau1/tau)
#print(D0, D12, tau3, tau, D32, tau, tau1)
print(A, AA, B)

#6 calculate E
E = np.dot(R2, rho2)
print(E)

#7 calculate a, b, and c
a = -(A**2 + 2*A*E + np.dot(R2, R2))
b = -2*mu*B*(A + E)
c = -mu**2*B**2
print(a, b, c)

#8 find the roots using Newton's method, F = x8 + ax6 + bx3 + c for x > 0
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
x = np.linspace(0, 10000, 10000)

def f(x, a, b, c):
	type(a)
	return (x**8 + a*x**6 + b*x**3 + c)

def ff(x, a, b, c):
	return (8*x**7 + 6*a*x**5 + 3*b*x**2)

fx = f(x, a, b, c)

ax1.plot(x, fx)
ax1.plot(x, [0 for i in range(len(x))], 'r--')
#plt.show()

root = optimize.newton(f, 9000, fprime=ff, args=(a,b,c))
r2 = root
print(f'r2 = {root}')

#9 calculate rho1, rho2, and rho3
rho1_val = 1 / D0 * ((6*(D31*tau1/tau3 + D21*tau/tau3)*r2**3 + mu*D31*(tau**2 - tau1**2)*tau1/tau3) / (6*r2**3 + mu*(tau**2-tau3**2)) - D11)
rho2_val = A + (mu*B/r2**3)
rho3_val = 1 / D0 * ((6*(D13*tau3/tau1 - D23*tau/tau1)*r2**3 + mu*D13*(tau**2 - tau3**2)*tau3/tau1) / (6*r2**3 + mu*(tau**2-tau1**2)) - D33)
print(rho1_val, rho2_val, rho3_val)

#10 calculate state vector r1, r2, and r3
r1_vect = R1 + rho1_val*rho1
r2_vect = R2 + rho2_val*rho2
r3_vect = R3 + rho3_val*rho3
r_vect_list = [r1_vect, r2_vect, r3_vect]
print('r1_r2_r3_vect')
print(r1_vect, r2_vect, r3_vect)

#11 calculate the lagrange coefficients f1, g1, f3, and g3
f1 = 1 - 1/2*mu*tau1**2/r2**3
f3 = 1 - 1/2*mu*tau3**2/r2**3
g1 = tau1 - 1/6*mu*tau1**3/r2**3
g3 = tau3 - 1/6*mu*tau3**3/r2**3
#print(f1, f3, g1, g3)

#12 calculate v2
v2_vect = 1 / (f1*g3 - f3*g1) * (-f3*r1_vect + f1*r3_vect)
print('v2_vect')
print(v2_vect)

#13 obtain the orbital elements with r2 and v2
def cal_orbit_ele(r_vect, v_vect):
	r = np.linalg.norm(r_vect) # km
	v = np.linalg.norm(v_vect) # km/s
	vradial = np.dot(r_vect, v_vect) / r  # km/s
	h_vect = np.cross(r_vect, v_vect) # km2/s
	h = np.sqrt(np.dot(h_vect, h_vect)) # km2/s
	i = np.arccos(h_vect[2]/h) # rad
	N_vect = np.cross(np.array([0,0,1]), h_vect) # km2/s
	N = np.sqrt(np.dot(N_vect, N_vect)) # km/s
	if N_vect[1] >= 0:
		Omega = np.arccos(N_vect[0] / N) # rad
	else:
		Omega = 2*np.pi - np.arccos(N_vect[0] / N)
	e_vect = 1 / mu * ((v**2 - mu/r)*r_vect - r*vradial*v_vect) # unitless
	e = np.sqrt(np.dot(e_vect, e_vect)) # unitless
	if e_vect[2] >= 0:
		omega = np.arccos(np.dot(N_vect, e_vect) / (N*e)) # rad
	else:
		omega = 2*np.pi - np.arccos(np.dot(N_vect, e_vect) / (N*e))
	if vradial >= 0 :
		theta = np.arccos(np.dot(e_vect, r_vect) / (e*r)) # rad
		theta1 = np.arccos(1/e*(h**2/(mu*r) -1))
	else:
		theta = 2*np.pi - np.arccos(np.dot(e_vect, r_vect) / (e * r))
		theta1 = 2*np.pi - np.arccos(1 / e * (h ** 2 / (mu * r) - 1))
	#print(rad2deg(i), h, e, rad2deg(Omega), rad2deg(omega), rad2deg(theta))
	print('i, h, e, Omega, omega, theta')
	print(i, h, e, Omega, omega, theta)

	return (i, h, e, Omega, omega, theta)

def sec2hour(sec):
	hour = sec / 3600
	return hour

#r_test = np.array([-6045, -3490, 2500])
#v_test = np.array([-3.457, 6.618, 2.533])
#elements = cal_orbit_ele(r_test, v_test)
elements = cal_orbit_ele(r2_vect, v2_vect)
print('i(deg), h(km2/s), e(unitless), Omega(rad), omega(rad), theta(rad)')
print('print i')
print(elements)

#14 calculate the true anomaly, the Eccentric anomaly, the Mean anomaly with time
def cal_kepler_equation(elements):
	incl, h, e, Omega, omega, theta = elements
	rp = h**2/mu/(1+e*np.cos(0.)) # km
	ra = h**2/mu/(1+e*np.cos(np.pi)) # km
	a = 1/2*(rp + ra) # semi_major km
	T = 2*np.pi/np.sqrt(mu)*a**(3/2) #sec

	t = np.linspace(0, T, 1000)
	M = 2*np.pi / T * t

	def fE(Ex, M, q):
		return Ex - e*np.sin(Ex) - M
	def fEprime(Ex, M, q):
		return 1 - e*np.cos(Ex)

	E_list = []
	for m in M:
		print(m)
		root = optimize.newton(fE, m+1, fprime=fEprime, args=(m, m+1))
		E_list.append(root)
	E = np.array(E_list)
	tH_rad = 2 * np.arctan(np.sqrt((1+e) / (1-e)) * np.tan(E / 2))
	tH_deg = rad2deg(tH_rad)
	tH_deg = np.array([360 + i if i < 0 else i for i in tH_deg])

	if e == 0:
		r = np.full_like(t, h**2/mu)
	elif 0 < e < 1:
		r = a * (1 - e**2) / (1 + e*np.cos(tH_rad))
	elif e == 1:
		r = h**2 / mu / (1 + np.cos(tH_rad))
	elif e > 1:
		r = a * (e**2 - 1) / (1 + e*np.cos(tH_rad))
		r2 = h**2 / mu / (1 + np.cos(tH_rad))
	else:
		print('Do you know e < 0 case?')
		breakpoint()

	x = r * np.cos(tH_rad)
	y = r * np.sin(tH_rad)
	z = np.zeros_like(x, dtype=np.float)
	#xyz = np.row_stack((x, y, z))

	Ome = [[np.cos(Omega), np.sin(Omega), 0], [-np.sin(Omega), np.cos(Omega), 0], [0, 0, 1]]
	inclin = [[1, 0, 0], [0, np.cos(incl), np.sin(incl)], [0, -np.sin(incl), np.cos(incl)]]
	ome = [[np.cos(omega), np.sin(omega), 0], [-np.sin(omega), np.cos(omega), 0], [0, 0, 1]]
	trans_matrix_Omega = np.array(Ome)
	trans_matrix_incl = np.array(inclin)
	trans_matrix_omega = np.array(ome)
	trans_matrix_Xx = np.dot(trans_matrix_omega, np.dot(trans_matrix_incl, trans_matrix_Omega))
	trans_matrix_xX = trans_matrix_Xx.T

	data = []
	for u, w, z in zip(x, y, z):
		tranxX = np.dot(trans_matrix_xX, np.array([u, w, z]))
		#print(type(tran))
		data.append(list(tranxX))
	data = np.array(data).T

	fig = plt.figure()
	ax = plt3.Axes3D(fig)

	def update(num, data, position):
		position.set_data(data[:2, num:num + 1])
		position.set_3d_properties(data[2, num:num + 1])

	position, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1], 'ro')
	ax.plot(data[0, :], data[1, :], data[2, :])

	N = len(data[0, :])

	# Setting the axes properties
	#ax.set_xlim3d([-1.0, 1.0])
	ax.set_xlabel('X')

	#ax.set_ylim3d([-1.0, 1.0])
	ax.set_ylabel('Y')

	#ax.set_zlim3d([0.0, 10.0])
	ax.set_zlabel('Z')

	ani = animation.FuncAnimation(fig, update, N, fargs=(data, position), interval=5000 / N, blit=False)
	#ani.save('matplot003.gif', writer='imagemagick')
	plt.show()
	print('end')

cal_kepler_equation(elements) # km, secs

#15 transformation coordinate from orbit own to the Geocentric Equatorial Coordinate


#13-2 improve the preliminary estimate of the orbit
def f_xi(xi, recipro_semajor, r, vradial, dt):
	z = recipro_semajor * xi ** 2
	if z > 0:
		C = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
		S = (1 - np.sqrt(np.cos(z))) / z
	elif z < 0:
		C = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3
		S = (np.cosh(np.sqrt(-z)) - 1) / (-z)
	else:
		C = 1 / 6
		S = 1 / 2
	f = r*vradial/np.sqrt(mu)*xi**2*C + (1 - recipro_semajor*r)*xi**3*S + r*xi - np.sqrt(mu)*dt
	return f

def fprime_xi(xi, recipro_semajor, r, vradial, dt):
	z = recipro_semajor * xi ** 2
	if z > 0:
		C = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
		S = (1 - np.sqrt(np.cos(z))) / z
	elif z < 0:
		C = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3
		S = (np.cosh(np.sqrt(-z)) - 1) / (-z)
	else:
		C = 1 / 6
		S = 1 / 2
	fprime = r*vradial/np.sqrt(mu)*xi*(1 - recipro_semajor*xi**2*S) + (1 - recipro_semajor*r)*xi**2*C + r
	return fprime

def cal_Lagrange_coeffcients(recipro_semajor, xi, r2, dt):
	z = recipro_semajor*xi**2
	if z > 0:
		C = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)) ** 3
		S = (1 - np.sqrt(np.cos(z))) / z
	elif z < 0:
		C = (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)) ** 3
		S = (np.cosh(np.sqrt(-z)) - 1) / (-z)
	else:
		C = 1 / 6
		S = 1 / 2
	f = 1 - xi**2/r2*C # unitless
	g = dt - 1/ np.sqrt(mu)*xi**3*S # seconds
	return f, g

def improve(r2_vect, v2_vect):
	r2 = np.linalg.norm(r2_vect) # km
	v2 = np.linalg.norm(v2_vect) # km/s
	recipro_semajor = 2/r2 - v2**2/mu # km-1
	v2radial = np.dot(v2_vect, r2_vect)/r2 # km/s
	xi1_init = np.sqrt(mu)*np.abs(recipro_semajor)*tau1
	xi3_init = np.sqrt(mu)*np.abs(recipro_semajor)*tau3
	print(r2, v2, recipro_semajor)

	xi1 = optimize.newton(f_xi, xi1_init, fprime=fprime_xi, args=(recipro_semajor, r2, v2radial, tau1))
	xi3 = optimize.newton(f_xi, xi3_init, fprime=fprime_xi, args=(recipro_semajor, r2, v2radial, tau3))
	print(xi1, xi3)

	f1_update, g1_update = cal_Lagrange_coeffcients(recipro_semajor, xi1, r2, tau1) # unitless, seconds
	f3_update, g3_update = cal_Lagrange_coeffcients(recipro_semajor, xi3, r2, tau3)
	print(f1_update, g1_update, f3_update, g3_update)

	c1 = g3_update / (f1_update*g3_update - f3_update*g1_update)
	c3 = -g1_update / (f1_update*g3_update - f3_update*g1_update)
	print(c1, c3)

	rho1_val_update = 1 / D0 * (-D11 + D21/c1 - c3/c1*D31)
	rho2_val_update = 1 / D0 * (-c1*D12 + D22 - c3*D32)
	rho3_val_update = 1 / D0 * (-c1/c3*D13 + 1/c3*D23 - D33)
	print(rho1_val_update, rho2_val_update, rho3_val_update)
	print(1/(-0.0015198)*(-782.15 + 784.72/0.50467 - 0.4989/0.50467*787.31))
	print(1 / (-0.0015198) * (-782.15 + (784.72 / c1) - c3 / c1 * 787.31))
	print(c1, c3)
	print(D0)

	r1_vect_update = R1 + rho1_val_update*rho1
	r2_vect_update = R2 + rho2_val_update*rho2
	r3_vect_update = R3 + rho3_val_update*rho3

	r_vect_list_update = [r1_vect_update, r2_vect_update, r3_vect_update]
	print(r1_vect_update, r2_vect_update, r3_vect_update)
	breakpoint()
	v2_vect_update = 1 / (f1_update*g3_update - f3_update*g1_update) * (-f3_update*r1_vect_update + f1_update*r3_vect_update)

	return rho1_val_update, rho2_val_update, rho3_val_update, r2_vect_update, v2_vect_update


'''
try:
	threadsh = 10e-3
	diff_rho1 = 10000
	diffrho2 = 10000
	diffrho3 = 10000
	while True:
		if diff_rho1 > threadsh:
			rho1_update, rho2_update, rho3_update, r2_update, v2_update =improve(r2_vect, v2_vect)
			diff_rho1 = abs(np.linalg.norm(rho1_update - rho1_val))
			diff_rho2 = abs(np.linalg.norm(rho2_update - rho2_val))
			diff_rho3 = abs(np.linalg.norm(rho3_update - rho3_val))
			r2_vect = r2_update
			v2_vect = v2_update
			print(r2_vect, v2_vect)
			print(rho1_val)
		else:
			break
except KeyboardInterrupt as e:
	print(e)

finally:
	print('The End')
'''

