import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as plt3
from matplotlib import animation
import astropy.constants as const
from usefultools import deg2rad, rad2deg, sec2hour, m2au, sec2day, km2au, read_HORIZONS
from astropy.coordinates import get_body_barycentric, get_sun, solar_system_ephemeris, \
	cartesian_to_spherical, CartesianRepresentation, SphericalRepresentation
from astropy.time import Time
import re
import pandas as pd
import plotly.express as px

class orbit():
	def __init__(self, obj):
# given the para
		#global mu, tau1, tau3, tau
# recalling the para
		self.obj = obj
		assert obj in ['sat', 'planet'], print("You choose 1 argument among in ['sat', 'planet']")
		if obj == 'sat':
			self.mu = 398600 # km3/sec2
			self.Re = 6378 # km ; the equatorial radius of the earth
			self.f = 0.003353 # unitless; the flattening factor
			print(f'distance: km, time: sec')

		elif obj == 'planet':
			self.mu = const.GM_sun.value * m2au(1, 3) / sec2day(1, 2)  # au3/day2
			print(f'Distance: AU, Time: day')

		else:
			print("missing 1 argument. choose 'sat' or 'planet'")

	#def cal_R(self, Re, f, height, lat, lst): # calculation for position vector of observer R at each observation time
	def cal_r2v2_vect(self, observation_data):
		if self.obj == 'planet':
			pass
		else:
			self.lst = observation_data[3] # deg
			self.height = observation_data[5][0] # km
		self.time = observation_data[0] # deg
		self.ra = observation_data[1] # deg
		self.dec = observation_data[2] # deg
		self.lat = observation_data[4][0] # deg : all same
		#self.height = observation_data[5][0] # km

		if self.obj == 'sat':
			self.RI = (self.Re / np.sqrt(1 - (2 * self.f - self.f ** 2) * np.sin(self.lat) ** 2) + self.height) * np.cos(self.lat) * (np.cos(self.lst))
			self.RJ = (self.Re / np.sqrt(1 - (2 * self.f - self.f ** 2) * np.sin(self.lat) ** 2) + self.height) * np.cos(self.lat) * (np.sin(self.lst))
			self.RK = (self.Re * (1-self.f)**2 / np.sqrt(1 - (2 * self.f - self.f ** 2) * np.sin(self.lat) ** 2) + self.height) * np.sin(self.lat)
			#return np.round_(self.RI, 1), np.round_(self.RJ, 1), np.round_(np.array(self.RK), 1)

#RI, RJ,     RK = cal_R(Re, f, height, lat, lst)
			self.R1 = np.array([self.RI[0], self.RJ[0], self.RK])
			self.R2 = np.array([self.RI[1], self.RJ[1], self.RK])
			self.R3 = np.array([self.RI[2], self.RJ[2], self.RK])

			self.R_list = [self.R1, self.R2, self.R3]

			print(f'R1 = {self.R1}')
			print(f'R2 = {self.R2}')
			print(f'R3 = {self.R3}')
		elif self.obj == 'planet':
			solar_system_ephemeris.set('jpl')
			#t = Time(['2020-09-10', '2020-09-15', '2020-09-20'], scale='utc')
			r1, r2, r3 = get_body_barycentric('earth', self.time)
			#print(r1)
			self.R1 = km2au(np.array(r1.xyz.value))
			self.R2 = km2au(np.array(r2.xyz.value))
			self.R3 = km2au(np.array(r3.xyz.value))
			self.R_list = [self.R1, self.R2, self.R3]
			#print(f'R_list: {self.R_list}')

		else:
			print('what type of object?')
			raise AssertionError

	#def cal_direction_cosine(self): # calculation for direction cosine of target at each observation time
		rhoI = np.cos(self.dec)*np.cos(self.ra)
		rhoJ = np.cos(self.dec)*np.sin(self.ra)
		rhoK = np.sin(self.dec)
		#return rhoI, rhoJ, rhoK

	#rhoI, rhoJ, rhoK = cal_direction_cosine(ra, dec)
		rho = ['rho1', 'rho2', 'rho3']

		for i in range(len(rhoI)):
			globals()['rho{}'.format(i+1)] = np.round_(np.array([rhoI[i], rhoJ[i], rhoK[i]]), 5)
		self.rho1_dc = rho1
		self.rho2_dc = rho2
		self.rho3_dc = rho3

		self.rho_dc_list = [self.rho1_dc, self.rho2_dc, self.rho3_dc]
		print('rho')
		print(self.rho1_dc, self.rho2_dc, self.rho3_dc)

	#def cal_rho(self):
#1 calculate the time intervals tau1, tau3, and tau
		#self.jd = self.time.jd1
		self.tau1 = np.round_(self.time[0] - self.time[1], 2)
		self.tau3 = np.round_(self.time[2] - self.time[1], 2)
		self.tau =  np.round_(self.time[2] - self.time[0], 2)
		print('tau')
		print(self.tau1, self.tau3, self.tau)

#2 calculate the cross products p1 = pho2^ X pho3^, p2 = pho1^ X pho3^ and p3 = pho1^ X pho2^
		p1 = np.round_(np.cross(self.rho2_dc, self.rho3_dc), 6)
		p2 = np.round_(np.cross(self.rho1_dc, self.rho3_dc), 6)
		p3 = np.round_(np.cross(self.rho1_dc, self.rho2_dc), 6)
		p_list = [p1, p2, p3]
		print('p_list')
		print(p_list)

#3 calculate D0 = rho1^ dot p1
		self.D0 = np.round_(np.dot(self.rho1_dc, p1), 7)

#4 calculate D11, D12, D13, D21, D22, D23, D31, D32, D33; D11 = R1 dot p1, D23 = R2 dot p3, D31 = R1 dot p1
		for i in range(3):
			for j in range(3):
				globals()['D{}{}'.format(i+1,j+1)] = np.round_(np.dot(self.R_list[i], p_list[j]), 2)
		self.D11 = D11
		self.D12 = D12
		self.D13 = D13
		self.D21 = D21
		self.D22 = D22
		self.D23 = D23
		self.D31 = D31
		self.D32 = D32
		self.D33 = D33
		self.D_list = [self.D0, self.D11, self.D12, self.D13, self.D21, self.D22, self.D23,
		               self.D31, self.D32, self.D33]
		print('D_list')
		print(self.D_list)

#5 calculate A, B
		A = 1 / self.D0 * (-self.D12*self.tau3/self.tau + self.D22 + self.D32*self.tau1/self.tau)
		AA = (-1646.5*119.47/237.58 + 1651.5 + 1656.6*(-118.10)/237.58) / -0.0015198
		#print(-D12, tau3, tau, D22, D32, tau1)

#print(D0, -D12, tau3, tau, D22, D32, tau1, tau)
		B = 1 / (6*self.D0) * (self.D12*(self.tau3**2 - self.tau**2)*self.tau3/self.tau +
		                       self.D32*(self.tau**2 - self.tau1**2)*self.tau1/self.tau)
		#print(D0, D12, tau3, tau, D32, tau, tau1)
		#print(A, AA, B)

#6 calculate E
		E = np.dot(self.R2, self.rho2_dc)
		#print(E)

#7 calculate a, b, and c
		a = -(A**2 + 2*A*E + np.dot(self.R2, self.R2))
		b = -2*self.mu*B*(A + E)
		c = -self.mu**2*B**2
		#print(a, b, c)

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
		plt.show()

		appnum = int(input('Input the approximate number to root:\n'))
		root = optimize.newton(f, appnum, fprime=ff, args=(a,b,c))
		r2 = root
		print(f'root = {root}')

#9 calculate rho1, rho2, and rho3
		rho1_val = 1 / self.D0 * ((6*(self.D31*self.tau1/self.tau3 + self.D21*self.tau/self.tau3)*r2**3
		                           + self.mu*self.D31*(self.tau**2 - self.tau1**2)*self.tau1/self.tau3)
		                          / (6*r2**3 + self.mu*(self.tau**2-self.tau3**2)) - self.D11)
		rho2_val = A + (self.mu*B/r2**3)
		rho3_val = 1 / self.D0 * ((6*(self.D13*self.tau3/self.tau1 - self.D23*self.tau/self.tau1)*r2**3
		                           + self.mu*self.D13*(self.tau**2 - self.tau3**2)*self.tau3/self.tau1)
		                          / (6*r2**3 + self.mu*(self.tau**2-self.tau1**2)) - self.D33)
		#print(rho1_val, rho2_val, rho3_val)

#10 calculate state vector r1, r2, and r3
		self.r1_vect = self.R1 + rho1_val*self.rho1_dc
		self.r2_vect = self.R2 + rho2_val*self.rho2_dc
		self.r3_vect = self.R3 + rho3_val*self.rho3_dc
		self.r_vect_list = [self.r1_vect, self.r2_vect, self.r3_vect]
		print('r1_r2_r3_vect')
		print(self.r1_vect, self.r2_vect, self.r3_vect)

#11 calculate the lagrange coefficients f1, g1, f3, and g3
		f1 = 1 - 1/2*self.mu*self.tau1**2/r2**3
		f3 = 1 - 1/2*self.mu*self.tau3**2/r2**3
		g1 = self.tau1 - 1/6*self.mu*self.tau1**3/r2**3
		g3 = self.tau3 - 1/6*self.mu*self.tau3**3/r2**3
		#print(f1, f3, g1, g3)

#12 calculate v2
		self.v2_vect = 1 / (f1*g3 - f3*g1) * (-f3*self.r1_vect + f1*self.r3_vect)
		#print(type(self.v2_vect))
		#print(self.v2_vect.shape)
		return self.r2_vect, self.v2_vect

#13 obtain the orbital elements with r2 and v2
	def cal_orbit_ele(self, r2_vect, v2_vect):
		if isinstance(r2_vect, np.ndarray) and isinstance(v2_vect, np.ndarray):
			pass
		else:
			raise TypeError
		r = np.linalg.norm(r2_vect) # km
		v = np.linalg.norm(v2_vect) # km/s
		vradial = np.dot(r2_vect, v2_vect) / r  # km/s
		h_vect = np.cross(r2_vect, v2_vect) # km2/s
		self.h = np.sqrt(np.dot(h_vect, h_vect)) # km2/s
		self.i = np.arccos(h_vect[2]/self.h) # rad
		N_vect = np.cross(np.array([0,0,1]), h_vect) # km2/s
		N = np.sqrt(np.dot(N_vect, N_vect)) # km/s
		if N_vect[1] >= 0:
			self.Omega = np.arccos(N_vect[0] / N) # rad
		else:
			self.Omega = 2*np.pi - np.arccos(N_vect[0] / N)
		e_vect = 1 / self.mu * ((v**2 - self.mu/r)*r2_vect - r*vradial*v2_vect) # unitless
		self.e = np.sqrt(np.dot(e_vect, e_vect)) # unitless
		if e_vect[2] >= 0:
			self.omega = np.arccos(np.dot(N_vect, e_vect) / (N*self.e)) # rad
		else:
			self.omega = 2*np.pi - np.arccos(np.dot(N_vect, e_vect) / (N*self.e))
		if vradial >= 0 :
			self.theta = np.arccos(np.dot(e_vect, r2_vect) / (self.e*r)) # rad
			theta1 = np.arccos(1/self.e*(self.h**2/(self.mu*r) -1))
		else:
			self.theta = 2*np.pi - np.arccos(np.dot(e_vect, r2_vect) / (self.e * r))
			theta1 = 2*np.pi - np.arccos(1 / self.e * (self.h ** 2 / (self.mu * r) - 1))

		self.rp = self.h ** 2 / self.mu / (1 + self.e * np.cos(0.))  # km
		self.ra = self.h ** 2 / self.mu / (1 + self.e * np.cos(np.pi))  # km
		self.a = 1 / 2 * (self.rp + self.ra)  # semi_major km
		self.T = 2 * np.pi / np.sqrt(self.mu) * self.a ** (3 / 2)  # sec
		#return (rad2deg(i), h, e, rad2deg(Omega), rad2deg(omega), rad2deg(theta))
		#print('i(deg), h(km2/s), e(unitless), Omega(rad), omega(rad), theta(rad)')
		#print(self.i, self.h, self.e, self.Omega, self.omega, self.theta)
		#return (self.i, self.h, self.e, self.Omega, self.omega, self.theta)
		self.elements = [self.i, self.h, self.e, self.Omega, self.omega, self.theta, self.rp, self.ra, self.a, self.T]
		print('i, h, e, Omega, omega, theta', 'r_peri', 'r_aposi', 'semi-major', 'Period')
		print(self.elements)
		return self.elements

	def cal_kepler_equation(self, elements):
		if isinstance(elements, list) and len(elements) == 10:
			pass
		else:
			print(f'The type of var elements is {type(elements)}, it should be List type.')
			raise TypeError
		self.i, self.h, self.e, self.Omega, self.omega, self.theta, self.rp, self.ra, self.a, self.T = elements
#14 calculate the true anomaly, the Eccentric anomaly, the Mean anomaly with time
		print(f'period: {self.T}')
		if self.obj == 'planet':
			pluto_T = 100000 # day
			t = np.arange(0, 100000)
		else:
			t = np.arange(0, 1000)
		M = 2*np.pi / self.T * t

		def fE(Ex, M, q):
			return Ex - self.e*np.sin(Ex) - M
		def fEprime(Ex, M, q):
			return 1 - self.e*np.cos(Ex)

		E_list = []
		for m in M:
			#print(m)
			root = optimize.newton(fE, m+1, fprime=fEprime, args=(m, m+1))
			E_list.append(root)
		E = np.array(E_list)
		tH_rad = 2 * np.arctan(np.sqrt((1+self.e) / (1-self.e)) * np.tan(E / 2))
		tH_deg = rad2deg(tH_rad)
		tH_deg = np.array([360 + i if i < 0 else i for i in tH_deg])

		if self.e == 0:
			r = np.full_like(t, self.h**2/self.mu)
		elif 0 < self.e < 1:
			r = self.a * (1 - self.e**2) / (1 + self.e*np.cos(tH_rad))
		elif self.e == 1:
			r = self.h**2 / self.mu / (1 + np.cos(tH_rad))
		elif self.e > 1:
			r = self.a * (self.e**2 - 1) / (1 + self.e*np.cos(tH_rad))
			r2 = self.h**2 / self.mu / (1 + np.cos(tH_rad))
		else:
			print('Do you know e < 0 case?')
			breakpoint()

		x = r * np.cos(tH_rad)
		y = r * np.sin(tH_rad)
		z = np.zeros_like(x, dtype=np.float)
		#xyz = np.row_stack((x, y, z))

		Ome = [[np.cos(self.Omega), np.sin(self.Omega), 0], [-np.sin(self.Omega), np.cos(self.Omega), 0], [0, 0, 1]]
		inclin = [[1, 0, 0], [0, np.cos(self.i), np.sin(self.i)], [0, -np.sin(self.i), np.cos(self.i)]]
		ome = [[np.cos(self.omega), np.sin(self.omega), 0], [-np.sin(self.omega), np.cos(self.omega), 0], [0, 0, 1]]

# calculation of transformation maxtrix from ijk to IJK or reverse
# calculation xyz and XYZ
		# [Q]Xx = [R3(omega)][R1(i)][R3(Omega)]
		trans_matrix_Omega = np.array(Ome)
		trans_matrix_incl = np.array(inclin)
		trans_matrix_omega = np.array(ome)
		trans_matrix_Xx = np.dot(trans_matrix_omega, np.dot(trans_matrix_incl, trans_matrix_Omega))
		trans_matrix_xX = trans_matrix_Xx.T

		Xxdata = []
		xXdata = []
		for t1, u, w, z in zip(t, x, y, z):
			tranxX = np.dot(trans_matrix_xX, np.array([u, w, z]))
			Xxdata.append([t1, u, w, z, self.T])
			xXdata.append([t1, tranxX[0], tranxX[1], tranxX[2], self.T])

		self.TXYZP = xXdata
		self.txyzp = Xxdata
		#print(self.TXYZ.shape)

		return self.txyzp, self.TXYZP

	def cal_kepler_equation_line(self, elements):
		if isinstance(elements, list) and len(elements) == 10:
			pass
		else:
			print(f'The type of var elements is {type(elements)}, it should be List type.')
			raise TypeError
		self.i, self.h, self.e, self.Omega, self.omega, self.theta, self.rp, self.ra, self.a, self.T = elements
		# 14 calculate the true anomaly, the Eccentric anomaly, the Mean anomaly with time
		t = np.linspace(0, self.T, 1000)
		M = 2 * np.pi / self.T * t

		def fE(Ex, M, q):
			return Ex - self.e * np.sin(Ex) - M

		def fEprime(Ex, M, q):
			return 1 - self.e * np.cos(Ex)

		E_list = []
		for m in M:
			# print(m)
			root = optimize.newton(fE, m + 1, fprime=fEprime, args=(m, m + 1))
			E_list.append(root)
		E = np.array(E_list)
		tH_rad = 2 * np.arctan(np.sqrt((1 + self.e) / (1 - self.e)) * np.tan(E / 2))
		tH_deg = rad2deg(tH_rad)
		tH_deg = np.array([360 + i if i < 0 else i for i in tH_deg])

		if self.e == 0:
			r = np.full_like(t, self.h ** 2 / self.mu)
		elif 0 < self.e < 1:
			r = self.a * (1 - self.e ** 2) / (1 + self.e * np.cos(tH_rad))
		elif self.e == 1:
			r = self.h ** 2 / self.mu / (1 + np.cos(tH_rad))
		elif self.e > 1:
			r = self.a * (self.e ** 2 - 1) / (1 + self.e * np.cos(tH_rad))
			r2 = self.h ** 2 / self.mu / (1 + np.cos(tH_rad))
		else:
			print('Do you know e < 0 case?')
			breakpoint()

		x = r * np.cos(tH_rad)
		y = r * np.sin(tH_rad)
		z = np.zeros_like(x, dtype=np.float)
		# xyz = np.row_stack((x, y, z))

		Ome = [[np.cos(self.Omega), np.sin(self.Omega), 0], [-np.sin(self.Omega), np.cos(self.Omega), 0], [0, 0, 1]]
		inclin = [[1, 0, 0], [0, np.cos(self.i), np.sin(self.i)], [0, -np.sin(self.i), np.cos(self.i)]]
		ome = [[np.cos(self.omega), np.sin(self.omega), 0], [-np.sin(self.omega), np.cos(self.omega), 0], [0, 0, 1]]

		# calculation of transformation maxtrix from ijk to IJK or reverse
		# calculation xyz and XYZ
		# [Q]Xx = [R3(omega)][R1(i)][R3(Omega)]
		trans_matrix_Omega = np.array(Ome)
		trans_matrix_incl = np.array(inclin)
		trans_matrix_omega = np.array(ome)
		trans_matrix_Xx = np.dot(trans_matrix_omega, np.dot(trans_matrix_incl, trans_matrix_Omega))
		trans_matrix_xX = trans_matrix_Xx.T

		Xxdata = []
		xXdata = []
		for t1, u, w, z in zip(t, x, y, z):
			tranxX = np.dot(trans_matrix_xX, np.array([u, w, z]))
			Xxdata.append([t1, u, w, z, self.T])
			xXdata.append([t1, tranxX[0], tranxX[1], tranxX[2], self.T])

		self.TXYZP = xXdata
		self.txyzp = Xxdata
		# print(self.TXYZP.shape)

		return self.txyzp, self.TXYZP

	def plot_3d(self, inp_data):
		fig = plt.figure()
		ax = plt3.Axes3D(fig)
		ax.set_xlabel('X')
		# ax.set_ylim3d([-1.0, 1.0])
		ax.set_ylabel('Y')
		# ax.set_zlim3d([0.0, 10.0])
		ax.set_zlabel('Z')

		def update(num, data, position):
			position.set_data(data[:2, num:num + 1])
			position.set_3d_properties(data[2, num:num + 1])

		for key in inp_data.keys():
			data1 = inp_data[key]
			ax.plot(data1[1, :], data1[2, :], data1[3, :])
			#ax.plot(data2[0, :], data2[1, :], data2[2, :])
			position1, = ax.plot(data1[1, 0:1], data1[2, 0:1], data1[3, 0:1], 'ro')
			#position2, = ax.plot(data2[0, 0:1], data2[1, 0:1], data2[2, 0:1], 'bo')
			N1 = len(data1[1, :])
			print(N1)
			globals()[key] = animation.FuncAnimation(fig, update, N1, fargs=(data1, position1), interval=100000 / N1, blit=False)
			#N2 = len(data2[0, :])
			#ani2 = animation.FuncAnimation(fig, update, N2, fargs=(data2, position2), interval=5000 / N2, blit=False)
			# ani.save('matplot003.gif', writer='imagemagick')
		plt.show()
		print('end')

	def plotly_plot(self, inp_data):
		if isinstance(inp_data, list):
			df = pd.DataFrame(inp_data, columns=['name', 'time', 'x', 'y', 'z', 'Period'])
			df1 = df[df['name'].isin(['sat1'])]
			df2 = df[df['name'].isin(['sat2'])]
		else:
			df = pd.read_csv(inp_data)
		print(df)

		'''
		fig = px.scatter_3d(df, x="x", y="y", z="z",
	                    range_x=[df['x'].min(), df['x'].max()], range_y=[df['y'].min(), df['y'].max()],
	                    range_z=[df['z'].min(), df['z'].max()], animation_frame='time')
		'''
		fig = px.scatter_3d(df, x="x", y="y", z='z', color='name', animation_group='name',
		                     range_x=[df['x'].min(), df['x'].max()], range_y=[df['y'].min(), df['y'].max()],
		                     range_z=[df['z'].min(), df['z'].max()], animation_frame='time')
		fig2 = px.line_3d(df1, x="x", y="y", z="z")
		fig3 = px.line_3d(df2, x='x', y='y', z='z')
		#fig.add_trace(fig2.data[0])
		#fig.show()
		'''
		import base64
		# set a local image as a background
		image_filename = 'image.png'
		plotly_logo = base64.b64encode(open(image_filename, 'rb').read())

		fig.update_layout(
			images=[dict(
				source='data:image/png;base64,{}'.format(plotly_logo.decode()),
				xref="paper", yref="paper",
				x=0, y=1,
				sizex=0.5, sizey=0.5,
				xanchor="left",
				yanchor="top",
				# sizing="stretch",
				layer="above")])
		'''

		return fig, fig2, fig3


#15 transformation coordinate from orbit own to the Geocentric Equatorial Coordinate

'''
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

if __name__ == ' __main__ ':
	sat_Nlist = ['sat1']
	# sat_Nlist = ['sat1', 'sat2']
	time1 = [0, 118.10, 237.58]  # sec
	ra1 = [43.537, 54.420, 64.318]  # deg
	ra1 = list(deg2rad(ra1))
	dec1 = [-8.7833, -12.074, -15.105]  # deg
	dec1 = list(deg2rad(dec1))
	lst1 = [44.506, 45.0, 45.499]  # deg
	lst1 = list(deg2rad(lst1))
	lat1 = [40.0] * len(time1)  # deg
	lat1 = list(deg2rad(lat1))
	height1 = [1.] * len(time1)  # km

	time2 = [0, 200.10, 400.58]  # sec
	ra2 = [68.537, 78.420, 85.318]  # deg
	ra2 = list(deg2rad(ra2))
	dec2 = [-8.7833, -12.074, -15.105]  # deg
	dec2 = list(deg2rad(dec2))
	lst2 = [44.506, 45.0, 45.499]  # deg
	lst2 = list(deg2rad(lst2))
	lat2 = [40.0] * len(time2)  # deg
	lat2 = list(deg2rad(lat2))
	height2 = [1.] * len(time2)  # km
	time, ra, dec, lst, lat, height = [time1, time2], [ra1, ra2], [dec1, dec2], [lst1, lst2], [lat1, lat2], [height1,
	                                                                                                         height2]

	sat = orbit('sat')
	observation_data = {}
	inp_data = []
	inp_data_line = []
	for i, name in enumerate(sat_Nlist):
		observation_data[name] = (time[i], ra[i], dec[i], lst[i], lat[i], height[i])
		r, v = sat.cal_r2v2_vect(observation_data[name])
		ele = sat.cal_orbit_ele(r, v)

		txyz, TXYZ = sat.cal_kepler_equation(ele)
		num = np.array(TXYZ).shape[0]
		for nlist in TXYZ:
			nlist.insert(0, name)
			inp_data.append(nlist)

		txyz_line, TXYZ_line = sat.cal_kepler_equation_line(ele)
		num = np.array(TXYZ_line).shape[0]
		for nlist in TXYZ_line:
			nlist.insert(0, name)
			inp_data_line.append(nlist)

	fig, fig2, fig3 = sat.plotly_plot(inp_data)
	fig.add_trace(fig2.data[0])
	# fig.add_trace(fig3.data[0])
	fig.show()

else:
	print('Please import orbit')


