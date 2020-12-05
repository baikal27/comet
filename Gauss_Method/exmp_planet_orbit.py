from orbital import orbit
from usefultools import *

''' This is for planet from 6 elements to plot keplerian motion '''

def add_rpraaT(ele):
	a, e, I, theta, o, O = ele
	I = deg2rad(I) ; O = deg2rad(O) ; o = deg2rad(o) ; theta = deg2rad(theta)

	mu = const.GM_sun.value * m2au(1, 3) / sec2day(1, 2)
	h = np.sqrt(2*mu*a / ((1 + 1/e/np.cos(0.)) + (1 + 1/e/np.cos(np.pi))))
	rp = h**2/mu/(1+e*np.cos(0.)) # km
	ra = h**2/mu/(1+e*np.cos(np.pi)) # km
	T = 2 * np.pi / np.sqrt(mu) * a ** (3 / 2)
	rep_a = 1/2*(rp + ra)
	#print(a, rep_a)
	trans_ele = [I, h, e, O, o, theta, rp, ra, a, T]
	print(f'The List of trans_ele order : [i, h, e, Omega, omega, theta, rp, ra, a, T]')
	return trans_ele

with open('planet_ele.txt', 'r') as f:
	data = f.readlines()[16:-1:2]
with open('planet_ele.txt', 'r') as f:
	columns = f.readlines()[13]
	columns = columns.split(' ')
	columns = ' '.join(columns).split()
	#print(f'This is planet_ele.txt order : {columns}')
planet_data = {}
for line in data:
	a = line.split(' ')
	a = ' '.join(a).split()
	dummy_tup = tuple([float(i) for i in a[1:]])
	planet_data[a[0]] = add_rpraaT(dummy_tup)
	print('The List of planet_data order : [i, h, e, Omega, omega, theta, rp, ra, a, T]')

planet = orbit('planet')
planet_Nlist = list(planet_data.keys())
inp_data = []
for i, name in enumerate(planet_Nlist):
	txyz, TXYZ = planet.cal_kepler_equation(planet_data[name])
	num = np.array(TXYZ).shape[0]
	for nlist in TXYZ:
		nlist.insert(0, name)
		inp_data.append(nlist)

with open('kepler_XYZ.csv', 'w') as f:
	df = pd.DataFrame(inp_data, columns=['name', 'time', 'x', 'y', 'z'])
	print(df)
	df1 = df[df['name'].isin(['sat1'])]

fig, fig2, fig3 = planet.plotly_plot(inp_data)
#fig.add_trace(fig2.data[0])
#fig.add_trace(fig3.data[0])
fig.show()