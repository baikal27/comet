from orbital import orbit
from usefultools import *
import csv
import pandas as pd
import plotly.graph_objs as go

''' This is for planet from 6 elements to plot keplerian motion '''
'''
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
inp_data_line = []
for i, name in enumerate(planet_Nlist):
	txyz, TXYZ = planet.cal_kepler_equation(planet_data[name])
	num = np.array(TXYZ).shape[0]
	for nlist in TXYZ:
		nlist.insert(0, name)
		inp_data.append(nlist)

	txyz_line, TXYZ_line = planet.cal_kepler_equation_line(planet_data[name])
	num = np.array(TXYZ_line).shape[0]
	for nlist in TXYZ_line:
		nlist.insert(0, name)
		inp_data_line.append(nlist)

with open('kepler_XYZ.csv', 'w', newline='') as csvf:
	csvwriter = csv.writer(csvf)
	csvwriter.writerow(['name', 'time', 'x', 'y', 'z', 'T'])
	csvwriter.writerows(inp_data)

with open('kepler_XYZ_line.csv', 'w', newline='') as csvf:
	csvwriter = csv.writer(csvf)
	csvwriter.writerow(['name', 'time', 'x', 'y', 'z', 'T'])
	csvwriter.writerows(inp_data_line)

with open('kepler_XYZ.csv', 'r', newline='') as csvf:
	csvreader = csv.reader(csvf)
	for row in csvreader:
		print(row)

with open('kepler_XYZ_line.csv', 'r', newline='') as csvf:
	csvreader = csv.reader(csvf)
	for row in csvreader:
		print(row)
'''
'''
fig = planet.plotly_plot('kepler_XYZ.csv')
#fig.add_trace(fig2.data[0])
#fig.add_trace(fig3.data[0])
fig.show()
'''

df_line = pd.read_csv('kepler_XYZ_line.csv')
#df1 = df_line[df_line['name'] == 'Mercury']
#df2 = df_line[df_line['name'] == 'Venus']
df = pd.read_csv('kepler_XYZ.csv', index_col='time')
df1 = df[df['name'] == 'Pluto']
df2 = df[df['name'] == 'Jupiter']
print(df1)
#df2 = df_line[df_line['name'] == 'Venus']


planets = df_line.drop_duplicates("name", keep='first')["name"].values
'''
traces = []
for name in planet_names:
   line = df[df['name'] == name]
   trace = go.Scatter3d(
   x = line['x'], y = line['y'], z = line['z'], mode = 'lines', marker = dict(
      size = 3,
      color = line['z'], # set color to an array/list of desired values
      colorscale = 'Viridis'
      ),
      name = name
   )
   traces.append(trace)
'''
traces = [go.Scatter3d(
      x = df_line[df_line['name'] == planet]['x'], y = df_line[df_line['name'] == planet]['y'], z = df_line[df_line['name'] == planet]['z'],
      mode = 'lines', marker = dict(
         size = 3,
         color = df_line[df_line['name'] == planet]['z'], # set color to an array/list of desired values
         colorscale = 'Viridis'
         ),
      name = planet
      )
   for planet in planets]


layout = go.Layout(
        title = '3D Scatter plot',
        xaxis=dict(range=[df1['x'].min(), df1['x'].max()], autorange=False, zeroline=False),
        yaxis=dict(range=[df1['y'].min(), df1['y'].max()], autorange=False, zeroline=False),
        title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])])
start = 0
end = 1000
step = 50

#xdata = [df[(df['time']==t) & (df['name']==name)]['x'] for t in range(start, end, step) for name in planet_names]
#ydata = [df[(df['time']==t) & (df['name']==name)]['y'] for t in range(start, end, step) for name in planet_names]
#zdata = [df[(df['time']==t) & (df['name']==name)]['z'] for t in range(start, end, step) for name in planet_names]
'''
frames = [go.Frame(data=[go.Scatter3d(x=df[(df['time']==t) & (df['name']==planet)]['x'],
                                      y=df[(df['time']==t) & (df['name']==planet)]['y'],
                                      z=df[(df['time']==t) & (df['name']==planet)]['z'],
                                      mode='markers', marker=dict(size=7, color=df1['z'], colorscale='Viridis'))])
                    for t in range(start, end, step) for planet in planets]
'''
'''
frames = [go.Frame(data=[go.Scatter3d(x=[df1['x'][i], df2['x'][i]], y=[df1['y'][i], df2['y'][i]], z=[df1['z'][i], df2['z'][i]],
                           mode='markers', marker=dict(size=7, color=df1['z'], colorscale='Viridis'))])
          for i in range(0, 5000, 50)]
'''
frames = []
for i in range(start, end, step):
    xdata, ydata, zdata = [], [], []
    for planet in planets:
        xdata.append(df[df['name']==planet]['x'][i])
        ydata.append(df[df['name']==planet]['y'][i])
        zdata.append(df[df['name']==planet]['z'][i])
    data = go.Scatter3d(x=xdata, y=ydata, z=zdata, mode='markers',
                        marker=dict(size=7, color=df[df['name']==planet]['z'], colorscale='Viridis'))
    frames.append(go.Frame(data=data))

fig = go.Figure(data=traces, layout=layout, frames=frames)
fig.show()

