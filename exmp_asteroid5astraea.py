from orbital import orbit
from usefultools import *
''' geocentric from jpl-horizons 2020-04-22 to 2020-07-21, asteroid 5 Astraea
# ra with HMS, dec with DMS
---Time: UT-----  ---R.A.__(ICRF)__DEC---  ---R.A.__(a-apparent)__DEC--(ref. frame defined by the Earth equator)
2020-04-22 00:00  08 27 26.03 +20 24 07.9  08 28 35.32 +20 20 07.6
2020-05-22 00:00  09 15 47.25 +17 58 11.6  09 16 54.14 +17 53 10.4
2020-06-22 00:00  10 11 56.52 +13 48 53.4  10 13 00.76 +13 42 57.6
'''

'''(DOAO)topo-site from jpl-horizons 2020-04-22 to 2020-07-21, asteroid 5 Astraea
---Time: UT-----  ---R.A.__(ICRF)__DEC---  ---R.A.__(a-apparent)__DEC--
2020-04-22 00:00  08 27 26.18 +20 24 04.1 08 28 35.46 +20 20 03.7
2020-05-22 00:00  09 15 47.43 +17 58 08.7 09 16 54.31 +17 53 07.3
2020-06-22 00:00  10 11 56.71 +13 48 51.1 10 13 00.93 +13 42 55.3
'''

#sat_Nlist = ['sat1']
sat_Nlist = ['sat1', 'sat2']
time1 = [0, 118.10, 237.58]  # sec
ra1 = [43.537, 54.420, 64.318]  # deg
ra1 = list(deg2rad(ra1))
dec1 = [-8.7833, -12.074, -15.105]  # deg
dec1 = list(deg2rad(dec1))
lst1 = [44.506, 45.0, 45.499]  # deg
lst1 = list(deg2rad(lst1))
lat1 = [40.0]*len(time1)  # deg
lat1 = list(deg2rad(lat1))
height1 = [1.]*len(time1)  # km

time2 = [0, 200.10, 400.58]  # sec
ra2 = [68.537, 78.420, 85.318]  # deg
ra2 = list(deg2rad(ra2))
dec2 = [-8.7833, -12.074, -15.105]  # deg
dec2 = list(deg2rad(dec2))
lst2 = [44.506, 45.0, 45.499]  # deg
lst2 = list(deg2rad(lst2))
lat2 = [40.0]*len(time2)  # deg
lat2 = list(deg2rad(lat2))
height2 = [1.]*len(time2)  # km
time, ra, dec, lst, lat, height = [time1, time2], [ra1, ra2], [dec1, dec2], [lst1, lst2], [lat1, lat2], [height1, height2]

'''
result = read_HORIZONS('mars_results.txt')
time = result['tjd'][:3]
ra = result['ra'][:3]
ra = list(deg2rad(ra))
dec = result['dec'][:3]
dec = list(deg2rad(dec))
lst = result['lst'][:3]
lst = list(deg2rad(lst))
observation_data = {}
observation_data['mars'] = (time, ra, dec, lst, lat)
mars = orbit('planet')
r, v = mars.cal_r2v2_vect(observation_data['mars'])
ele = mars.cal_orbit_ele(r,v)
print(f'ele: {ele}')
breakpoint()
xyz, XYZ = mars.cal_kepler_equation(ele)
plot_inpdata = {}
plot_inpdata['mars'] = XYZ
mars.plot_3d(plot_inpdata)
'''

sat = orbit('sat')
observation_data = {}
inp_data = []
for i, name in enumerate(sat_Nlist):
	observation_data[name] = (time[i], ra[i], dec[i], lst[i], lat[i], height[i])
	r, v = sat.cal_r2v2_vect(observation_data[name])
	ele = sat.cal_orbit_ele(r, v)
	txyzp, TXYZP = sat.cal_kepler_equation_line(ele)
	num = np.array(TXYZP).shape[0]
	for nlist in TXYZP:
		nlist.insert(0, name)
		inp_data.append(nlist)
		#inp_data = ['name', 'time', 'x', 'y', 'z', 'Period']

fig, fig2, fig3 = sat.plotly_plot(inp_data)
fig.add_trace(fig2.data[0])
fig.add_trace(fig3.data[0])
fig.show()
