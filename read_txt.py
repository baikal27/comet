import pandas as pd
import re
import numpy as np
from astropy.time import Time

def read_HORIZONS(fname):
	with open(fname) as f:
		lines = f.readlines()
		spattern = '$$SOE'
		epattern = '$$EOE'
		for i, line in enumerate(lines):
			if spattern in line:
				snum = i
				print(f'startnum: {snum}')
			elif epattern in line:
				enum = i
				print(f'endnum: {enum}')
		lines = lines[snum+1:enum-1]
		newlines = []
		for line in lines:
			line = line.split('\n').pop(0).split(',')
			line = [i.lstrip() for i in line]
			newlines.append(line)
			print(line)

	df = pd.DataFrame(newlines, columns=['date', 'dum1', 'dum2', 'ra', 'dec', 'lst', 'dum3'])
	df[['ra_h', 'ra_m', 'ra_s']] = df['ra'].str.split(' ', n=3, expand=True)
	df[['lst_h', 'lst_m', 'lst_s']] = df['lst'].str.split(' ', n=3, expand=True)
	df[['dec_d', 'dec_m', 'dec_s']] = df['dec'].str.split(' ', n=3, expand=True)
	df[['ra_h', 'ra_m', 'ra_s']] = df[['ra_h', 'ra_m', 'ra_s']].astype(np.float)
	df[['lst_h', 'lst_m', 'lst_s']] = df[['lst_h', 'lst_m', 'lst_s']].astype(np.float)
	df[['dec_d', 'dec_m', 'dec_s']] = df[['dec_d', 'dec_m', 'dec_s']].astype(np.float)
	df['ra_deg'] = df.apply(lambda x: (x['ra_s']/3600 + x['ra_m']/60 + x['ra_h'])*15., axis=1)
	df['lst_deg'] = df.apply(lambda x: (x['lst_s']/3600 + x['lst_m']/60 + x['lst_h'])*15., axis=1)
	df['dec_deg'] = df.apply(lambda x: (x['dec_s']/3600 + x['dec_m']/60 + x['dec_d']), axis=1)

	ra = [i for i in set(df['ra_deg'])]
	lst = [i for i in set(df['lst_deg'])]
	dec = [i for i in set(df['dec_deg'])]
	t_list = [i for i in set(df['date'])]
	times = []
	dict = {'Jan':'1', 'Feb':'2', 'Mar':'3', 'Apr':'5', 'May':'5', 'Jun':'6',
	        'Jul':'7', 'Aug':'8', 'Sep':'9', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
	rep = re.compile('[a-zA-Z]+')
	for t in t_list:
		ob = rep.search(t).group()
		print(ob)
		if ob in list(dict.keys()):
			replaced = re.sub(ob, dict[ob], t)
			times.append(replaced)
	#print(times)
	#ra_dec = ['08 23 25.09 +14 41 41.9', '07 57 33.96 +17 38 16.0', '07 44 16.69 +20 09 57.3']
	time = Time(times, format='iso', scale='utc')
	#print(time.jd1)
	#print(f'ra: {ra}')
	#print(f'dec:{dec}')
	#print(f'lst: {lst}')
	#print(f't_list: {tjd}')
	result_dic = {'time':time, 'ra':ra, 'dec':dec, 'lst':lst}
	print(result_dic['ra'])
	print(result_dic['dec'])

	return result_dic
read_HORIZONS('mars_results.txt')



