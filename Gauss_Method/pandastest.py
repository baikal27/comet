import pandas as pd
df_line = pd.read_csv('kepler_XYZ_line.csv')
df1 = df_line[df_line['name'] == 'Mercury']
df2 = df_line[df_line['name'] == 'Venus']
df = pd.read_csv('kepler_XYZ.csv', index_col='time')
planets = df_line.drop_duplicates("name", keep='first')["name"].values
print(planets)


scene_options = {
	'Planets': dict(
		xaxis = dict(nticks=4, range=[-2,2],),
        yaxis = dict(nticks=4, range=[-2,2],),
        zaxis = dict(nticks=4, range=[-2,2],),
    	width=700,
    	margin=dict(r=20, l=10, b=10, t=10)
    	),
	'Comets': dict(
		xaxis = dict(nticks=4, range=[-2,2],),
        yaxis = dict(nticks=4, range=[-2,2],),
        zaxis = dict(nticks=4, range=[-2,2],),
    	width=700,
    	margin=dict(r=20, l=10, b=10, t=10)
    	),
	'Asteroids': dict(
		xaxis = dict(nticks=4, range=[-2,2],),
        yaxis = dict(nticks=4, range=[-2,2],),
        zaxis = dict(nticks=4, range=[-2],),
    	width=700,
    	margin=dict(r=20, l=10, b=10, t=10)
    	),
	'Rocky': dict(
		xaxis = dict(nticks=4, range=[-2,2],),
        yaxis = dict(nticks=4, range=[-2,2],),
        zaxis = dict(nticks=4, range=[-2,2],),
    	width=700,
    	margin=dict(r=20, l=10, b=10, t=10)
    	),
	'Fluid': dict(
		xaxis = dict(nticks=4, range=[-2,2],),
        yaxis = dict(nticks=4, range=[-2,2],),
        zaxis = dict(nticks=4, range=[-2,2],),
    	width=700,
    	margin=dict(r=20, l=10, b=10, t=10)
    	)
	}
