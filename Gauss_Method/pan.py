import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import random
import pandas as pd

app = dash.Dash(__name__)
app.layout = html.Div(
	html.Div([
		html.H4('TERRA Satellite Live Feed'),
		html.Div(id='live-update-text'),
		dcc.Graph(id='live-update-graph'),
		dcc.Interval(
			id='interval-component',
			interval=1*1000, # in milliseconds
			n_intervals=0
		)
	])
)

global num
start=0
m = 5000 # data 개수
global internum
internum = 10 # 속도와 관련 10~50
global df
df = pd.read_csv('kepler_XYZ.csv', index_col='time')
global planets
planets = list(df.drop_duplicates("name", keep='first')["name"].values)
data = []
for planet in planets:
	datadict = dict(
			name = planet,
			x = df[df['name']==planet]['x'][start:internum*m:internum].values,
			y = df[df['name']==planet]['y'][start:internum*m:internum].values,
			z = df[df['name']==planet]['z'][start:internum*m:internum].values
		)
	data.append(datadict)


dfx = []
dfy = []
dfz = []
n=1
# Collect some dataura
for idata in data:
	dfx.append(idata['x'][n])
	dfy.append(idata['y'][n])
	dfz.append(idata['z'][n])

print(f'dfx: {dfx}')

