import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import plotly

from usefultools import *
import csv
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from pyorbital.orbital import Orbital
import datetime


import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
from dash.dependencies import Input, Output

# pip install pyorbital
from pyorbital.orbital import Orbital
satellite = Orbital('TERRA')

global tt
tt = 0

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
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


@app.callback(Output('live-update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Longitude: {0:.2f}'.format(lon), style=style),
        html.Span('Latitude: {0:.2f}'.format(lat), style=style),
        html.Span('Altitude: {0:0.2f}'.format(alt), style=style)
    ]


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', 'figure'),
              Input('interval-component', 'n_intervals'))
def update_graph_live(n):
    satellite = Orbital('TERRA')
    data = {
        'time': [],
        'Latitude': [],
        'Longitude': [],
        'Altitude': []
    }
    global tt
    ttt = tt + 100

    # Collect some data
    for i in range(180):
        time = datetime.datetime.now() - datetime.timedelta(seconds=i*20)
        lon, lat, alt = satellite.get_lonlatalt(
            time
        )
        data['Longitude'].append(lon)
        data['Latitude'].append(lat)
        data['Altitude'].append(alt)
        data['time'].append(time)

    df = pd.read_csv('kepler_XYZ.csv', index_col='time')
    venus = df[df['name']=='Venus']

    venusx = venus['x'][:2]
	
    trace1 = go.Scatter3d(
		x = list(df[df['name']=='Venus']['x'][tt:ttt]), y = list(df[df['name']=='Venus']['y'][tt:ttt]),
		z = list(df[df['name']=='Venus']['z'][tt:ttt]), mode='lines+markers',
		marker = dict(
             size = 3,
             color = df[df['name'] == 'Venus']['z'], # set color to an array/list of desired values
             colorscale = 'Viridis'
             ))


    # Create the graph with subplots
    fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{'type':'scene'}],[{'type':'scene'}]])
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }

    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    fig.append_trace(go.Scatter3d(
        x = data['time'],
        y = data['Altitude'],
        z = data['Latitude'],
        name = 'Altitude',
        mode = 'lines+markers',
        type = 'scatter3d'
    ), 1, 1)

    fig.append_trace(go.Scatter3d(
        x = data['time'],
        y = data['Altitude'],
        z = data['Latitude'],
        name = 'Altitude',
        mode = 'lines+markers',
        type = 'scatter3d'
    ), 2, 1)

    tt = ttt

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)









@app.callback(Output('live-update-text', 'children'),
			  Input('interval-component', 'n_intervals'))
def update_metrics(n):
	global num
	global internum
	global df
	numnext = num + internum
	style = {'padding': '5px', 'fontSize': '16px'}

	dfx = data['Venus']['x'][n]
	dfy = data['Venus']['y'][n]
	dfz = data['Venus']['z'][n]

	num = numnext

	return [
		html.Span('Number: {0:.2f}'.format(n), style=style),
		html.Span('dfx: {0:.2f}'.format(dfx), style=style), 
		html.Span('dfy: {0:.2f}'.format(dfy), style=style),
		html.Span('dfz: {0:.2f}'.format(dfz), style=style),
		html.Span('planet: {}'.format('Venus'))
	]






@app.callback(Output('live-update-graph', 'figure'),
			  Input('interval-component', 'n_intervals'))
def update_graph_live(n):
	global num
	global internum
	global df
	numnext = num + internum

	data = {
		'time': [],
		'Latitude': [],
		'Longitude': [],
		'Altitude': []
	}

	dfx = []
	dfy = []
	dfz = []
	# Collect some dataura
	for planet in ['Mercury', 'Venus', 'EM_Bary', 'Mars', 'Jupiter', 'Saturn', 'Uranus']:
		dfx.append(df[df['name']==planet]['x'][numnext])
		dfy.append(df[df['name']==planet]['y'][numnext])
		dfz.append(df[df['name']==planet]['z'][numnext])

	num = numnext

	
	fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{'type':'scene'}],[{'type':'scene'}]])
	fig['layout']['margin'] = {
		'l': 30, 'r': 10, 'b': 30, 't': 10
	}
	fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

	fig.add_trace(
		go.Scatter3d(
		x = dfx,
		y = dfy,
		z = dfz,
		mode = 'markers'
	), 1, 1)

	return fig


