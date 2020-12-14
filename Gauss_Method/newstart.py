import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
import random
import pandas as pd

all_options = {
	'Planets': ['Mercury', 'Venus', 'EM_Bary', 'Mars', 'Jupiter', 'Saturn',
	 'Uranus', 'Neptune', 'Pluto'],
	'Comets' : ['halley'],
	'Asteroids' : ['hello'],
	'Rocky' : ['Mercury', 'Venus', 'EM_Bary', 'Mars'],
	'Fluid' : ['Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
}

scene_options = {
	'Planets' : dict(
			xaxis = dict(nticks=4, range=[-10,10]),
			yaxis = dict(nticks=4, range=[-10,10]),
			zaxis = dict(nticks=4, range=[-10,10])
			),
	'Comets' : dict(
			xaxis = dict(nticks=4, range=[-2,2]),
			yaxis = dict(nticks=4, range=[-2,2]),
			zaxis = dict(nticks=4, range=[-2,2])
			),
	'Asteroid' : dict(
			xaxis = dict(nticks=4, range=[-2,2]),
			yaxis = dict(nticks=4, range=[-2,2]),
			zaxis = dict(nticks=4, range=[-2,2])
			),
	'Rocky' : dict(
			xaxis = dict(nticks=4, range=[-2,2]),
			yaxis = dict(nticks=4, range=[-2,2]),
			zaxis = dict(nticks=4, range=[-2,2])
			),
	'Fluid' : dict(
			xaxis = dict(nticks=4, range=[-2,2]),
			yaxis = dict(nticks=4, range=[-2,2]),
			zaxis = dict(nticks=4, range=[-2,2])
			)

	}




df = pd.read_csv('kepler_XYZ.csv', index_col='time')

app = dash.Dash(__name__)
app.layout = html.Div(
	html.Div([
		dcc.Dropdown(
        id='planet-dropdown',
        options=[
            {'label': name, 'value': name} for name in all_options.keys()],
        value='Planets'
    ),
		html.H4('TERRA Satellite Live Feed'),
		html.Div(id='hidden-value', style={'display': 'none'}),
		html.Div(id='live-update-text'),
		dcc.Graph(id='live-update-graph'),
		dcc.Interval(
			id='interval-component',
			interval=1*1000, # in milliseconds
			n_intervals=0
		)
	])
)


@app.callback(Output('hidden-value', 'children'),
				Input('planet-dropdown', 'value'))
def upload_data(selected_option):
		# variables
	start=0
	m = 10000 # data 개수
	internum = 100 # 속도와 관련 10~50
	
	planets = all_options[selected_option]
	selected_data = []
	for planet in planets:
		datadict = dict(
				name = planet,
				x = df[df['name']==planet]['x'][start:internum*m:internum].values,
				y = df[df['name']==planet]['y'][start:internum*m:internum].values,
				z = df[df['name']==planet]['z'][start:internum*m:internum].values
			)
		selected_data.append(datadict)
	return selected_data


@app.callback(Output('live-update-text', 'children'),
				Input('hidden-value', 'children'),
			  Input('interval-component', 'n_intervals'))
def update_metrics(selected_data, n):

	style = {'padding': '5px', 'fontSize': '16px'}

	dfx = selected_data[2]['x'][n]
	dfy = selected_data[2]['y'][n]
	dfz = selected_data[2]['z'][n]

	return [
		html.Span('Group: {}'.format('planet'), style=style),
		html.Span('Number: {0:.2f}'.format(n), style=style),
		html.Span('dfx: {0:.2f}'.format(dfx), style=style), 
		html.Span('dfy: {0:.2f}'.format(dfy), style=style),
		html.Span('dfz: {0:.2f}'.format(dfz), style=style),
		html.Span('planet: {}'.format(selected_data[2]['name']))
	]

@app.callback(Output('live-update-graph', 'figure'),
				Input('planet-dropdown', 'value'),
				Input('hidden-value', 'children'),
			  Input('interval-component', 'n_intervals'),
			  State('live-update-graph', 'relayoutData')
			  )
def update_graph_live(selected_option, selected_data, n, layoutdata):
	traces = []
	# Collect some dataura
	for idata in selected_data:
		traces.append(go.Scatter3d(
				x = [idata['x'][n]],
				y = [idata['y'][n]],
				z = [idata['z'][n]],
				name = idata['name'],
				mode = 'markers',
				marker=dict(size=7, color=idata['z'][n], colorscale='Viridis')
			)
		)
	
	fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{'is_3d':True}],[{'is_3d':True}]])
	fig['layout']['margin'] = {
		'l': 30, 'r': 10, 'b': 30, 't': 10
	}
	fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
	#fig['layout']['selectdirection'] = 'any'
	#fig['layout']['selectionrevision'] = True
	if layoutdata and 'scene.camera' in layoutdata:
		fig.update_layout(scene_camera=layoutdata['scene.camera'])
		print(layoutdata['scene.camera'])
	else:
		print('stay layout')

	fig.update_scenes(scene_options[selected_option]
	)
	fig.add_traces(
		traces, 1, 1)

	return fig

if __name__ == '__main__':
	app.run_server(debug=True, port=4040)

