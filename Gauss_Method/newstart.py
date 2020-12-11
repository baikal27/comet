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

# variables
start=0
m = 10000 # data 개수
internum = 100 # 속도와 관련 10~50

df = pd.read_csv('kepler_XYZ.csv', index_col='time')
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


@app.callback(Output('live-update-text', 'children'),
			  Input('interval-component', 'n_intervals'))
def update_metrics(n):

	style = {'padding': '5px', 'fontSize': '16px'}

	dfx = data[1]['x'][n]
	dfy = data[1]['y'][n]
	dfz = data[1]['z'][n]

	return [
		html.Span('Number: {0:.2f}'.format(n), style=style),
		html.Span('dfx: {0:.2f}'.format(dfx), style=style), 
		html.Span('dfy: {0:.2f}'.format(dfy), style=style),
		html.Span('dfz: {0:.2f}'.format(dfz), style=style),
		html.Span('planet: {}'.format(data[1]['name']))
	]

@app.callback(Output('live-update-graph', 'figure'),
			  Input('interval-component', 'n_intervals'))
def update_graph_live(n):
	traces = []
	# Collect some dataura
	for idata in data:
		traces.append(go.Scatter3d(
				x = [idata['x'][n]],
				y = [idata['y'][n]],
				z = [idata['z'][n]],
				name = idata['name'],
				mode = 'markers',
				marker=dict(size=7, color=idata['z'][n], colorscale='Viridis')
			)
		)
	
	fig = plotly.tools.make_subplots(rows=2, cols=1, vertical_spacing=0.2, specs=[[{'type':'scene'}],[{'type':'scene'}]])
	fig['layout']['margin'] = {
		'l': 30, 'r': 10, 'b': 30, 't': 10
	}
	fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

	fig.add_traces(
		traces, 1, 1)

	return fig

if __name__ == '__main__':
	app.run_server(debug=True, port=4040)

