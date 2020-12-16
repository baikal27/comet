import plotly
import plotly.graph_objects as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash


app = dash.Dash(__name__)
app.layout = html.Div(
	html.Div([
		html.H4('test for sub_plots'),
		dcc.Graph(id='subplots'),
		dcc.Interval(
			id='interval-component',
			interval=1*1000, # in milliseconds
			n_intervals=0)
        ])
	)

fig = plotly.tools.make_subplots()
fig.add_traces(
    	[go.Scatter(y=[2, 3, 1])]
    )

@app.callback(Output('subplots', 'figure'),
				Input('interval-component', 'n_intervals'))
def update_figure(valuess):
	return fig


if __name__ == '__main__':
	app.run_server(debug=True, port=5050)