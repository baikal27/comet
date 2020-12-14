from datetime import date, datetime
import dash
import dash_html_components as html
import dash_core_components as dcc
import re
from astropy.time import Time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
	dcc.DatePickerRange(
		id='my-date-picker-range',
		min_date_allowed=date(1970, 1, 1),
		max_date_allowed=date(2070, 1, 1),
		initial_visible_month=date(2020, 1, 1),
		start_date = date.today(),
		end_date=date(2050, 8, 25)
	),
	html.Div(id='output-container-date-picker-range')
])

global start_date_object, end_date_object
start_date_object = datetime.fromisoformat('1994-10-03')
start_date_object = datetime.fromisoformat('1994-10-03')
global sjd, ejd
sjd = 0.
ejd = 0.

@app.callback(
	dash.dependencies.Output('output-container-date-picker-range', 'children'),
	dash.dependencies.Input('my-date-picker-range', 'start_date'),
	dash.dependencies.Input('my-date-picker-range', 'end_date'))
def update_output(start_date, end_date):
	global start_date_object, sjd, ejd
	global end_date_object
	string_prefix = 'You have selected: '
	if start_date is not None:
		start_date_object = datetime.fromisoformat(start_date)
		sjd = Time(start_date_object).jd
		start_date_string = start_date_object.strftime('%Y-%m-%d')
		#starting = datetime.fromisoformat(start_date_string)
		string_prefix = string_prefix + 'Start Date: ' + start_date_string + ' | '
	if end_date is not None:
		#end_date_object = date.fromisoformat(end_date)
		end_date_object = datetime.fromisoformat(end_date)
		ejd = Time(start_date_object).jd
		end_date_string = end_date_object.strftime('%Y-%m-%d')
		#ending = datetime.fromisoformat(end_date_string)
		string_prefix = string_prefix + 'End Date: ' + end_date_string
	if len(string_prefix) == len('You have selected: '):
		return 'Select a date to see it displayed here'
	else:
		return [
			html.Span(f'JD of sjd : {sjd}'),
			html.Span(f'JD of sjd : {ejd}')
		]


if __name__ == '__main__':
	app.run_server(debug=True, port=8085)