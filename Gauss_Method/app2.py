# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from orbital import orbit
from usefultools import *
import csv
import pandas as pd
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

colors = {
    'background': '#111111',
    'text': '#7FDBFF',
    'text2': '#6effff'
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


df_line = pd.read_csv('kepler_XYZ_line.csv')
df1 = df_line[df_line['name'] == 'Mercury']
df2 = df_line[df_line['name'] == 'Venus']
df = pd.read_csv('kepler_XYZ.csv', index_col='time')
planets = df_line.drop_duplicates("name", keep='first')["name"].values

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

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'left',
        'color': colors['text2']
    }),

    dcc.Graph(
        id='example-graph-2',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(port=4020)