import dash
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


app = dash.Dash(__name__)

X, Y, Z = np.random.random((3, 50))


app.layout = html.Div([
        dcc.Graph(
            id='main-figure',
            figure=go.Figure(data=go.Scatter3d(x=X, y=Y, z=Z))
            ),
        dcc.Graph(
            id='other-figure',
            figure=go.Figure(data=go.Scatter3d(x=X, y=Y, z=Z))
            ),

        ]
        )

@app.callback(
    Output('other-figure', 'figure'),
    [Input('main-figure', 'relayoutData')])
def sync1(data):
    if data and 'scene.camera' in data:
        fig=go.Figure(data=go.Scatter3d(x=X, y=Y, z=Z))
        fig.update_layout(scene_camera=data['scene.camera'])
        return fig
    else:
        raise PreventUpdate


@app.callback(
    Output('main-figure', 'figure'),
    [Input('other-figure', 'relayoutData')])
def sync2(data):
    if data and 'scene.camera' in data:
        fig=go.Figure(data=go.Scatter3d(x=X, y=Y, z=Z))
        fig.update_layout(scene_camera=data['scene.camera'])
        return fig
    else:
        raise PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True, port=8065)