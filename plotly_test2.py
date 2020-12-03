from orbital import orbit
from usefultools import *
import csv
import pandas as pd
import plotly.graph_objs as go
import random

df_line = pd.read_csv('kepler_XYZ_line.csv')
df = pd.read_csv('kepler_XYZ.csv')
df = df.set_index(df['time'])
#df1 = df[df['name'] == 'Pluto'].reset_index()
planets = df.drop_duplicates("name", keep='first')["name"].values
color_discrete_map = {}
random.seed(300)
for planet in planets:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color_discrete_map[planet] = f'rgb({r},{g},{b})'
#color_discrete_map = {'virginica': 'rgb(255,0,0)', 'setosa': 'rgb(0,255,0)', 'versicolor': 'rgb(0,0,255)'}
#fig = px.scatter(df[df.species.isin(['virginica', 'setosa'])], x="sepal_width", y="sepal_length", color="species", color_discrete_map=color_discrete_map)

# make figure
fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
}

# fill in most of layout
fig_dict["layout"]["xaxis"] = {"range": [df['x'].min(), df['x'].max()], "title": "X"}
fig_dict["layout"]["yaxis"] = {"range": [df['y'].min(), df['y'].max()], "title": "Y"}
#fig_dict["layout"]["zaxis"] = {"range": [df['z'].min(), df['z'].max()], "title": "Z"}
fig_dict["layout"]["hovermode"] = "closest"
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 300,
                                                                    "easing": "quadratic-in-out"}}],
                "label": "Play",
                "method": "animate"
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate"
            }
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 87},
        "showactive": False,
        "type": "buttons",
        "x": 0.1,
        "xanchor": "right",
        "y": 0,
        "yanchor": "top"
    }
]

sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "JD:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 300, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 50},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}

# make data
traces1 = [go.Scatter3d(
      x = df_line[df_line['name'] == planet]['x'], y = df_line[df_line['name'] == planet]['y'], z = df_line[df_line['name'] == planet]['z'],
      mode = 'lines', marker = dict(
         size = 3,
        color = color_discrete_map[planet], # set color to an array/list of desired values
         colorscale = 'Viridis'
         ),
      name = planet
      )
   for planet in planets]
traces2 = [go.Scatter3d(
      x = df_line[df_line['name'] == planet]['x'], y = df_line[df_line['name'] == planet]['y'], z = df_line[df_line['name'] == planet]['z'],
      mode = 'lines', marker = dict(
         size = 3,
         color = color_discrete_map[planet], # set color to an array/list of desired values
         colorscale = 'Viridis'
         ),
      name = planet, showlegend=False
      )
   for planet in planets]
fig_dict["data"] = traces1 + traces2

# make frames
start = 0
end = 10000
step = 50
frames = []
for t in range(start, end, step):
    data = []
    for planet in planets:
        xdata , ydata, zdata = [], [], []
        xdata.append(df[df['name']==planet]['x'][t])
        ydata.append(df[df['name']==planet]['y'][t])
        zdata.append(df[df['name']==planet]['z'][t])
        data.append(go.Scatter3d(x=xdata, y=ydata, z=zdata, mode='markers',
                        marker=dict(size=4, color=color_discrete_map[planet], colorscale='Viridis'
                                    )))
    frames.append(go.Frame(data=data))

    slider_step = {"args": [
        [t],
        {"frame": {"duration": 300, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 300}}
    ],
        "label": t,
        "method": "animate"}
    sliders_dict["steps"].append(slider_step)

fig_dict['frames'] = frames
fig_dict["layout"]["sliders"] = [sliders_dict]

layout = go.Layout(
        title = '3D Scatter plot',
        xaxis=dict(range=[df['x'].min(), df['x'].max()], autorange=False, zeroline=False),
        yaxis=dict(range=[df['x'].min(), df['x'].max()], autorange=False, zeroline=False),
        title_text="Kinematic Generation of a Planar Curve", hovermode="closest",
        updatemenus=[dict(type="buttons",
                          buttons=[dict(label="Play",
                                        method="animate",
                                        args=[None])])])
fig_dict['layout'] = layout
#fig = go.Figure(fig_dict)
fig = go.Figure(fig_dict)
fig.write_html("orbit_planets.html")
fig.show()