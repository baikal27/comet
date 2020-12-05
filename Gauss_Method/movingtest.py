# Importing plotly modules
import plotly.offline as ofl
import plotly.graph_objs as go

# Importing numpy to create data
import numpy as np

# Important to initialize notebook mode to visualize plots within notebook
#ofl.init_notebook_mode()

# Creating first data trace
x, y, z = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

# Creating second data trace
x2, y2, z2 = np.random.multivariate_normal(np.array([0,0,0]), np.eye(3), 200).transpose()
trace2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        color='rgb(127, 127, 127)',
        size=12,
        symbol='circle',
        line=dict(
            color='rgb(204, 204, 204)',
            width=1
        ),
        opacity=0.9
    )
)

# Defining data
data = [trace1, trace2]

# Defining layout
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

# Creating Figure object
fig = go.Figure(data=data, layout=layout)

# Visualizing the plot
#ofl.iplot(fig, filename='simple-3d-scatter')
fig.show()