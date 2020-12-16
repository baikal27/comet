import numpy as np
import plotly.graph_objects as go
import skimage.io as sio
'''
x = np.linspace(-100,100, 1600)
y = np.linspace(-60,60, 900)
x, z = np.meshgrid(x,y)
y = np.sin(x**2*z)
'''
x = np.linspace(-2, 2, 197)
y = np.linspace(-2, 2, 255)
X, Y = np.meshgrid(x,y)
z = (X+Y)/(2+np.cos(X)*np.sin(Y))

fig = go.Figure(go.Surface(x=x, y=y, z=z,
                           colorscale='RdBu', 
                           showscale=False))
image = sio.imread ("milkyway.jpeg") 
print(image.shape)
img = image[:,:, 1] 
Y = 0.5 * np.ones(y.shape)
fig.add_surface(x=x, y=y, z=z, 
                surfacecolor=np.flipud(img), 
                colorscale='gray',
                #colorscale='solar', 
                #colorscale='viridis', 
                showscale=False)
fig.update_layout(width=600, height=600, 
                  scene_camera_eye_z=0.6, 
                  scene_aspectratio=dict(x=0.9, y=1, z=1));
fig.show()

'''
['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
             'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
             'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
             'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
             'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
             'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
             'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
             'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
             'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
             'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
             'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
             'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
             'ylorrd']
             '''
