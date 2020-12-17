import plotly.graph_objects as go
from skimage import io
#from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot
#init_notebook_mode(connected=True)
#import chart_studio.plotly as py
#%matplotlib inline

import numpy as np

x = np.linspace(0,5, 1000)
y = np.linspace(0, 5, 1000)
X, Y = np.meshgrid(x,y)
z = (X+Y)/(2+np.cos(x)*np.sin(y))

image = io.imread('milkyway2.jpg') #https://github.com/empet/Datasets/blob/master/Images/lena.png
print(image.shape)
image = image[:,:,2]

surfcolor = np.fliplr(image[:,:])
pl_grey =[[0.0, 'rgb(0, 0, 0)'],
 [0.05, 'rgb(13, 13, 13)'],
 [0.1, 'rgb(29, 29, 29)'],
 [0.15, 'rgb(45, 45, 45)'],
 [0.2, 'rgb(64, 64, 64)'],
 [0.25, 'rgb(82, 82, 82)'],
 [0.3, 'rgb(94, 94, 94)'],
 [0.35, 'rgb(108, 108, 108)'],
 [0.4, 'rgb(122, 122, 122)'],
 [0.45, 'rgb(136, 136, 136)'],
 [0.5, 'rgb(150, 150, 150)'],
 [0.55, 'rgb(165, 165, 165)'],
 [0.6, 'rgb(181, 181, 181)'],
 [0.65, 'rgb(194, 194, 194)'],
 [0.7, 'rgb(206, 206, 206)'],
 [0.75, 'rgb(217, 217, 217)'],
 [0.8, 'rgb(226, 226, 226)'],
 [0.85, 'rgb(235, 235, 235)'],
 [0.9, 'rgb(243, 243, 243)'],
 [0.95, 'rgb(249, 249, 249)'],
 [1.0, 'rgb(255, 255, 255)']]


surf = go.Surface(x=x, y=y, z=z,
				  surfacecolor=surfcolor,
				  colorscale=pl_grey,
				  showscale=False) 
layout = go.Layout(
		 title='Mapping an image onto a surface', 
		 font_family='Balto',
		 width=800,
		 height=800,
		 scene=dict(xaxis_visible=True,
					 yaxis_visible=True, 
					 zaxis_visible=True, 
					 aspectratio=dict(x=1,
									  y=1,
									  z=0.5
									 ),
					 camera_eye = {'z':0.1}
					))

fig = go.Figure(data=[surf], layout=layout)
fig.show()
#py.iplot(fig, filename='mappingLenaSurf')


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

'''
	changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
	if 'btn_default' in changed_id:
		fig.update_scenes['camera'] = camera_options['default']
	elif 'btn_below' in changed_id:
		fig.update_scenes['camera'] = camera_options['below']
	elif 'btn_side' in changed_id:
		fig.update_scenes['camera'] = camera_options['side']
'''	

'''
	if layoutdata and 'scene.camera' in layoutdata:
		fig.update_layout(scene_camera=layoutdata['scene.camera'])
		print(layoutdata['scene.camera'])
	else:
		print('stay layout')
'''