import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.optimize import newton

def Gen_RandLine(length, dims, elements, index):
    lineData = np.empty((dims, length))
    a, e, I, peri, node, T = elements
    M = 2 * np.pi * t / T
    E = newton(lambda E: E - e * np.sin(E) - M, M)
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    r = a * (1 - e * np.cos(E + peri))
    I, node, peri = np.radians((I, node, peri))

    x = r * (np.cos(node) * np.cos(peri + theta) - np.sin(node) * np.sin(peri + theta) * np.cos(I))
    y = r * (np.sin(node) * np.cos(peri + theta) + np.cos(node) * np.sin(peri + theta) * np.cos(I))
    z = r * np.sin(peri + theta) * np.sin(I)
    ax.plot(x, y, z, '--')
    ax.text(x[index], y[index], z[index], f'{name_planet[index]}', color='k')

    for i in range(length):
        lineData[0, i] = x[i]
        lineData[1, i] = y[i]
        lineData[2, i] = z[i]
        #ax.text(x[i], y[i], z[i], 'planet', color='k')
    return lineData

def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, num-1:num])
        line.set_3d_properties(data[2, num-1:num])
    return lines

def update_texts(num, dataLines, texts):
    for text, data in zip(texts, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        text.set_data(data[0:2, num-1:num])
        text.set_3d_properties(data[2, num-1:num])
    return texts

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
orb_elements = [
    [2.5073015178854776, 0.375644335854439, 6.803999338660359, -69.92224985118337, -29.121714435891704, 3.9702671398694944],
    #[1.4321499750543671, 0.2291938430658519, 40.99618734405878, -10.64346336819083, -18.006414363590157 ],#moshup
    [2.5740373, 0.1909134, 5.36742, 358.64840, 141.57102, 4.13], #Astraea
    [0.387098, 0.205633, 7.0045, 29.0882, 48.2163, 0.241],     #Mercury
    [0.723330, 0.006778, 3.3945, 54.8420, 76.5925, 0.615],        #Venus
    [0.997319, 0.016710, 0.00005, 114.20783, 348.73936, 1.000],    #Earth
    [1.523688, 0.093396, 1.8498, 286.3978, 49.4826, 1.881], #Mars
    [5.20256, 0.048482, 1.3036, 273.8194, 100.3561,11.87] #Jupiter
    #[9.55475, 0.055580, 2.4890, 339.2884, 113.5787, 29.47] #Saturn
    ]
name_planet = ['mosthup', 'Astraea', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']
num_bodies = len(orb_elements)
t = np.linspace(0, 100, 10000)
t_length = len(t)
dims = 3

data = [Gen_RandLine(t_length, dims, orb_elements[index], index) for index in range(num_bodies)]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0:1, 0], dat[0:1, 1], dat[0:1, 2], 'ro')[0] for dat in data]

'''
# Setthe axes properties
ax.set_xlim3d([0.0, 8*np.pi])
ax.set_xlabel('X')
ax.set_ylim3d([-1.0, 1.0])
ax.set_ylabel('Y')
ax.set_zlim3d([-1.0, 1.0])
ax.set_zlabel('Z')
ax.set_title('3D Test')
'''

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, t_length, fargs=(data, lines),
                                   interval=50, blit=False)

plt.show()

