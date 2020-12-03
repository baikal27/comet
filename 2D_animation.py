import matplotlib
matplotlib.use('Qt5Agg') #use Qt5 as backend, comment this line for default backend
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.optimize import newton


fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

orb_elements = np.array([
    [2.5073015178854776, 0.375644335854439, 6.803999338660359, -69.92224985118337, -29.121714435891704, 3.9702671398694944],
    #[1.4321499750543671, 0.2291938430658519, 40.99618734405878, -10.64346336819083, -18.006414363590157 ],#moshup
    [2.5740373, 0.1909134, 5.36742, 358.64840, 141.57102, 4.13], #Astraea
    [0.387098, 0.205633, 7.0045, 29.0882, 48.2163, 0.241],     #Mercury
    [0.723330, 0.006778, 3.3945, 54.8420, 76.5925, 0.615],        #Venus
    [0.997319, 0.016710, 0.00005, 114.20783, 348.73936, 1.000],    #Earth
    [1.523688, 0.093396, 1.8498, 286.3978, 49.4826, 1.881], #Mars
    [5.20256, 0.048482, 1.3036, 273.8194, 100.3561,11.87] #Jupiter
    #[9.55475, 0.055580, 2.4890, 339.2884, 113.5787, 29.47] #Saturn
    ])

N = len(orb_elements)
circles = [plt.plot([], [])[0] for _ in range(N)] #lines to animate

def init():
    #init lines
    for circle in circles:
        circle.set_data([], [])
    return circles #return everything that must be updated

def animate(i):
    #animate lines
    for j,circle in enumerate(circles):
        a, e, I, peri, node, T = orb_elements[j]
        E = cal_kepler(i, T)
        theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
        r = a * (1 - e * np.cos(E + peri))
        circle.set_data(theta, r)
    return circles #return everything that must be updated

def cal_kepler(i, T):
    t = i / 20
    M = 2 * np.pi * t / T
    E = newton(lambda E: E - e * np.sin(E) - M, M)
    return E

for i, orbit in enumerate(orb_elements):
    a, e, I, peri, node, T = orbit
    t = np.linspace(0,100, 10000)
    M = 2 * np.pi * t / T
    E = newton(lambda E: E - e * np.sin(E) - M, M)
    theta = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    r = a * (1 - e * np.cos(E + peri))
    ax.plot(theta, r)

circles = [ax.plot([], [], 'go')[0] for _ in range(N)]
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(1e6), interval=100, blit=True)
plt.show()


def plot_3d(self, data):
    fig = plt.figure()
    ax = plt3.Axes3D(fig)
    data1 = data
    # data2 = data[1]
    ax.plot(data1[0, :], data1[1, :], data1[2, :])
    # ax.plot(data2[0, :], data2[1, :], data2[2, :])
    position1, = ax.plot(data1[0, 0:1], data1[1, 0:1], data1[2, 0:1], 'ro')
    # position2, = ax.plot(data2[0, 0:1], data2[1, 0:1], data2[2, 0:1], 'bo')

    # Setting the axes properties
    # ax.set_xlim3d([-1.0, 1.0])
    ax.set_xlabel('X')
    # ax.set_ylim3d([-1.0, 1.0])
    ax.set_ylabel('Y')
    # ax.set_zlim3d([0.0, 10.0])
    ax.set_zlabel('Z')

    def update(num, data, position):
        position.set_data(data[:2, num:num + 1])
        position.set_3d_properties(data[2, num:num + 1])

    N1 = len(data1[0, :])
    ani1 = animation.FuncAnimation(fig, update, N1, fargs=(data1, position1), interval=5000 / N1, blit=False)
    # N2 = len(data2[0, :])
    # ani2 = animation.FuncAnimation(fig, update, N2, fargs=(data2, position2), interval=5000 / N2, blit=False)
    # ani.save('matplot003.gif', writer='imagemagick')
    plt.show()
    print('end')