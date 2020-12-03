import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sympy

r = np.linspace(-100, 100, 1000)

y1 = r
y2 = np.sqrt(4 + 1/(r**6) + 1/(r**3))
y3 = -np.sqrt(4 + 1/(r**6) + 1/(r**3))

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(r, y1, 'r-')
ax.plot(r, y2)
ax.plot(r, y3, color='b')
plt.show()

r = optimize.bisect(lambda r: r**2 - 4 - 1/(r**6) - 1/(r**3), 1, 3)
print(f'r = {r:.6f}')
