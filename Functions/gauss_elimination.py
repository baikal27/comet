import numpy as np
import sympy
import matplotlib.pyplot as plt

def POLYNRegression(t_list, ra_list, order=2):
    A = np.array([[1, sum(t_list), sum([i ** 2 for i in t_list])],
                  [sum(t_list), sum([i ** 2 for i in t_list]), sum([i ** 3 for i in t_list])],
                  [sum([i ** 2 for i in t_list]), sum([i ** 3 for i in t_list]), sum([i ** 4 for i in t_list])]])

    B = np.array([sum(ra_list), sum([t_list[i] * ra_list[i] for i in range(len(t_list))]), \
                  sum([t_list[i] * ra_list[i] ** 2 for i in range(len(t_list))])]).reshape(3, 1)
    print(A)
    print(B)
    return (A, B)

def GUSElim(arrayA, arrayB):
    AB = np.concatenate((arrayA, arrayB), axis=1)
    row1 = AB[0,:].reshape(1,4)
    base1 = (AB[0,:] / AB[0,0]).reshape(1,4)
    row2 = (AB[1,:] - (base1 * AB[1,0])).reshape(1,4)
    row3 = (AB[2,:] - (base1 * AB[2,0])).reshape(1,4)
    base2 = (row2[:] / row2[0,1]).reshape(1,4)
    row3 = row3 - (base2 * row3[0,1])

    AB = np.concatenate((row1,row2,row3), axis=0)
    '''
    AB = np.array([[1, 3, 5, 6],
                       [0, -4, -6, -12],
                       [0, 0, 1, 0]])
    U = np.array([[1, 3, 5],
                        [0, -4, -6],
                        [0, 0, 1]])
    L = np.array([6, -12, 0])
    '''
    af01 = AB[0,1] / AB[0,0]
    af02 = AB[0,2] / AB[0,0]
    bf0 = AB[0,3] / AB[0,0]
    af11 = AB[1,1] - AB[1,0]*af01
    af12 = AB[1,2] - AB[1,0]*af02
    bf1 = AB[1,3] - AB[1,0]*bf0
    af21 = AB[2,1] - AB[2,0]*af01
    af22 = AB[2,2] - AB[2,0]*af02
    bf2 = AB[2,3] - AB[2,0]*bf0
    aff12 = af12 / af11
    bff1 = bf1 / af11
    aff22 = af22 - af21*aff12
    bff2 = bf2 - af21*bff1

    C = bff2 / aff22
    B = (bf1 - af12*C) / af11
    A = (AB[0,3] - AB[0,1]*B - AB[0,2]*C) / AB[0,0]

    return (A,B,C, np.linalg.solve(arrayA, arrayB))

t_list = [1, 2, 3]
ra_list = [2, 6, 10]
t_array, ra_array = POLYNRegression(t_list, ra_list, 2)
sol = GUSElim(t_array, ra_array)
print(sol)

t = np.linspace(1, 3, 100)
f_ra = sol[2] + sol[1]*t + sol[0]*t**2

fig, ax = plt.subplots(num=1)
ax.plot(t_list, ra_list, 'ro')
ax.plot(t, f_ra)
ax.set_xlabel('t')
ax.set_ylabel('f(t)')
plt.show()

fig, ax = plt.subplots(num=2)
t = np.array(t_list)
y = np.array(ra_list)
z = np.polyfit(t, y, 2)
p = np.poly1d(z)
tt = np.linspace(1, 3, 100)
ax.plot(t, y, '.', tt, p(tt), '-')
plt.show()



