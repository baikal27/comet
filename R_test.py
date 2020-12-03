import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
import astropy
import astropy.units as u
import sympy
import pandas as pd
from scipy.interpolate import UnivariateSpline

def deg_to_rad(deg):
    return deg*np.pi/180.
def deg_to_rad2(deg):
    deg = deg - 360.*(deg//360)
    return deg*np.pi/180.

def gauss_elim(A,b):
    n = len(b)
    b = b.reshape(3,1)
    newarr = np.concatenate((A,b), axis=1)
    line0 = newarr[0,:].copy()
    line1 = newarr[1,:].copy()
    line2 = newarr[2,:].copy()

    Jline = line1 - line0/line0[0]*line1[0]
    line2 = line2 - line0/line0[0]*line2[0]
    Kline = line2 - Jline/Jline[1]*line2[1]
    #print(line0)
    x3 = Kline[3] / Kline[2]
    x2 = (Jline[3] - Jline[2]*x3) / Jline[1]
    x1 = (line0[3] - line0[1]*x2 - line0[2]*x3) / line0[0]
    zeroarr = np.vstack((line0, Jline, Kline))

    return (x3, x2, x1)


def LagrangeInterpolPolynominal3(x, y, tjd_given=None):
    if not len(x) == len(y):
        breakpoint()
    global vx
    vx = sympy.Symbol('vx')
    L0 = (vx - x[1]) / (x[0] - x[1]) * (vx - x[2]) / (x[0] - x[2])
    L1 = (vx - x[0]) / (x[1] - x[0]) * (vx - x[2]) / (x[1] - x[2])
    L2 = (vx - x[0]) / (x[2] - x[0]) * (vx - x[1]) / (x[2] - x[1])
    f = L0 * y[0] + L1 * y[1] + L2 * y[2]
    df = sympy.diff(f, vx)
    ddf = sympy.diff(df, vx)
    print(f' f: {f.subs(vx, tjd_given[1])}')
    print(f' df: {df.subs(vx, tjd_given[1])}')
    print(f' ddf: {ddf}')

    df2 = 1/(2*(x[2]-x[1]))*(-y[0] + y[2])
    ddf2 = 1/ (x[2]-x[1])*(y[0] - 2*y[1] + y[2])
    print(f'f2: {f}')
    print(f'df2: {df2}')
    print(f'ddf2: {ddf2}')
    breakpoint()
    '''
    xlist = np.linspace(x[0], x[-1], 100)
    ylist = exp_func(xlist)
    fig, ax = plt.subplots(num=1)
    ax.plot(x, y, 'ro')
    ax.plot(xlist, ylist, 'b--')
    ax.plot(xlist[30], ylist[30], 'bo')
    plt.show()
    '''
    if type(tjd_given) is np.float64:
        y_new = f.subs(vx, tjd_given)
        ydot_new = df.subs(vx, tjd_given)
        yddot_new = ddf.subs(vx, tjd_given)  # 3차라서 두번 미분하면 이건 상수. ddf.subs 하는 게 큰 의미는 없다.
        print(y_new, ydot_new, yddot_new)
        return (y_new, ydot_new, yddot_new)
    elif type(tjd_given) is np.ndarray:
        exp_func = sympy.lambdify(vx, f, 'numpy')
        y_new_arr = exp_func(tjd_given)
        dot_func = sympy.lambdify(vx, df, 'numpy')
        ydot_new_arr = dot_func(tjd_given)
        yddot_new_arr = np.array([np.float64(ddf)] * len(tjd_given))  # ddf is constant
        return (y_new_arr, ydot_new_arr, yddot_new_arr)
    elif tjd_given is None:
        exp_func = sympy.lambdify(vx, f, 'numpy')
        dot_func = sympy.lambdify(vx, df, 'numpy')
        ddot_scalar = np.float64(ddf)
        return (exp_func, dot_func, ddot_scalar)
    else:
        print('what kind of the tjd_given?')

#def cal_heliocenter_R_Rdot_Rddot(gi, gidot, giddot, intpoled_ra_arr, intpoled_dec_arr):
def cal_heliocenter_R_Rdot_Rddot(helio_ra_arr, helio_radot_arr, helio_raddot_arr, helio_dec_arr, helio_decdot_arr, helio_decddot_arr):
    Gix = 1. * np.cos(helio_dec_arr)*np.cos(helio_ra_arr) # 1 indicates 1 au
    Giy = 1. * np.cos(helio_dec_arr)*np.sin(helio_ra_arr) # 1 indicates 1 au
    Giz = 1. * np.sin(helio_dec_arr)                      # 1 indicates 1 au

    Rix = Gix
    Riy = Giy
    Riz = Giz

    ra_ = helio_ra_arr
    radot = helio_radot_arr
    raddot = helio_raddot_arr
    dec_ = helio_dec_arr
    decdot = helio_decdot_arr
    decddot = helio_decddot_arr

    Gixdot = -np.sin(dec_)*np.cos(ra_)*decdot - np.cos(dec_)*np.sin(ra_)*radot
    Giydot = -np.sin(dec_)*np.sin(ra_)*decdot + np.cos(dec_)*np.cos(ra_)*radot
    Gizdot = np.cos(dec_)*decdot
    Gixddot = -np.cos(dec_)*np.cos(ra_)*decdot**2 + np.sin(dec_)*np.sin(ra_)*decdot*radot - np.sin(dec_)*np.cos(ra_)*decddot \
              +np.sin(dec_)*np.sin(ra_)*radot*decdot - np.cos(dec_)*np.cos(ra_)*radot**2 - np.cos(dec_)*np.sin(ra_)*raddot
    Giyddot = -np.cos(dec_)*np.sin(ra_)*decdot**2 - np.sin(dec_)*np.cos(ra_)*decdot*radot - np.sin(dec_)*np.sin(ra_)*decddot \
              -np.sin(dec_)*np.cos(ra_)*radot*decdot - np.cos(dec_)*np.sin(ra_)*radot**2 - np.cos(dec_)*np.cos(ra_)*raddot
    Gizddot = -np.sin(dec_)*decdot**2 + np.cos(dec_)*decddot

    Rixdot = Gixdot
    Riydot = Giydot
    Rizdot = Gizdot
    Rixddot = Gixddot
    Riyddot = Giyddot
    Rizddot = Gizddot

    return (Rix, Riy, Riz), (Rixdot, Riydot, Rizdot), (Rixddot, Riyddot, Rizddot)

tjd_given = 2446374.57284
time = Time(tjd_given, format='jd')
time_arr = np.array([time-1, time, time+1])
tjd_arr = np.array([time.value-1, time.value, time.value+1])
#print(time.to_datetime())
ra_list = []
dec_list = []
sun_loc = []
sun_x = []
sun_y = []
sun_z = []
for tim in time_arr:
    sun_loc = astropy.coordinates.get_sun(tim)
    ra_list.append(sun_loc.ra.value)
    dec_list.append(sun_loc.dec.value)
    sun_x.append(sun_loc.cartesian.x.value)
    sun_y.append(sun_loc.cartesian.y.value)
    sun_z.append(sun_loc.cartesian.z.value)
ra_arr = np.array(ra_list)
dec_arr = np.array(dec_list)
sun_x_arr = np.array(sun_x)
sun_y_arr = np.array(sun_y)
sun_z_arr = np.array(sun_z)
print(f'tjd_arr: {tjd_arr}')
print(f'ra_arr: {ra_arr}')
print(f'dec_arr: {dec_arr}')
print(f'sun_x:  {sun_x_arr}')
print(f'sun_y:  {sun_y_arr}')
print(f'sun_z:  {sun_z_arr}')

#LagrangeInterpolPolynominal3_test(tjd_arr, ra_arr, tjd_given)

int_alp, int_alpdot, int_alpddot = LagrangeInterpolPolynominal3(tjd_arr, ra_arr, tjd_arr)
int_delt, int_deltdot, int_deltddot = LagrangeInterpolPolynominal3(tjd_arr, dec_arr, tjd_arr)
print(int_alpdot[1])
int_alp_arr = deg_to_rad(int_alp)
int_alpdot_arr = deg_to_rad(int_alpdot)
int_alpddot_arr = deg_to_rad(int_alpddot)
int_delt_arr = deg_to_rad(int_delt)
int_deltdot_arr = deg_to_rad(int_deltdot)
int_deltddot_arr = deg_to_rad(int_deltddot)

zra = np.polyfit(tjd_arr, ra_arr, 3)  # 3차나 4차에 따라서 차이가 난다. 특히 fdot, fddot이 점점 커진다.
p = np.poly1d(zra)
pdot = np.polyder(p)
pddot = np.polyder(pdot)
int_alp_arr = deg_to_rad(p(tjd_given))
int_alpdot_arr = deg_to_rad(pdot(tjd_given))
int_alpddot_arr = deg_to_rad(pddot(tjd_given))
zra = np.polyfit(tjd_arr, dec_arr, 3)  # 3차나 4차에 따라서 차이가 난다. 특히 fdot, fddot이 점점 커진다.
p = np.poly1d(zra)
pdot = np.polyder(p)
pddot = np.polyder(pdot)
int_delt_arr = deg_to_rad(p(tjd_given))
int_deltdot_arr = deg_to_rad(pdot(tjd_given))
int_deltddot_arr = deg_to_rad(pddot(tjd_given))

R, Rdot, Rddot = cal_heliocenter_R_Rdot_Rddot(int_alp_arr, int_alpdot_arr, int_alpddot_arr, int_delt_arr, int_deltdot_arr, int_deltddot_arr)
print(f'R: {R}')
print(f'Rdot: {Rdot}')
print(f'Rddot: {Rddot}')

spl = UnivariateSpline(tjd_arr, ra_arr, k=2, s=0)
print(spl.get_coeffs())
print(spl.derivative(1).get_coeffs())
print(spl.derivative(2).get_coeffs())
print(' ')
