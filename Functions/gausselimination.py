import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
import astropy
import astropy.units as u
import sympy
import pandas as pd
from scipy import optimize

def deg_to_rad(deg):
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

def POLYRegression3(X, Y):
    #print(f'len(X) : {len(X)}')
    #print(f'len(Y) : {len(Y)}')
    if not len(X) == len(Y):
        breakpoint()
    xsum = np.sum(X)
    ysum = np.sum(Y)
    xsquare = np.sum(X**2)
    xtriple = np.sum(X**3)
    xforth = np.sum(X**4)
    xysum = np.sum(X*Y)
    xsquarey = np.sum(X**2*Y)

    newarr = np.array([[xsquare, xtriple, xforth],
                   [xsum, xsquare, xtriple],
                   [len(X), xsum, xsquare]])
    newb = np.array([xsquarey, xysum, ysum])

    a, b, c = gauss_elim(newarr, newb)   # ax**2 + bx + c
    unkn = sympy.Symbol('unkn')
    y = a*unkn**2 + b*unkn + c
    print(f'y: {y}')
    k = 0.017202099
    ydot = sympy.diff(y, unkn) / k
    print(f' ydot: {ydot}')
    yddot = sympy.diff(ydot, unkn) / (k**k)
    print(f' yddot: {yddot}')
    y_func = sympy.lambdify(unkn, y, 'numpy')
    ydot_func = sympy.lambdify(unkn, ydot, 'numpy')
    yddot_func = sympy.lambdify(unkn, yddot, 'numpy')
    '''
    y = y_func(X)
    print(f'y_func: {y_func}')
    print(f'ydot_func: {ydot_func}')
    print(f'yddot_func: {yddot_func}')
    fig, ax = plt.subplots(num=1)
    ax.plot(X, Y, 'ro')
    ax.plot(X, y, '--')
    plt.show()
    '''
    #print(y, ydot, yddot)
    return (y_func, ydot_func, yddot_func)

def meanequi_to_standequi(g_vector, tjd_given):
    J = (tjd_given - 2451545.) / 36525      # J is the number of Julian centuries from J2000.0

    p11 = 1.0 - 0.00029724*J**2 - 0.00000013*J**3
    p12 = -0.02236172*J - 0.00000677*J**2 + 0.00000222*J**3
    p13 = -0.00971717*J + 0.00000207*J**2 + 0.00000096*J**3
    p21 = -p12
    p22 = 1.0 - 0.00025002*J**2 - 0.00000015*J**3
    p23 = -0.00010865*J**2
    p31 = -p13
    p32 = p23
    p33 = 1.0 - 0.00004721*J**2

    P_arr = np.array([[p11, p12, p13],
                      [p21, p22, p23],
                      [p31, p32, p33]])
    R_vector = np.dot(P_arr, g_vector.T)

    return R_vector

def LagrangeInterpolPolynominal3(x, y, tjd_given=None):
    if not len(x) == len(y):
        breakpoint()
    global vx
    vx = sympy.Symbol('vx')
    L0 = (vx - x[1]) / (x[0] - x[1]) * (vx - x[2]) / (x[0] - x[2])
    L1 = (vx - x[0]) / (x[1] - x[0]) * (vx - x[2]) / (x[1] - x[2])
    L2 = (vx - x[0]) / (x[2] - x[0]) * (vx - x[1]) / (x[2] - x[1])
    f = L0*y[0] + L1*y[1] + L2*y[2]
    k = 0.017202099
    df = sympy.diff(f, vx) / k
    ddf = sympy.diff(df, vx) / (k**2)
    '''
    print(f'f: {f}')
    print(f'df: {df}')
    print(f'ddf: {ddf}')
    breakpoint()
    
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
        yddot_new = ddf.subs(vx, tjd_given) # 3차라서 두번 미분하면 이건 상수. ddf.subs 하는 게 큰 의미는 없다.
        return (y_new, ydot_new, yddot_new)
    elif type(tjd_given) is np.ndarray:
        exp_func = sympy.lambdify(vx, f, 'numpy')
        y_new_arr = exp_func(tjd_given)
        dot_func = sympy.lambdify(vx, df, 'numpy')
        ydot_new_arr = dot_func(tjd_given)
        yddot_new_arr = np.array([ddf]*len(tjd_given)) # ddf is constant
        return (y_new_arr, ydot_new_arr, yddot_new_arr)
    elif tjd_given is None:
        exp_func = sympy.lambdify(vx, f, 'numpy')
        dot_func = sympy.lambdify(vx, df, 'numpy')
        ddot_scalar = ddf
        return (exp_func, dot_func, ddot_scalar)
    else:
        print('what kind of the tjd_given?')

def cal_L_vector(tjd_arr, ra_arr, dec_arr, tjd_given):
    alpha_func, alphadot_func, alphaddot_func = POLYRegression3(tjd_arr, ra_arr)
    delta_func, deltadot_func, deltaddot_func = POLYRegression3(tjd_arr, dec_arr)
    alpha = alpha_func(tjd_given)
    delta = delta_func(tjd_given)
    print(f'tjd_given: {tjd_given}')
    print(f'ra_arr: {ra_arr}, dec_arr: {dec_arr}')
    print(f'alpha: {alpha}, delta: {delta}')
    alphadot = alphadot_func(tjd_given)
    deltadot = deltadot_func(tjd_given)
    alphaddot = alphaddot_func(tjd_given)
    deltaddot = deltaddot_func(tjd_given)
    print(alpha, alphadot, alphaddot, delta, deltadot, deltaddot)
    alpha = deg_to_rad(alpha - 360*(alpha//360))
    alphadot = deg_to_rad(alphadot - 360*(alphadot//360))
    alphaddot = deg_to_rad(alphaddot - 360*(alphaddot//360))
    delta = deg_to_rad(delta - 360*(delta//360))
    deltadot = deg_to_rad(deltadot - 360*(deltadot//360))
    deltaddot = deg_to_rad(deltaddot - 360*(deltaddot//360))
    alpha = deg_to_rad(alpha)
    alphadot = deg_to_rad(alphadot)
    alphaddot = deg_to_rad(alphaddot)
    delta = deg_to_rad(delta)
    deltadot = deg_to_rad(deltadot)
    deltaddot = deg_to_rad(deltaddot)
    '''
    print('result of cal_L_vector')
    print(alpha1, alphadot1, alphaddot1, delta1, deltadot1, deltaddot1)
    alpha2 = deg_to_rad(alpha)
    alphadot2 = deg_to_rad(alphadot)
    alphaddot2 = deg_to_rad(alphaddot)
    delta2 = deg_to_rad(delta)
    deltadot2 = deg_to_rad(deltadot)
    deltaddot2 = deg_to_rad(deltaddot)
    print('result of cal_L_vector')
    print(alpha2, alphadot2, alphaddot2, delta2, deltadot2, deltaddot2)
    '''
    Lx = np.cos(delta) * np.cos(alpha)
    Ly = np.cos(delta) * np.sin(alpha)
    Lz = np.sin(delta)

    cx = np.sin(delta) * np.cos(alpha)
    cy = np.sin(delta) * np.sin(alpha)
    cz = np.cos(delta)

    cxdot = Lx * deltadot - cy * alphadot
    cydot = Ly * deltadot + cx * alphadot
    czdot = -Lz * deltadot

    Lxdot = -cx * deltadot - Ly * alphadot
    Lydot = -cy * deltadot + Lx * alphadot
    Lzdot = cz * deltadot

    Lxddot = -cxdot * deltadot - cx * deltaddot - Lydot * alphadot - Ly * alphaddot
    Lyddot = -cydot * deltadot - cy * deltaddot + Lxdot * alphadot + Lx * alphaddot
    Lzddot = czdot * deltadot + cz * deltaddot

    L_vector = np.array([Lx, Ly, Lz])
    Ldot_vector = np.array([Lxdot, Lydot, Lzdot])
    Lddot_vector = np.array([Lxddot, Lyddot, Lzddot])

    return L_vector, Ldot_vector, Lddot_vector

def cal_L_vector2(tjd_arr, ra_arr, dec_arr, tjd_given, order):
    zra = np.polyfit(tjd_arr, ra_arr, order)  # 3차나 4차에 따라서 차이가 난다. 특히 fdot, fddot이 점점 커진다.
    p = np.poly1d(zra)
    pdot = np.polyder(p)
    pddot = np.polyder(pdot)
    print(f'2 ra pddot : {pddot}')
    alp = deg_to_rad(p(tjd_given) - 360*(p(tjd_given)//360))
    alpdot = deg_to_rad(pdot(tjd_given) - 360*(pdot(tjd_given)//360))
    alpddot = deg_to_rad(pddot(tjd_given) - 360*(pddot(tjd_given)//360))
    zdec = np.polyfit(tjd_arr, dec_arr, order)  # 3차나 4차에 따라서 차이가 난다. 특히 fdot, fddot이 점점 커진다.
    p = np.poly1d(zdec)
    pdot = np.polyder(p)
    pddot = np.polyder(pdot)
    print(f'2 dec pddot: {pddot}')
    delt = deg_to_rad(p(tjd_given) - 360*(p(tjd_given)//360))
    deltdot = deg_to_rad(pdot(tjd_given) - 360*(pdot(tjd_given)//360))
    deltddot = deg_to_rad(pddot(tjd_given) - 360*(pddot(tjd_given)//360))
    print('result of cal_L_vector2')
    print(alp, alpdot, alpddot, delt, deltdot, deltddot)
    lx = np.cos(delt) * np.cos(alp)
    ly = np.cos(delt) * np.sin(alp)
    lz = np.sin(delt)
    kx = np.sin(delt) * np.cos(alp)
    ky = np.sin(delt) * np.sin(alp)
    kz = np.cos(delt)
    kxdot = lx * deltdot - ky * alpdot
    kydot = ly * deltdot + kx * alpdot
    kzdot = -lz * deltdot
    lxdot = -kx * deltdot - ly * alpdot
    lydot = -ky * deltdot + lx * alpdot
    lzdot = kz * deltdot
    lxddot = -kxdot * deltdot - kx * deltddot - lydot * alpdot - ly * alpddot
    lyddot = -kydot * deltdot - ky * deltddot + lxdot * alpdot + lx * alpddot
    lzddot = kzdot * deltdot + kz * deltddot
    L_vector = np.array([lx, ly, lz])
    Ldot_vector = np.array([lxdot, lydot, lzdot])
    Lddot_vector = np.array([lxddot, lyddot, lzddot])

    return L_vector, Ldot_vector, Lddot_vector

def utc_to_tjd_lmst(time_obs, location_obs):
    loc_Elong = location_obs['lon_deg']  # The location of observatory is considered E longitude
    time = Time(time_obs, format='iso', scale='utc')
    tjd_arr = time.jd  # utc to jd , equal to J0
    lmst_list = []
    for i in range(len(tjd_arr)):
        J = (tjd_arr[i] - 2451545.) / 36525  # J is the number of Julian centuries from J2000.0
        ut = time_obs[i].split(' ')[-1].split(':')
        if len(ut) == 3:
            UT = int(ut[0]) + int(ut[1]) / 60 + int(ut[2]) / 3600
        elif len(ut) == 2:
            UT = int(ut[0]) + int(ut[1]) / 60
        else:
            UT = int(ut[0])

        theta0 = 100.4606184 + 36000.77004 * J + 0.000387933 * J ** 2
        GMST = theta0 + 360.98564724 * UT / 24
        LMST = GMST + loc_Elong  # 동경은 더하고, 서경은 빼는데, 빼면 (-)가 나오니 360-서경 해서 동경으로 맞추고, 그냥 GMST - 동경 하는 게 낫다.
        LMST = LMST - 360. * (LMST // 360)  # 단순화
        lmst = deg_to_rad(LMST)
        lmst_list.append(lmst)
    lmst_arr = np.array(lmst_list)

    return tjd_arr, lmst_arr

def cal_geocenter_g_gdot_gddot(lmst_arr):
    lat = deg_to_rad(location_obs['lat_deg'])  # It is considered North latitude
    lat_arr = np.array([lat]*len(lmst_arr))
    f = 1 / 298.257
    F = np.sqrt(1 - (2 * f - f ** 2) * np.sin(lat_arr) ** 2)
    Gc = 1 / F  # + DOAO_site2['height_m']
    Gs = (1 - f) ** 2 / F  # + DOAO_site2['height_m']
    gix = -Gc * np.cos(lat_arr) * np.cos(lmst_arr)
    giy = -Gc * np.cos(lat_arr) * np.sin(lmst_arr)
    giz = -Gs * np.sin(lat_arr)
    ke = 0.07436680
    londot = 0.0043752695 / ke  # londot is constant
    gixdot = Gc * np.cos(lat_arr) * np.sin(lmst_arr) * londot
    giydot = -Gc * np.cos(lat_arr) * np.cos(lmst_arr) * londot
    gizdot = 0.
    gixddot = -giydot * londot
    giyddot = gixdot * londot
    gizddot = 0.

    return (gix, giy, giz), (gixdot, giydot, gizdot), (gixddot, giyddot, gizddot)

def cal_heliocenter_R_Rdot_Rddot(gi, gidot, giddot, intpoled_ra_arr, intpoled_dec_arr):
    gix, giy, giz = gi
    gixdot, giydot, gizdot = gidot
    gixddot, giyddot, gizddot = giddot

    helio_ra_arr, helio_radot_arr, helio_raddot_arr = intpoled_ra_arr
    helio_dec_arr, helio_decdot_arr, helio_decddot_arr = intpoled_dec_arr
    helio_ra_arr = deg_to_rad(helio_ra_arr - 360.*(helio_ra_arr//360))
    helio_radot_arr = deg_to_rad(helio_radot_arr - 360.*(helio_radot_arr//360))
    helio_raddot_arr = deg_to_rad(helio_raddot_arr - 360.*(helio_raddot_arr//360))
    helio_dec_arr = deg_to_rad(helio_dec_arr - 360.*(helio_dec_arr//360))
    helio_decdot_arr = deg_to_rad(helio_decdot_arr - 360.*(helio_decdot_arr//360))
    helio_decddot_arr = deg_to_rad(helio_decddot_arr - 360.*(helio_decddot_arr//360))
    print(intpoled_ra_arr[0])
    print(helio_ra_arr)

    # conversion radius of Earth to the AU unit of the heliocentric system
    Ae = 4.263523e-5 # 6400 km / 1.5e8 km
    gix = Ae * gix
    giy = Ae * giy
    giz = Ae * giz
    gixdot = Ae * gixdot
    giydot = Ae * giydot
    gizdot = Ae * gizdot
    gixddot = Ae * gixddot
    giyddot = Ae * giyddot
    gizddot = Ae * gizddot

    Gix = 1. * np.cos(helio_dec_arr)*np.cos(helio_ra_arr) # 1 indicates 1 au
    Giy = 1. * np.cos(helio_dec_arr)*np.sin(helio_ra_arr) # 1 indicates 1 au
    Giz = 1. * np.sin(helio_dec_arr)                      # 1 indicates 1 au

    Rix = gix + Gix
    Riy = giy + Giy
    Riz = giz + Giz

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

    Rixdot = gixdot + Gixdot
    Riydot = giydot + Giydot
    Rizdot = gizdot + Gizdot
    Rixddot = gixddot + Gixddot
    Riyddot = giyddot + Giyddot
    Rizddot = gizddot + Gizddot

    return (Rix, Riy, Riz), (Rixdot, Riydot, Rizdot), (Rixddot, Riyddot, Rizddot)

def cal_(L, Ldot, Lddot, R, Rdot, Rddot):
    if not L.shape == Ldot.shape == Lddot.shape == R.shape == Rdot.shape == Rddot.shape:
        breakpoint()
    D0 = np.dot(L, np.cross(Ldot, Lddot).T)
    A = np.dot(L, np.cross(Ldot, Rddot).T) / D0
    B = np.dot(L, np.cross(Ldot, R).T) / D0
    C = np.dot(L, np.cross(Rddot, Lddot).T) / (2*D0)
    D = np.dot(L, np.cross(R, Lddot).T) / (2*D0)
    E = -2*np.dot(L, R.T)
    F = np.dot(R, R.T)
    print(A, B, C, D, E, F)
    r = sympy.Symbol('r', positive=True)
    print(r)
    mu = 1.
    p = A + (mu*B)/r**3
    pdot = C + (mu*D)/r**3

    # find the r (root)
    a = -(A**2 + A*E + F)
    b = -mu*(2*A*B + B*E)
    c = -(mu**2)*(B**2)

    f = r**8 + a*r**6 + b*r**3 + c
    df = sympy.diff(f, r)
    f =sympy.lambdify(r, f, 'numpy')
    df = sympy.lambdify(r, df, 'numpy')

    fig, ax = plt.subplots(num=1)
    x = np.linspace(-10, 10, 100)
    y = f(x)
    ax.plot(x, y, 'ro')
    plt.show()
    for i in range(len(x)):
        print(x[i], y[i])
    breakpoint()


    # find the root using the method of Newton-Raphson
    xi = 10.
    tolerence = 1.e-10
    while -tolerence < f(xi) < tolerence:
        xs = xi - f(xi)/df(xi)
        xi = xs
        print('1')
    r_scalar = xi
    print(f' r(Newton-Raphson): {r_scalar}, {f(r_scalar)}')
    # find the root using the method of Bisection
    #rr = optimize.bisect(f, 1, 3)
    #print(f' r(Bisection): {rr}')

    p_scalar = p.subs(r, xi)
    pdot_scalar = pdot.subs(r, xi)
    print(r_scalar, p_scalar, pdot_scalar)

    r_vector = p_scalar*L - R
    rdot_vector = pdot_scalar*L + p_scalar*Ldot - Rdot
    r_vector = r_vector.astype(np.float64)
    rdot_vector = rdot_vector.astype(np.float64)

    return r_vector, rdot_vector

def rv_vec_to_6elems(r, v):
    todeg = 180 / np.pi
    torad = np.pi / 180
    mu = 1.

	#r = np.array([3 * np.sqrt(3) / 4, 3 / 4, 0])
	#v = np.array([-1 / np.sqrt(8), np.sqrt(3 / 8), 1 / np.sqrt(2)])
    #print(f'r:{r}')
    #print(f'v:{v}')
    print(np.linalg.norm(r))
    k = np.array([0, 0, 1])

    h = np.cross(r, v)
    e = 1 / mu * ((np.dot(v, v.T) - mu / np.linalg.norm(r)) * r - (np.dot(r, v.T) * r))
    n = np.cross(k, h)

    print('angular momentum: {}'.format(h))
    print('eccentricity: {}'.format(e))
    print('ascending node: {} \n'.format(n))

    i = np.arccos(h[2] / np.linalg.norm(h))
    omega = np.arccos(n[0] / np.linalg.norm(n))
    smallomega = np.arccos(np.dot(n, e) / (np.linalg.norm(n) * np.linalg.norm(e)))
    nuzero = np.arccos(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r)))
    uzero = np.arccos(np.dot(n, r) / (np.linalg.norm(n) * np.linalg.norm(r)))

    i = i * todeg
    omega = omega * todeg
    smallomega = smallomega * todeg
    nuzero = nuzero * todeg
    uzero = uzero * todeg

    if n[1] < 0:
    	omega = 360 - omega

    if e[2] < 0:
    	smallomega = 360 - smallomega

    if np.dot(r, v) < 0:
    	nuzero = 360 - nuzero

    if r[2] < 0:
    	uzero = 360 - uzero

    print('eccentricity: {}'.format(np.linalg.norm(e)))
    print('semi-major axis: {}'.format(np.dot(h, h) / (mu * (1 - np.dot(e, e)))))
    print('inclination: {}'.format(i))
    print('omega: {}'.format(omega))
    print('small omega: {}'.format(smallomega))
    print('nu0: {}'.format(nuzero))
    print('u0: {}'.format(uzero))

'''
---Time: UT-----  ---R.A.__(ICRF)__DEC---  ---R.A.__(a-apparent)__DEC--
2020-04-22 00:00  08 27 26.17 +20 24 04.0 08 28 35.45 +20 20 03.6
2020-05-22 00:00  09 15 47.42 +17 58 08.6 09 16 54.30 +17 53 07.2
2020-06-22 00:00  10 11 56.70 +13 48 51.0 10 13 00.92 +13 42 55.2
--- Solar position with GHA, DEC ---------- LHA = GHA + Longi if LHA > 360deg, LHA = LHA - 360deg
2020-Apr-22 00:00     01 59 31.15 +12 11 10.3
2020-May-22 00:00     03 55 50.94 +20 22 35.4
2020-Jun-22 00:00     06 03 21.23 +23 26 03.9
'''
'''
time_obs = ['2019-01-22 00:00', '2019-02-22 00:00', '2019-03-22 00:00']
ra_obs = [' 00 12 45.01', '00 52 58.46', '01 34 36.69']
dec_obs = [' -03 01 02.6', ' +01 28 45.3', '+05 47 44.9']
time_sun = ['2019-01-22 00:00', '2019-02-22 00:00', '2019-03-22 00:00', '2019-04-22 00:00']
ra_sun = ['20 15 35.94 ', '22 20 01.19 ', '00 03 57.46', '01 57 44.05']
dec_sun = ['-19 47 25.1', '-10 22 53.3', ' +00 25 36.9', '+12 01 25.4']
'''

#DOAO_site = {'lon_dms':'127d26m49s', 'lat_dms':'34d41m34s', 'height_m':81}
location_obs = {'lon_deg':127.44694444444447, 'lat_deg':34.692777777777785, 'height_m':81} # DOAO
#location_obs = {'lon_deg':77.59472, 'lat_deg':37.52583, 'height_m':100} # In textbook
# radec_obs is the apparent RA-DEC of 5 Astraea at DOAO topo-site.

time_obs = ['2019-01-22 00:00', '2019-02-22 00:00', '2019-03-22 00:00']
ra_obs = [' 00 12 45.01', '00 52 58.46', '01 34 36.69']
dec_obs = [' -03 01 02.6', ' +01 28 45.3', '+05 47 44.9']
time_sun = ['2019-01-22 00:00', '2019-02-22 00:00', '2019-03-22 00:00', '2019-04-22 00:00']
ra_sun = ['20 15 35.94 ', '22 20 01.19 ', '00 03 57.46', '01 57 44.05']
dec_sun = ['-19 47 25.1', '-10 22 53.3', ' +00 25 36.9', '+12 01 25.4']

'''
time_obs = ['2020-05-20 00:00', '2020-05-30 00:00', '2020-06-09 00:00']
ra_obs = ['09 12 18.88', '09 29 55.75', '09 48 00.35']
dec_obs = ['+18 11 02.8', '+17 02 21.5', '+15 43 52.1']
time_sun = ['2020-05-20 00:00', '2020-05-30 00:00', '2020-06-09 00:00', '2020-06-19 00:00']
ra_sun = ['03 47 50.30', '04 28 15.39', '05 09 21.41', '05 50 52.57']
dec_sun = ['+19 58 09.8', '+21 45 43.7', '+22 55 30.6', '+23 25 08.7']
ra_obs_ap = ['09 13 25.94', '09 31 01.96', '09 49 05.68']
ra_obs_ap = ['+18 06 05.5', '+16 57 04.6', '+15 38 17.0']
ra_sun_ap = ['03 48 58.63', '04 29 25.51', '05 10 32.83', '05 52 04.60']
dec_sun_ap = ['+20 01 43.0', '+21 48 15.8', '+22 56 55.5', '+23 25 22.9']
'
time_obs = ['2020-01-20 00:00', '2020-04-20 00:00', '2020-07-20 00:00']
ra_obs = ['08 09 30.83', '08 25 45.49', '11 05 18.12']
dec_obs = ['+16 18 07.5', '+20 26 01.3', '+08 55 34.7']
time_sun = ['2020-01-20 00:00', '2020-04-20 00:00', '2020-07-20 00:00']
ra_sun = ['20 06 05.58', '01 53 05.31', '07 59 19.26']
dec_sun = ['-20 16 56.0', '+11 36 05.1', '+20 36 31.2']

target radot, decdot : degree
-32.0809  15.21911
50.11619  -6.96533
69.58144  -27.3680
sun radot, decdot : degree
148.3433  31.22390
136.1100  51.60732
139.3398  -27.8128
'''
ra = Angle(ra_obs, unit=(u.hourangle)).deg
ra_arr = np.array(ra)
dec = Angle(dec_obs, unit=(u.deg)).deg
dec_arr = np.array(dec)
sun_ra = Angle(ra_sun, unit=(u.hourangle)).deg
sun_ra_arr = np.array(sun_ra)
sun_dec = Angle(dec_sun, unit=(u.deg)).deg
sun_dec_arr = np.array(sun_dec)

# ------------------------------- The vector of topocentric geocenter ------------------------------------------- #
tjd_arr, lmst_arr = utc_to_tjd_lmst(time_obs, location_obs)
#print(type(tjd_arr), tjd_arr.shape)

#(gix, giy, giz), (gixdot, giydot, gizdot), (gixddot, giyddot, gizddot) = cal_geocenter_g_gdot_gddot(lmst_arr)
gi, gidot, giddot = cal_geocenter_g_gdot_gddot(lmst_arr) # gi = (gix, giy, giz), gidot = (gixdot, giydot, gizdot)

#df = pd.DataFrame({'obs_time':time_obs, 'tjd': tjd_arr, 'lmst': lmst_arr})
'''
g_vector = np.array([gx, gy, gz])
gdot_vector = np.array([gxdot, gydot, gzdot])
gddot_vector = np.array([gxddot, gyddot, gzddot])

R_vector = meanequi_to_standequi(g_vector, tjd_given)
Rdot_vector = meanequi_to_standequi(gdot_vector, tjd_given)
Rddot_vector = meanequi_to_standequi(gddot_vector, tjd_given)

print(f'g_vector: {g_vector}, R_vector: {R_vector}')
print(f'gdot_vector: {gdot_vector}, Rdot_vector: {Rdot_vector}')
print(f'gddot_vector: {gddot_vector}, Rddot_vector: {Rddot_vector}')
'''
# ------------------------------- The vector of topocentric Heliocenter ------------------------------------------- #
time_sun = Time(time_sun, format='iso', scale='utc')
tjd_sun_arr = time_sun.jd # utc to jd , equal to J0, numpy.ndarray
#print(tjd_sun_arr.shape)
#tjd_given = tjd_sun_arr[0] + 0.5

#intpoled_ra_ = LagrangeInterpolPolynominal3(tjd_sun_arr, sun_ra_arr, tjd_given=tjd_given)
#intpoled_dec = LagrangeInterpolPolynominal3(tjd_sun_arr, sun_dec_arr, tjd_given=tjd_given)
intpoled_sun_ra_arr = LagrangeInterpolPolynominal3(tjd_sun_arr, sun_ra_arr, tjd_given=tjd_arr) # ra_arr, radot_arr, raddot_arr = intpoled_ra_arr
intpoled_sun_dec_arr = LagrangeInterpolPolynominal3(tjd_sun_arr, sun_dec_arr, tjd_given=tjd_arr) # dec_arr, decdot_arr, decddot_arr = intpoled_dec_arr
#intpoled_ra_func = LagrangeInterpolPolynominal3(tjd_sun_arr, sun_ra_arr) # ra_func, radot_func, raddot_func = intpoled_ra_func
#intpoled_dec_func = LagrangeInterpolPolynominal3(tjd_sun_arr, sun_dec_arr) # dec_func, decdot_func, decddot_func = intpoled_dec_func
print(f' tjd_arr: {tjd_arr}')
print(f' tjd_sun_arr: {tjd_sun_arr}')
int_sun_ra_arr, _, _ = intpoled_sun_ra_arr
int_sun_dec_arr, _, _ = intpoled_sun_dec_arr
print(f'int_sun_ra_arr: {int_sun_ra_arr}')
print(f'sun_ra_arr: {sun_ra_arr}')
print(f'int_sun_dec_arr: {int_sun_dec_arr}')
print(f'sun_dec_arr: {sun_dec_arr}')

Ri, Ridot, Riddot = cal_heliocenter_R_Rdot_Rddot(gi, gidot, giddot, intpoled_sun_ra_arr, intpoled_sun_dec_arr)

gix, giy, giz = gi
gixdot, giydot, gizdot = gidot
gixddot, giyddot, gizddot = giddot
Rix, Riy, Riz = Ri
Rixdot, Riydot, Rizdot = Ridot
Rixddot, Riyddot, Rizddot = Riddot

df = pd.DataFrame({'obs_time':time_obs, 'tjd': tjd_arr, 'lmst': lmst_arr, 'ra':ra_arr, 'dec':dec_arr, #'gix': gix, 'giy':giy, 'giz':giz,
                   #'gixdot':gixdot, 'giydot':giydot, 'gizdot':gizdot, 'gixddot':gixddot, 'giyddot':giyddot, 'gizddot':gizddot,
                   #'helio_ra_arr':helio_ra_arr, 'helio_radot_arr':helio_radot_arr, 'helio_raddot_arr':helio_raddot_arr,
                   #'helio_dec_arr':helio_dec_arr, 'helio_decdot_arr':helio_decdot_arr, 'helio_decddot_arr':helio_decddot_arr,
                   'Rix': Rix, 'Riy': Riy, 'Riz': Riz, 'Rixdot': Rixdot, 'Riydot': giydot, 'Rizdot': gizdot,
                   'Rixddot': Rixddot, 'Riyddot': Riyddot, 'Rizddot': Rizddot})

given_idx = 1
tjd_given = tjd_arr[given_idx]
R_vector = np.array([Rix[given_idx], Riy[given_idx], Riz[given_idx]])
Rdot_vector = np.array([Rixdot[given_idx], Riydot[given_idx], Rizdot[given_idx]])
Rddot_vector = np.array([Rixddot[given_idx], Riyddot[given_idx], Rizddot[given_idx]])
print(R_vector)
print(Rdot_vector)
print(Rddot_vector)
'''
tjd_arr = np.array([95.08167, 96.08167, 97.08167, 98.08167, 99.08167, 100.08167, 101.08167, 102.08167, 103.08167])
ra_arr = np.array([2.09624, 2.73443, 3.25454, 3.68088, 4.03544, 4.33550, 4.59395, 4.82023, 5.02129])
dec_arr = np.array([2.84935, 2.39724, 1.78343, 1.08638, 0.35237, -0.39552, -1.14379, -0.88538, -2.61638])
tjd_given = 99.0816700
'''
print(f'tjd_given: {tjd_given}')
L_vector, Ldot_vector, Lddot_vector = cal_L_vector(tjd_arr, ra_arr, dec_arr, tjd_given)
#L_vector, Ldot_vector, Lddot_vector = cal_L_vector2(tjd_arr, ra_arr, dec_arr, tjd_given, 3)
#print(L_vector, L_vector2)
#print(Ldot_vector, Ldot_vector2)
#print(Lddot_vector, Lddot_vector2)

print(f'utc_given: {time_obs[given_idx]}')
print(f'tjd_given: {tjd_given}')
print(f'L_vector: {L_vector}')
print(f'R_vector: {R_vector}')
print(f'Rdot_vector: {Rdot_vector}')
print(f'Rddot_vector: {Rddot_vector}')

print(f'R_vector(calculated topocentric heliocenter vector: {R_vector}') # the topocentric heliocenter vector calculated by functions
#print(intpoled_sun_ra_arr[given_idx], intpoled_sun_dec_arr[given_idx])
sun_loc = astropy.coordinates.get_sun(Time(time_obs[given_idx], format='iso', scale='utc')).cartesian # geocentric heiliocenter vector
#sun_loc = astropy.coordinates.get_sun(Time(2446374.57284, format='jd')).cartesian # geocentric heiliocenter vector
print(f'sun_loc_geocentric(from get_sun in astropy): {sun_loc}')

sun_loc_topo = np.array([sun_loc.x.value + gix[1]*4.263523e-5, sun_loc.y.value + giy[1]*4.263523e-5, sun_loc.z.value + giz[1]*4.263523e-5])
print(f'sun_loc_topocentric(add topocentric geocenter vector): {sun_loc_topo}') # topocentric heliocenter vector

r_vector, rdot_vector = cal_(L_vector, Ldot_vector, Lddot_vector, R_vector, Rdot_vector, Rddot_vector)
print(r_vector, rdot_vector)

rv_vec_to_6elems(r_vector, rdot_vector)

'''
nitial IAU76/J2000 heliocentric ecliptic osculating elements (au, days, deg.):
  EPOCH=  2452007.5 ! 2001-Apr-08.00 (TDB)         Residual RMS= .27148        
   EC= .1926114352345891   QR= 2.078044083084556   TP= 2451379.0306617534      
   OM= 141.6911125304881   W=  357.6756137044301   IN= 5.368439871633877       
  Equivalent ICRF heliocentric equatorial cartesian coordinates (au, au/d):
   X= 1.443478822928470E+00  Y=-2.478165309189651E+00  Z=-9.528687747591245E-01
  VX= 8.214197484389779E-03 VY= 3.582781462309442E-03 VZ= 7.438178504214057E-04
Asteroid physical parameters (km, seconds, rotational period in hours):        
   GM= n.a.                RAD= 53.3495            ROTPER= 16.806              
   H= 6.9                  G= .150                 B-V= .826                   
                           ALBEDO= .274            STYP= S

# t100ddest for LagrangeInterpolPolynominal3
testx = [15, 17, 19, 21]
testy = [0.9238643, 0.9099594, 0.8949607, 0.8788878]
func, dfunc, ddfunc = LagrangeInterpolPolynominal3(testx, testy)
print(func(18), dfunc(18), ddfunc)

# test for POLYRegression3 function
# 1. Satellite GEOS
testx = np.array([95.08167, 96.08167, 97.08167, 98.08167, 99.08167, 100.08167, 101.08167, 102.08167, 103.08167])
testalpha = np.array([2.09624, 2.73443, 3.25454, 3.68088, 4.03544, 4.33550, 4.59395, 4.82023, 5.02129])
testdelta = np.array([2.84935, 2.39724, 1.78343, 1.08683, 0.35237, -0.39552, -1.14379, -1.88538, -2.61638])

f, fdot, fddot = POLYRegression3(testx, testalpha)
print(f'f(99.08167): {f(99.08167)}')
print(f'fdot(99.08167): {fdot(99.08167)}')
print(f'fddot(99.08167): {fddot}')

f, fdot, fddot = POLYRegression3(testx, testdelta)
print(f'f(99.08167): {f(99.08167)}')
print(f'fdot(99.08167): {fdot(99.08167)}')
print(f'fddot(99.08167): {fddot(99.08167)}')

z = np.polyfit(testx, testdelta, 4) # 3차나 4차에 따라서 차이가 난다. 특히 fdot, fddot이 점점 커진다.
p = np.poly1d(z)
print(f' f(99.08167): {p(99.08167)}')
p2 = np.polyder(p)
print(f' fdot(99.08167): {p2(99.08167)}')
p3 = np.polyder(p2)
print(f' fdot(99.08167): {p3(99.08167)}')

# 2. Comet Rebek-Jewel
t = np.array([2446370.57744,2446371.57632,2446372.57518,2446373.57402,2446374.57284,2446375.57163,2446376.57040,2446377.56916,2446378.56789,2446379.56659])
t = t - 2440000
ra = np.array([5.41652, 5.35088, 5.28090, 5.20632, 5.12686, 5.04222, 4.95212, 4.85627, 4.75436, 4.64614])
dec = np.array([21.85272, 21.92988, 22.00440, 22.07521, 22.14104, 22.20041, 22.25161, 22.29266, 22.32127, 22.33487])
print(len(t), len(ra), len(dec))
f, fdot, fddot = POLYRegression3(t, ra)
givent = 6374.57284
print(f'f(99.08167): {f(givent)}')
print(f'fdot(99.08167): {fdot(givent)}')
print(f'fddot(99.08167): {fddot(givent)}')

z = np.polyfit(t, ra, 4) # 3차나 4차에 따라서 차이가 난다. 특히 fdot, fddot이 점점 커진다.
p = np.poly1d(z)
print(p)
print(f' f(99.08167): {p(givent)}')
p2 = np.polyder(p)
print(f' fdot(99.08167): {p2(givent)}')
p3 = np.polyder(p2)
print(f' fdot(99.08167): {p3(givent)}')
'''