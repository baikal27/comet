from astropy.io import fits
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.cm as cm
from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation, GCRS
import astropy.units as u
from astropy.time import Time
import astropy.coordinates as coo
import astropy.coordinates.solar_system as ss

working_dir = os.path.join(os.getcwd(), '41P-Tuttle-Giacobini-Kresak/20170323')
#os.chdir(working_dir)
name_fits = 'output.fits'
hdu = fits.open(name_fits)[0]
data = hdu.data
header = hdu.header
for i in header:
	print(i, header[i])

vmin = np.percentile(data, 90)
vmax = np.percentile(data, 99)

fig, ax = plt.subplots(num=1)
cax = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cm.magma)
plt.colorbar(cax)
#plt.show()

w = wcs.WCS(header)
pixcrd = np.array([[1024, 512],
                   [1024, 512]], dtype=np.float64)
world = w.wcs_pix2world(pixcrd, 0)
print(world)

header_list = ['SITELAT', 'SITELONG', 'JD', 'LST', 'RA', 'DEC', 'AZ', 'ALT', 'DATE-OBS', 'EXPOSURE']

ra, dec = header['RA'], header['DEC']
c = SkyCoord(ra*u.deg, dec*u.deg, frame='gcrs', distance=770*u.pc)
print(f'gcrs_x: {c.cartesian.x}')
print(f'gcrs_y: {c.cartesian.y}')
print(f'gcrs_z: {c.cartesian.z}')
c2 = c.transform_to('icrs')
print(f'icrs_x: {c2.cartesian.x}')
print(f'icrs_y: {c2.cartesian.y}')
print(f'icrs_z: {c2.cartesian.z}')
a = '2010-01-01T00:00:00'
times = ['2020-01-01 00:00', '2020-02-01 00:00', '2020-02-01 00:00']
#ra_dec = ['08 23 25.09 +14 41 41.9', '07 57 33.96 +17 38 16.0', '07 44 16.69 +20 09 57.3']
t = Time(times, format='iso', scale='utc')
print(f't: {t.jd}')

sun_gcrs = coo.get_sun(t)
sun_icrs = sun_gcrs.transform_to('icrs')
print(f'sun_gcrs: {sun_gcrs}')
print(f'sun_icrs: {sun_icrs}')

loc_lon_dms = header['SITELONG'].split(' ')
loc_lon_dms = loc_lon_dms[0] + 'd' + loc_lon_dms[1] + 'm' + loc_lon_dms[2] +'s'
loc_la_dms = header['SITELAT'].split(' ')
loc_la_dms = loc_la_dms[0] + 'd' + loc_la_dms[1] + 'm' + loc_la_dms[2] +'s'
print(f'la: {loc_la_dms}')
print(f'long: {loc_lon_dms}')

loc = EarthLocation.from_geodetic(loc_lon_dms, loc_la_dms, 81*u.m)
loc_posvel = loc.get_gcrs_posvel(t[0])
print(f'loc_xyz : {loc.geocentric}')
print(f'loc_pos: {loc_posvel[0]}')
print(f'loc_vel: {loc_posvel[1]}')

print(EarthLocation.get_site_names())
#print(header['DATE-OBS'])
#u = EarthLocation.get_gcrs_posvel(Time(header['DATE-OBS']))
#print(f'gcrs_posvel: {u}')


print(ra, dec)
coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='gcrs', obstime=header['DATE-OBS'])
coord_icrs = coord.transform_to('icrs')
coord_icrs.representation_type = 'cartesian'
print(coord_icrs)






