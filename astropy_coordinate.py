from astropy.coordinates import get_body_barycentric, get_sun, solar_system_ephemeris, cartesian_to_spherical, CartesianRepresentation, SphericalRepresentation
from astropy.time import Time
from astropy import units as u
import numpy as np

solar_system_ephemeris.set('jpl')
t = Time(['2020-09-10', '2020-09-15', '2020-09-20'], scale='utc')
t1 = Time.now()
r1, r2, r3 = get_body_barycentric('earth', t)
r1 = r1.xyz.value
r2 = r2.xyz.value
r3 = r3.xyz.value


'''
rp = get_body_barycentric('earth', t1)

sr1 = SphericalRepresentation.from_cartesian(r1)
sr2 = SphericalRepresentation.from_cartesian(r2)
sr3 = SphericalRepresentation.from_cartesian(r3)
dist = np.array([sr1.distance.value, sr2.distance.value, sr3.distance.value])
lat = np.array([sr1.lat.value, sr1.lat.value, sr3.lat.value])
lon = np.array([sr1.lon.value, sr2.lon.value, sr3.lon.value])

dc = sr1.unit_vectors()
#x, y, z = r1
#sun1, sun2 = get_sun(t)
print(f'dist: {dist}')
print(f'lat: {lat}')
print(f'lon: {lon}')
'''