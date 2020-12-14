from astropy.time import Time
from datetime import datetime, date

times = date(1970, 1, 1).isoformat()
print(times)
#print(datetime(times))

t = datetime.fromisoformat('1994-10-03')
tt = Time(t)
print(tt.jd)