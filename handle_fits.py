from astropy.io import fits
import os

working_dir = os.path.join(os.getcwd(), '41P-Tuttle-Giacobini-Kresak/20170323')
os.chdir(working_dir)
name_fits = 'object-000159.fits'

# fits 파일의 프레임 정보를 보여줌.
hdulist = fits.open(name_fits)
print(hdulist.info())

# hdulist[0]는 첫번째 프레임 객체, hdulist[1]는 두번째 프레임 객체를 의미함.

# 첫번째 프레임 객체의 header 정보를 한꺼번에 꺼집어 냄.
print(hdulist[0].header)
# 첫번째 프레임 객체의 header의 정보 수를 표현
print(len(hdulist[0].header))

# 헤더 키워드를 알면 더 쉽게 정보를 파악할 수 있음. 이 키워드는 관측자도 쉽게 바꿀 수 있기 때문에 fits 파일을 열고 직접
# 확인을 해봐야 정확하게 알 수 있음.
# EXPTIME, TELESCOP, TARGNAME, DATE
print('expose time(sec): {}'.format(hdulist[0].header['EXPTIME']))
print('name of telescope: {}'.format(hdulist[0].header['TELESCOP']))
print('date of observation: {}'.format(hdulist[0].header['DATE-OBS']))
print('JD : {}'.format(hdulist[0].header['JD']))

hdulist.close()

data, header = fits.getdata(name_fits, extent=0, header=True)
header['TARGET'] = '41P-Tuttle-Giacobini-Kresak'
#fits.writeto('output.fits', data, header, overwrite=False)

