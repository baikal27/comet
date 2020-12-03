from astropy.io import fits
import os

working_dir = os.path.join(os.getcwd(), '41P-Tuttle-Giacobini-Kresak/20170323')
os.chdir(working_dir)

name_fits = 'output.fits'

# method 1
# ext 옵션은 프레임 number를 의미. 안쓰면 default로 ext=0 설정.
data = fits.getdata(name_fits)
header = fits.getheader(name_fits, ext=0)
print(header)

# method 2
# header 옵션을 True로 하면, header 객체도 같이 얻을 수 있다.
data, header = fits.getdata(name_fits, header=True)
print(header)
print(header['TARGET'])