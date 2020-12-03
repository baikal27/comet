from astropy.io import fits
import os
import matplotlib.pyplot as plt

working_dir = os.path.join(os.getcwd(), '41P-Tuttle-Giacobini-Kresak/20170323')
os.chdir(working_dir)

name_fits = 'output.fits'
data = fits.getdata(name_fits)
print(type(data), data.shape)

plt.imshow(data, cmap='gray')
plt.colorbar()
plt.show()