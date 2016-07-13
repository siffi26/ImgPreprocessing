import numpy as np
import matplotlib.pyplot as plt
from skimage import io 
from skimage import img_as_uint

from skimage import data, img_as_float
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral

from skimage import io 
from skimage import color
from skimage import img_as_uint,img_as_float


astro = img_as_float(io.imread('/home/mallikarjuna/Desktop/new1/s.png'))

noisy = astro + 0.6 * astro.std() * np.random.random(astro.shape)
noisy = np.clip(noisy, 0, 1)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5), sharex=True,
                       sharey=True, subplot_kw={'adjustable': 'box-forced'})

plt.gray()

ax.imshow(denoise_bilateral(noisy, sigma_range=0.05, sigma_spatial=15))
io.imsave('/home/mallikarjuna/Desktop/pic.png',img_as_uint(denoise_bilateral(noisy, sigma_range=0.05, sigma_spatial=15)))
ax.axis('off')
ax.set_title('Bilateral')

image = color.rgb2gray(io.imread('/home/mallikarjuna/Desktop/pic.png'))
io.imsave('/home/mallikarjuna/Desktop/black.png',img_as_uint(image))

fig.tight_layout()

plt.show()







