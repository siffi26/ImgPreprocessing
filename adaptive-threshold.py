import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io 
from skimage import filter 
from skimage import img_as_uint 
from skimage import data


image = io.imread('/home/mallikarjuna/Desktop/white.png')
#image2 = data.page()
#print (type(image2))
import numpy as np
#print(np.shape(image2))
global_thresh = filter.threshold_otsu(image)
binary_global = image > global_thresh
io.imsave('/home/mallikarjuna/Desktop/global2.png',img_as_uint(binary_global))

block_size = 35
binary_adaptive = filter.threshold_adaptive(image, block_size, offset=10)
io.imsave('/home/mallikarjuna/Desktop/adaptive2.png',img_as_uint(binary_adaptive))
fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

for ax in axes:
    ax.axis('off')

plt.show()

