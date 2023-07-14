import cv2
import numpy as np
from matplotlib import pyplot as plt

""" Step1 """
img_1 = cv2.imread('test1.tif',0)
img_2 = cv2.imread('test2.tif',0)
ft_1 = np.fft.fft2(img_1)
ft_2 = np.fft.fft2(img_2)
ft_shift_1 = np.fft.fftshift(ft_1)
ft_shift_2 = np.fft.fftshift(ft_2)

# get the magnitudes of spectrum
spec_mag_1 = np.abs(ft_shift_1)
spec_mag_2 = np.abs(ft_shift_2)

plt.axis('off'),plt.imshow(np.log(spec_mag_1+1), cmap = 'gray')
plt.savefig('spectrum_1.tif', bbox_inches='tight', pad_inches=0)
plt.axis('off'),plt.imshow(np.log(spec_mag_2+1), cmap = 'gray')
plt.savefig('spectrum_2.tif', bbox_inches='tight', pad_inches=0)

""" Step2 """
#############################
# mask1
def mask1(img):
    row_1, col_1 = img.shape
    img[0:320, 335:340] = 0
    img[350:row_1, 335:340] = 0
    # row1=662 col1=675
    return img
    
"""row_1, col_1 = ft_1.shape
row_n1, col_n1 = row_1 // 2 , col_1 // 2

d1 = 30 # cut-off frequency
mask1 = np.zeros((row_1,col_1))
for i in range(ft_1.shape[0]):
        for j in range(ft_1.shape[1]):
            mask1[i, j] = np.exp(-((i - row_n1)**2 + (j - col_n1)**2) / (2 * d1**2))"""

# mask2
def mask2(img):
    row_2, col_2 = img.shape
    img[25:50, 40:65] = 0
    img[75:100, 40:65] = 0
    img[155:180, 40:65] = 0
    img[205:230, 40:65] = 0
    img[25:50, 105:130] = 0
    img[75:100, 105:130] = 0
    img[155:180, 105:130] = 0
    img[205:230, 105:130] = 0
    # row2=246 col2 = 168
    #print(col_2)
    return img
"""row_2, col_2 = ft_2.shape
row_n2, col_n2 = row_2 // 2 , col_2 // 2

d2 = 20 # cut-off frequency
mask2 = np.zeros((row_2,col_2))
for i in range(ft_2.shape[0]):
        for j in range(ft_2.shape[1]):
            mask2[i, j] = np.exp(-((i - row_n2)**2 + (j - col_n2)**2) / (2 * d2**2))"""

#############################
# apply mask1&2
filtered_spec_1 = mask1(spec_mag_1)
filtered_spec_2 = mask2(spec_mag_2)

plt.axis('off'),plt.imshow(np.log(filtered_spec_1+1), cmap = 'gray')
plt.savefig('reduce_noise_1.tif', bbox_inches='tight', pad_inches=0)
plt.axis('off'),plt.imshow(np.log(filtered_spec_2+1), cmap = 'gray')
plt.savefig('reduce_noise_2.tif', bbox_inches='tight', pad_inches=0)

""" Step3 """
# apply mask1&2
ft1_filtered = mask1(ft_shift_1)
ft2_filtered = mask2(ft_shift_2)
ift_shift_1 = np.fft.ifftshift(ft1_filtered)
ift_shift_2 = np.fft.ifftshift(ft2_filtered)

ift_1 = np.fft.ifft2(ift_shift_1)
ift_2 = np.fft.ifft2(ift_shift_2)
plt.axis('off'),plt.imshow(ift_1.astype(np.float64), cmap = 'gray')
plt.savefig('fin_1.tif', bbox_inches='tight', pad_inches=0)
plt.axis('off'),plt.imshow(ift_2.astype(np.float64), cmap = 'gray')
plt.savefig('fin_2.tif', bbox_inches='tight', pad_inches=0)
