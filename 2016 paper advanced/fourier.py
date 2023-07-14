import cv2
import numpy as np
from matplotlib import pyplot as plt

""" Step1 """
img_1 = cv2.imread('Input_images/29.jpg',0)
ft_1 = np.fft.fft2(img_1)
ft_shift_1 = np.fft.fftshift(ft_1)

# get the magnitudes of spectrum
spec_mag_1 = np.abs(ft_shift_1)

plt.axis('off'),plt.imshow(np.log(spec_mag_1+1), cmap = 'gray')
plt.savefig('spec_29.jpg', bbox_inches='tight', pad_inches=0)

""" Step2 """
#############################
"""row_1, col_1 = ft_1.shape
row_n1, col_n1 = row_1 // 2 , col_1 // 2

d1 = 30 # cut-off frequency
mask1 = np.zeros((row_1,col_1))
for i in range(ft_1.shape[0]):
        for j in range(ft_1.shape[1]):
            mask1[i, j] = np.exp(-((i - row_n1)**2 + (j - col_n1)**2) / (2 * d1**2))"""

# mask1
def mask1(img):
    row_2, col_2 = img.shape
    img[85:145, 200:350] = 0
    img[85:145, 500:650] = 0
    img[85:145, 850:1000] = 0
    img[255:315, 200:350] = 0
    img[255:315, 500:650] = 0
    img[545:605, 200:350] = 0
    img[545:605, 850:1000] = 0
    img[545:605, 1200:1350] = 0
    img[735:795, 200:350] = 0
    img[735:795, 500:650] = 0
    img[735:795, 850:1000] = 0
    img[735:795, 1200:1350] = 0
    # row2=939 col2 = 1513
    print(row_2,col_2)
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

plt.axis('off'),plt.imshow(np.log(filtered_spec_1+1), cmap = 'gray')
plt.savefig('reduce_noise_29.jpg', bbox_inches='tight', pad_inches=0)

""" Step3 """
# apply mask1&2
ft1_filtered = mask1(ft_shift_1)
ift_shift_1 = np.fft.ifftshift(ft1_filtered)

ift_1 = np.fft.ifft2(ift_shift_1)
plt.axis('off'),plt.imshow(ift_1.astype(np.float64), cmap = 'gray')
plt.savefig('fin_29.jpg', bbox_inches='tight', pad_inches=0)
