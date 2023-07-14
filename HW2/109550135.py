import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def hist_equl(img):
    # calculate histogram
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    # get cumulative distribution function , then normalize it
    cdf = hist.cumsum()
    cdf_norm = cdf * hist.max() / cdf.max()
    
    # calculate equalized histogram
    res = np.interp(img.flatten(), bins[:-1], cdf_norm)
    # reshape equalized image to its original shape
    res = res.reshape(img.shape)
    
    return res
#########
def hist_spec(target, reference):
    # calculate histograms target & reference img
    hist_1, bins_1 = np.histogram(target.ravel(), 256, [0, 256])
    hist_2, bins_2 = np.histogram(reference.ravel(), 256, [0, 256])
    # get cumulative distribution functions , then normalize them
    cdf_1 = hist_1.cumsum()
    cdf_2 = hist_2.cumsum()
    cdf_1_norm = cdf_1 / cdf_1.max()
    cdf_2_norm = cdf_2 / cdf_2.max()
    
    # generate lookup table to map the intensities of reference to target
    l_table = np.interp(cdf_1_norm, cdf_2_norm, np.arange(0, 256))
    # use lookup table to get matched image of the target
    res = l_table[target]

    return res.astype(np.uint8)
#########
def gs_kernel(size, sigma):
    # generate 2D gaussian kernel
    x, y = np.meshgrid(np.arange(-size//2+1,size//2+1), np.arange(-size//2+1,size//2+1))
    kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
    kernel = kernel / np.sum(kernel)
    
    return kernel

def gs_filter(img, size, sigma):
    # generate the gaussian kernel
    kernel = gs_kernel(size, sigma)
    # pad the img with 0s
    padded_img = cv2.copyMakeBorder(img, size//2, size//2, size//2, size//2, cv2.BORDER_CONSTANT, value=0)

    # apply the gaussian filter to each channel
    filtered_channels = []
    for channel in cv2.split(padded_img):
        filtered_channel = cv2.filter2D(channel, -1, kernel)
        filtered_channels.append(filtered_channel)

    # merge filtered channels into single image
    filtered_img = cv2.merge(filtered_channels)
    # remove the padding
    filtered_img = filtered_img[size//2:-size//2, size//2:-size//2]

    return filtered_img

##############################################

# Q1
img = cv2.imread('Q1.jpg', 0)
# do histogram equalization
equalized_img = hist_equl(img)

plt.axis('off'),plt.imshow(equalized_img, cmap='gray')
plt.savefig('Q1_ans.jpg', bbox_inches='tight', pad_inches=0)
tmp_1 = cv2.imread('Q1_ans.jpg')
tmp_1 = cv2.resize(tmp_1, (840, 548), interpolation=cv2.INTER_AREA)
cv2.imwrite("Q1_ans.jpg", tmp_1)
#########

# Q2
target = np.array(mpimg.imread('Q1.jpg', 0))
reference = np.array(mpimg.imread('Q2.jpg', 0))
# do histogram specification on two imgs
matched_img = hist_spec(target, reference)

plt.axis('off'),plt.imshow(matched_img, cmap='gray')
plt.savefig('Q2_ans.jpg', bbox_inches='tight', pad_inches=0)
tmp_2 = cv2.imread('Q2_ans.jpg')
tmp_2 = cv2.resize(tmp_2, (840, 548), interpolation=cv2.INTER_AREA)
cv2.imwrite("Q2_ans.jpg", tmp_2)
#########

# Q3
img = cv2.imread('Q3.jpg', 0)
# do gaussian filter with requested size
filtered_img = gs_filter(img, 5, 25)

plt.axis('off'),plt.imshow(filtered_img, cmap='gray')
plt.savefig('Q3_ans.jpg', bbox_inches='tight', pad_inches=0)
tmp_3 = cv2.imread('Q3_ans.jpg')
tmp_3 = cv2.resize(tmp_3, (1024, 677), interpolation=cv2.INTER_AREA)
cv2.imwrite("Q3_ans.jpg", tmp_3)