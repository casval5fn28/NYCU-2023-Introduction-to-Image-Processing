import cv2
import numpy as np

def median_filter(image, kernel_size):
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image


for i in range(1,31):#1,31
    ss_1 = "Input_images/"+str(i)+".jpg"
    image = cv2.imread(ss_1)

    # Apply median filter with kernel size 3x3
    filtered_image = median_filter(image, 5)

    # Save the filtered image
    ss_2 = "new_img/new_"+str(i)+".jpg"#"Result_images/test_"+str(i)+".jpg"
    cv2.imwrite(ss_2, filtered_image)

