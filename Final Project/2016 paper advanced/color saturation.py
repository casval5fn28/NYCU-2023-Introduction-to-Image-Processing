import cv2
import numpy as np

def modify_saturation(image, saturation_factor):
    # Convert image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split the HSV image into separate channels
    h, s, v = cv2.split(hsv_image)
    
    # Modify the saturation channel
    s = np.clip(s * saturation_factor, 0, 255).astype(np.uint8)
    
    # Merge the modified channels back into the HSV image
    modified_hsv = cv2.merge([h, s, v])
    
    # Convert the modified HSV image back to BGR color space
    modified_image = cv2.cvtColor(modified_hsv, cv2.COLOR_HSV2BGR)
    
    return modified_image

# Load the image
image = cv2.imread('res_29.jpg')

# Modify the saturation of the image
modified_image = modify_saturation(image, saturation_factor=1.5)

# Display the original and modified images
cv2.imwrite('test_sat.jpg', modified_image)
"""cv2.imshow('Original Image', image)
cv2.imshow('Modified Image', modified_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
