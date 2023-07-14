import cv2
import numpy as np

def white_balance(image):
    # Convert image to float32 for accurate calculations
    image = image.astype(np.float32)
    
    # Compute average values for each color channel
    avg_b = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_r = np.mean(image[:, :, 2])
    
    # Compute the average gray value
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    # Compute the scaling factors for each channel
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    
    # Apply scaling to each channel
    image[:, :, 0] *= scale_b
    image[:, :, 1] *= scale_g
    image[:, :, 2] *= scale_r
    
    # Clip values to the valid range [0, 255]
    image = np.clip(image, 0, 255)
    
    # Convert image back to uint8 data type
    image = image.astype(np.uint8)
    
    return image

# Load the image
image = cv2.imread('res_29.jpg')#Input_images/11.jpg

# Apply white balancing
balanced_image = white_balance(image)

# Display the original and balanced images
cv2.imwrite('test_white.jpg', balanced_image)
"""cv2.imshow('Original Image', image)
cv2.imshow('Balanced Image', balanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
