import cv2
import numpy as np

def unsharp_masking(image, sigma=1.0, strength=1.5):
    # Convert image to float32 for accurate calculations
    image = image.astype(np.float32)
    
    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Calculate the difference between the blurred image and the original image
    mask = image - blurred
    
    # Multiply the mask by the strength factor
    mask *= strength
    
    # Add the mask back to the original image
    sharpened = image + mask
    
    # Clip values to the valid range [0, 255]
    sharpened = np.clip(sharpened, 0, 255)
    
    # Convert image back to uint8 data type
    sharpened = sharpened.astype(np.uint8)
    
    return sharpened

# Load the image
image = cv2.imread('res_29.jpg')

# Apply unsharp masking
sharpened_image = unsharp_masking(image, sigma=2.0, strength=1)

# Display the original and sharpened images
cv2.imwrite('test_unsharp.jpg', sharpened_image)
"""cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
