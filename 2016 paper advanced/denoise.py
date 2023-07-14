import cv2

def denoise_image(image, h, hColor, templateWindowSize=7, searchWindowSize=21):
    # Perform non-local means denoising on the image
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, hColor, templateWindowSize, searchWindowSize)
    return denoised_image

# Load the noisy image
image = cv2.imread('test_sat.jpg')#Input_images/29.jpg#test_sat.jpg

# Perform image denoising
denoised_image = denoise_image(image, h=4, hColor=4, templateWindowSize=7, searchWindowSize=21)

# Display the original and denoised images
cv2.imwrite('test_issat.jpg', denoised_image)#Input_images/29.jpg
"""cv2.imshow('Noisy Image', image)
cv2.imshow('Denoised Image', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""
