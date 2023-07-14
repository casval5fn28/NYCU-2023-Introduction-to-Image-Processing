import numpy as np
import cv2
from PIL import Image 
import math

def bilinear_itp(img,dst_h,dsr_w):
    src_h,src_w,_ = img.shape
    img = np.pad(img,((0,1),(0,1),(0,0)),'constant')
    new_size = np.zeros((dst_h,dsr_w,3),dtype=np.uint8)
    for i in range(dst_h):
        for j in range(dsr_w):
            src_x = (i+2)*(src_h/dst_h)-1
            src_y = (j+2)*(src_w/dsr_w)-1
            x = math.floor(src_x)
            y = math.floor(src_y)
            u = src_x-x
            v = src_y-y
            new_size[i,j] = (1-u)*(1-v)*img[x,y] + u*(1-v)*img[x+1,y] + (1-u)*v*img[x,y+1] + u*v*img[x+1,y+1]
    return new_size

def weights(x):
    x = abs(x)
    if x <= 1:
        return 1-2*(x**2)+(x**3)
    elif x < 2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0
    
def bicubic_itp(img,dst_h,dst_w):
    src_h,src_w,_ = img.shape
    new_size = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    for i in range(dst_h):
        for j in range(dst_w):
            src_x = i*(src_h/dst_h)
            src_y = j*(src_w/dst_w)
            x = math.floor(src_x)
            y = math.floor(src_y)
            u = src_x-x
            v = src_y-y
            tmp = 0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii < 0 or y+jj < 0 or x+ii >= src_h or y+jj >= src_w:
                        continue
                    tmp += img[x+ii,y+jj]*weights(ii-u)*weights(jj-v)
            new_size[i,j] = np.clip(tmp,0,255)
    return new_size

image = cv2.imread("test.jpg")
# make a copy
blank = np.zeros((360,600,3), np.uint8)
blank = image.copy()

# 1.Exchange position
tmp1 = np.zeros((120,200,3), np.uint8)
tmp1 = image[0:120, 0:200].copy()
tmp2 = np.zeros((120,200,3), np.uint8)
tmp2 = image[0:120, 400:600].copy()
blank[0:120, 0:200] = tmp2
blank[0:120, 400:600] = tmp1

# 2.Gray scale
for row in range(240,360):
        for col in range(0,200):
            blank[row, col] = sum(blank[row, col])/3
# 3.Intensity Resolution ###
norm = np.array(cv2.imread('test.jpg',0))
div = int(256/4)
for row in range(240,360):
    for col in range(400,600):
        levels=int((norm[row, col])/div)
        #print(levels)
        #print((norm[row, col])/div)  
        norm[row, col] = levels*div
norm = Image.fromarray(norm.astype('uint8')).convert('RGB')
norm.save('gray.jpg')
blank[240:360, 400:600] = cv2.imread("gray.jpg")[240:360, 400:600] 
            
# 4.Color Filter – Red
for row in range(120,240):
        for col in range(0,200):
            if blank[row,col,2]<=150 or blank[row,col,2]*0.6 <= blank[row,col,0] or blank[row,col,2]*0.6 <= blank[row,col,1]:
                blank[row, col] = sum(blank[row, col])/3
# 5.Color Filter – Yellow
for row in range(120,240):
        for col in range(400,600):
            if (int(blank[row,col,1])+int(blank[row,col,2]))*0.3 <= blank[row,col,0] or abs(int(blank[row,col,1]) - int(blank[row,col,2]) ) >= 50 :
                blank[row, col] = sum(blank[row, col])/3
# 6.Channel Operation(Double the value of green channel)
for row in range(240,360):
        for col in range(200,400):
            # deal with overflow
            if blank[row,col,1]*2 > 255:
                blank[row,col,1] = 255
            else:
                blank[row,col,1]*=2
            
for_itps = 'test.jpg'
image_np = np.array(Image.open(for_itps))            
# 7. Bilinear Interpolation – 2x
image_np_1 = image_np[0:120, 200:400]
np_tmp1 = bilinear_itp(image_np_1, image_np_1.shape[0]*2, image_np_1.shape[1]*2)
np_tmp1 = Image.fromarray(np_tmp1.astype('uint8')).convert('RGB')
np_tmp1.save('bli.jpg')

bl = cv2.imread("bli.jpg")
pic_1 = np.zeros((240,400,3), np.uint8)
pic_1 = bl.copy()
blank[0:120, 200:400] = pic_1[0:120, 0:200]

# 8. Bicubic Interpolation – 2x
image_np_2 = image_np[120:240, 200:400]
np_tmp2 = bicubic_itp(image_np_2,image_np_2.shape[0]*2,image_np_2.shape[1]*2)
np_tmp2 = Image.fromarray(np_tmp2.astype('uint8')).convert('RGB')
np_tmp2.save('bci.jpg')

bc = cv2.imread("bci.jpg")
pic_2 = np.zeros((240,400,3), np.uint8)
pic_2 = bc.copy()
blank[120:240, 200:400] = pic_2[0:120, 0:200]
#cv2.imshow("Test 1", blk_1)
"""cv2.imshow("My image", blank)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#save
cv2.imwrite("output.jpg", blank)