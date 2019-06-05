import cv2
import numpy as np
#from pylab import *
#from PIL import Image
import ExtractBloodVessel
import sys
import tensorflow as tf


def rotate_images_90(img):
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)

    angle90 = 90
    angle180 = 180

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    M = cv2.getRotationMatrix2D(center, angle90, scale)
    rotated90 = cv2.warpAffine(img, M, (h, w))
    cv2.imshow('Image rotated by 90 degrees', rotated90)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image


def rotate_images_180(img):
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)

    angle180 = 180

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    # 180 degrees
    M = cv2.getRotationMatrix2D(center, angle180, scale)
    rotated180 = cv2.warpAffine(img, M, (h, w))
    cv2.imshow('Image rotated by 180 degrees', rotated180)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

def rotate_images_270(img):
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)

    angle270 = 270

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    # 90 degrees
    M = cv2.getRotationMatrix2D(center, angle270, scale)
    rotated270 = cv2.warpAffine(img, M, (h, w))
    cv2.imshow('Image rotated by 270 degrees', rotated270)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image

img = cv2.imread("C:/Users/user/Desktop/FYPImplementation/Retinal/123/3/20051020_57622_0100_PP.tif")
img_size = 256
img = cv2.resize(img, (256, 256), 50, 50)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", gray);cv2.waitKey()
## (2) Threshold
th, threshed = cv2.threshold(gray, 275, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

## (3) Find the min-area contour
_, cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea)
for cnt in cnts:
    if cv2.contourArea(cnt) > 100:
        break

## (4) Create mask and do bitwise-op
mask = np.zeros(img.shape[:2],np.uint8)
cv2.drawContours(mask, [cnt],-1, 255, -1)
dst = cv2.bitwise_and(img, img, mask=mask)
#cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", dst);cv2.waitKey()
lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
#cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", lab);cv2.waitKey()

b = 80. # brightness
c = 50.  # contrast -> alpha

#call addWeighted function, which performs:
#    dst = src1*alpha + src2*beta + gamma
# we use beta = 0 to effectively only operate on src1
lab = cv2.addWeighted(lab, 1. + c/127., lab, 0, b-c)

cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", lab);cv2.waitKey()

#rotated_imgs90 = rotate_images_90(lab)
#rotated_imgs270 = rotate_images_270(lab)
#rotated_imgs180 = rotate_images_180(lab)
# run a 5x5 gaussian blur then a 3x3 gaussian blr
blur5 = cv2.GaussianBlur(lab, (5, 5), 0)
blur3 = cv2.GaussianBlur(img,(3,3),0)
blur1 = cv2.GaussianBlur(img, (1,1), 0)
new_blur = blur5 - blur3
cv2.imshow('Guassian blur', new_blur)
cv2.waitKey()

blur = cv2.bilateralFilter(new_blur,9,75,75)
cv2.imshow('Guassian blur', blur)
cv2.waitKey()

blu = cv2.GaussianBlur(img, (5, 5), 0)
filtered_image = 5*(img - blu) + 128
cv2.imshow('Guassian blur', filtered_image)
cv2.waitKey()
blur = cv2.bilateralFilter(filtered_image,9,75,75)
cv2.imshow('Guassian blur', blur)
cv2.waitKey()
'''
gray = img[11:120+128, 45:80+128]
cv2.imshow("cropped", gray)
cv2.waitKey(0)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", gray);cv2.waitKey()
## (2) Threshold
th, threshed = cv2.threshold(gray, 275, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

## (3) Find the min-area contour
_, cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea)
for cnt in cnts:
    if cv2.contourArea(cnt) > 100:
        break

## (4) Create mask and do bitwise-op
mask = np.zeros(img.shape[:2],np.uint8)
cv2.drawContours(mask, [cnt],-1, 255, -1)
dst = cv2.bitwise_and(img, img, mask=mask)
#cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", dst);cv2.waitKey()
lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
#cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", lab);cv2.waitKey()

b = 40. # brightness
c = 150.  # contrast -> alpha

#call addWeighted function, which performs:
#    dst = src1*alpha + src2*beta + gamma
# we use beta = 0 to effectively only operate on src1
lab = cv2.addWeighted(lab, 1. + c/127., lab, 0, b-c)
cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", lab);cv2.waitKey()

## (2) Threshold
#th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

## (3) Find the min-area contour
_, cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea)
for cnt in cnts:
    if cv2.contourArea(cnt) > 100:
        break

## (4) Create mask and do bitwise-op
mask = np.zeros(img.shape[:2],np.uint8)
cv2.drawContours(mask, [cnt],-1, 255, -1)
dst = cv2.bitwise_and(img, img, mask=mask)


l, a, b = cv2.split(dst)
cv2.imshow('l_channel', l)
cv2.imshow('a_channel', a)
cv2.imshow('b_channel', b)

#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)
cv2.imshow('CLAHE output', cl)

#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
limg = cv2.merge((cl,a,b))
cv2.imshow('limg', limg)

#-----Converting image from LAB Color model to RGB model--------------------
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
cv2.imshow('final', final)

#

#cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.tif", edged);cv2.waitKey()

#bloodvessel = ExtractBloodVessel.extract_bv(img)

#cv2.imshow("F:/Year3_Sem 2/Image Processing/Retinal/123/train/dst.png", bloodvessel);cv2.waitKey()
'''