# Written By Mark Dhruba Sikder (26529548) & Asif Rana (27158632)

import cv2
import os
import glob


def rotate_images_270(img):
    """
    This function allows the images to be rotated 270 degrees
    :param img: input image to be rotated
    :return: rotated image
    """
    assert (img is not None), "No image to augment."
    # get image height, width
    (h, w) = img.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)

    angle270 = 270

    scale = 1.0

    # Perform the counter clockwise rotation holding at the center
    # 270 degrees
    M = cv2.getRotationMatrix2D(center, angle270, scale)
    rotated270 = cv2.warpAffine(img, M, (h, w))
    return rotated270

def rotate_images_180(img):
    """
    This function allows the images to be rotated 180 degrees
    :param img: input image to be rotated
    :return: rotated 180 degree image
    """
    assert (img is not None), "No image to augment."

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

    return rotated180


def augment_images():
    """
    This function allows us to use the rotating functions and rotate the images and eventually write the
    images in a folder with a different name.
    :return:
    """
    mypath = 'C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/FYP_final_Mark&Rana/train_final_process/Severe DR'
    path = os.path.join(mypath, '*tif')
    files = glob.glob(path)
    image_size = 512
    images = []
    for fl in files:
        image = cv2.imread(fl)
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
        rotated_image270 = rotate_images_270(image)   # rotate 270
        rotated_image180 = rotate_images_180(image)   # rotate 180
        images.append(rotated_image270)  # images = np.array(images)
        images.append(rotated_image180)

    assert (len(images) > 0), "No images were augmented. Maybe the image folder is empty."

    for i, face in enumerate(images):
        cv2.imwrite( "C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/FYP_final_Mark&Rana/train_final_process/Severe DR/Severe_aug-" + str(i) + ".tif", face)

if __name__ == "__main__":
     augment_images()