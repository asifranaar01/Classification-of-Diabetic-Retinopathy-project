# Written By Mark Dhruba Sikder (26529548) & Asif Rana (27158632)

import cv2, glob, os
import numpy as np

def estimate_radius(img):
    """
    This function estimates the radius of the retina
    :param img:
    :return:
    """
    mx = img[img.shape[0] // 2, :, :].sum(1)
    rx = (mx > mx.mean() / 10).sum() / 2

    my = img[:, img.shape[1] // 2, :].sum(1)
    ry = (my > my.mean() / 10).sum() / 2

    return (ry, rx)

def crop_img(img, h, w):
    """
    This function crops the image in a specific fashion
    :param img:
    :param h:
    :param w:
    :return:
    """
    h_margin = (img.shape[0] - h) // 2 if img.shape[0] > h else 0
    w_margin = (img.shape[1] - w) // 2 if img.shape[1] > w else 0

    crop_img = img[h_margin:h + h_margin,w_margin:w + w_margin,:]

    return crop_img

def subtract_gaussian_blur(img):
    """
    This function helps to subtract the blurs in the image by usign the guassian elimination function
    :param img:
    :return:
    """
    gb_img = cv2.GaussianBlur(img, (0, 0), 5)

    return cv2.addWeighted(img, 4, gb_img, -4, 128)


def remove_outer_circle(a, p, r):
    """
    Removes the outer border of the retina
    :param a:
    :param p:
    :param r:
    :return:
    """
    b = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(r * p), (1, 1, 1), -1, 8, 0)

    return a * b + 128 * (1 - b)


def place_in_square(img, r, h, w):
    """
    Places the image in a square
    :param img:
    :param r:
    :param h:
    :param w:
    :return:
    """
    new_img = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
    new_img += 128
    new_img[r - h // 2:r - h // 2 + img.shape[0], r - w // 2:r - w // 2 + img.shape[1]] = img

    return new_img


if __name__ == "__main__":

    mypath = 'C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/train_final_process/Severe DR'
    path = os.path.join(mypath, '*tif')
    files = glob.glob(path)
    image_size = 512
    scale = 256  # target radius
    images = []

    for fl in files:
        img = cv2.imread(fl)
        ry, rx = estimate_radius(img)
        resize_scale = scale / max(rx, ry)
        w = min(int(rx * resize_scale * 2), scale * 2)
        h = min(int(ry * resize_scale * 2), scale * 2)
        img_resize = cv2.resize(img.copy(), (0, 0), fx=resize_scale, fy=resize_scale)
        img_crop = crop_img(img_resize.copy(), h, w)
        img_gbs = subtract_gaussian_blur(img_crop.copy())
        img_remove_outer = remove_outer_circle(img_gbs.copy(), 0.9, scale)
        new_img = place_in_square(img_remove_outer.copy(), scale, h, w)
        images.append(new_img)
    outpath = os.path.join('C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/train_final_process/Severe DR', '*tif')
    for i, face in enumerate(images):
        print("Writing to file..")
        cv2.imwrite("C:/Users/fit3162-03/AppData/Local/Continuum/anaconda3/envs/tensorflow/train_final_process/Severe DR/Severe DR processed-" + str(i) + ".tif", face)
        print("write complete")
