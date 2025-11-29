import torch
import numpy as np
from PIL import Image
from PIL import ImageCms
import cv2
from skimage import color

def visualize(data, filename):
    '''
    :param data:     input is 3d tensor of a image,whose size is (H*W*C)
    :param filename:
    :return:         saved into filename positions
    '''
    assert (len(data.shape) == 3)  # height*width*channels
    # print data
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img

def rgb2gray(rgb):
    assert (len(rgb.shape)==4)   #4D arrays
    # 将 NumPy 数组转换为 PyTorch 张量

    assert (rgb.shape[1]==3)
    bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
    return bn_imgs

# Fives数据集=======================================================================================================
# def rgb2gray(rgb):
#     assert (len(rgb.shape) == 4)  # 确保是4D数组
#     assert (rgb.shape[2] == 3)  # 确保输入有3个通道（RGB）
#
#     # 第一步：变换通道，从 (9600, 512, 3, 512) -> (9600, 3, 512, 512)
#     rgb = np.transpose(rgb, (0, 2, 1, 3))  # (9600, 3, 512, 512)
#
#     # 第二步：计算灰度图，使用RGB的加权平均值
#     bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
#
#     # 第三步：将灰度图的形状转换为 (9600, 1, 512, 512)
#     bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
#
#     return bn_imgs

# Fives数据集=======================================================================================================


# def rgb2gray_test1(rgb):
#     assert (len(rgb.shape)==4)   #4D arrays
#     assert (rgb.shape[1]==3)
#     bn_imgs = rgb[:,0,:,:]*0.299 + rgb[:,1,:,:]*0.587 + rgb[:,2,:,:]*0.114
#     bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],1,rgb.shape[2],rgb.shape[3]))
#     return bn_imgs


def PreProc1(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.   #reduce to 0-1 range
    return train_imgs

#==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = cv2.equalizeHist(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

# CLAHE (Contrast Limited Adaptive Histogram Equalization)
def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized

# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs

#=========================================================================================================

if __name__ == '__main__':
    # cat_photo()
    # print(torch.nn.ConvTranspose2d(12,32, kernel_size=3, stride=2, padding=1, output_padding=1)(torch.rand(size=(2,12,128,128))).size())
    # pro_re()
    # pro_re2()
    # print('<->')



    pass