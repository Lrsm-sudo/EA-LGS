import sys

sys.path.append('../data_process/')
sys.path.append('../networks/')
sys.path.append('../networks/common/')
sys.path.append('../networks/MESnet/')
sys.path.append('../networks/threestage/')
sys.path.append('../networks/ACPNet/')
sys.path.append('../networks/DCNet/')
sys.path.append('./GAN_train_test/')
sys.path.append('../GAN_train_test/')
sys.path.append('../networks/mynets/')

sys.path.append('../../data_process/')
sys.path.append('../../networks/')
sys.path.append('../../networks/common/')
sys.path.append('../../networks/MESnet/')
sys.path.append('../../networks/threestage/')
sys.path.append('../../networks/ACPNet/')
sys.path.append('../../networks/DCNet/')
print(sys.path)

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optims
import torch.utils.data as data
from torch.autograd import Variable as V
import sklearn.metrics as metrics
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualize(data,filename):
    '''
    :param data:      the visual data must be channel last ! H W C
    :param filename:
    :return:          save into filename path .png format !
    '''
    assert (len(data.shape)==3) #height*width*channels
    img = None
    if data.shape[2]==1:  #in case it is black and white
        data = np.reshape(data,(data.shape[0],data.shape[1]))
    if np.max(data)>1:
        img = Image.fromarray(data.astype(np.uint8))   #the image is already 0-255
    else:
        img = Image.fromarray((data*255).astype(np.uint8))  #the image is between 0-1
    img.save(filename + '.png')
    # print('===========================>visualize function have saved into ',filename + '.png')
    return img

def crop_order_images(image, mask, crop_size = 400, row = 3, col = 4,  mode = 'orders', rands = 12):
    if mode =='orders':
        image, s_h, s_w= padding_img(image, crop_size, row, col)
        mask, s_h, s_w = padding_img(mask, crop_size, row, col)
        assert (crop_size > s_h and crop_size > s_w)
        img_set, mask_set = [], []
        for m in range(0, row):
            for n in range(0, col):
                dh = m * s_h
                dw = n * s_w
                img_set.append(image[:,:,dh:dh+crop_size, dw:dw+crop_size])
                mask_set.append(mask[:, :, dh:dh + crop_size, dw:dw + crop_size])
        return torch.cat([img_set[i] for i in range(0, len(img_set))], dim=0), \
               torch.cat([mask_set[i] for i in range(0, len(img_set))], dim=0),
    elif mode == 'random':
        # random select center point to expand patches ! (compliment)
        import  random
        img_set, mask_set = [], []
        for i in range(0, rands):
            center_y = random.randint(crop_size//2, image.size()[2] - crop_size//2)
            center_x = random.randint(crop_size//2, image.size()[3] - crop_size//2)
            crops_img = image[:,:,center_y - crop_size//2:center_y + crop_size//2,center_x - crop_size//2:center_x + crop_size//2]
            img_set.append(crops_img)
            crops_mask = mask[:,:,center_y - crop_size//2:center_y + crop_size//2,center_x - crop_size//2:center_x + crop_size//2]
            mask_set.append(crops_mask)
        return torch.cat([img_set[i] for i in range(0, len(img_set))], dim=0)\
            ,torch.cat([mask_set[i] for i in range(0, len(img_set))], dim=0)






def crop_eval_new_V1(net, image, crop_size = 300, row=3, col=4):
    h_o, w_o = image.size()[2], image.size()[3]
    group_image = torch.zeros_like(torch.rand(size=(row, col, image.size()[0], image.size()[1], crop_size,  crop_size)))
    image, s_h, s_w= padding_img(image, crop_size, row, col)
    h, w = image.size()[2], image.size()[3]
    assert (crop_size > s_h and crop_size > s_w)
    merge_img = torch.zeros_like(image)
    for m in range(0, row):
        for n in range(0, col):
            dh = m * s_h
            dw = n * s_w
            if net is None:
                group_image[m,n,:,:,:,:,] = image[:,:,dh:dh+crop_size, dw:dw+crop_size]
            else:
                group_image[m,n,:,:,:,:,] = net(image[:,:,dh:dh+crop_size, dw:dw+crop_size])

    # print(image.size(),'-------------------- here has cropped -----------------------',s_h, s_w, h, w, crop_size)

    for i in range(0, h):
        for j in range(0, w):
           p, q = np.maximum((i - crop_size)// s_h, -1) + 1, np.maximum((j - crop_size)// s_w, -1) + 1
           p_h, p_w = i - p * s_h, j - q * s_w
           # print(p, q, p_h, p_w)
           j_s = 1
           merge_img[:, :, i, j] = group_image[p, q, :, :, p_h, p_w]
           for k1 in range(0, row):
               for k2 in range(0, col):
                   if (k1 < p_h / s_h and k1 < row - p):
                       merge_img[:,:, i, j] += group_image[p + k1, q, :, :, p_h - k1 * s_h, p_w]
                       j_s +=1
                   if (k1 < (crop_size-p_h) / s_h and k1 <= p):
                       merge_img[:, :, i, j] += group_image[p - k1, q, :, :, p_h + k1 * s_h, p_w]
                       j_s += 1
                   if (k2 < p_w / s_w and k2 < col - q):
                       merge_img[:,:, i, j] += group_image[p, q + k2, :, :, p_h, p_w - k2 * s_w]
                       j_s += 1
                   if (k2 < (crop_size - p_w) / s_w and k2 <= q):
                       merge_img[:,:, i, j] += group_image[p, q - k2, :, :, p_h, p_w + k2 * s_w]
                       j_s += 1
           merge_img[:, :, i, j] /= j_s
    print('================= has finished this picture prediction ================== ')
    return merge_img[:, :, 0:h_o, 0:w_o].to(device)

def padding_img(image, crop_size, rows, clos):
    pad_h, pad_w = (image.size()[2] - crop_size)%(rows-1), (image.size()[3] - crop_size)%(clos-1)
    image = padding_hw(image, dims='h', ns= 0 if pad_h==0 else rows -1 -pad_h)
    image = padding_hw(image, dims='w', ns= 0 if pad_w==0 else clos -1 -pad_w)
    return image.to(device), (image.size()[2] - crop_size)//(rows-1), (image.size()[3] - crop_size)//(clos-1)

def padding_hw(img, dims = 'h', ns = 0):
    if ns ==0:
        return img
    else:
        after_expanding = None
        if dims == 'h':
            pad_img = torch.zeros_like(img[:,:,0 : ns,:,])
            after_expanding = torch.cat([img, pad_img], dim=2)
        elif dims == 'w':
            pad_img = torch.zeros_like(img[:,:,:,0:ns])
            after_expanding = torch.cat([img, pad_img], dim=3)
        return after_expanding


if __name__ == '__main__':
    # path = '../log/weights_save/STARE/unet/116.iter'
    # test_vessel(path)

    # outs = padding_hw(torch.rand(size=(10,3,43,65)))
    # print(outs.size(),'--------------')

    # outs,_,_ = padding_img(torch.rand(size=(10,3,100,120)), 48, 4)
    # print(outs.size(),'--------------')

    imgs= np.asarray(Image.open('./0prob.png'))
    xx_imgs = torch.from_numpy(np.expand_dims(np.expand_dims(imgs.copy(), axis = 0), axis = 0)).float()
    print('this image size is :' , xx_imgs.size())

    merge_img = crop_eval_new_V1(net=None, image=xx_imgs)
    visualize(np.transpose(merge_img.numpy()[0,:,:,:,],(1, 2, 0)), './here')

    # rs = crop_order_images(xx_imgs, mode='random')
    # print(rs.size())


    pass