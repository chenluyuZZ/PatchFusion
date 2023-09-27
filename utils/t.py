# import cv2
# import torch
# from  pipeline import *
# from save_image import *
# import numpy
# a = torch.from_numpy(cv2.imread('/home/sstl/fht/mask_train/40_36_epoch.png').transpose(2,1,0)).unsqueeze(0).to(torch.float)
# img,list= dilation_erosion(a,dilate_size=9,erode_size=8,blur_size=5)

# 


import cv2
import torch
from  pipeline import *
from save_image import *
import numpy
a = torch.from_numpy(cv2.imread('/home/sstl/fht/mask_train/sample/pre_project/test_1/cam_mask/0_6_epoch.png').transpose(2,0,1)).to(torch.float)
for i in range(5,15):
    for j in range(1,15):
        for k in range(1,15):
            try:
                img_dilate = tensor_dilate(a.unsqueeze(0),ksize=i)
                img_erode = tensor_erode(img_dilate,ksize=j)
                mask_label = filters.median_blur(img_erode,(k,k))
                mask_label = torch.where(mask_label>0.5,1.0,0.0)
            except:
                continue
            if mask_label.sum()==0:
                break
            save_images(mask_label,[f'{i}_{j}_{k}.png'],'/home/sstl/fht/mask_train/dil_test/')