import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from model import *
from utils import *
from data import *
from torch.nn.functional import interpolate

import numpy as np
import cv2


def superimpose(img_rgb, cam, thresh,file_name): # tensor类型
    
    '''
    Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.
    

    Args:
      image: (img_width x img_height x 3) numpy array
      grad-cam heatmap: (img_width x img_width) numpy array
      threshold: float
      emphasize: boolean

    Returns 
      uint8 numpy array with shape (img_height, img_width, 3)

    '''


    img_rgb = unorm(torch.tensor(img_rgb))
    img_rgb = img_rgb.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # heatmap = sigmoid(cam[0], 50, thresh, 1)
    heatmap = np.uint8(255 * cam)
    cv2.imwrite(file_name,heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = 0.5
    superimposed_img = heatmap * hif + img_rgb * 0.9
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    superimposed_img_bgr = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)
    
    return superimposed_img_bgr,heatmap





class Gradmodel():
    def __init__(self) -> None:
        super(Gradmodel).__init__()

        self.model = resnet18_10cls_fune.cuda()
        self.flag = True
        
        self.cam_feature = []
        self.cam_weight = []
        self.cam = []
        self.cam_hook()

    def setup(self,img:torch.tensor,label:torch.tensor):

        self.data = (img,label)
        self.cam_feature.clear()
        self.cam_weight.clear()
        self.target_logit = self.model(img)[range(len(label)),label.tolist()].sum()
        self.target_logit.backward()
        self.get_cam_mask()
    

        
    def forward_hook(self,modules,input,output):
        temp = output.clone().detach().mean(dim=1)
        

        self.cam_feature.insert(0,temp) # 前向过程 lay1 lay2 lay3 lay4 
        
    
    def backward_hook(self,modules,input,output):
        temp = input[0].clone().detach().mean(dim=1)
        self.cam_weight.append(temp) # 从头插入

    def cam_hook(self):
        

        self.model[1].layer1[0].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer1[0].conv2.register_backward_hook(self.backward_hook) 
        self.model[1].layer1[1].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer1[1].conv2.register_backward_hook(self.backward_hook) 

        self.model[1].layer2[0].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer2[0].conv2.register_backward_hook(self.backward_hook) 
        self.model[1].layer2[1].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer2[1].conv2.register_backward_hook(self.backward_hook) 

        self.model[1].layer3[0].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer3[0].conv2.register_backward_hook(self.backward_hook) 
        self.model[1].layer3[1].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer3[1].conv2.register_backward_hook(self.backward_hook) 

        self.model[1].layer4[0].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer4[0].conv2.register_backward_hook(self.backward_hook) 
        self.model[1].layer4[1].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer4[1].conv2.register_backward_hook(self.backward_hook) 

    def get_cam_mask(self): # 8 16 1 224 224
        for i in range(len(self.cam_feature)):
            self.cam.append(interpolate(tran_tanh(self.cam_weight[i].unsqueeze(1) * self.cam_feature[i].unsqueeze(1)), size=(224,224), mode='bilinear', align_corners=True)) # [[16,1,224,224],[16,1,224,224]
       
        self.cam = torch.concat(self.cam,dim = 1) # 每张图的cam 合并在一起 16 8 224 224 
    

grad_model = Gradmodel()

# for i,(img,label,index) in enumerate(dataloader_list[0]):
#     img = img.cuda();label = label.cuda()
#     grad_model.setup(img,label)
#     for k in range(2):
#         all_layer = []
#         for j in range(8):
#             img_add_cam,heatmap =superimpose(img[k].cpu().numpy(),grad_model.cam[k][j].cpu().numpy(),0.5, f'/home/sstl/fht/mask_train/sample/multi_cam/gray/{k}_img_{j}.png' ) # 3 224 224  224 224 
#             all_layer.append(heatmap) # 224 224 3  
#             cv2.imwrite(f'/home/sstl/fht/mask_train/sample/multi_cam/cam/{k}_img_{j}.png',img_add_cam)

#     break

