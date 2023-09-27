import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from model import *
from utils import *
from data import *
import numpy as np
import random
from utils import *
from torch.nn.functional import interpolate



def sigmoid(x,a,c,throde):
    return c/(1+torch.exp(-a*(x-throde)))


model_path = '/home/sstl/fht/mask_train/model/models/'





class Gener_Mask():
    def __init__(self,label) -> None:
        super(Gener_Mask).__init__()
        checkpoint = torch.load(f'/home/sstl/fht/mask_train/sample/test_9/pth/model_{label}.pth')
        self.g = Unet().cuda()
        self.g.load_state_dict(checkpoint['model'])
        self.model = resnet18_10cls_fune.cuda()
        self.trainer = torch.optim.SGD([{'params':self.g.parameters()}], lr=1e-4)
        self.trainer.load_state_dict(checkpoint['optimizer'])
        self.flag = True
        self.cam_feature = []
        self.cam_weight = []
        
        self.cam_hook()
        
    
    def setup(self,img:torch.tensor,label:torch.tensor,index,epoch):

        self.data = (img,label)
        self.cam_feature.clear()
        self.cam_weight.clear()
        self.target_logit = self.model(img)[range(len(label)),label.tolist()].sum()
        self.target_logit.backward()
        self.Get_Cam_Mask = Get_Cam_Mask(self.g,self.cam_feature,self.cam_weight)
        self.Get_Cam_Mask.forward(index)
    
        
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



class Get_Cam_Mask(torch.nn.Module):
    def __init__(self,model,cam_feature,cam_weight):
        super(Get_Cam_Mask,self).__init__()
        self.g = model
        self.cam_feature = cam_feature
        self.cam_weight = cam_weight
        self.cam = []

    def cam_concat(self):
        for i in range(len(self.cam_feature)): 
            temp = tran_tanh(self.cam_weight[i].unsqueeze(1) * self.cam_feature[i].unsqueeze(1))
            temp = interpolate(temp, size=(224,224), mode='bilinear', align_corners=True)
            for j in range(temp.shape[0]):
                temp[j] = sigmoid(temp[j],50,1,torch.median(temp[j]+1e-3))
            self.cam.append(temp) # [[16,1,224,224],[16,1,224,224]


    def forward(self,index): # 8cam 8batch 1 224 224 
        self.cam_concat()
        self.cam = torch.concat(self.cam,dim = 1) # 每张图的cam 合并在一起 16 8 224 224 
        mask = torch.ones_like(self.cam[:,0,:,:].unsqueeze(1)) # 8 1 16 16
        for i in range(0,4):
            init_input = torch.cat((mask,self.cam[:,i*2,:,:].unsqueeze(1)),dim=1) # 0 2 4 6 
            mask = self.g(init_input)
        
        self.cam = []

        self.mask = mask # 16 1 224 224 

    
    def dilation_Erosion(self):
        
        img_dilate = tensor_dilate(self.mask,ksize=7)
        img_erode = tensor_erode(img_dilate,ksize=7)
        self.mask_label = filters.median_blur(img_erode,(5,5))
        self.mask_label = torch.where(self.mask_label>0.5,1.0,0.0)
        return self.mask_label







class Pgd_Attack():
    def __init__(self,subst_model,attack_model,save_path,dataloader,label,epsilon=(-2.1179,2.64),alpha=8/256,iters=30) -> None:
        super(Pgd_Attack).__init__()
        self.subst_model = subst_model
        self.attack_model = attack_model
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
            if not os.path.exists(self.save_path+'attack_g'):
                os.mkdir(self.save_path+'attack_g')
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        
        self.generator = Gener_Mask(label) 


    
    def attack(self):
        acc = [0,0]
        for i,(images,label,index) in enumerate(self.dataloader):

            images = images.cuda()
            
            label = label.cuda()
            
            self.generator.setup(images,label,index,epoch=0)
            g_mask= self.generator.Get_Cam_Mask.dilation_Erosion()
            mask = g_mask.cuda()
            # PGD攻击
           
            delta = torch.zeros_like(images).uniform_(self.epsilon[0], self.epsilon[1]) # 生成随机数
            delta.requires_grad = True
            for j in range(self.iters):
                
                loss = F.cross_entropy(self.subst_model(images*(1-mask) + delta*mask), label.long())
                loss.backward()
                delta.data = (delta + self.alpha * delta.grad.detach().sign()).clamp(self.epsilon[0], self.epsilon[1])
                delta.grad.zero_()

            attack_img  = (images*(1-mask)+delta*mask).clamp(self.epsilon[0], self.epsilon[1]) # 8 3 224 224 

            preds = self.attack_model(attack_img).argmax(dim=1) # 改变这个模型可以进行黑盒攻击
            
            acc[0] +=(preds!=label).sum()
            acc[1] += label.shape[0]
            print((preds!=label).sum(),label.shape[0])
          
            for k in range(label.shape[0]):

                save_images(unorm(attack_img.detach()[k]).unsqueeze(0),[f'{index[k]}_ori_label{label[k]}_preds{preds[k]}.png'],self.save_path+'/attack_g')
                save_images(mask[k].unsqueeze(0),[f'{index[k]}_mask.png'],self.save_path+'attack_g')

        # return (images + delta).detach()
        with open(self.save_path+'ACC.txt','a') as file:
            file.write(f'Batch {i}   ACC:{acc[0]/acc[1]}  acc:{acc[0]}  selected_img:{acc[1]}  \n')




    


subst_model = resnet18_10cls_fune
subst_model = subst_model.cuda()

attack_model = resnet18_10cls.cuda()

index = 9

for i in [1,2]:
    with open(path_list[index]+'ACC.txt','a') as file:
        file.write(f'model_{i} test resut \n')
    Attack = Pgd_Attack(subst_model,subst_model,path_list[index],dataloader_list[index],i)
    Attack.attack()

