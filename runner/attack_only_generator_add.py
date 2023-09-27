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


# 数据集

Data = imagenet10_dataloader_bs32

class loss_function(nn.Module):
    def __init__(self) -> None:
        super(loss_function).__init__()
        self.lambda1 = 200
        self.lambda2 = 1e-3
        self.lambda3 = 1/(224*224*3*32)
    def forward(self, x_logit:torch.Tensor, m_logit:torch.Tensor,mask:torch.Tensor) -> torch.Tensor:

        self.loss1 = -torch.functional.F.kl_div(x_logit.log(), m_logit) *self.lambda1# +mask * eps 的kl散度， 该损失会让mask 不断增大 直至mask全为1
        self.loss2 = mask.norm(p=2) *self.lambda2  
        self.loss3 = ((mask-0.5)*(mask-0.5)).sum()*self.lambda3 # 使得mask 数值 小的更快趋于0 大的趋于1 突变的情况减少
        #print(self.loss1,self.loss2,self.loss3)

        return self.loss1 + self.loss2





class gener_mask():
    def __init__(self) -> None:
        super(gener_mask).__init__()
        checkpoint = torch.load('/home/sstl/fht/mask_train/model/pth/generator_cls10_epoch29.pth')
        self.g = Generator().cuda()
        self.g.load_state_dict(checkpoint['model'])
        self.loss = loss_function()
        self.model = resnet18_10cls_fune.cuda()
        self.trainer = torch.optim.SGD([{'params':self.g.parameters()}], lr=1e-4)
        self.trainer.load_state_dict(checkpoint['optimizer'])
        self.cam_hook()
        self.flag = True
    
    def setup(self,img,label):

        self.flag= True # 只有setup 阶段才会获取cam
        self.target_logit = self.model(img)[range(len(label)),label.tolist()].sum()#图像均为第 0 类 计算其梯度
        self.target_logit.backward()
        self.get_cam_mask() 
        self.flag =False
        
    
    def forward_hook(self,modules,input,output):
        if self.flag:
            self.cam_feature=output.clone().detach()
            
    
    def backward_hook(self,modules,input,output):
        if self.flag:
            self.cam_weight=input[0].clone().detach()

    def cam_hook(self):
        self.model[1].layer3[1].conv2.register_forward_hook(self.forward_hook) 
        self.model[1].layer3[1].conv2.register_backward_hook(self.backward_hook) 
    
    def get_cam_mask(self):
        self.cam_feature = self.cam_feature.mean(dim=1).unsqueeze(1)
        self.cam_weight = self.cam_weight.mean(dim=1).unsqueeze(1)
        self.cam = self.cam_feature * self.cam_weight
        self.cam_feature = self.g(self.cam_feature).unsqueeze(1)
        self.cam_weight = self.g(self.cam_weight).unsqueeze(1)
        
        self.cam_mask  =  self.cam_feature * self.cam_weight # 32  1 224 224 # mean: 0.05
        self.bin_mask = torch.where(tran_tanh(self.cam_mask)>0.5,1.0,0.0)

        a = self.cam_mask.norm(p=2)**2/(32*224*224*0.2)
        # print(f'sp:{a}  cam_mask.mean{self.cam_mask.mean()}')
    
    def dilation_Erosion(self):
        
        return dilation_erosion(self.bin_mask)



model = resnet18_10cls_fune
model = model.cuda()


    


class pgd_attack():
    def __init__(self,model,epsilon=(-2.1179,2.64),alpha=8/256,iters=40) -> None:
        super(pgd_attack).__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.generator = gener_mask() 
    

    
    def attack(self):
        acc = [0,0]
        for i,(images,label,index) in enumerate(Data['dataloader']['train']):

            images = images.cuda()
            
            label = label.cuda()
            self.generator.setup(images,label)
            g_mask,cor_num = self.generator.dilation_Erosion()
            mask = torch.from_numpy(g_mask).cuda()
            # PGD攻击
           
            delta = torch.zeros_like(images).uniform_(self.epsilon[0], self.epsilon[1]) # 生成随机数
            delta.requires_grad = True
            for j in range(self.iters):
                
                loss = F.cross_entropy(model(images+delta*mask.unsqueeze(1)), label)
                loss.backward() # 求出的梯度都是以 label 0为目标的
                delta.data = (delta + self.alpha * delta.grad.detach().sign()).clamp(self.epsilon[0]*0.1, self.epsilon[1]*0.1)
                delta.grad.zero_()

            attack_img  = (images+delta*mask.unsqueeze(1)).clamp(self.epsilon[0], self.epsilon[1])
            preds = model(attack_img).argmax(dim=1)
            mask_list  =[j for j  in range(len(mask)) if mask[j].sum()>500]
            acc[0] +=(preds[mask_list]!=label[mask_list]).sum()
            acc[1] += cor_num
            print((preds[mask_list]!=label[mask_list]).sum(),cor_num)
          
            for k in mask_list:
                save_images(unorm(attack_img.detach()[k]).unsqueeze(0),[f'{index[k]}_ori_label{label[k]}_preds{preds[k]}.png'],'/home/sstl/fht/mask_train/runner/attack_g_add')
                save_images(mask[k].unsqueeze(0).unsqueeze(0),[f'{index[k]}_mask.png'],'/home/sstl/fht/mask_train/runner/attack_g_add')

        # return (images + delta).detach()
            print(f'current ACC:{acc[0]/acc[1]}')


Attack = pgd_attack(model)
Attack.attack()

    





