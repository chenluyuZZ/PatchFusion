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

    
    def dilation_Erosion(self):
        
        return dilation_erosion(self.bin_mask)



mask_model = resnet18_10cls_fune
mask_model = mask_model.cuda()


    


class pgd_attack():
    def __init__(self,model,epsilon=(-2.1179,2.64),alpha=30/256,iters=20) -> None:
        super(pgd_attack).__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.generator = gener_mask() 
        self.target_mask = torch.load('/home/sstl/fht/mask_train/sample/gv0/cls_0~9_0.02423.pth') # 10 224 224 
    

    
    def attack(self):

        acc = [0,0]
        for i,(images,label,index) in enumerate(Data['dataloader']['train']):

            images = images.cuda()
            
            label = label.cuda()

            # # generator mask
            # self.generator.setup(images,label)
            # g_mask,cor_list = self.generator.dilation_Erosion()
            # mask = torch.from_numpy(g_mask).cuda()
            # PGD攻击
        
            delta = torch.zeros_like(images).uniform_(self.epsilon[0], self.epsilon[1]) # 生成随机数
            delta.requires_grad = True



            # top-k mask
            slice_mask = torch.load("/home/sstl/fht/mask_train/sample/gv0/cls_0~9_0.02423.pth")
            logits = mask_model(images)
            target_label = find_second_largest(logits) # 取第二高的类
            target_mask = slice_mask[target_label]
            target_label = torch.tensor(target_label,device='cuda:0')

            # total_mask = (mask+target_mask).clamp(0,1).unsqueeze(1).cuda() # 32 1 224 224
            total_mask = target_mask.unsqueeze(1)
            total_mask_sp = [ (iter_mask>0).sum()/(224*224)  for iter_mask in total_mask]
            for j in range(self.iters):

                loss = F.cross_entropy(mask_model(images  + delta * total_mask), target_label)# 有目标攻击 攻击目标为target_label
                loss.backward() 
                delta.data = (delta + self.alpha * -delta.grad.detach().sign()).clamp(self.epsilon[0], self.epsilon[1])
                delta.grad.zero_()

            attack_img  = (images + delta * total_mask).clamp(self.epsilon[0], self.epsilon[1])
            preds = mask_model(attack_img).argmax(dim=1)
            

            if i ==0:
                save_images(unorm(attack_img.detach()),[f'{index[k]}_ori_label{label[k]}_pred{preds[k]}_target{target_label[k]}.png' for k in range(len(label))],'/home/sstl/fht/mask_train/runner/attack_g_s')
                save_images(total_mask,[f'{index[k]}_mask_sp{total_mask_sp[k]:.4f}.png' for k in range(len(label))],'/home/sstl/fht/mask_train/runner/attack_g_s')
            acc[0] += (preds!=label).sum()
            acc[1] += len(label)
            print(f'current ACC :{acc[0]/acc[1]}  {acc[0]}_{acc[1]}')
            # for k in cor_list:
            #     save_images(unorm(attack_img.detach()[k]).unsqueeze(0),[f'{index[k]}_ori_label{label[k]}_preds{preds[k]}.png'],'/home/sstl/fht/mask_train/runner/attack_g_s')
            #     save_images(mask[k].unsqueeze(0).unsqueeze(0),[f'{index[k]}_mask.png'],'/home/sstl/fht/mask_train/runner/attack_g_s')

        # return (images + delta).detach()


Attack = pgd_attack(mask_model)
Attack.attack()

    





