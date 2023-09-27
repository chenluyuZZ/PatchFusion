import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
from model import *
from utils import *
from data import *
import numpy as np
from utils import *
from torch.nn.functional import interpolate
from torchvision import models

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def sigmoid(x,a,c,throde):
    return c/(1+torch.exp(-a*(x-throde)))



class Gener_Mask():
    def __init__(self) -> None:
        super(Gener_Mask).__init__()
        # checkpoint = torch.load(f'/home/sstl/fht/mask_train/sample/temp_img_label0/pth/model_15.pth')
        checkpoint = torch.load(f'/home/sstl/fht/patchfusion/sample/handle_optim/class_reoptim4/pth/model_1.pth')
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

        img.requires_grad =  True
        self.data = (img,label)
        self.cam_feature.clear()
        self.cam_weight.clear()
        self.logit  = self.model(img)
        self.target_logit = self.logit[range(len(label)),label.tolist()].sum()
        self.target_logit.backward()
        self.Get_Cam_Mask = Get_Cam_Mask(self.g,self.cam_feature,self.cam_weight)
        self.Get_Cam_Mask.forward(index)
    
        
    def forward_hook(self,modules,input,output):
        temp = output.clone().detach().mean(dim=1)
        self.cam_feature.insert(0,temp) # forward lay1 lay2 lay3 lay4 
        
    def backward_hook(self,modules,input,output):
        temp = input[0].clone().detach().mean(dim=1)
        self.cam_weight.append(temp)

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
            save_images(tran_tanh(mask[0]).unsqueeze(0),[f'multi_mask_{index[0]}_{i*2}.png'],'/home/sstl/fht/mask_train/merge_mask') 
            mask_l = torch.where(tran_tanh(mask[0])>0.5,1.0,0.0)
            save_images(mask_l.unsqueeze(0),[f'multi_mask_bri_{index[0]}_{i*2}.png'],'/home/sstl/fht/mask_train/merge_mask') 
        
        self.cam = []

        self.mask = mask # 16 1 224 224 

    
    def dilation_Erosion(self):
        
        img_dilate = tensor_dilate(self.mask,ksize=7)
        img_erode = tensor_erode(img_dilate,ksize=7)
        self.mask_label = filters.median_blur(img_erode,(5,5))
        self.mask_label = torch.where(self.mask_label>0.5,1.0,0.0)
        return self.mask_label




class Pgd_Attack():
    def __init__(self,subst_model,attack_model,save_path,dataloader,epsilon=[-2.1179,2.64],alpha=8/256,iters=128,rate=1.0) -> None: 
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
        
        self.generator = Gener_Mask() 
        self.limit_eps = [self.epsilon[0]*rate , self.epsilon[1]*rate] 
        

    def attack(self):
        acc = [[0,0],[0,0]]
        mask_Area = 0

        
        for epoch in range(20):
            avg_loss =0
            optim.zero_grad()
            for i,(images,label,index) in enumerate(self.dataloader):
                images = images.cuda()
                images.required_grad = True
                label = label.cuda()
                self.generator.setup(images,label,index,epoch=0)
                g_mask= self.generator.Get_Cam_Mask.dilation_Erosion()
                
                ori_pred = self.subst_model(images)
                
                # a = a_model(ori_pred_new).abs()[:,:,None,None]
                a = 1
                temp = self.attack_model(images)
                target_label = find_second_largest(temp,label)
                target_label = torch.tensor(target_label,device='cuda:0')
                
                # fore_mask = torch.ones((224,224),device='cuda:0')
                # fore_mask[:112] = 0
                # target_mask = target_mask * fore_mask
                # g_mask = g_mask *fore_mask
                # merge_mask = merge_mask * fore_mask
                
                # PGDattack


                #(g_mask[0].unsqueeze(0),[f'multi_mask_bri_{index[0]}_{j}.png'],'/home/sstl/fht/mask_train/merge_mask') 



                for j in [0,1]:
                        mask = g_mask
                        delta = torch.zeros_like(images).uniform_(self.limit_eps[0], self.limit_eps[1]) # 生成随机数
                        # delta.requires_grad = True
                        # for _ in range(self.iters):
                        #     if j ==0: # no-target  attack
                        #         loss = F.cross_entropy(self.subst_model(images*(1-mask) + delta*mask), label.long())
                        #         loss.backward()
                        #         delta.data = (delta + self.alpha * delta.grad.detach().sign()).clamp(self.limit_eps[0], self.limit_eps[1])
                        #         delta.grad.zero_()
                        #     if j ==1:  # target  attack
                        #         loss = F.cross_entropy(self.subst_model(images*(1-mask) + delta*mask), target_label)
                        #         loss.backward()
                        #         # target  attack
                        #         delta.data = (delta + self.alpha * -delta.grad.detach().sign()).clamp(self.limit_eps[0], self.limit_eps[1])
                        #         delta.grad.zero_()

                        attack_img  = (images*(1-mask)+ a* delta*mask).clamp(self.epsilon[0], self.epsilon[1]) # 8 3 224 224 
                        save_images(tran_tanh(unorm(attack_img[0].unsqueeze(0))),[f'new_{i}_{j}.png'],'/home/sstl/fht/patchfusion/merge_mask')

                        pred_prob = self.attack_model(attack_img)
                        preds = pred_prob.argmax(dim=-1)
                        

                        if j ==0:# no-target  attack
                            acc[j][0] +=(preds!=label).sum()
                            acc[j][1] += label.shape[0]
                            
                        if j ==1: # target  attack
                            acc[j][0] +=(preds==target_label).sum()
                            acc[j][1] += label.shape[0]


                    
                # loss = -F.cross_entropy(pred_prob,label.long()) 
                
                # (loss/162).backward(retain_graph=True)
                # if i % 162 ==161 :
                #     optim.step()
                #     optim.zero_grad()
                # avg_loss += loss.data.item()
                # #print(f'{loss.item()}   {a.squeeze().squeeze()}')
                # if i%162 ==161:
                #     a_loss = avg_loss/(i+1)
                #     print(f'{a_loss}   {a.squeeze().squeeze()}')
                if i%10 ==9:
                    print(f'un-target acc:{acc[0][0]/acc[0][1]}  target acc:{acc[1][0]/acc[1][1]}')
            # print(f'{a.squeeze().squeeze()}')
                            
        
                # if i %20 ==19:
                #     torch.save(a_model.state_dict(),f'model/models/eps_model.pt')
            print(f'epoch :{epoch}')
            break




subst_model = resnet18_10cls_fune
subst_model = subst_model.cuda()

attack_model = densenet_10cls.cuda()





a_model =purtube_model
a_model.train()
a_model = a_model.cuda() 

optim = torch.optim.SGD([{'params':a_model.parameters()}], lr=5e-3)



sacle = 255
save_path = "/home/sstl/fht/patchfusion/cls_train/total/"
Attack = Pgd_Attack(subst_model,attack_model,save_path,imagenet10_dataloader_bs8,rate=sacle/255)
Attack.attack()






