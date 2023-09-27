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


model_path = '/home/sstl/fht/mask_train/model/models/'





class Gener_Mask():
    def __init__(self,label,index) -> None:
        super(Gener_Mask).__init__()
        # checkpoint = torch.load(f'/home/sstl/fht/mask_train/sample/temp_img_label0/pth/model_15.pth')
        checkpoint = torch.load(f'/home/sstl/fht/mask_train/sample/test_{label}/pth/model_{index}.pth')
        self.g = Unet().cuda()
        self.g.load_state_dict(checkpoint['model'])
        self.model = resnet18_10cls_fune.cuda()
        self.trainer = torch.optim.SGD([{'params':self.g.parameters()}], lr=1e-4)
        self.trainer.load_state_dict(checkpoint['optimizer'])
        self.flag = True
        self.cam_feature = []
        self.cam_weight = []
        
        self.cam_hook()
        
    
    def setup(self,img:torch.tensor,label:torch.tensor,epoch):

        img.requires_grad =  True
        self.data = (img,label)
        self.cam_feature.clear()
        self.cam_weight.clear()
        self.logit  = self.model(img)
        self.target_logit = self.logit[range(len(label)),label.tolist()].sum()
        self.target_logit.backward()
        self.Get_Cam_Mask = Get_Cam_Mask(self.g,self.cam_feature,self.cam_weight)
        self.Get_Cam_Mask.forward()
    
        
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


    def forward(self): # 8cam 8batch 1 224 224 
        self.cam_concat()
        self.cam = torch.concat(self.cam,dim = 1) # 每张图的cam 合并在一起 16 8 224 224 
        mask = torch.ones_like(self.cam[:,0,:,:].unsqueeze(1)) # 8 1 16 16
        for i in range(0,4):
            init_input = torch.cat((mask,self.cam[:,i*2,:,:].unsqueeze(1)),dim=1) # 0 2 4 6 
            mask = self.g(init_input)
            mask_l = torch.where(tran_tanh(mask[0])>0.5,1.0,0.0)
        self.cam = []

        self.mask = mask # 16 1 224 224 

    
    def dilation_Erosion(self):
        
        img_dilate = tensor_dilate(self.mask,ksize=7)
        img_erode = tensor_erode(img_dilate,ksize=7)
        self.mask_label = filters.median_blur(img_erode,(5,5))
        self.mask_label = torch.where(self.mask_label>0.5,1.0,0.0)
        return self.mask_label






# black box 0 = white box  1 = googleNet  2= InceptionV3 3= denseNet  4=VIT   2 需要变换 5  8 /256   30 / 50 
class Pgd_Attack():
    def __init__(self,subst_model,attack_model,label,index,epsilon=[-2.1179,2.64],alpha=8/256,iters=128,black_box = 0,rate=0.2,target=False) -> None: 
        super(Pgd_Attack).__init__()
        self.subst_model = subst_model
        self.attack_model = attack_model

        self.black_box = black_box
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        self.target =target
        self.generator = Gener_Mask(label,index) 
        self.black_index= [145,153,289,404,405,510,805,817,867,950]
        self.limit_eps = [self.epsilon[0]*rate , self.epsilon[1]*rate] 
        

    def attack(self,images,label):

        images = images.cuda()
        images.required_grad = True
        label = label.cuda()
        
        self.generator.setup(images,label,epoch=0)
        g_mask= self.generator.Get_Cam_Mask.dilation_Erosion()
        
        # tok -k mask
        slice_mask = torch.load("/home/sstl/fht/mask_train/sample/slice_window/cls_0~9_0.02423.pth")

        
        if self.black_box>0:
            if self.black_box==2:
                images_v3 = interpolate(images, size=(299,299), mode='bilinear', align_corners=True)
                temp = self.attack_model(images_v3)
            else:
                temp = self.attack_model(images)
            target_label = find_second_largest(temp,label)
        else:
            target_label = find_second_largest(self.generator.logit)

        
        target_mask = slice_mask[target_label].unsqueeze(1) # 8 224 224
        
        target_label = torch.tensor(target_label,device='cuda:0')
        
        merge_mask = (g_mask+target_mask).clamp(0,1).cuda()
        
        mask_list = [g_mask,target_mask,merge_mask]
        mask = g_mask
        delta = torch.zeros_like(images).uniform_(self.limit_eps[0], self.limit_eps[1]) # 生成随机数
        delta.requires_grad = True
        for _ in range(self.iters):
            
            loss = F.cross_entropy(self.subst_model(images*(1-mask) + delta*mask), label.long())
            loss.backward()
            
            delta.data = (delta + self.alpha * (1-2*self.target)*delta.grad.detach().sign()).clamp(self.limit_eps[0], self.limit_eps[1])
            delta.grad.zero_()


        attack_img  = (images*(1-mask)+ delta*mask).clamp(self.epsilon[0], self.epsilon[1]) # 8 3 224 224 
        if self.black_box>0:
            if self.black_box ==2:
                attack_img = interpolate(attack_img, size=(299,299), mode='bilinear', align_corners=True)
                pred_prob = self.attack_model(attack_img) # 改变这个模型可以进行黑盒attack
            else:
                pred_prob = self.attack_model(attack_img) # 改变这个模型可以进行黑盒attack
        else:
            pred_prob = self.attack_model(attack_img)
        
        preds = pred_prob.argmax(dim=-1)
        
        return attack_img,pred_prob
                



 
            
    


# subst_model = resnet18_10cls_fune
# subst_model = subst_model.cuda()

# ori_model = resnet18_10cls.cuda()





# model_list =[2,5,2,1,0,11,2,4,8,1]

# # model_0_list = [i for i in range(2,7)] # 2 3 4 5 6 
# label = 3

# index = model_list[label]
# # Inceptionv3 = models.inception_v3(pretrained=True).cuda()  # 输入 3 299 299
# # # black box 0 = 白盒  1 = googleNet  2= InceptionV3 3= denseNet   Vgg = 4  Resnet50=5   2 需要变换   vit =6
# DenseNet = models.densenet121(pretrained=True).cuda()
# # GoogleNet = models.googlenet(pretrained=True).cuda()
# # vit = models.vit_b_32(pretrained=True).cuda()
# # Resnet50 = models.resnet50(pretrained=True).cuda()
# #Vgg= models.vgg16_bn(pretrained=True).cuda()
# # moblieNet = models.mobilenet_v2(pretrained=True).cuda()
# #efficientNet = models.efficientnet_b3(pretrained=True).cuda()

# sacle = 255

# with open(path_list[label]+'ACC.txt','a') as file:
#     file.write(f'fintune model   black box   efficientNet sacle PGD  scale:{sacle}/255  attack model {index}\n')


# ASPGD_training = False
# if ASPGD_training:
#     a_model =purtube_model
#     # a_model.load_state_dict(torch.load('/home/sstl/fht/mask_train/model/models/eps_model.pt'))
#     a_model.train()
#     a_model = a_model.cuda() 

#     optim = torch.optim.SGD([{'params':a_model.parameters()}], lr=5e-3)




# target = True
# Attack = Pgd_Attack(subst_model,DenseNet,label,index,black_box=3,rate=sacle/255,target=target)
# Attack.attack()

# Attack = Pgd_Attack(subst_model,DenseNet,path_list[label],dataloader_list[label],label,index,black_box=3,rate=sacle/255)




