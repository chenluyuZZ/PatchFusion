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
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from art.estimators.classification import PyTorchClassifier

def sigmoid(x,a,c,throde):
    return c/(1+torch.exp(-a*(x-throde)))


model_path = '/home/sstl/fht/mask_train/model/models/'





class Gener_Mask():
    def __init__(self,label,index) -> None:
        super(Gener_Mask).__init__()
        checkpoint = torch.load(f'/home/sstl/fht/mask_train/sample/test_{label}/pth/model_{index}.pth')
        #checkpoint = torch.load(f'/home/sstl/fht/mask_train/sample/temp_img_label0/pth/model_11.pth')
        # checkpoint  = torch.load('/home/sstl/fht/mask_train/model/models/model_0.pth')
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






# black box 0 = 白盒  1 = googleNet  2= InceptionV3 3= denseNet  4=VIT   2 需要变换 5  8 /256   30 / 50 
class Pgd_Attack():
    def __init__(self,subst_model,attack_model,save_path,dataloader,label,index,epsilon=[-2.1179,2.64],alpha=8/256,iters=128,black_box = 0,rate=1.0) -> None: 
        super(Pgd_Attack).__init__()
        self.subst_model = subst_model
        self.attack_model = attack_model
        self.save_path = save_path
        self.black_box = black_box
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.exists(self.save_path+'attack_g'):
            os.mkdir(self.save_path+'attack_g')
        self.dataloader = dataloader
        self.epsilon = epsilon
        self.alpha = alpha
        self.iters = iters
        
        self.generator = Gener_Mask(label,index) 
        self.black_index= [145,153,289,404,405,510,805,817,867,950]
        self.limit_eps = [self.epsilon[0]*rate , self.epsilon[1]*rate] 
        

    def attack(self):
        acc = [[[0,0],[0,0],[0,0]],[[0,0],[0,0],[0,0]]]
        mask_Area = [0,0,0]
        file_write=[['可变mask no_target','slice window no-target attack','可变mask + slice window no-target attack'],['可变mask target  attack','slice window target  attack','可变mask + slice window target  attack']]
        # no-target ，3种    target 3种 mask 的面积
        
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.subst_model.parameters(), lr=0.001) 
        self.subst_model = PyTorchClassifier(
                model=self.subst_model,
                loss=criterion,
                input_shape=(3,224,224),
                nb_classes=10,
                optimizer=optimizer
            )
        
        
        
        for i,(images,label,index) in enumerate(self.dataloader):

            images = images.cuda()
            images.required_grad = True
            label = label.cuda()
            
            self.generator.setup(images,label,index,epoch=0)
            g_mask= self.generator.Get_Cam_Mask.dilation_Erosion()
            
            # tok -k mask
            slice_mask = torch.load("/home/sstl/fht/mask_train/sample/slice_window/cls_0~9_0.02423.pth")


            
            if self.black_box>0:
                if self.black_box==2:
                    images_v3 = interpolate(images, size=(299,299), mode='bilinear', align_corners=True)
                    temp = self.attack_model(images_v3).logits[:,self.black_index]
                else:
                    temp = self.attack_model(images)[:,self.black_index]
                target_label = find_second_largest(temp)
            else:
                target_label = find_second_largest(self.generator.logit)

            target_mask = slice_mask[target_label].unsqueeze(1) # 8 224 224
            target_label = torch.tensor(target_label,device='cuda:0')
            merge_mask = (g_mask+target_mask).clamp(0,1).cuda()
            
            # PGDattack
            mask_list = [g_mask.cpu().detach().numpy(),target_mask.cpu().detach().numpy(),merge_mask.cpu().detach().numpy()]
            # for j in range(3):
            #     save_images(mask_list[j][0].unsqueeze(0),[f'multi_mask_bri_{index[0]}_{j}.png'],'/home/sstl/fht/mask_train/merge_mask') 


            for j in range(3):
                mask_Area[j] += mask_list[j].sum()/1300.0  # 数据集图片的数量
            
            

            images = images.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            target_label = target_label.cpu().numpy()
            
            for j in [False,True]:
                apgd = AutoProjectedGradientDescent(self.subst_model,eps=0.3,eps_step=8/255,nb_random_init=1,max_iter=100,targeted=j,batch_size=8,loss_type='cross_entropy')
                for k in range(3):
                    if j ==0:
                        x_adv = apgd.generate(x=images,y=label,mask=mask_list[k])
                    else:
                        x_adv = apgd.generate(x=images,y=target_label,mask=mask_list[k])
                    preds = self.attack_model(torch.tensor(x_adv).cuda())[:,self.black_index].argmax(dim=-1)
                    
                    
                    if j ==0:# no-target  attack
                        acc[j][k][0] +=(preds.cpu().numpy()!=label).sum()
                        acc[j][k][1] += label.shape[0]
                        
                        print(f' {file_write[j][k]}    Accuracy{acc[j][k][0]/acc[j][k][1]:.6f}    NO target Attack Success img:{acc[j][k][0]}  selected_img:{acc[j][k][1]}')

                    if j ==1: # target  attack
                        acc[j][k][0] +=(preds.cpu().numpy()==target_label).sum()
                        acc[j][k][1] += label.shape[0]

                        print(f' {file_write[j][k]}    Accuracy{acc[j][k][0]/acc[j][k][1]:.6f}   target Attack Success img:{acc[j][k][0]}  selected_img:{acc[j][k][1]}')
              

        
        for j in range(2):
            for k  in range(3):
                with open(self.save_path+'ACC.txt','a') as file:
                    file.write(f'{file_write[j][k]}   Accuracy:{acc[j][k][0]/acc[j][k][1]}  Attack Success img:{acc[j][k][0]}  selected_img:{acc[j][k][1]}  \n')


        for index,j in enumerate(['可变mask 补丁均值 ','slice window 补丁均值','可变mask + slice window  补丁均值']):
            with open(self.save_path+'ACC.txt','a') as file:
                    file.write(f'{j}: {mask_Area[index]} \n')

        
        with open(self.save_path+'ACC.txt','a') as file:
            file.write(f'\n')
    


subst_model = resnet18_10cls_fune
subst_model = subst_model.cuda()

ori_model = resnet18_10cls.cuda()





model_list =[2,5,2,1,0,11,2,4,8,1]

label = 3

index = model_list[label]
# Inceptionv3 = models.inception_v3(pretrained=True).cuda()  # 输入 3 299 299
# # black box 0 = 白盒  1 = googleNet  2= InceptionV3 3= denseNet   Vgg = 4  Resnet50=5   2 需要变换   vit =6
#DenseNet = models.densenet121(pretrained=True).cuda()
# GoogleNet = models.googlenet(pretrained=True).cuda()
# vit = models.vit_b_32(pretrained=True).cuda()
Resnet50 = models.resnet50(pretrained=True).cuda()
#Vgg= models.vgg16_bn(pretrained=True).cuda()
# moblieNet = models.mobilenet_v2(pretrained=True).cuda()
#efficientNet = models.efficientnet_b3(pretrained=True).cuda()

sacle = 255

with open(path_list[label]+'ACC.txt','a') as file:
    file.write(f'fintune model   black box   efficientNet sacle PGD  scale:{sacle}/255  attack model {index}\n')
    



Attack = Pgd_Attack(subst_model,Resnet50,path_list[label],dataloader_list[label],label,index,black_box=6,rate=sacle/255)
Attack.attack()



