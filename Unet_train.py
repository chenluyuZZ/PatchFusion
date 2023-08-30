import torch
import torch.nn as nn
import torchvision
from  torch.utils.data import DataLoader,Dataset
import os
import sys
from model import *
from utils import *
from data import *
from kornia import filters
from torch.nn.functional import interpolate


# 改用10类数据进行处理
# 仅当初始LOSS2 >=0.95 才可用此参数

# 45 个的时候出现loss1 = Nan


class loss_function(nn.Module):
    def __init__(self) -> None:
        super(loss_function).__init__()
        self.batch = 8
        self.lambda1 = 20
        self.L2_param = 0.0034 # 平衡参数
        self.L2_lim =0.02
        self.L1_flag = False
        self.L2_flag = False
        self.L3_flag = False
        self.lambda2 = 0
        self.lambda3 = 0
        self.mask_size = 8 * 224* 224




    def forward(self, x_logit:torch.Tensor, m_logit:torch.Tensor,mask:torch.Tensor) -> torch.Tensor:

        img_dilate = tensor_dilate(mask,ksize=7)
        img_erode = tensor_erode(img_dilate,ksize=7)
        self.mask_label = filters.median_blur(img_erode,(5,5))
        self.mask_label = torch.where(self.mask_label>0.5,1.0,0.0)
        
        # The mask is added to limit the area where the perturbation appears
        # self.fore_mask = torch.ones((self.mask_label.shape[-2:])).cuda()
        # self.fore_mask[:112] = 0
        # self.mask_label = self.fore_mask * self.mask_label
        
        if self.L2_flag:
            self.lambda2 = 1e-3 + 5e-4
            if self.L3_flag:
                self.lambda3 = 1e-4
    
            
       
        
        self.loss1 = torch.functional.F.kl_div(x_logit, m_logit) *self.lambda1# +mask * eps 的kl散度， 该损失会让mask 不断增大 直至mask全为1
        # self.loss1 = -(torch.log(m_logit/x_logit) * m_logit).sum() * self.lambda1
        self.mask_L2 = (mask.norm(p=2) - self.L2_param * self.mask_size * self.L2_lim).norm(p=2) # 60-600
        self.loss2 = self.mask_L2* self.lambda2

        self.loss3 = (mask-self.mask_label).norm(p=2) * self.lambda3

       
        
        # print(f'LOSS1: {self.loss1.item()}    LOSS2: {self.mask_L2*1e-3}          {self.loss3}')

        return self.loss1 +self.loss2 + self.loss3


def eval(norm_list):
    sum = 0
    len = len(norm_list) # 最佳是1 代表patch 大小均为1100
    for i in norm_list: #
        sum += -1/10000*(i-200)*(i-2000)/81
    return sum/len

def sigmoid(x,a,c,throde): 
    return c/(1+torch.exp(-a*(x-throde)))



class train_G():
    def __init__(self,save_path,dataset,dataloader) -> None:
        super(train_G).__init__()
        self.save_path = save_path
        self.dataset = dataset
        self.dataloader = dataloader
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(save_path+'ori')
            os.mkdir(save_path+'cam_mask')
            os.mkdir(save_path+'dilate_erode')
            os.mkdir(save_path+ 'pth')
            os.mkdir(save_path + 'gener_mask')


        self.g = Unet().cuda()
        
        self.loss = loss_function()
        self.model = resnet18_10cls_fune.cuda()
        self.trainer = torch.optim.SGD([{'params':self.g.parameters()}], lr=1e-4)
        self.flag = True
        self.cam_feature = []
        self.cam_weight = []
        self.cam = []
        self.cam_hook()




    def setup(self,img:torch.tensor,label:torch.tensor,index,epoch):

        self.data = (img,label)
        self.cam_feature.clear()
        self.cam_weight.clear()
        self.target_logit = self.model(img)[range(len(label)),label.tolist()].sum()
        self.target_logit.backward()
        self.get_cam_mask(index,epoch)
    

        
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

    def get_cam_mask(self,index,epoch): # 8cam 8batch 1 224 224 
        for i in range(len(self.cam_feature)): 
            temp = tran_tanh(self.cam_weight[i].unsqueeze(1) * self.cam_feature[i].unsqueeze(1))
            temp = interpolate(temp, size=(224,224), mode='bilinear', align_corners=True)
            for j in range(temp.shape[0]):
                temp[j] = sigmoid(temp[j],10,1,torch.median(temp[j]+1e-3))
            self.cam.append(temp) # [[16,1,224,224],[16,1,224,224]


        self.cam = torch.concat(self.cam,dim = 1) # 每张图的cam 合并在一起 16 8 224 224 

        mask = torch.ones_like(self.cam[:,0,:,:].unsqueeze(1)) # 8 1 16 16
        for i in range(0,4):
            init_input = torch.cat((mask,self.cam[:,i*2,:,:].unsqueeze(1)),dim=1) # 0 2 4 6 
            mask = self.g(init_input)
            save_images(tran_tanh(mask[0]).unsqueeze(0),[f'multi_mask_{index[0]}_{i*2}.png'],self.save_path+'gener_mask') 
        

        self.mask = mask  # 不是归一化后的值


    def train(self):
        epochs = 30
        max_eval = -10000
        eval_bacth = 30
        batch = self.loss.batch
        lambda1  = 20
        for epoch in range(epochs):
            avg_l = 0
            load_len = len(self.dataset)
            loss1_sum = 0
            loss2_sum = 0 
            loss3_sum = 0
            eval_num = 0 
            
            if self.loss.L1_flag and self.loss.mask_L2*1e-3>0.02:
                self.loss.lambda1 = self.loss.lambda1 * 0.91
            for i,(data,label,index) in enumerate(self.dataloader):
                img = data
                img = img.cuda()
                label = label.cuda()
                self.setup(img,label,index,epoch)

                self.trainer.zero_grad()
                x_logit = torch.functional.F.softmax(torch.pow(1.02,self.model(img)),dim=-1)  # From original logits to pdf
                m_logit = torch.functional.F.softmax(torch.pow(1.02,self.model(img+self.mask*torch.rand([img.shape[0],1,224,224],device='cuda:0'))),dim=-1) 
                l = self.loss.forward(x_logit,m_logit,self.mask) 
                l.backward()
                self.trainer.step()
            
                save_list = [j for j in range(len(index)) if index[j]%20==0]
                for j in save_list:
                    if epoch==0:
                        save_images(tran_tanh(unorm(img[j].unsqueeze(0))),[f'{index[j]}_ori_{label[j]}.png'],self.save_path+'ori')
                    if epoch>=0:
                        save_images(tran_tanh(self.mask[j].unsqueeze(1)),[f'{index[j]}_{epoch}_epoch.png' ],self.save_path+'cam_mask') # 8 4 224 224 
                        save_images(torch.where(self.loss.mask_label[j].unsqueeze(1)>0.5,1.0,0.0),[f'{index[j]}_{epoch}_epoch.png'],self.save_path+'dilate_erode') 
                        #save_images(tran_tanh(self.cam[j].unsqueeze(0)),[f'{index[j]}_{epoch}_cam.png'],'/home/sstl/fht/mask_train/sample/test_params2/cam/') 
                
                
                ##### 评估 该参数指标是否优秀，是否保存
                eval_num += eval((torch.where(self.loss.mask_label>0.5,1.0,0.0).norm(p=1,dim=(1,2,3))).tolist())
                
                avg_l += l.item()/load_len*batch
                loss1_sum += self.loss.loss1
                loss2_sum += self.loss.mask_L2 * 1e-3 
                loss3_sum += self.loss.loss3 
                if i%eval_bacth == (eval_bacth-1):
                    print(f'batch{i} Loss1_AVG:{loss1_sum/eval_bacth/self.loss.lambda1*lambda1:.6f},   Loss2_AVG:{loss2_sum/eval_bacth:.6f},   Loss3_AVG:{loss3_sum/eval_bacth:.6f}   lambda1:{self.loss.lambda1:.6f} lambda2:{self.loss.lambda2:.6f} lambda3:{self.loss.lambda3:.6f}  eval:{eval_num/i/batch:.3f}')
                    
                    try:
                        with open(self.save_path+'pth/pth_val.txt','a') as file:
                            file.write(f'batch{i}   Loss1_AVG:{loss1_sum/eval_bacth/self.loss.lambda1*lambda1:.6f},   Loss2_AVG:{loss2_sum/eval_bacth:.6f},   Loss3_AVG:{loss3_sum/eval_bacth:.6f} \n ')
                    except FileNotFoundError:
                        with open(self.save_path+'pth/pth_val.txt','w') as file:
                            file.write(f'Loss1_AVG:{loss1_sum/eval_bacth/self.loss.lambda1*lambda1:.6f},   Loss2_AVG:{loss2_sum/eval_bacth:.6f},   Loss3_AVG:{loss3_sum/eval_bacth:.6f} \n ')
                    
                    if abs(loss1_sum/eval_bacth) >0.3: #  使mask 中的点足够多 才开始收敛 kl散度足够大
                        self.loss.L2_flag =True 
                    if abs(loss1_sum/eval_bacth)>0.50: # 使Loss1 不过大
                        self.loss.L1_flag = True
                    if abs(loss2_sum/eval_bacth<0.5): # 使开始时不让loss1 loss2 产生对抗，使得收敛很慢
                        self.loss.L3_flag = True
                    loss1_sum=0;loss2_sum=0;loss3_sum=0
                    # eval_mask(self.cam_mask) 


            print(f"Epoch:{epoch}  Avg Loss: ", avg_l)

            
            if epoch >= 0:
                with open(self.save_path+'pth/pth_val.txt','a') as file:
                    file.write(f'epoch{epoch}  eval:  {eval_num/load_len:.3f}    \n\n')
                torch.save({'model':self.g.state_dict(),'optimizer':self.trainer.state_dict()},self.save_path+f'pth/model_{epoch}.pth')
            





        
run = train_G('/home/sstl/fht/patchfusion/sample/temp_img_label0/',dataset_list[0],dataloader_list[0])
run.train()






        



