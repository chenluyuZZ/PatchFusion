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
import shutil
import warnings
warnings.simplefilter("ignore")
# 改用10类数据进行处理
# 仅当初始LOSS2 >=0.95 才可用此参数
#   0.8 0.02 下降  0.8 0.03 下降 0.6 0.02 上升 0.7 0.02 上升 0.8 0.03 下降
# 0.4  0.03



class loss_function(nn.Module):
    def __init__(self) -> None:
        super(loss_function).__init__()
        self.batch = 8
        self.L2_lim =0.02
        self.lambda1 = 20
        self.lambda2 = 0
        self.lambda3 = 0
        self.mask_size = 224* 224




    def forward(self, x_logit:torch.Tensor, m_logit:torch.Tensor,mask:torch.Tensor) -> torch.Tensor:

        img_dilate = tensor_dilate(mask,ksize=7)
        img_erode = tensor_erode(img_dilate,ksize=7)
        self.mask_label = filters.median_blur(img_erode,(5,5))
        self.mask_label = torch.where(self.mask_label>0.5,1.0,0.0)
        
        
        self.loss1 = -torch.functional.F.kl_div(x_logit.log(), m_logit) *self.lambda1
        self.loss2 = (mask.sum()/mask.shape[0] -  self.mask_size * self.L2_lim) *1e-3 * self.lambda2
   

        self.loss3 = (mask-self.mask_label).norm(p=2) * self.lambda3 # 会使mask 分化严重，根据划分点来，0.5以下的更多20
        

        if self.loss2.isnan():
            print('erro')
        
        # print(f'LOSS1: {self.loss1.item()}    LOSS2: {self.mask_L2*1e-3}          {self.loss3}')

        return self.loss1  +self.loss2+ self.loss3



def sigmoid(x,a,c,throde):
    return c/(1+torch.exp(-a*(x-throde)))

generate_model_patch = '/home/sstl/fht/patchfusion/sample/handle_optim/class_reoptim8/pth/model_1.pth'


class train_G():
    def __init__(self,save_path,dataset,dataloader) -> None:
        super(train_G).__init__()
        self.save_path = save_path
        self.dataset = dataset
        self.dataloader = dataloader
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.mkdir(save_path)
        os.mkdir(save_path+'ori')
        os.mkdir(save_path+'cam_mask')
        os.mkdir(save_path+'dilate_erode')
        os.mkdir(save_path+ 'pth')
        os.mkdir(save_path + 'gener_mask')

        self.g = Unet().cuda()
        self.g.load_state_dict(torch.load(generate_model_patch)['model'])
        self.loss = loss_function()
        self.model = resnet18_10cls_fune.cuda().eval()
        self.trainer = torch.optim.SGD([{'params':self.g.parameters()}], lr=1e-4)
        self.trainer.load_state_dict(torch.load(generate_model_patch)['optimizer'])
        self.cam_feature = []
        self.cam_weight = []
        self.cam_list = []
        self.cam_hook()


    def setup(self,img:torch.tensor,label:torch.tensor,index,epoch):
        # get cam_mask
        self.data = (img,label)
        self.cam_feature.clear()
        self.cam_weight.clear()
        self.target_logit = self.model(img)[range(len(label)),label.tolist()].sum()
        self.target_logit.backward()
        self.get_cam_mask(index,epoch) 
    

        
    def forward_hook(self,modules,input,output):
        temp = output.clone().detach().mean(dim=1)
        self.cam_feature.insert(0,temp) 
        
    
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

    def get_cam_mask(self,index,epoch): 
        self.cam_list = []
        for i in range(len(self.cam_feature)): 
            temp = norm_img(self.cam_weight[i].unsqueeze(1) * self.cam_feature[i].unsqueeze(1)) 
            temp = interpolate(temp, size=(224,224), mode='bilinear', align_corners=True)
            self.cam_list.append(temp) # 8个grad_cam


        self.cam = torch.concat(self.cam_list,dim = 1)
        mask = torch.ones_like(self.cam[:,0,:,:].unsqueeze(1))
        for i in range(0,4):
            init_input = torch.cat((mask,self.cam[:,i*2,:,:].unsqueeze(1)),dim=1) # 0 2 4 6 
            mask = norm_img(self.g(init_input))
            mask = sigmoid(mask,2,1,2.5) # 

            save_images(mask[0].unsqueeze(0),[f'multi_mask_{index[0]}_{i*2}.png'],self.save_path+'gener_mask') 
            
        self.mask = mask 

        
        

    def train(self):
        epochs = 100
        max_eval = -10000
        eval_bacth = 10
        batch = self.loss.batch
        lambda1  = 20
        for epoch in range(epochs):

            self.loss.lambda1=float(input('lambda1,defate:20: '))
            self.loss.lambda2=float(input('lambda2,defate:0.8: ')) 
            self.loss.lambda3=float(input('lambda3,defate:0.02: ')) 
            min_batchs = int(input('min batchs:'))
            with open(self.save_path+'pth/pth_val.txt','a') as file:
                file.write('==='*10)
                file.write(f'Epoch:{epoch}    lambda1: {self.loss.lambda1}   lambda2: {self.loss.lambda2}  lambda3: {self.loss.lambda3} \n')
            
            for min_batch in range(min_batchs):
                avg_l = 0
                load_len = len(self.dataset)
                loss1_sum = 0
                loss2_sum = 0 
                loss3_sum = 0
                eval_num = 0 
                no_success = 0
                for i,(data,label,index) in enumerate(self.dataloader):
                    img = data
                    img = img.cuda()
                    label = label.cuda()
                    self.setup(img,label,index,epoch)

                    self.trainer.zero_grad()# 72 -29 pred
                    x_logit = torch.functional.F.softmax(torch.pow(1.02,self.model(img)),dim=-1)  # From original logits to pdf
                    m_logit = torch.functional.F.softmax(torch.pow(1.02,self.model(img+self.mask*norm(rand_mat_with_mask(self.mask)))),dim=-1) 
                    # if (x_logit.argmax(dim=-1)==m_logit.argmax(dim=-1)).sum():
                    no_success += (x_logit.argmax(dim=-1)==m_logit.argmax(dim=-1)).sum()
                    #     print(no_success/((i+1)*12))
                        
                    #m_logit = torch.functional.F.softmax(torch.pow(1.025,self.model(img+self.mask*norm(torch.rand([img.shape[0],1,224,224],device='cuda:0')))),dim=-1) 
                    l = self.loss.forward(x_logit,m_logit,self.mask) 
                    l.backward()
                    self.trainer.step()
                
                    save_list = [j for j in range(len(index)) if index[j]%20==0]
                    for j in save_list:
                        if epoch==0:
                            save_images(tran_tanh(unorm(img[j].unsqueeze(0))),[f'{index[j]}_ori_{label[j]}.png'],self.save_path+'ori')
                        if epoch>=0:
                            save_images(tran_tanh(self.mask[j].unsqueeze(1)),[f'{index[j]}__{epoch}_{min_batch}.png' ],self.save_path+'cam_mask') # 8 4 224 224 
                            save_images(torch.where(self.loss.mask_label[j].unsqueeze(1)>0.5,1.0,0.0),[f'{index[j]}_{epoch}_{min_batch}.png'],self.save_path+'dilate_erode') 
                            
                    
                    
                    
                    avg_l += l.item()/load_len*batch
                    loss1_sum += self.loss.loss1/self.loss.lambda1 *20
                    loss2_sum += self.loss.loss2 /self.loss.lambda2 
                    loss3_sum += self.loss.loss3 / self.loss.lambda3 * 0.01
                
                    if i%eval_bacth == (eval_bacth-1):
                        print(f'spot check{i} Loss1_AVG:{loss1_sum/i/self.loss.lambda1*lambda1:.6f},   Loss2_AVG:{loss2_sum/i:.6f}   Loss3_AVG:{loss3_sum/i:.6f} mask: {(self.mask>0.5).sum()/8}   lambda1:{self.loss.lambda1:.6f} lambda2:{self.loss.lambda2:.6f} lambda3:{self.loss.lambda3:.6f}  no_succes:{no_success/(i+1)/12}')
                        
                        try:
                            with open(self.save_path+'pth/pth_val.txt','a') as file:
                                file.write(f'       spot check{i}   Loss1_AVG:{loss1_sum/i/self.loss.lambda1*lambda1:.6f},   Loss2_AVG:{loss2_sum/i:.6f},   Loss3_AVG:{loss3_sum/i:.6f}  mask: {(self.mask>0.5).sum()/8}  no_succes:{no_success/(i+1)/12}\n')
                        except FileNotFoundError:
                            with open(self.save_path+'pth/pth_val.txt','w') as file:
                                file.write(f'       spot check{i}  Loss1_AVG:{loss1_sum/i/self.loss.lambda1*lambda1:.6f},   Loss2_AVG:{loss2_sum/i:.6f},   Loss3_AVG:{loss3_sum/i:.6f} mask: {(self.mask>0.5).sum()/8} no_succes:{no_success/(i+1)/12}\n ')

                with open(self.save_path+'pth/pth_val.txt','a') as file:
                    file.write(f"Epoch:{epoch} min_batch: {min_batch}   Avg Loss: {avg_l} \n")
                print(f"Epoch:{epoch}  Avg Loss: ", avg_l)

            
            if epoch >= 0:
                with open(self.save_path+'pth/pth_val.txt','a') as file:
                    file.write(f'epoch{epoch}  eval:  {eval_num/load_len:.3f}    \n\n')
                torch.save({'model':self.g.state_dict(),'optimizer':self.trainer.state_dict()},self.save_path+f'pth/model_{epoch}.pth')
            





        
run = train_G('/home/sstl/fht/patchfusion/sample/handle_optim/class_reoptim9/',imagenet10_dataset,imagenet10_dataloader_bs12)
run.train()






        



