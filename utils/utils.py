import torch
def find_second_largest(tensors,label):
    
    indexs = []
    
    for j,tensor in enumerate(tensors):
        index = [-1,-1] # max_index, secend_max_idx
        max_num = float('-inf')
        second_max_num = float('-inf')
        for i,num in enumerate(tensor):
            if num > max_num:
                second_max_num = max_num; 
                max_num = num
                index[1] = index[0]
                index[0] = i
            elif num > second_max_num:
                second_max_num = num
                index[1] = i
        if label[j] == index[0]:
            indexs.append(index[1])
        else:
            indexs.append(index[0])
    return indexs


def eval_mask(img):
    print(img.std(),img.mean())


def tran_tanh(img):
    img_list = []

    for i in range(len(img)):

        max_pix = img[i].max()
        min_pix = img[i].min()
        img_list.append((img[i]-min_pix)/(max_pix-min_pix))

    return  torch.stack(img_list)


def norm_img(img):
    img_clone = img.clone()
    for i in range(img_clone.shape[0]):
        mean_ = img[i].mean()
        std_ = img[i].std()
        img_clone[i] = (img[i]-mean_)/std_
    return img_clone



import numpy as np
import matplotlib.pyplot as plt



def pdf(img):
    data = img.reshape(img.shape[0],-1).detach().cpu().numpy()

    for i in range(8):
        mu = np.mean(data[i]) 
        sigma = np.std(data[i])

        # 绘制直方图
        plt.hist(data[i], bins=500, density=True) 

        # 绘制pdf曲线
        x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        plt.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) ))

            
        # 保存图像
        plt.title('Group {}'.format(i+1)) 
        plt.savefig('/home/sstl/fht/patchfusion/show_img/group{}_dist.png'.format(i+1)) 
        plt.clf()


def minmax_sacle(tensor):
    out = tensor.clone()
    for i in range(tensor.shape[0]):
        min_ = tensor[i].min()
        max_ = tensor[i].max()
        out[i] = (tensor[i]-min_) / (max_-min_)
    return out


def rand_mat_with_mask(mask):
    mask = mask.detach()
    mean = torch.zeros_like(mask)
    std = mask
    return torch.normal(mean,std).cuda()