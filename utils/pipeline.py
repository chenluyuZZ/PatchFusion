import torch
import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import os
import torch.nn.functional as F
from kornia import filters




def cluster_pipeline(input_path,out_path, cluster=1, dilate_size=7, erode_size=8, blur_size=5):
    # read mask
    mask_list = sorted(os.listdir(input_path),key= lambda x:(int(x.split("_")[1]),int(x.split("_")[0])))
    for mask_name in mask_list[::-1]:
        mask  = cv2.imread(os.path.join(input_path,mask_name))
        mask = torch.from_numpy(mask)
        mask = mask.to(torch.device('cpu'))
        mask = mask[:,:,0]
        # convert pixels into np.array
        points = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                tmp = []
                if mask[i][j] != 0:
                    tmp.append(j)
                    tmp.append(223 - i)
                    points.append(tmp)
        points = np.array(points)

        # kmeans clustering
        kmeans = KMeans(n_clusters=cluster, init='k-means++').fit(X=points)
        labels = kmeans.predict(points)
        centers = kmeans.cluster_centers_
        # save original cluster plot with cv into ori
        ori = np.where(mask == 0, 0, 255)
        ori = np.expand_dims(ori, -1)
        ori = np.tile(ori, (1,1,3))
        centers = np.int32(centers)
        ori = np.float32(ori)
        for c in centers:
            cv2.circle(ori, (c[0], 223-c[1]), 2, [255,0,0], 2)
        mat = np.where(mask == 0, 0, 255)

        # record points to drop using GaussianMixture model
        drop = []
        # if multiple clusters or not
        if cluster == 1:
            cluster_data = labels == 0
            X = points[cluster_data]
            cluster_data = [i for i, x in enumerate(cluster_data) if x]
            gm = GaussianMixture(n_components=1, random_state=0).fit(X)
            tmp = gm.score_samples(X) < -10
            tmp = [i for i, x in enumerate(tmp) if x]
            for t in tmp:
                drop.append(cluster_data[t])
        elif cluster > 1:
            for i in range(cluster):
                cluster_data = labels == i
                X = points[cluster_data]
                cluster_data = [i for i, x in enumerate(cluster_data) if x]
                gm = GaussianMixture(n_components=1, random_state=0).fit(X)
                tmp = gm.score_samples(X) < -10
                tmp = [i for i, x in enumerate(tmp) if x]
                for t in tmp:
                    drop.append(cluster_data[t])
        # drop samples with GM model and save into mat
        for p in points[drop]:
            x = 223 - p[1]
            y = p[0]
            mat[x][y] = 0
        mat = np.expand_dims(mat, -1)
        mat = np.tile(mat, (1, 1, 3))
        centers = np.int32(centers)
        mat = np.float32(mat)
        for c in centers:
            cv2.circle(mat, (c[0], 223-c[1]), 2, [255,0,0], 2)
        
        # dilate
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
        morph = cv2.morphologyEx(mat, cv2.MORPH_DILATE, se)
        # erode
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
        morph2 = cv2.morphologyEx(morph, cv2.MORPH_ERODE, se2)
        # medianBlur
        mat_blur = cv2.medianBlur(morph2, blur_size) 
        # combined = np.hstack([ori, mat, morph, morph2, mat_blur])
        cv2.imwrite(os.path.join(out_path,mask_name), mat_blur)



input_path = "/home/sstl/fht/mask_train/sample/test_params1_log4/cam_mask/"
out_path = "/home/sstl/fht/mask_train/sample/test_params1_log4/medianBlur/"

# cluster_pipeline(input_path,out_path, 5, 9, 8, 5)






def dilation_erosion(masks, cluster=5, dilate_size=7, erode_size=3, blur_size=5):
    # read mask
    masks = masks.mean(dim=1) #32 1 224 224 -> 32 224 224
    mask_list = []
    correct_mask = []
    for k in range(masks.shape[0]):
        mask = masks[k].to(torch.device('cpu'))
        # if (mask>0).sum()<50: # (mask>0).sum()>2500
        #     mask_list.append(np.zeros_like(mask))
        #     continue #跳过当前
        # convert pixels into np.array
        points = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                tmp = []
                if mask[i][j] != 0:
                    tmp.append(j)
                    tmp.append(223 - i)
                    points.append(tmp)
        points = np.array(points)

        # kmeans clustering
        try:
            kmeans = KMeans(n_clusters=cluster, init='k-means++').fit(X=points)
        except:
            mask_list.append(np.zeros_like(mask))
            continue
        labels = kmeans.predict(points)
        centers = kmeans.cluster_centers_
        # save original cluster plot with cv into ori
        ori = np.where(mask == 0, 0, 255)
        ori = np.expand_dims(ori, -1)
        ori = np.tile(ori, (1,1,3))
        centers = np.int32(centers)
        ori = np.float32(ori)
        for c in centers:
            cv2.circle(ori, (c[0], 223-c[1]), 2, [255,0,0], 2)
        mat = np.where(mask == 0, 0, 255)

        # record points to drop using GaussianMixture model
        drop = []
        # if multiple clusters or not
        if cluster == 1:
            cluster_data = labels == 0
            X = points[cluster_data]
            cluster_data = [i for i, x in enumerate(cluster_data) if x]
            gm = GaussianMixture(n_components=1, random_state=0).fit(X)
            tmp = gm.score_samples(X) < -10
            tmp = [i for i, x in enumerate(tmp) if x]
            for t in tmp:
                drop.append(cluster_data[t])
        elif cluster > 1:
            for i in range(cluster):
                cluster_data = labels == i
                X = points[cluster_data]
                cluster_data = [i for i, x in enumerate(cluster_data) if x]
                try:
                    gm = GaussianMixture(n_components=1, random_state=0).fit(X)
                except:
                    mask_list.append(np.zeros_like(mask))
                    break
                tmp = gm.score_samples(X) < -10
                tmp = [i for i, x in enumerate(tmp) if x]
                for t in tmp:
                    drop.append(cluster_data[t])
        # drop samples with GM model and save into mat
        if len(mask_list)==k+1:
            continue #意味上面已执行try catch 跳过此循环
        for p in points[drop]:
            x = 223 - p[1]
            y = p[0]
            mat[x][y] = 0
        mat = np.expand_dims(mat, -1)
        mat = np.tile(mat, (1, 1, 3))
        centers = np.int32(centers)
        mat = np.float32(mat)
        for c in centers:
            cv2.circle(mat, (c[0], 223-c[1]), 2, [255,0,0], 2)
        
        # dilate
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_size, dilate_size))
        morph = cv2.morphologyEx(mat, cv2.MORPH_DILATE, se)
        # erode
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
        morph2 = cv2.morphologyEx(morph, cv2.MORPH_ERODE, se2)
        # medianBlur
        mat_blur = cv2.medianBlur(morph2, blur_size) 
        mask_list.append(mat_blur[:,:,1]/255)
        if (mat_blur[:,:,1]>0).sum()<20000 and(mat_blur[:,:,1]>0).sum()>500:
            correct_mask.append(k)
        #ori, mat, morph, morph2, mat_blur
    correct_mask = range(masks.shape[0])
    return np.stack(mask_list,axis=0),correct_mask







def tensor_dilate(bin_img, ksize=7): #
    # 首先为原图加入 padding，防止图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    # 取每个 patch 中最小的值，i.e., 0
    dilate, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return dilate

def tensor_erode(bin_img, ksize=7): # 已测试 
    #先为原图加入 padding，防止腐蚀后图像尺寸缩小
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
    # 将原图 unfold 成 patch
    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)
    # B x C x H x W x k x k
    # 取每个 patch 中最小的值
    eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
    return eroded



import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
def main():

    # 加载二值图像并转换为PIL图像对象
    img = Image.open('/home/sstl/fht/mask_train/0_10_epoch.png').convert('1')

    # 将PIL图像对象转换为PyTorch张量
    img_tensor = TF.to_tensor(img)

    img_dilation_tensor = tensor_dilate(img_tensor.unsqueeze(0))
    img_erosion_tensor = tensor_erode(img_dilation_tensor)
    img_avg_tensor = filters.median_blur(img_erosion_tensor,kernel_size=(5,5))
    
    img_dilation = TF.to_pil_image(img_dilation_tensor.squeeze())
    img_dilation.save('dilated_image.png')
    img_erosion = TF.to_pil_image(img_erosion_tensor.squeeze())
    img_erosion.save('eroded_image.png')
    img_avg = TF.to_pil_image(img_avg_tensor.squeeze())
    img_avg.save('avg_image.png')



