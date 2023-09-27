from torchvision import models
from data import *

Inceptionv3 = models.inception_v3(pretrained=True)  # 输入 3 299 299
# 0: 145  1:153  2:289  3:404   4: 405  5:510   6:805  7:817  8: 867  9:950  
pred_index=[145,153,289,404,405,510,805,817,867,950]

GoogleNet = models.googlenet(pretrained=True) # 3 224 224 
# 0: 145  1:153  2:289  3:404   4: 405  5:510   6:805  7:817  8: 867  9:950
denseNet = models.densenet121(pretrained=True) # 3 224 224
# 0: 145  1:153  2:289  3:404   4: 405  5:510   6:805  7:817  8: 867  9:950
ViT = models.vit_b_16(pretrained=True) # 3 224 224

Vgg = models.vgg16_bn(pretrained=True)
Resnet50 = models.resnet50(pretrained=True)
res = []
for j in range(10):
    res.append([])

for i,(images,label,index) in enumerate(imagenet10_dataloader_bs32):


    pred = Vgg(images).argmax(dim=1)

    

    label_list = label.tolist()
    pred_list = pred.tolist()


    for j in range(10):
        for k in range(len(label_list)):
            if label_list[k] == j:
                res[j].append(pred_list[k])

    
    if i ==5:
        print(res)
        break

