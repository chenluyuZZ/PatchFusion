from torch import nn
from torchvision.models import resnet18, resnet34,densenet121
from utils.normalization import norm
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def remove_batchnorm_recursive(module):
    """
    递归地删除模型中的批归一化（Batch Normalization）模块
    Args:
        module (nn.Module): 需要删除批归一化模块的模块

    Returns:
        nn.Module: 不包含批归一化模块的新模块
    """
    if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
        for i, child in enumerate(module.children()):
            module[i] = remove_batchnorm_recursive(child)
        return module
    
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        return nn.Identity()
    
    elif isinstance(module, nn.Module):
        for name, child in module.named_children():
            module.__setattr__(name, remove_batchnorm_recursive(child))
        return module
    else:
        return module
    
resnet18_10cls = resnet18(pretrained=True)
resnet18_10cls.fc = nn.Linear(resnet18_10cls.fc.in_features, 10)

resnet18_10cls = remove_batchnorm_recursive(resnet18_10cls)
resnet18_10cls_fune = nn.Sequential(norm, resnet18_10cls)
resnet18_10cls_fune.load_state_dict(torch.load('/home/sstl/LaneDetection/Code/DRA/exp/res18_bn10/Finetune/epoch9_acc0.964.pth'))
#resnet18_10cls_fune.load_state_dict(torch.load('/home/sstl/LaneDetection/Code/DRA/exp/exp3_lam6_1/Finetune/epoch5_acc0.959.pth'))
setattr(resnet18_10cls_fune, "name", "resnet18")





# resnet34_10cls = resnet34(pretrained=True)
# resnet34_10cls.fc = nn.Linear(resnet34_10cls.fc.in_features, 10)
# resnet34_10cls = nn.Sequential(norm, resnet34_10cls)
# setattr(resnet34_10cls, "name", "resnet34")


densenet_10cls = densenet121(pretrained=True)
densenet_10cls = remove_batchnorm_recursive(densenet_10cls)
densenet_10cls = nn.Sequential(densenet_10cls,nn.Linear(densenet_10cls.classifier.out_features,10))
densenet_10cls.load_state_dict(torch.load('/home/sstl/Jerry/DRA-BlackBoxAttack/exp/dense/Train/acc_0.91.pth'))
setattr(densenet_10cls, "name", "densenet")


purtube_model = nn.Sequential(
    nn.Linear(10,10),
    nn.ReLU(),
    nn.Linear(10,10),
    nn.ReLU(),
    nn.Linear(10,1)
)