import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

class GradCam():
    def __init__(self, model, savepath="/home/sstl/fht/mask_train/sample"):
        self.model = model.cuda()
        self.submodule_dict = dict(self.model.named_modules())
        self.target_layer = self.submodule_dict['1.layer4']
        self.hook_a, self.hook_g = None, None
        self.hook1 = self.target_layer.register_forward_hook(self._hook_a)
        self.hook2 = self.target_layer.register_backward_hook(self._hook_g)
        self.savepath = savepath

    def _hook_a(self, module, inp, out):
        self.hook_a = out

    def _hook_g(self, module, inp, out):
        self.hook_g = out[0]
    
    def cam_(self, img_path, class_idx, show=False, campp = True):
        img = Image.open(img_path, mode='r').convert('RGB')
        img_tensor = normalize(to_tensor(resize(img, (224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda()
        scores = self.model(img_tensor.unsqueeze(0))
        scores = self.model(img_tensor.unsqueeze(0))
        loss = scores[:,class_idx].sum()
        loss.backward()
        self.hook1.remove()
        self.hook2.remove()
        if campp:
            grad_2 = self.hook_g.pow(2)
            grad_3 = grad_2 * self.hook_g
            denom = 2 * grad_2 + (grad_3 * self.hook_a).sum(dim=(2, 3), keepdim=True)
            nan_mask = grad_2 > 0
            grad_2[nan_mask].div_(denom[nan_mask])
            weights = grad_2.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(dim=(1, 2))
        else:
            weights = self.hook_g.squeeze(0).mean(dim=(1,2))

        cam = (weights.view(*weights.shape, 1, 1) * self.hook_a.squeeze(0)).sum(0)
        cam = F.relu(cam)
        cam.sub_(cam.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cam.div_(cam.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        cam = cam.data.cpu().numpy()

        if show:
            heatmap = to_pil_image(cam, mode='F')
            overlay = heatmap.resize(img.size, resample=Image.BICUBIC)
            cmap = cm.get_cmap('jet')
            overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
            alpha = .7
            result = (alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8)
            plt.imsave(os.path.join(self.savepath, "cam.jpg"), result)
        return cam
    
    def cam(self, img_tensor, class_idx, campp = True):
        scores = self.model(img_tensor.unsqueeze(0))
        scores = self.model(img_tensor.unsqueeze(0))
        loss = scores[:,class_idx].sum()
        loss.backward()
        self.hook1.remove()
        self.hook2.remove()
        if campp:
            grad_2 = self.hook_g.pow(2)
            grad_3 = grad_2 * self.hook_g
            denom = 2 * grad_2 + (grad_3 * self.hook_a).sum(dim=(2, 3), keepdim=True)
            nan_mask = grad_2 > 0
            grad_2[nan_mask].div_(denom[nan_mask])
            weights = grad_2.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(dim=(1, 2))
        else:
            weights = self.hook_g.squeeze(0).mean(dim=(1,2))

        cam = (weights.view(*weights.shape, 1, 1) * self.hook_a.squeeze(0)).sum(0)
        cam = F.relu(cam)
        cam.sub_(cam.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cam.div_(cam.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        cam = cam.data

        return cam
    
    def save(self, cam, path):
        cam = cam.clone().detach()
        cam = cam.cpu().numpy()
        heatmap = to_pil_image(cam.squeeze(), mode='F')
        overlay = heatmap.resize((224, 244), resample=Image.BICUBIC)
        cmap = cm.get_cmap('jet')
        overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
        plt.imsave(path, overlay)
        


if __name__ == "__main__":
    from torch import nn
    from torchvision.models import resnet18
    
    import torch

    import torch
    from torch import nn

    class Normalize(nn.Module):
        def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            super(Normalize, self).__init__()
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)

        def forward(self, x):
            if bool(torch.max(x) <= 1 and torch.min(x) >= 0):
                return x
            return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

    norm = Normalize()


    class UnNormalize(object):
        def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):
            for t, m, s in zip(tensor, self.mean, self.std):
                t.mul_(s).add_(m)
            return tensor

    unorm = UnNormalize()
    resnet18_10cls = resnet18(pretrained=True)
    resnet18_10cls.fc = nn.Linear(resnet18_10cls.fc.in_features, 10)
    resnet18_10cls = nn.Sequential(norm, resnet18_10cls)
    resnet18_10cls.load_state_dict(torch.load('/home/sstl/Jerry/DRA-BlackBoxAttack/exp/resnet18_finetune/epoch4_acc0.962.pth'))
    setattr(resnet18_10cls, "name", "resnet18")
    
    gc = GradCam(resnet18_10cls)
    gc.cam_("/home/sstl/Jerry/experiments/Semantic_Adversarial_Patch_Combination/data/imagenet-10/n02056570/n02056570_25505.JPEG", 6, True)

