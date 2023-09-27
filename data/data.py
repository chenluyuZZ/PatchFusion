
import torch
import torch.nn as nn
import torchvision
from  torch.utils.data import DataLoader,Dataset
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from typing import Union
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from torchvision.datasets.folder import default_loader,IMG_EXTENSIONS
from PIL import Image
import os
from PIL import Image





data_list = ['n02056570','n02085936','n02128757', 'n02690373','n02692877','n03095699','n04254680','n04285008','n04467665','n07747607']
class MY_ImageFolder(torchvision.datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target,index

class MyData(Dataset):

    def __init__(self, root_dir,transform,target):
        self.root_dir = root_dir
        self.path = os.path.join(root_dir)
        self.img_path = sorted(os.listdir(self.path),key = lambda x: int(x.split('.')[0].split('_')[1]))

        self.transform = transform
        self.target = target

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = torch.tensor(self.target,dtype=torch.int32)
        return img,target,idx
    def __len__(self) -> int:
        return len(self.img_path)

normal_transforms = \
        torchvision.transforms.Compose([ 
        torchvision.transforms.Resize((224,224)),
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

Inceptionv3_transforms = \
        torchvision.transforms.Compose([ 
        torchvision.transforms.Resize((299,299)),
        # torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


imagenet10_dataset = MY_ImageFolder("/opt/Data/imagenet-10/", transform=normal_transforms)
imagenet10_dataloader_bs32 = DataLoader(imagenet10_dataset, batch_size=32, num_workers=4,shuffle=True)


imagenet10_dataloader_bs12 = DataLoader(imagenet10_dataset, batch_size=12, num_workers=4,shuffle=True)

imagenet10_dataloader_bs8 = DataLoader(imagenet10_dataset, batch_size=8, num_workers=4,shuffle=True)


dataset_list = []  
dataloader_list= []
path_list = [f'/home/sstl/fht/mask_train/cls_train/cls{i}/'for i in range(10)]
for i in range(len(data_list)):
    dataset_list.append(MyData(f"/opt/Data/imagenet-10/{data_list[i]}/", transform=normal_transforms,target=i))
    dataloader_list.append(DataLoader(dataset_list[i], batch_size=8, num_workers=4))



