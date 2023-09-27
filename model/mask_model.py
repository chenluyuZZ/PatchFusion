import torch
import torch.nn as nn
import torchvision

class mask_model(nn.Module):
    def __init__(self):
        super(mask_model,self).__init__()
        # input_size 3*224*224


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False), #64*112*112
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.maxpool= nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),return_indices=True) #64 56 56 
        
        self.conv2=nn.Sequential( # 64 56 56 
            nn.Conv2d(64,64,(3,3),(1,1),(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential( # 112 28 28
            nn.Conv2d(64,112,(3,3),(2,2),(1,1)),
            nn.BatchNorm2d(112),
            nn.ReLU(True)
        )
        self.conv3Trans = nn.Sequential( # 64 55 55
            nn.ConvTranspose2d(112,64,(4,4),(2,2),(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.conv2Trans = nn.Sequential( # 64 56 56 
            nn.ConvTranspose2d(64,64,(3,3),(1,1),(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.maxuppool = nn.MaxUnpool2d(kernel_size=(2,2),stride=(2,2)) # 64 112 112
        self.conv1Trans = nn.Sequential( # 3 224 224
            nn.ConvTranspose2d(64,3,(8,8),(2,2),(3,3),bias=False), #3 224 224
            nn.BatchNorm2d(3),
        )
    
    def forward(self,input):
        x = self.conv1(input)
        x,indice = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv3Trans(x)
        x = self.conv2Trans(x)
        x = self.maxuppool(x,indice)
        x = self.conv1Trans(x)

        return x

        




