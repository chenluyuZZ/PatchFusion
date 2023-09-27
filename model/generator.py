import torch.nn as nn

# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.net = nn.Sequential(
#             # input size 1 x 7 x 7
#             nn.ConvTranspose2d(1, 64, (2, 2), (1, 1), (0, 0), bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # output size 64 x 8 x 8
#             nn.ConvTranspose2d(64, 128, (4, 4), (3, 3), (0, 0), bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # output size 128 x 25 x 25
#             nn.ConvTranspose2d(128, 256, (3, 3), (2, 2), (0, 0), bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # output size 256 x 51 x 51
#             nn.ConvTranspose2d(256, 128, (4, 4), (1, 1), (0, 0), bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # output size 128 x 54 x 54
#             nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (0, 0), bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # output size 64 x 110 x 110
#             nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (0, 0), bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(True),
#             # output size 32 x 221 x 221
#             nn.ConvTranspose2d(32, 1, (3, 3), (1, 1), (0, 0), bias=True),
#             nn.Tanh()
#         )
    
#     def forward(self, x):
#         if len(x.shape) == 2:
#             x = x.unsqueeze(0).unsqueeze(0)
#         return (0.5 * self.net(x) + 0.5).squeeze()





class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # input size 1 x 14 x 14
            nn.ConvTranspose2d(1, 64, (3, 3), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # output size 64 x 16 x 16
            nn.ConvTranspose2d(64, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # output size 128 x 31 x 31
            nn.ConvTranspose2d(128, 256, (2, 2), (2, 2), (2, 2), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # output size 256 x 58 x 58
            nn.ConvTranspose2d(256, 128, (3, 3), (1, 1), (2, 2), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # output size 128 x 56 x 56
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # output size 64 x 112 x 112
            nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # output size 32 x 224 x 224
            nn.ConvTranspose2d(32, 1, (3, 3), (1, 1), (1, 1), bias=True),
            nn.Tanh()
        )
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        return (0.5 * self.net(x) + 0.5).squeeze()


if __name__ == '__main__':
    import torch
    inp = torch.rand([1,14,14])
    g = Generator()
    opt = g(inp.unsqueeze(0))
    print(opt.shape)