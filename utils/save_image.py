
import torch
import cv2
import os
def save_images(input_tensors, filenames, output_dir):
    assert (len(input_tensors.shape) == 4)
    for i in range(input_tensors.shape[0]):
        input_tensor = input_tensors[i,:,:,:]
        filename = filenames[i]
        input_tensor = input_tensor.clone().detach()
        input_tensor = input_tensor.to(torch.device('cpu'))
        input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, filename), input_tensor)




import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_image(image, img_size,save_path=None):
    
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
        
    image = torch.nn.functional.interpolate(image, size=img_size, mode='bilinear', align_corners=True)[0]
    image = image.cpu().detach().numpy()
    
    normalized_image = (image - image.min()) / (image.max() - image.min())
    
    
    plt.figure()
    plt.imshow(normalized_image.transpose(1, 2, 0))
    
    plt.axis('off')  # 关闭坐标轴的显示
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 去掉白边
        print(f"Image saved to: {save_path}")
    else:
        plt.show()