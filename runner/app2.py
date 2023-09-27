

import argparse

from model.resnet import resnet18_10cls_fune,densenet_10cls
from data.data import normal_transforms
from PIL import Image
from model.Unet import Unet
import argparse
from torchvision import models
from target_attack_two_patch_api import Pgd_Attack
from PIL  import Image 
from utils.save_image import visualize_image
import torch
import gradio as gr
import argparse
from data import *

model_list =[2,5,2,1,0,11,2,4,8,1]
img_list = [145,153,289,404,405,510,805,817,867,950]
label_text= ['企鹅','狗','豹子' ,'飞机' ,'飞艇' ,'轮船','足球','小汽车','卡车' ,'橘子']




def tensor_to_PIL(tensor):
    # 将tensor转换为numpy数组
    image = tensor.detach().cpu().numpy()[0].transpose(1,2,0)
    image = (image - image.min())*255 / (image.max() - image.min())
    # 将numpy数组转换为PIL图像
    seg_image = Image.fromarray(image.astype('uint8'))
    return seg_image

def ImageProcessor(image):
    
    model =resnet18_10cls_fune.cuda()
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    img_shape = img.size[::-1]
    
    img = normal_transforms(img).cuda()
    pred = model(img.unsqueeze(0)).softmax(dim=-1)
    label = pred.argmax(dim=-1).data.item()
    return {label_text[i]: float(pred[0][i].item()) for i in range(10)}

        

if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.Markdown("ImageNet + 类检测.")
        with gr.Tab(""):
            with gr.Row(): #并行显示，可开多列
                with gr.Column(): # 并列显示，可开多行
                    inputs = gr.inputs.Image(label='图像').style(height=410)
                    
                with gr.Column():
                    label_input = gr.outputs.Label(num_top_classes=10,label='预测分数')
                    

    gr.Interface(fn=ImageProcessor, inputs=inputs, outputs=[label_input],title='Imagenet 十类检测',description='企鹅 ,狗 ,豹子 ,飞机 ,飞艇 ,轮船 ,足球 ,小汽车 ,卡车,橘子').launch()