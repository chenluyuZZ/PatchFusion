

import argparse
import sys
sys.path.append('/home/sstl/fht/patchfusion/')

from model.resnet import resnet18_10cls_fune,densenet_10cls
from data.data import normal_transforms
from PIL import Image
from model.Unet import Unet
import argparse
from torchvision import models
from target_attack_two_patch_api import Pgd_Attack
from PIL  import Image, ImageOps
from utils.save_image import visualize_image
import torch
import gradio as gr
import argparse


model_list =[2,5,2,1,0,11,2,4,8,1]

label_text= ['企鹅','狗','豹子' ,'飞机' ,'飞艇' ,'轮船','足球','小汽车','卡车' ,'橘子']




def tensor_to_PIL(tensor):
    # 将tensor转换为numpy数组
    image = tensor.detach().cpu().numpy()[0].transpose(1,2,0)
    image = (image - image.min())*255 / (image.max() - image.min())
    
    # 将numpy数组转换为PIL图像
    seg_image = Image.fromarray(image.astype('uint8'))

    return seg_image

def resize_image(img, width, height):
    resized_image = img.resize((width, height))
    return resized_image

def ImageProcessor(image):
    # resize = [image.size[0], image.size[1]]
    # center = (0.5, 0.5)
    # image = ImageOps.fit(img, resize, centering=center)
    model = resnet18_10cls_fune.cuda()
    img = Image.fromarray(image.astype('uint8'), 'RGB')
    
    
    
    img = normal_transforms(img).cuda()
    pred = model(img)
    label = pred.argmax(dim=-1).data.item()
    attack_model = densenet_10cls.cuda()
    Attack = Pgd_Attack(model,attack_model,label,model_list[label],black_box=3,rate=25/255,target=True)
    att_img,att_pred = Attack.attack(img.unsqueeze(0),torch.tensor([label]))
    att_pred = att_pred.softmax(dim=-1)[0].data
    ori_pred = pred.softmax(dim=-1)[0].data
    # res = resize_image(tensor_to_PIL(att_img), img_shape[1], img_shape[0]),{label_text[i]: float(ori_pred[i].item()) for i in range(10)},{label_text[i]: float(att_pred[i].item()) for i in range(10)}
    res = tensor_to_PIL(att_img), {label_text[i]: float(ori_pred[i].item()) for i in range(10)},{label_text[i]: float(att_pred[i].item()) for i in range(10)}

    return res

        

if __name__ == '__main__':
    
    

    with gr.Blocks() as demo:
        gr.Markdown("Patchfusion demo.")
        with gr.Tab("attack demo"):
            with gr.Row(): #并行显示，可开多列
                with gr.Column(): # 并列显示，可开多行
                    inputs = gr.inputs.Image(shape=(800,800), label='原始图像').style(height=410)
                    label_input = gr.outputs.Label(num_top_classes=3,label='原始预测分数')
                    
                with gr.Column():
                    image_out = gr.outputs.Image(type="pil",label='攻击后图像').style(height=410)
                    label_input = gr.outputs.Label(num_top_classes=3,label='原始预测分数')
                    label_out = gr.outputs.Label(num_top_classes=3,label='攻击后预测分数')
                    

    gr.Interface(fn=ImageProcessor, inputs=inputs, outputs=[image_out,label_input,label_out],title="物理世界对抗攻击展示").launch()
    
