# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import gc
import json
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from PIL import Image, ImageFile
from numpy.lib._iotools import str2bool

from detection_models import networks
from detection_util.util import *

warnings.filterwarnings("ignore", category=UserWarning)

ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
该函数 data_transforms 根据参数 full_size 对图像进行不同方式的缩放处理：
如果 full_size 为 "full_size"，则将图像尺寸调整为最接近的16的倍数。
如果 full_size 为 "scale_256"，则先将图像短边缩放到256，再调整为最接近的16的倍数。
'''
def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    elif full_size == "scale_256":
        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)

'''
该函数用于调整图像张量的尺寸。首先获取图像的宽度和高度，根据较短边调整为默认尺寸（256），并按比例缩放较长边。
然后将高度和宽度调整为16的倍数，最后使用双线性插值法进行缩放。
'''
def scale_tensor(img_tensor, default_scale=256):
    _, _, w, h = img_tensor.shape
    if w < h:
        ow = default_scale
        oh = h / w * default_scale
    else:
        oh = default_scale
        ow = w / h * default_scale

    oh = int(round(oh / 16) * 16)
    ow = int(round(ow / 16) * 16)

    return F.interpolate(img_tensor, [ow, oh], mode="bilinear")

'''
该函数用于将图像与掩码进行融合。具体步骤如下：
将输入图像转换为浮点数类型的NumPy数组。
计算融合后的图像：原图像乘以（1 - 掩码）加上掩码乘以255，然后转换为8位无符号整型。
将结果转换为RGB格式的PIL图像并返回。
'''
def blend_mask(img, mask):

    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")


def main(config):
    print("初始化 Dataloader")
    # 创建UNet实例
    model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )

    ## 加载模型，从本地加载
    print("开始加载本地模型参数文件...")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints/detection/FT_Epoch_latest.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print("模型参数加载完成")
    if config.mps:
        model.to("mps")
        print("加载到MPS")
    else:
        if config.GPU >= 0:
            print("加载到GPU")
            model.to(config.GPU)
        else:
            print("加载到CPU")
            model.cpu()
    print("设置模型为模型评估模式")
    model.eval()

    ## 数据加载和转换
    print("测试图片目录: " + config.test_path)
    # 加载当前目录下所有文件
    imagelist = os.listdir(config.test_path)
    imagelist.sort()
    total_iter = 0

    P_matrix = {}
    save_url = os.path.join(config.output_dir)
    mkdir_if_not(save_url)

    input_dir = os.path.join(save_url, "input")
    output_dir = os.path.join(save_url, "mask")
    # blend_output_dir=os.path.join(save_url, 'blend_output')
    mkdir_if_not(input_dir)
    mkdir_if_not(output_dir)
    # mkdir_if_not(blend_output_dir)

    idx = 0

    results = []
    # 对照片遍历
    for image_name in imagelist:

        idx += 1

        print("开始处理图片:", image_name)

        scratch_file = os.path.join(config.test_path, image_name)
        if not os.path.isfile(scratch_file):
            print("跳过无效文件 %s" % image_name)
            continue
        # 转换为RGB格式
        scratch_image = Image.open(scratch_file).convert("RGB")
        # 获取图片宽，高
        w, h = scratch_image.size
        # 对原始图像应用数据转换，转换为PIL图像格式
        transformed_image_PIL = data_transforms(scratch_image, config.input_size)
        # 将转换后的图像转换为灰度模式
        scratch_image = transformed_image_PIL.convert("L")
        # 将灰度图像转换为PyTorch张量
        scratch_image = tv.transforms.ToTensor()(scratch_image)
        # 对图像数据进行归一化处理
        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)
        # 在维度0处扩展图像张量，以匹配神经网络输入的形状要求
        scratch_image = torch.unsqueeze(scratch_image, 0)

        # 获取未处理图像的宽度和高度
        _, _, ow, oh = scratch_image.shape
        # 对图像进行缩放处理
        scratch_image_scale = scale_tensor(scratch_image)
        # 根据配置决定是否使用GPU加速计算
        if config.GPU >= 0:
            scratch_image_scale = scratch_image_scale.to(config.GPU)
        else:
            scratch_image_scale = scratch_image_scale.cpu()
        # 在评估模式下运行模型，关闭梯度计算以节省内存
        with torch.no_grad():
            # 使用sigmoid函数将模型的输出转换为概率
            P = torch.sigmoid(model(scratch_image_scale))

        P = P.data.cpu()
        # 使用最近邻插值法调整P的尺寸到原始图像的宽度ow和高度oh
        P = F.interpolate(P, [ow, oh], mode="nearest")

        # 保存概率图，阈值为0.4，以PNG格式保存到输出目录
        tv.utils.save_image(
            (P >= 0.4).float(),  # 将概率P转换为二值图，概率大于等于0.4的像素点设为1，否则设为0，并将类型转换为float
            os.path.join(
                output_dir,  # 输出目录
                image_name[:-4] + ".png",  # 保存的文件名，使用原始文件名去掉扩展名后加上".png"
            ),
            nrow=1,  # 每行显示图片数量为1
            padding=0,  # 图片间的填充像素为0
            normalize=True,  # 对图像进行归一化处理
        )

        # 保存变换后的图像到输入目录，文件名同上
        transformed_image_PIL.save(os.path.join(input_dir, image_name[:-4] + ".png"))

        # 清理内存和显存
        gc.collect()  # 调用垃圾回收器清理内存
        torch.cuda.empty_cache()  # 清空CUDA缓存，释放显存


if __name__ == "__main__":
    # 获取参数
    parser = argparse.ArgumentParser()
    # parser.add_argument('--checkpoint_name', type=str, default="FT_Epoch_latest.pt", help='Checkpoint Name')
    parser.add_argument("--GPU", type=int, default=0)
    parser.add_argument("--mps", type=str2bool, default=False, help="if use mps acceleration, set true")
    parser.add_argument("--test_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--input_size", type=str, default="scale_256", help="resize_256|full_size|scale_256")
    config = parser.parse_args()

    main(config)
