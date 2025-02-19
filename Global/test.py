# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
from models.mapping_model import Pix2PixHDModel_Mapping
import util.util as util
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2
'''
该函数用于对图像进行变换，主要功能包括：
获取图像的原始尺寸。
根据 scale 参数调整图像尺寸，确保短边为 256 像素。
将调整后的尺寸四舍五入到最接近的 4 的倍数。
如果调整后的尺寸与原始尺寸相同，则直接返回原图；否则，返回调整后的图像。
'''
def data_transforms(img, method=Image.BILINEAR, scale=False):

    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)

'''
该函数用于处理图像，确保其尺寸至少为256x256。
如果图像的宽度或高度小于256，则将其缩放到256x256，最后进行中心裁剪至256x256。
'''
def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)

'''
该函数用于在图像上合成不规则的孔洞。具体步骤如下：
将输入的图像和掩码转换为NumPy数组，并将数据类型设置为uint8。
将掩码数组归一化到0-1之间。
根据掩码生成新的图像，保留非掩码区域的像素值，将掩码区域填充为白色（255）。
将处理后的数组转换回PIL图像并返回。
'''
def irregular_hole_synthesize(img, mask):

    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img

'''
该函数 parameter_set 用于设置模型训练和推理的参数
'''
def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    if opt.Quality_restore:
        opt.name = "mapping_quality"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = "combine"
        opt.non_local = "Setting_42"
        opt.name = "mapping_scratch"
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = "mapping_Patch_Attention"

'''
此方法是不带划痕的图片修复
'''
if __name__ == "__main__":

    # 读取参数
    opt = TestOptions().parse(save=False)
    # 设置参数
    parameter_set(opt)

    # 初始化模型
    model = Pix2PixHDModel_Mapping()
    model.initialize(opt)
    model.eval()

    # 创建输出目录
    if not os.path.exists(opt.outputs_dir + "/" + "input_image"):
        os.makedirs(opt.outputs_dir + "/" + "input_image")
    if not os.path.exists(opt.outputs_dir + "/" + "restored_image"):
        os.makedirs(opt.outputs_dir + "/" + "restored_image")
    if not os.path.exists(opt.outputs_dir + "/" + "origin"):
        os.makedirs(opt.outputs_dir + "/" + "origin")

    # 初始化数据集大小
    dataset_size = 0

    # 加载输入图像
    input_loader = os.listdir(opt.test_input)
    dataset_size = len(input_loader)
    input_loader.sort()

    # 如果提供了掩码路径，则加载掩码图像
    if opt.test_mask != "":
        mask_loader = os.listdir(opt.test_mask)
        dataset_size = len(os.listdir(opt.test_mask))
        mask_loader.sort()

    # 图像和掩码的预处理
    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mask_transform = transforms.ToTensor()

    # 处理每个输入图像
    for i in range(dataset_size):

        input_name = input_loader[i]
        input_file = os.path.join(opt.test_input, input_name)
        if not os.path.isfile(input_file):
            print("Skipping non-file %s" % input_name)
            continue
        input = Image.open(input_file).convert("RGB")

        print("Now you are processing %s" % (input_name))

        # 如果使用掩码
        if opt.NL_use_mask:
            mask_name = mask_loader[i]
            mask = Image.open(os.path.join(opt.test_mask, mask_name)).convert("RGB")
            if opt.mask_dilation != 0:
                kernel = np.ones((3,3),np.uint8)
                mask = np.array(mask)
                mask = cv2.dilate(mask,kernel,iterations = opt.mask_dilation)
                mask = Image.fromarray(mask.astype('uint8'))
            origin = input
            input = irregular_hole_synthesize(input, mask)
            mask = mask_transform(mask)
            mask = mask[:1, :, :]  ## Convert to single channel
            mask = mask.unsqueeze(0)
            input = img_transform(input)
            input = input.unsqueeze(0)
        else:
            # 根据不同的测试模式进行图像变换
            if opt.test_mode == "Scale":
                input = data_transforms(input, scale=True)
            if opt.test_mode == "Full":
                input = data_transforms(input, scale=False)
            if opt.test_mode == "Crop":
                input = data_transforms_rgb_old(input)
            origin = input
            input = img_transform(input)
            input = input.unsqueeze(0)
            mask = torch.zeros_like(input)
        ### Necessary input

        # 使用模型进行推理
        try:
            with torch.no_grad():
                generated = model.inference(input, mask)
        except Exception as ex:
            print("Skip %s due to an error:\n%s" % (input_name, str(ex)))
            continue

        # 保存结果图像
        if input_name.endswith(".jpg"):
            input_name = input_name[:-4] + ".png"

        image_grid = vutils.save_image(
            (input + 1.0) / 2.0,
            opt.outputs_dir + "/input_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )
        image_grid = vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            opt.outputs_dir + "/restored_image/" + input_name,
            nrow=1,
            padding=0,
            normalize=True,
        )

        origin.save(opt.outputs_dir + "/origin/" + input_name)
