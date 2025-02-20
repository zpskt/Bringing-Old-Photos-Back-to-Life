# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import skimage.io as io

# from face_sdk import FaceDetection
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from PIL import Image, ImageFilter
import torch.nn.functional as F
import torchvision as tv
import torchvision.utils as vutils
import time
import cv2
import os
from skimage import img_as_ubyte
import json
import argparse
import dlib

'''
该函数用于计算累积分布函数（CDF），并将其归一化。具体步骤如下：
计算直方图的累积和，得到累积分布函数（CDF）。
将CDF除以其最大值，实现归一化。
返回归一化的CDF。
'''
def calculate_cdf(histogram):
    """
    此方法计算累积分布函数
    :p aram array histogram：直方图的值
    ：return： normalized_cdf： 归一化累积分布函数
    ：rtype： 数组
    """
    # 获取元素的累积和
    cdf = histogram.cumsum()

    # 规范化 cdf
    normalized_cdf = cdf / float(cdf.max())

    return normalized_cdf

'''
此函数用于创建查找表，将源图像的累积分布函数（CDF）映射到参考图像的CDF。具体步骤如下：
初始化一个长度为256的查找表和一个查找值。
遍历源图像的CDF，对于每个像素值，在参考图像的CDF中找到最匹配的像素值。
将匹配的像素值存入查找表对应位置。
返回构建完成的查找表。
'''
def calculate_lookup(src_cdf, ref_cdf):
    """
    此方法创建查找表
    :param array src_cdf: The cdf for the source image
    :param array ref_cdf: The cdf for the reference image
    :return: lookup_table: The lookup table
    :rtype: array
    """
    # 初始化查找表，长度为256，用于后续的像素值映射
    lookup_table = np.zeros(256)
    # 初始化查找值，用于记录参考图像像素值
    lookup_val = 0

    # 遍历源图像的累积分布函数（CDF），为每个像素值找到最匹配的参考图像像素值
    for src_pixel_val in range(len(src_cdf)):
        # 对每个源像素值，遍历参考图像的累积分布函数
        for ref_pixel_val in range(len(ref_cdf)):
            # 当参考图像的累积分布函数值大于等于源图像的相应值时
            if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                # 将当前的参考像素值赋给查找值
                lookup_val = ref_pixel_val
                # 找到匹配值后，跳出循环，继续处理下一个源像素值
                break
        # 将查找值赋给查找表中对应源像素值的位置，完成该像素值的映射
        lookup_table[src_pixel_val] = lookup_val

    # 返回构建完成的查找表
    return lookup_table

'''
此方法将源图像的直方图与参考图像的直方图进行匹配，具体步骤如下：
将源图像和参考图像拆分为蓝、绿、红三个颜色通道。
分别计算每个颜色通道的直方图。
计算每个颜色通道的累积分布函数（CDF）并归一化。
为每个颜色通道创建查找表，使源图像的CDF与参考图像的CDF匹配。
使用查找表转换源图像的颜色值。
合并转换后的颜色通道，生成最终匹配后的图像。
'''
def match_histograms(src_image, ref_image):
    """
    此方法将源图像直方图与参考信号
    :param image src_image: 原始源图像
    :param image  ref_image: 参考图像
    :return: image_after_matching
    :rtype: image (array)
    """
    # 将图像分割成不同的颜色通道
    # B 表示蓝色，G 表示绿色，R 表示红色
    src_b, src_g, src_r = cv2.split(src_image)
    ref_b, ref_g, ref_r = cv2.split(ref_image)

    # 分别计算 b、g 和 r 直方图
    # flatten（） Numpy 方法返回折叠为一维的数组的副本。
    src_hist_blue, bin_0 = np.histogram(src_b.flatten(), 256, [0, 256])
    src_hist_green, bin_1 = np.histogram(src_g.flatten(), 256, [0, 256])
    src_hist_red, bin_2 = np.histogram(src_r.flatten(), 256, [0, 256])
    ref_hist_blue, bin_3 = np.histogram(ref_b.flatten(), 256, [0, 256])
    ref_hist_green, bin_4 = np.histogram(ref_g.flatten(), 256, [0, 256])
    ref_hist_red, bin_5 = np.histogram(ref_r.flatten(), 256, [0, 256])

    # 计算源图像和参考图像的归一化 cdf
    src_cdf_blue = calculate_cdf(src_hist_blue)
    src_cdf_green = calculate_cdf(src_hist_green)
    src_cdf_red = calculate_cdf(src_hist_red)
    ref_cdf_blue = calculate_cdf(ref_hist_blue)
    ref_cdf_green = calculate_cdf(ref_hist_green)
    ref_cdf_red = calculate_cdf(ref_hist_red)

    # 为每种颜色制作一个单独的查找表
    blue_lookup_table = calculate_lookup(src_cdf_blue, ref_cdf_blue)
    green_lookup_table = calculate_lookup(src_cdf_green, ref_cdf_green)
    red_lookup_table = calculate_lookup(src_cdf_red, ref_cdf_red)

    # 使用 lookup 函数转换原始
    # 源图像
    blue_after_transform = cv2.LUT(src_b, blue_lookup_table)
    green_after_transform = cv2.LUT(src_g, green_lookup_table)
    red_after_transform = cv2.LUT(src_r, red_lookup_table)

    # 将经过直方图匹配后的蓝色、绿色和红色通道重新组合成一个完整的图像
    image_after_matching = cv2.merge([blue_after_transform, green_after_transform, red_after_transform])
    # 将图像数据转换为8位无符号整数格式（0到255）
    image_after_matching = cv2.convertScaleAbs(image_after_matching)

    return image_after_matching

'''
生成标准化的人脸特征点坐标，做归一化处理，即将坐标映射到一个特定的范围（例如，[-1, 1]）
'''
def _standard_face_pts():
    pts = (
        np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32) / 256.0
        - 1.0
    )

    return np.reshape(pts, (5, 2))

'''
定义了一个包含10个浮点数的数组，并将其重塑为5行2列的二维数组，表示5个面部特征点的坐标。
'''
def _origin_face_pts():
    pts = np.array([196.0, 226.0, 316.0, 226.0, 256.0, 286.0, 220.0, 360.4, 292.0, 360.4], np.float32)

    return np.reshape(pts, (5, 2))

'''
该函数用于计算图像的仿射变换矩阵。具体步骤如下：
获取标准面部特征点，并根据目标缩放因子调整这些点的位置。
获取图像尺寸，如果需要归一化，则对输入的landmark进行归一化处理。
创建一个相似变换对象，并估计从目标特征点到输入特征点的变换矩阵。
返回计算好的变换矩阵。
'''
def compute_transformation_matrix(img, landmark, normalize, target_face_scale=1.0):
    # 生成归一化的标准人脸特征点
    std_pts = _standard_face_pts()  # [-1,1]
    # 将标准人脸特征点缩放到目标尺度并转换到图像坐标系中 加1 除以2以后就转换到了[0,1],再乘256，就直接到了图片的坐标系中
    target_pts = (std_pts * target_face_scale + 1) / 2 * 256.0

    # print(target_pts)

    h, w, c = img.shape
    # 如果需要归一化，那么执行操作
    if normalize == True:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    # print(landmark)

    affine = SimilarityTransform()

    affine.estimate(target_pts, landmark)

    return affine

'''
该函数用于计算图像的逆变换矩阵。主要步骤如下：
获取标准面部特征点并缩放到目标尺度。
根据图像尺寸和是否归一化，调整输入的面部特征点。
使用相似变换估计从输入特征点到目标特征点的变换矩阵。
'''
def compute_inverse_transformation_matrix(img, landmark, normalize, target_face_scale=1.0):

    std_pts = _standard_face_pts()  # [-1,1]
    target_pts = (std_pts * target_face_scale + 1) / 2 * 256.0

    # print(target_pts)

    h, w, c = img.shape
    if normalize == True:
        landmark[:, 0] = landmark[:, 0] / h * 2 - 1.0
        landmark[:, 1] = landmark[:, 1] / w * 2 - 1.0

    # print(landmark)

    affine = SimilarityTransform()

    affine.estimate(landmark, target_pts)

    return affine


def show_detection(image, box, landmark):
    plt.imshow(image)
    print(box[2] - box[0])
    plt.gca().add_patch(
        Rectangle(
            (box[1], box[0]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none"
        )
    )
    plt.scatter(landmark[0][0], landmark[0][1])
    plt.scatter(landmark[1][0], landmark[1][1])
    plt.scatter(landmark[2][0], landmark[2][1])
    plt.scatter(landmark[3][0], landmark[3][1])
    plt.scatter(landmark[4][0], landmark[4][1])
    plt.show()

'''
该函数用于在图像上显示检测结果。具体功能如下：
显示图像。
打印框的宽度。
在图像上绘制矩形框，表示检测到的目标位置。
在图像上绘制5个特征点。
'''
def affine2theta(affine, input_w, input_h, target_w, target_h):
    # param = np.linalg.inv(affine)
    param = affine
    theta = np.zeros([2, 3])
    theta[0, 0] = param[0, 0] * input_h / target_h
    theta[0, 1] = param[0, 1] * input_w / target_h
    theta[0, 2] = (2 * param[0, 2] + param[0, 0] * input_h + param[0, 1] * input_w) / target_h - 1
    theta[1, 0] = param[1, 0] * input_h / target_w
    theta[1, 1] = param[1, 1] * input_w / target_w
    theta[1, 2] = (2 * param[1, 2] + param[1, 0] * input_h + param[1, 1] * input_w) / target_w - 1
    return theta

'''
该函数 blur_blending 实现了两张图像的融合，并通过模糊掩码进行平滑过渡。具体步骤如下：
将掩码缩放到255并进行腐蚀操作，减少边缘影响。
将掩码和图像转换为PIL格式。
对掩码进行高斯模糊处理。
使用原始掩码和模糊后的掩码分别进行图像合成，最终返回融合后的图像。
'''
def blur_blending(im1, im2, mask):

    mask *= 255.0

    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    mask = Image.fromarray(mask.astype("uint8")).convert("L")
    im1 = Image.fromarray(im1.astype("uint8"))
    im2 = Image.fromarray(im2.astype("uint8"))

    mask_blur = mask.filter(ImageFilter.GaussianBlur(20))
    im = Image.composite(im1, im2, mask)

    im = Image.composite(im, im2, mask_blur)

    return np.array(im) / 255.0


def blur_blending_cv2(im1, im2, mask):

    # mask *= 255.0
    # 将mask转换为0-255的灰度图像
    mask = mask * 255.0
    # 创建一个9x9的全一数组作为卷积核，用于后续的腐蚀操作
    kernel = np.ones((9, 9), np.uint8)
    # 对mask进行腐蚀操作，去除小的噪声点，iterations=3表示进行三次腐蚀
    mask = cv2.erode(mask, kernel, iterations=3)

    # 对mask进行高斯模糊，使用25x25的核，标准偏差为0，这有助于平滑mask的边缘
    mask_blur = cv2.GaussianBlur(mask, (25, 25), 0)
    # 将mask_blur归一化到0-1之间，以便于后续的图像融合
    mask_blur /= 255.0

    # 根据mask_blur融合两张图像，mask_blur中的高值区域保留im1的像素，低值区域保留im2的像素
    im = im1 * mask_blur + (1 - mask_blur) * im2

    # 将融合后的图像归一化到0-1之间，以便于后续的图像处理
    im /= 255.0
    # 对图像进行裁剪，确保所有像素值都在0.0到1.0的范围内，避免像素值溢出
    im = np.clip(im, 0.0, 1.0)

    # 返回融合处理后的图像
    return im

def Poisson_blending(im1, im2, mask):

    # mask=1-mask
    mask *= 255
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask /= 255
    mask = 1 - mask
    mask *= 255

    mask = mask[:, :, 0]
    width, height, channels = im1.shape
    center = (int(height / 2), int(width / 2))
    result = cv2.seamlessClone(
        im2.astype("uint8"), im1.astype("uint8"), mask.astype("uint8"), center, cv2.MIXED_CLONE
    )

    return result / 255.0


def Poisson_B(im1, im2, mask, center):

    mask *= 255

    result = cv2.seamlessClone(
        im2.astype("uint8"), im1.astype("uint8"), mask.astype("uint8"), center, cv2.NORMAL_CLONE
    )

    return result / 255

'''
该函数 seamless_clone 实现了将新脸无缝融合到旧脸上。具体步骤如下：
获取旧脸的尺寸并缩小一半。
根据掩码确定新脸和掩码的裁剪区域，并计算中心点。
将新脸和掩码转换为合适的格式。
对旧脸进行填充，生成先验图像。
处理掩码，获取边界矩形。
如果边界矩形太小，直接返回先验图像；否则使用 OpenCV 的无缝克隆算法融合图像。
返回融合后的图像。
'''
def seamless_clone(old_face, new_face, raw_mask):
    # 获取旧人脸图像的形状信息，包括高度、宽度和通道数
    height, width, _ = old_face.shape
    # 将旧人脸图像的高度和宽度各减半，可能用于后续处理中降低分辨率或裁剪图像
    height = height // 2
    width = width // 2

    # 使用np.nonzero获取非零元素的索引，这些索引表示在raw_mask中 nonzero elements 的位置
    y_indices, x_indices, _ = np.nonzero(raw_mask)
    # 计算y轴方向的最小和最大索引，用于确定裁剪区域
    y_crop = slice(np.min(y_indices), np.max(y_indices))
    # 计算x轴方向的最小和最大索引，用于确定裁剪区域
    x_crop = slice(np.min(x_indices), np.max(x_indices))

    # 计算y轴方向的中心点索引，考虑到高度补偿
    y_center = int(np.rint((np.max(y_indices) + np.min(y_indices)) / 2 + height))
    # 计算x轴方向的中心点索引，考虑到宽度补偿
    x_center = int(np.rint((np.max(x_indices) + np.min(x_indices)) / 2 + width))

    # 将新面孔图像的裁剪区域转换为适合插入的格式
    insertion = np.rint(new_face[y_crop, x_crop] * 255.0).astype("uint8")
    # 将原始掩码的裁剪区域转换为适合插入的格式，并强化非零值以清晰地区分前景和背景
    insertion_mask = np.rint(raw_mask[y_crop, x_crop] * 255.0).astype("uint8")
    # 将新面孔图像的掩码中非零值设置为25
    insertion_mask[insertion_mask != 0] = 255
    # 对旧面孔图像进行填充，以适应可能的尺寸变化，并确保类型适合进一步处理
    prior = np.rint(np.pad(old_face * 255.0, ((height, height), (width, width), (0, 0)), "constant")).astype(
        "uint8"
    )

    # if np.sum(insertion_mask) == 0:
    n_mask = insertion_mask[1:-1, 1:-1, :]
    n_mask = cv2.copyMakeBorder(n_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    print(n_mask.shape)
    x, y, w, h = cv2.boundingRect(n_mask[:, :, 0])
    if w < 4 or h < 4:
        blended = prior
    else:
        blended = cv2.seamlessClone(
            insertion,  # pylint: disable=no-member
            prior,
            insertion_mask,
            (x_center, y_center),
            cv2.NORMAL_CLONE,
        )  # pylint: disable=no-member

    blended = blended[height:-height, width:-width]

    return blended.astype("float32") / 255.0

'''
该函数用于从面部特征点中获取指定ID的特征点坐标。具体步骤如下：
根据ID从face_landmarks中获取对应的part对象。
从part对象中提取x和y坐标。
返回一个包含x和y坐标的元组。
'''
def get_landmark(face_landmarks, id):
    part = face_landmarks.part(id)
    if hasattr(part, 'img'):
        x = part.img
    else:
        x = part.x
    y = part.y

    return (x, y)


def search(face_landmarks):

    x1, y1 = get_landmark(face_landmarks, 36)
    x2, y2 = get_landmark(face_landmarks, 39)
    x3, y3 = get_landmark(face_landmarks, 42)
    x4, y4 = get_landmark(face_landmarks, 45)

    # 获取鼻子的坐标
    x_nose, y_nose = get_landmark(face_landmarks, 30)

    # 获取左嘴角的坐标
    x_left_mouth, y_left_mouth = get_landmark(face_landmarks, 48)
    # 获取右嘴角的坐标
    x_right_mouth, y_right_mouth = get_landmark(face_landmarks, 54)

    x_left_eye = int((x1 + x2) / 2)
    y_left_eye = int((y1 + y2) / 2)
    x_right_eye = int((x3 + x4) / 2)
    y_right_eye = int((y3 + y4) / 2)

    results = np.array(
        [
            [x_left_eye, y_left_eye],
            [x_right_eye, y_right_eye],
            [x_nose, y_nose],
            [x_left_mouth, y_left_mouth],
            [x_right_mouth, y_right_mouth],
        ]
    )

    return results

'''
人脸检测阶段
'''
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_url", type=str, default="./", help="origin images")
    parser.add_argument("--replace_url", type=str, default="./", help="restored faces")
    parser.add_argument("--save_url", type=str, default="./save")
    opts = parser.parse_args()

    origin_url = opts.origin_url
    replace_url = opts.replace_url
    save_url = opts.save_url

    if not os.path.exists(save_url):
        os.makedirs(save_url)

    # 使用 dlib 的内置人脸检测功能初始化人脸检测器
    face_detector = dlib.get_frontal_face_detector()

    # 使用预先训练的模型初始化面部特征定位器
    landmark_locator = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    count = 0

    for img in os.listdir(origin_url):
        img_url = os.path.join(origin_url, img)
        pil_img = Image.open(img_url).convert("RGB")

        origin_width, origin_height = pil_img.size
        image = np.array(pil_img)

        # 记录开始时间，用于计算图像检测的处理时间
        start = time.time()
        # 使用face_detector对图像进行人脸检测，结果保存在faces变量中
        faces = face_detector(image)
        # 记录完成时间，用于计算图像检测的处理时间
        done = time.time()

        if len(faces) == 0:
            print("警告：没有检测到人脸 %s" % (img))
            continue

        # 将图像数据赋值给变量blended，以进行后续的图像处理或融合操作
        blended = image
        for face_id in range(len(faces)):

            # 根据face_id获取当前人脸的特征
            current_face = faces[face_id]
            # 使用landmark_locator函数检测图像中当前人脸的地标
            face_landmarks = landmark_locator(image, current_face)
            # 通过search函数处理人脸地标，以获取进一步的面部特征
            current_fl = search(face_landmarks)
            # 初始化一个与图像大小相同，值为1的掩码，用于后续处理
            forward_mask = np.ones_like(image).astype("uint8")
            # 计算转换矩阵，用于将人脸对齐到目标位置和大小
            affine = compute_transformation_matrix(image, current_fl, False, target_face_scale=1.3)
            # 应用转换矩阵，将图像中的当前人脸对齐到目标位置和大小
            aligned_face = warp(image, affine, output_shape=(256, 256, 3), preserve_range=True)
            # 对掩码进行相同的变换，以匹配对齐后的人脸
            forward_mask = warp(
                forward_mask, affine, output_shape=(256, 256, 3), order=0, preserve_range=True
            )

            # 获取当前人脸对齐的逆矩阵，用于后续可能的逆操作
            affine_inverse = affine.inverse
            # 将当前处理的对齐后的人脸赋值给cur_face，以便进行进一步的处理
            cur_face = aligned_face
            # 检查是否存在需要替换的URL，以决定是否执行替换操作
            if replace_url != "":
                face_name = img[:-4] + "_" + str(face_id + 1) + ".png"
                cur_url = os.path.join(replace_url, face_name)
                restored_face = Image.open(cur_url).convert("RGB")
                restored_face = np.array(restored_face)
                cur_face = restored_face

            ## 直方图颜色匹配
            # 将对齐后的人脸图像从RGB颜色空间转换到BGR颜色空间
            A = cv2.cvtColor(aligned_face.astype("uint8"), cv2.COLOR_RGB2BGR)
            # 将当前人脸图像从RGB颜色空间转换到BGR颜色空间
            B = cv2.cvtColor(cur_face.astype("uint8"), cv2.COLOR_RGB2BGR)

            # 调整当前人脸图像的直方图，使其与对齐后的人脸图像直方图匹配
            B = match_histograms(B, A)

            # 将处理后的当前人脸图像转换回RGB颜色空间
            cur_face = cv2.cvtColor(B.astype("uint8"), cv2.COLOR_BGR2RGB)

            # 将当前人脸图像变形回到原始位置
            warped_back = warp(
                cur_face,
                affine_inverse,
                output_shape=(origin_height, origin_width, 3),
                order=3,
                preserve_range=True,
            )

            # 将前向变形的掩码变形回到原始位置，使用最近邻插值
            backward_mask = warp(
                forward_mask,
                affine_inverse,
                output_shape=(origin_height, origin_width, 3),
                order=0,
                preserve_range=True,
            )  ## Nearest neighbour

            # 使用模糊融合方法将变形后的当前人脸图像与已融合的图像进行融合
            blended = blur_blending_cv2(warped_back, blended, backward_mask)
            # 将融合后的图像恢复到0-255的范围
            blended *= 255.0

            # 保存处理后的图像
            io.imsave(os.path.join(save_url, img), img_as_ubyte(blended / 255.0))

            # 计数处理的图像数量
            count += 1

            # 每处理1000张图像，打印一次进度信息
            if count % 1000 == 0:
                print("%d 已完成 ..." % (count))

