# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import shutil
import sys
from subprocess import call

def run_cmd(command):
    try:
        call(command, shell=True)
    except KeyboardInterrupt:
        print("Process interrupted")
        sys.exit(1)

if __name__ == "__main__":
    # 读取可选参数输入
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="./test_images/old", help="Test images")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Restored images, please use the absolute path",
    )
    parser.add_argument("--GPU", type=str, default="6,7", help="0,1,2")
    parser.add_argument("--mps", type=bool, default=False, help="if use M chip, set true")
    parser.add_argument(
        "--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint"
    )
    parser.add_argument("--with_scratch", action="store_true")
    parser.add_argument("--HR", action='store_true')
    opts = parser.parse_args()

    gpu1 = opts.GPU

    # 在更改目录之前解析相对路径
    opts.input_folder = os.path.abspath(opts.input_folder)
    opts.output_folder = os.path.abspath(opts.output_folder)
    if not os.path.exists(opts.output_folder):
        os.makedirs(opts.output_folder)
    # 获取当前工作目录路径
    main_environment = os.getcwd()

    ## 第一阶段：整体质量提升
    print("运行阶段 1: 整体修复")
    # 切换到工作目录Global
    os.chdir("./Global")
    stage_1_input_dir = opts.input_folder
    # 设定阶段1 输出路径
    stage_1_output_dir = os.path.join(opts.output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    if not opts.with_scratch:
        # 如果参数中没有带划痕参数，那么执行 test脚本进行图像质量恢复
        stage_1_command = (
            "python test.py --test_mode Full --Quality_restore --test_input "
            + stage_1_input_dir
            + " --outputs_dir "
            + stage_1_output_dir
            + " --gpu_ids "
            + gpu1
            + " --mps "
            + str(opts.mps)
        )
        run_cmd(stage_1_command)
    else:
        # 如果参数中有带划痕参数，那么生成mask并保存
        #
        mask_dir = os.path.join(stage_1_output_dir, "masks")
        new_input = os.path.join(mask_dir, "input")
        new_mask = os.path.join(mask_dir, "mask")
        stage_1_command_1 = (
            "python detection.py --test_path "
            + stage_1_input_dir
            + " --output_dir "
            + mask_dir
            + " --input_size full_size"
            + " --GPU "
            + gpu1
            + " --mps "
            + str(opts.mps)
        )
        # 判断是否启用高分辨率模式
        if opts.HR:
            HR_suffix=" --HR"
        else:
            HR_suffix=""
        # 执行test脚本进行划痕质量修复
        stage_1_command_2 = (
            "python test.py --Scratch_and_Quality_restore --test_input "
            + new_input
            + " --test_mask "
            + new_mask
            + " --outputs_dir "
            + stage_1_output_dir
            + " --gpu_ids "
            + gpu1 + HR_suffix
            + " --mps "
            + str(opts.mps)
        )
        # 构建完命令以后，分别执行
        run_cmd(stage_1_command_1)
        run_cmd(stage_1_command_2)

    ## 解决旧照片中没有人脸的情况
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    for x in os.listdir(stage_1_results):
        img_dir = os.path.join(stage_1_results, x)
        shutil.copy(img_dir, stage_4_output_dir)

    print("完成第 1 阶段 ...")
    print("\n")

    ## 第 2 阶段：人脸检测

    print("运行阶段 2：人脸检测")
    os.chdir(".././Face_Detection")
    # 将第一阶段的输出路径作为2阶段的输入路径
    stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_2_output_dir = os.path.join(opts.output_folder, "stage_2_detection_output")
    if not os.path.exists(stage_2_output_dir):
        os.makedirs(stage_2_output_dir)
    if opts.HR:
        stage_2_command = (
            "python detect_all_dlib_HR.py --url " + stage_2_input_dir + " --save_url " + stage_2_output_dir
        )
    else:
        stage_2_command = (
            "python detect_all_dlib.py --url " + stage_2_input_dir + " --save_url " + stage_2_output_dir
        )
    run_cmd(stage_2_command)
    print("完成第 2 阶段 ...")
    print("\n")

    ## 第 3 阶段：面部修复
    print("运行阶段 3：面部增强")
    os.chdir(".././Face_Enhancement")
    stage_3_input_mask = "./"
    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")
    if not os.path.exists(stage_3_output_dir):
        os.makedirs(stage_3_output_dir)
    # 根据质量参数决定模型使用什么参数
    if opts.HR:
        opts.checkpoint_name='FaceSR_512'
        stage_3_command = (
            "python test_face.py --old_face_folder "
            + stage_3_input_face
            + " --old_face_label_folder "
            + stage_3_input_mask
            + " --tensorboard_log --name "
            + opts.checkpoint_name
            + " --gpu_ids "
            + gpu1
            + " --mps "
            + str(opts.mps)
            + " --load_size 512 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 1 --results_dir "
            + stage_3_output_dir
            + " --no_parsing_map"
        ) 
    else:
        stage_3_command = (
            "python test_face.py --old_face_folder "
            + stage_3_input_face
            + " --old_face_label_folder "
            + stage_3_input_mask
            + " --tensorboard_log --name "
            + opts.checkpoint_name
            + " --gpu_ids "
            + gpu1
            + " --mps "
            + str(opts.mps)
            + " --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 --results_dir "
            + stage_3_output_dir
            + " --no_parsing_map"
        )
    run_cmd(stage_3_command)
    print("完成第 3 阶段 ...")
    print("\n")

    ## 第 4 阶段：替换人脸
    print("运行阶段 4：混合图像")
    os.chdir(".././Face_Detection")
    # 将第一阶段的输出和第三阶段的输出结合起来
    stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    if opts.HR:
        stage_4_command = (
            "python align_warp_back_multiple_dlib_HR.py --origin_url "
            + stage_4_input_image_dir
            + " --replace_url "
            + stage_4_input_face_dir
            + " --save_url "
            + stage_4_output_dir
        )
    else:
        stage_4_command = (
            "python align_warp_back_multiple_dlib.py --origin_url "
            + stage_4_input_image_dir
            + " --replace_url "
            + stage_4_input_face_dir
            + " --save_url "
            + stage_4_output_dir
        )
    run_cmd(stage_4_command)
    print("完成第 4 阶段 ...")
    print("\n")

    print("所有处理都已完成。请检查结果.")

