# Old Photo Restoration (Official PyTorch Implementation)

<img src='imgs/0001.jpg'/>

### [Project Page](http://raywzy.com/Old_Photo/) | [Paper (CVPR version)](https://arxiv.org/abs/2004.09484) | [Paper (Journal version)](https://arxiv.org/pdf/2009.07047v1.pdf) | [Pretrained Model](https://hkustconnect-my.sharepoint.com/:f:/g/personal/bzhangai_connect_ust_hk/Em0KnYOeSSxFtp4g_dhWdf0BdeT3tY12jIYJ6qvSf300cA?e=nXkJH2) | [Colab Demo](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing)  | [Replicate Demo & Docker Image](https://replicate.ai/zhangmozhe/bringing-old-photos-back-to-life) :fire:

**Bringing Old Photos Back to Life, CVPR2020 (Oral)**

**Old Photo Restoration via Deep Latent Space Translation, TPAMI 2022**

[Ziyu Wan](http://raywzy.com/)<sup>1</sup>,
[Bo Zhang](https://www.microsoft.com/en-us/research/people/zhanbo/)<sup>2</sup>,
[Dongdong Chen](http://www.dongdongchen.bid/)<sup>3</sup>,
[Pan Zhang](https://panzhang0212.github.io/)<sup>4</sup>,
[Dong Chen](https://www.microsoft.com/en-us/research/people/doch/)<sup>2</sup>,
[Jing Liao](https://liaojing.github.io/html/)<sup>1</sup>,
[Fang Wen](https://www.microsoft.com/en-us/research/people/fangwen/)<sup>2</sup> <br>
<sup>1</sup>City University of Hong Kong, <sup>2</sup>Microsoft Research Asia, <sup>3</sup>Microsoft Cloud AI, <sup>4</sup>USTC

<!-- ## Notes of this project
The code originates from our research project and the aim is to demonstrate the research idea, so we have not optimized it from a product perspective. And we will spend time to address some common issues, such as out of memory issue, limited resolution, but will not involve too much in engineering problems, such as speedup of the inference, fastapi deployment and so on. **We welcome volunteers to contribute to this project to make it more usable for practical application.** -->

## :sparkles: News
**2022.3.31**: Our new work regarding old film restoration will be published in CVPR 2022. For more details, please refer to the [project website](http://raywzy.com/Old_Film/) and [github repo](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life).

The framework now supports the restoration of high-resolution input.

<img src='imgs/HR_result.png'>

Training code is available and welcome to have a try and learn the training details. 

You can now play with our [Colab](https://colab.research.google.com/drive/1NEm6AsybIiC5TwTU_4DqDkQO0nFRB-uA?usp=sharing) and try it on your photos. 

## Requirement
该代码在安装了 Nvidia GPU 和 CUDA 的 Ubuntu 上进行了测试。需要 Python>=3.6 才能运行代码。
## 安装教程
windows安装注意：对于下载包以后得cp和解压操作，一定要确认你的项目目录层级然后再放，不然会显示找不到。
Clone the Synchronized-BatchNorm-PyTorch repository for
切到指定目录下，然后clone项目,递归将文件放到当前目录下
```
cd Face_Enhancement/models/networks/
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../../
```

```
cd Global/detection_models
git clone https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
cp -rf Synchronized-BatchNorm-PyTorch/sync_batchnorm .
cd ../../
```

下载特征点检测预训练模型
```
cd Face_Detection/
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ../
```

下载预训练模型，将文件 'Face_Enhancement/checkpoints.zip' 放在 './Face_Enhancement' 下，
将文件 'Global/checkpoints.zip' 放在 './Global' 下。然后分别解压缩它们。
```
cd Face_Enhancement/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/face_checkpoints.zip
unzip face_checkpoints.zip
cd ../
cd Global/
wget https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip
unzip global_checkpoints.zip
cd ../
```

安装依赖项：
```bash
pip install -r requirements.txt
```

## :rocket: How to use?

**Note**: GPU can be set 0 or 0,1,2 or 0,2; use -1 for CPU

### 1) Full Pipeline

安装并下载预训练模型后，您可以使用一个简单的命令轻松恢复旧照片。

对于没有划痕的图像：

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0
```
```shell
 python run.py --input_folder .\test_images\qiang --output_folder .\output\ --GPU -1
```
对于有划痕的图像：
```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0 \
              --with_scratch
```

```shell
 python run.py --input_folder .\test_images\person2 --output_folder .\output\ --GPU -1 --with_scratch --HR
```
**对于有划痕的高分辨率图像**:

```
python run.py --input_folder [test_image_folder_path] \
              --output_folder [output_path] \
              --GPU 0 \
              --with_scratch \
              --HR
```

注意：请尝试使用绝对路径。最终结果将保存在 './output_path/final_output/' 中。您还可以在 'output_path' 中检查不同步骤的生成结果。
### 2) 划痕检测

目前，我们不打算直接发布带有标签的 scratched old photos 数据集。如果您想获取配对数据，您可以使用我们的预训练模型来测试收集的图像以获取标签。
```
cd Global/
python detection.py --test_path [test_image_folder_path] \
                    --output_dir [output_path] \
                    --input_size [resize_256|full_size|scale_256]
```

<img src='imgs/scratch_detection.png'>

### 3) Global Restoration

提出了一种三重态域翻译网络来解决旧照片的结构化降级和非结构化降级问题。
<p align="center">
<img src='imgs/pipeline.PNG' width="50%" height="50%"/>
</p>

```
cd Global/
python test.py --Scratch_and_Quality_restore \
               --test_input [test_image_folder_path] \
               --test_mask [corresponding mask] \
               --outputs_dir [output_path]

python test.py --Quality_restore \
               --test_input [test_image_folder_path] \
               --outputs_dir [output_path]
```

<img src='imgs/global.png'>


### 4) 面部增强

我们使用渐进式生成器来优化旧照片的面部区域。更多详细信息可以在我们的期刊提交和 './Face_Enhancement' 文件夹中找到。
<p align="center">
<img src='imgs/face_pipeline.jpg' width="60%" height="60%"/>
</p>


<img src='imgs/face.png'>

> *NOTE*: 
> This repo is mainly for research purpose and we have not yet optimized the running performance. 
> 
> Since the model is pretrained with 256*256 images, the model may not work ideally for arbitrary resolution.

### 5) GUI

用户友好的 GUI，可按用户输入图像并在相应的窗口中显示结果。
#### How it works:

1. 运行 GUI.py 文件.
2. 单击浏览并从 test_images/old_w_scratch 文件夹中选择您的图像以去除划痕.
3. 点击 修改照片 按钮。
4. 稍等片刻，然后在 GUI 窗口上查看结果。
5. 通过单击退出窗口退出窗口并在输出文件夹中获取结果图像。

<img src='imgs/gui.PNG'>

## How to train?

### 1) Create Training File

Put the folders of VOC dataset, collected old photos (e.g., Real_L_old and Real_RGB_old) into one shared folder. Then
```
cd Global/data/
python Create_Bigfile.py
```
Note: Remember to modify the code based on your own environment.

### 2) Train the VAEs of domain A and domain B respectively

```
cd ..
python train_domain_A.py --use_v2_degradation --continue_train --training_dataset domain_A --name domainA_SR_old_photos --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 100 --no_html --gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder]

python train_domain_B.py --continue_train --training_dataset domain_B --name domainB_old_photos --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder]  --no_instance --resize_or_crop crop_only --batchSize 120 --no_html --gpu_ids 0,1,2,3 --self_gen --nThreads 4 --n_downsample_global 3 --k_size 4 --use_v2 --mc 64 --start_r 1 --kl 1 --no_cgan --outputs_dir [your_output_folder]  --checkpoints_dir [your_ckpt_folder]
```
注意：对于 --name 选项，请确保您的实验名称包含“domainA”或“domainB”，这将用于选择不同的数据集。
### 3) Train the mapping network between domains

训练无划痕的映射：```
python train_mapping.py --use_v2_degradation --training_dataset mapping --use_vae_which_epoch 200 --continue_train --name mapping_quality --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 80 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [ckpt_of_domainA_SR_old_photos] --load_pretrainB [ckpt_of_domainB_old_photos] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder]
```


使用 scrache 训练映射：```
python train_mapping.py --no_TTUR --NL_res --random_hole --use_SN --correlation_renormalize --training_dataset mapping --NL_use_mask --NL_fusion_method combine --non_local Setting_42 --use_v2_degradation --use_vae_which_epoch 200 --continue_train --name mapping_scratch --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 36 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [ckpt_of_domainA_SR_old_photos] --load_pretrainB [ckpt_of_domainB_old_photos] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder] --irregular_mask [absolute_path_of_mask_file]
```

使用 scraches 训练映射（用于 HR 输入的 Multi-Scale Patch Attention）：```
python train_mapping.py --no_TTUR --NL_res --random_hole --use_SN --correlation_renormalize --training_dataset mapping --NL_use_mask --NL_fusion_method combine --non_local Setting_42 --use_v2_degradation --use_vae_which_epoch 200 --continue_train --name mapping_Patch_Attention --label_nc 0 --loadSize 256 --fineSize 256 --dataroot [your_data_folder] --no_instance --resize_or_crop crop_only --batchSize 36 --no_html --gpu_ids 0,1,2,3 --nThreads 8 --load_pretrainA [ckpt_of_domainA_SR_old_photos] --load_pretrainB [ckpt_of_domainB_old_photos] --l2_feat 60 --n_downsample_global 3 --mc 64 --k_size 4 --start_r 1 --mapping_n_block 6 --map_mc 512 --use_l1_feat --niter 150 --niter_decay 100 --outputs_dir [your_output_folder] --checkpoints_dir [your_ckpt_folder] --irregular_mask [absolute_path_of_mask_file] --mapping_exp 1
```


## Citation

如果您发现我们的工作对您的研究有用，请考虑引用以下论文:)
```bibtex
@inproceedings{wan2020bringing,
title={Bringing Old Photos Back to Life},
author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={2747--2757},
year={2020}
}
```

```bibtex
@article{wan2020old,
  title={Old Photo Restoration via Deep Latent Space Translation},
  author={Wan, Ziyu and Zhang, Bo and Chen, Dongdong and Zhang, Pan and Chen, Dong and Liao, Jing and Wen, Fang},
  journal={arXiv preprint arXiv:2009.07047},
  year={2020}
}
```

If you are also interested in the legacy photo/video colorization, please refer to [this work](https://github.com/zhangmozhe/video-colorization).

## Maintenance

This project is currently maintained by Ziyu Wan and is for academic research use only. If you have any questions, feel free to contact raywzy@gmail.com.

## License

The codes and the pretrained model in this repository are under the MIT license as specified by the LICENSE file. We use our labeled dataset to train the scratch detection model.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
