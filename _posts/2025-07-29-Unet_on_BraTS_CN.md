---
layout:     post
title:      "在BraTS数据集上使用Unet进行分割"
subtitle:   "第一个Unet!"
date:       2025-07-30 00:00:01
author:     "董占彬"
header-img: "../img/Unet/seg_res.png"
catalog: true
tags:
    - Deep Learning
---

# 在BraTS数据集上使用Unet进行分割
## ABSTRACT
本项目基于经典的语义分割模型 UNet，使用 PyTorch 框架进行了**独立复现**，并在脑部肿瘤医学影像数据集 **BraTS** 上进行了训练和评估。考虑到脑部肿瘤多位于图像中心区域，本项目采用对整幅图像直接输入模型的方法，避免了边缘信息干扰和显存浪费。模型采用了 **Dice Loss 与 BCE Loss 的联合损失函数设计**，既提高了对少数前景区域的敏感性，又保证了像素级分类准确性。在训练过程中，模型的损失稳定下降，最终在测试集上达到了较高的 Dice 系数与 Sensitivity，验证了模型**良好的分割性能和泛化能力**。此外，本文还实现了可视化分割效果，对模型的预测能力进行了直观评估。

本项目已开源在：[Github](https://github.com/ZhB-Dong/Unet_on_BraTS)
## 1. Introduction
Unet 是一种流行的图像分割模型。它可以被视为一种分类模型。该网络不包含全连接层，因此分割图只包含有输入图像中所有有完整可用上下文的像素，这使得对像素的聚类只依靠其周围的语义，减小了更远位置的像素对语义的干扰[[1]](https://arxiv.org/abs/1505.04597)。
该网络结构如下：
<!-- ![Unet](../img/Unet/fig1-unet.png "Unet") -->
![Unet](https://github.com/ZhB-Dong/ZhB-Dong.github.io/raw/d85b9ebc9b7efda048dbcae77a0602726f25b2b1/img/_2025-7-29_unet.png "Unet")
本文将利用python-torch库对Unet模型进行**独立复现**，并在Brain Tumer Image Segmentation (BraTS) 数据集上进行训练和测试。将Dice和Sensitivity作为评价指标。

## 2. Dataset
本文选择了语义信息较明确的医疗场景的数据集BraTS，该数据集包含了如下信息

|Dataset|Image/Label number|
|---|---|
|Train set|1502|
|Test set|215|

使用[该博客](https://zhuanlan.zhihu.com/p/1895864478723186793)提供的数据集加载程序对数据进行加载以及对标注进行生成(`./Dataset/data.py`)。图像和标注实例如下
![Label](https://github.com/ZhB-Dong/ZhB-Dong.github.io/raw/d85b9ebc9b7efda048dbcae77a0602726f25b2b1/img/Unet/label.png "label")

## 3. Model
本项目**独立复现**了Ronneberger等人于2015年提出的一个经典的用于语义分割的模型Unet[[1]](https://arxiv.org/abs/1505.04597)。

该模型先通过max pooling对图像进行5次下采样，提升模型对各级结构特征的感知力。然后通过5次上采样并将结果与下采样前的图像拼接来提升模型对局部结构的敏感度。原文为了提升模型对边缘部分信息的感知力，使用了Overlap-tile对边缘填充，然后利用多次无padding的卷积降低图像分辨率至label的分辨率。**由于脑MRI不需要关注图像边缘部分（通常是噪声）**，为了保证图像大小一致，本项目的模型中的每次卷积均对图像进行0填充。模型在`./models/unet2.py`

## 4. Training & Evaluation
### 4.1 Data prepartion
在原文中作者为了让图像边缘的语义连续，使用Overlap-tile的策略对图像的边缘进行镜像填充，**但是对于脑MRI的数据集，图像主要分割的对象-脑部肿瘤通常位于图像的中央，边缘位置的语义重要性较低；填充后的图像通常占据更大的现存并导致更长的训练时间**。因此本文中对数据的预处理并未包含对图像进行Overlap-tile的填充。
数据的预处理部分主要包含对图像的**resize和normalization**。

Batch size设定为1，并且将完整的图像输入网络而不是将其分割成小patch，这因为脑肿瘤在大脑的分布比较离散，**如果使用较大Batch以及较小的patch，网络的学习效率可能会下降，还有可能造成对噪声的过拟合。**

在原文中，为了提升网络对边缘信息的敏感性，提升了分割结果边缘的权重。对于本文面对的问题：脑部肿瘤分割，该对象在脑中分布比较离散（多数label仅有1片联通的mask），**边缘权重并不会显著影响模型的分割效果，因此并未仿照论文对边缘权重进行增强。**

### 4.2 激活函数与损失函数设计
在分割任务中，BCE + Dice Loss 的组合能够同时兼顾像素级准确性与整体区域预测质量。BCE（Binary Cross Entropy）关注每个像素的分类正确性，有助于模型优化细粒度边界，而 Dice Loss 强调预测区域与真实区域的重叠程度，能有效缓解类别不平衡问题（如前景区域远少于背景时）。将两者结合，能综合提升模型对小目标区域的敏感性和全局结构的准确度。

同时，使用 Sigmoid 激活函数是因为二分类分割任务的输出是每个像素属于前景（1）或背景（0）的概率，Sigmoid 将每个像素的输出映射到 [0, 1] 区间，适合作为概率解释，并与 BCE/Dice 等损失函数配合使用，形成稳定有效的训练目标。损失函数在`./utils/loss.py`

### 4.3 训练与评估方法
实验使用GPU RTX4090作为模型的训练和测试硬件平台。训练参数如下

|Paras|Values|
|---|---|
|Learning rate|1e-4|
|Batch size|1|
|Epoch|50|
|Image size|512*512|

训练部分代码在`./train.py`

将Dice和Sensitivity作为评价指标，Dice系数用来评估衡量预测与标签的重叠程度，可以表示为：

$$Dice = \frac{2 \times TP}{2 \times TP + FP + FN}$$

where TP（真正例）表示预测为正且真实为正的像素数, FP（假正例）表示预测为正但真实为负的像素数, FN（假反例）表示预测为负但真实为正的像素数。

Sensitivity用来评估衡量前景（Positive）被预测出来的比例，可以表示为：

$$Sensitivity = \frac{TP}{TP + FN}$$

### 4.4 训练结果与准确性评估
损失下降结果如下
![loss](https://github.com/ZhB-Dong/ZhB-Dong.github.io/raw/d85b9ebc9b7efda048dbcae77a0602726f25b2b1/img/Unet/loss.png "loss")
如图所示损失在n epoch后逐渐稳定，这表示模型在逐渐收敛。测试集loss在12epoch后上升趋势，这表明模型在此后可能出现过拟合。

模型在测试集上的Dice和Sensitivity指标随着模型训练的变化如下
![sens](https://github.com/ZhB-Dong/ZhB-Dong.github.io/raw/d85b9ebc9b7efda048dbcae77a0602726f25b2b1/img/Unet/testDiceSens.png "sens")
如图所示，在模型训练过程中，Dice和Sensitivity均体现了逐渐上升的趋势，但在12epoch后上升趋势逐渐减缓，这表明在当前学习率下模型已基本完成对label的学习。Dice和Sensitivity均接近0.8，这表明模型的分类性能较好：分类的预测结果即和标签重合度较高，同时前景被预测出的比例也较高。

在测试集上随机数据的分割效果展示如下
![Segmentation](https://github.com/ZhB-Dong/ZhB-Dong.github.io/raw/d85b9ebc9b7efda048dbcae77a0602726f25b2b1/img/Unet/seg_res.png "segmentation")

如图所示红色区域和绿色的区域的重合度较高，同时分割的区域基本覆盖了病变区域，这表明模型学习到了病变上下文的语义信息。在Subject A和B中的预测结果更少地覆盖了为病变但是被划分近标注中的大脑结构，这表明相对于大脑结构，模型更对病变区域更加敏感，因此模型可以更准确地区分大脑的正常结构和病变结构。

## 5. Discussion
本项目对Unet网络进行了独立复现。本项目首先利用torch库对Unet网络进行搭建，并在BraTS数据集上进行训练和评估。结果表明模型可以更准确地区分大脑的正常结构和病变结构，预测结果与标注结果具有较高重合度。

## 6. Future Work
- 为了进一步提升网络泛化性，需要对数据进行数据增强。在未来将通过模拟随机拉伸形变的方式对数据集进行扩充以提升模型的泛化性。
- 寻找可用的3DMRI影像数据集，在现有网络基础上进行改进实现3DUnet