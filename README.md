# attention-set
# 各种即插即用模块的实现

### 1. GAM 注意力

论文地址：https://arxiv.org/pdf/2112.05561v1.pdf

摘要：为了提高计算机视觉任务的性能，人们研究了各种注意力机制。然而，以往的方法忽略了保留通道和空间方面的信息以增强跨维度交互的重要性。因此，本文提出了一种通过减少信息弥散和放大全局交互表示来提高深度神经网络性能的全局注意力机制。本文引入了3D-permutation 与多层感知器的通道注意力和卷积空间注意力子模块。在CIFAR-100和ImageNet-1K上对所提出的图像分类机制的评估表明，本文的方法稳定地优于最近的几个注意力机制，包括ResNet和轻量级的MobileNet。

![GAM](img/GAM.jpg)



### 2、STN模块

论文地址：https://proceedings.neurips.cc/paper/2015/file/33ceb07bf4eeb3da587e268d663aba1a-Paper.pdf

摘要：卷积神经网络定义了一类非常强大的模型，但仍受限于缺乏以计算和参数效率的方式对输入数据进行空间不变的能力。在这项工作中，我们引入了一个新的可学习模块--空间变换器，它明确地允许在网络中对数据进行空间操作。这个可区分的模块可以插入到现有的卷积结构中，使神经网络具有主动对特征图进行空间转换的能力，以特征图本身为条件，不需要任何额外的训练监督或对优化过程的修改。我们表明，使用空间变换器的结果是，模型学会了对平移、缩放、旋转和更通用的扭曲的不变性，从而在一些基准和一些变换类别上获得了最先进的性能。



![GAM](img/STN.png)


源码：

[torch](./1-STN/pytorch)  参考：[AlexHex7/**[Spatial-Transformer-Networks_pytorch]**](https://github.com/AlexHex7)

[tensorflow](./1-STN/tensorflow/) 参考：[kevinzakka/**[spatial-transformer-network]**](https://github.com/kevinzakka)



### 3、SE模块

论文地址：https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf

摘要：卷积神经网络是建立在卷积操作之上的，它通过在局部感受野内将空间和通道信息融合在一起来提取信息特征。为了提高网络的表示能力，最近的一些方法显示了加强空间编码的好处。在这项工作中，我们把重点放在通道关系上，并提出了一个新的结构单元，我们称之为 "挤压和激励"（SE）块，它通过明确地重新校准通道特征反应来适应性地调整 通过明确地模拟通道之间的相互依存关系，自适应地重新校准通道的特征响应。我们证明，通过将这些模块堆叠在一起 我们证明，通过将这些模块堆叠在一起，我们可以构建SENet架构，在具有挑战性的数据集上有非常好的通用性。最重要的是，我们发现SE模块对现有的最新技术产生了明显的性能改进。我们发现，SE块能以最小的额外计算成本为现有的最先进的深度架构带来重大的性能改进。SENets构成了我们的ILSVRC的基础 2017年分类报告的基础，该报告赢得了第一名，并且 显著降低了前五名的误差至2.251%，与2016年的获奖作品相比，实现了25%的相对改进。



![GAM](img/SE.png)

源码：

[torch](./2-SENet/pytorch)  参考：[miraclewkf/**[SENet-PyTorch]**](https://github.com/miraclewkf)

[tensorflow](./2-SENet/tensorflow/) 参考：[taki0112/**[SENet-Tensorflow]**](https://github.com/taki0112)



### 4、ODConv动态卷积

论文地址：https://openreview.net/pdf?id=DmpCfq6Mg39

摘要：在每个卷积层中学习单个静态卷积核是现代卷积神经网络 (CNN) 的常见训练范式。相反，最近对动态卷积的研究表明，学习 n 个卷积核的线性组合，该卷积核加权与其输入相关的注意力可以显着提高轻量级 CNN 的准确性，同时保持有效的推理。然而，我们观察到现有的工作通过内核空间的一维（关于卷积核数）赋予卷积核动态属性，但其他三个维度（关于空间大小、输入通道数和输出通道数）每个卷积核）都被忽略了。受此启发，我们提出了全维动态卷积（ODConv），一种更通用但更优雅的动态卷积设计，以推进这一研究方向。ODConv 利用一种新颖的多维注意力机制和并行策略，在任何卷积层沿内核空间的所有四个维度学习卷积内核的互补注意力。作为常规卷积的直接替代品，ODConv 可以插入到许多 CNN 架构中。在 ImageNet 和 MS-COCO 数据集上的大量实验表明，ODConv 为各种流行的 CNN 主干网（包括轻量级和大型主干网）带来了可靠的准确度提升，例如，3.77%~5.71%|1.86%~3.72% 的绝对 top-1 改进MobivleNetV2|ImageNet 数据集上的 ResNet 系列。有趣的是，由于其改进的特征学习能力，即使只有一个内核的 ODConv 也可以与现有的具有多个内核的动态卷积对应物竞争或优于现有的动态卷积对应物，从而大大减少了额外的参数。此外，ODConv 在调制输出特征或卷积权重方面也优于其他注意模块。代码和模型将在https://github.com/OSVAI/ODConv。



![GAM](img/ODConv.png)

源码： [torch](https://github.com/OSVAI/ODConv)  参考：[OSVAI/**[ODConv]**](https://github.com/OSVAI/ODConv)



### 5、FAN注意力

论文地址：https://arxiv.org/abs/2204.12451

摘要：最近的研究表明，视觉变换器(ViTs)表现出强大的鲁棒性对各种腐败。虽然这种特性在一定程度上归因于自我注意机制，但仍缺乏系统的认识。本文研究了自我注意在鲁棒表征学习中的作用。我们的研究是由视觉变换器中新兴的视觉分组的有趣特性所激发的，这表明自我注意可以通过改进中层表征来促进健壮性。我们进一步提出了一系列完全注意网络(FAN) ，它们通过结合注意通道处理设计来加强这种能力。我们在不同的层次主干上全面验证了设计。我们的模型在 ImageNet-1k 和 ImageNet-C 上实现了最先进的87.1% 的准确率和35.8% 的 mCE，参数为76.8 M。我们还展示了最先进的准确性和鲁棒性在两个下游任务: 语义分割和目标检测。



![GAM](img/FAN.png)

源码：  [torch](./4-FAN)  参考：[NVlabs/**[FAN]**](https://github.com/NVlabs/FAN)

