# Enhanced Training of Query-Based Object Detection via Selective Query Recollection [arxiv](https://arxiv.org/abs/2212.07593)

> [**Enhanced Training of Query-Based Object Detection via Selective Query Recollection**](https://arxiv.org/abs/2212.07593)<br>
> Fangyi Chen, Han Zhang, Kai Hu, Yu-Kai Huang, Chenchen Zhu, Marios Savvides<br>Carnegie Mellon University, Meta AI


## 📰 News
**2023.06** We fixed an [issue](https://github.com/Fangyi-Chen/SQR/issues/9#issue-1751787799) in the inference of SQR-Deformable DETR, which logically exists but does not impact the final results.\
**2023.03** This work has been accepted by [CVPR 2023](https://cvpr2023.thecvf.com/).\
**2023.03** The experiments and code on SQR-adamixer and SQR-Deformable DETR have been released.\
**2022.12** The code is available now. 

## 🤔 Motivation
### 🌧 One phenomenon where query-based object detectors mispredict at the last decoding stage but correctly predict at intermediate stages.

The decoding procedure of DETR implies that detection should be stage-by-stage enhanced in terms of IOU and confidence score. Indeed, monotonically improved AP is empirically achieved by this procedure. However, when visualizing the stage-wise predictions, we surprisingly observe that decoder makes mistakes in a decent proportion of cases where the later stages degrade true- positives and upgrade false-positives from the former stages.

### ⭕ Two limitations of training 
1. The responsibility that each stage takes is unbalanced, while supervision applied to them is analogous.  
2.  Due to the sequential structure of the decoder, an intermediate query refined by a stage - no matter this refinement brings positive or negative effects - will be cascaded to the following stage

## 🚀 Selective Query Recollection
As a training strategy that fit most query-based object detectors (DETR family), SQR cumulatively collects intermediate queries as stages go deeper, and feeds the collected queries to the downstream stages aside from the sequential structure.



## ➡️ Guide to Code

This repo provide the implementation of SQR-Adamixer and SQR-deformable DETR. The code structure follows the MMDetection framework. [Adamixer](https://arxiv.org/abs/2203.16507) is a typical query-based object detector that enjoys fast convergence and high AP performance. 
[Deformable DETR](https://arxiv.org/abs/2010.04159) is known for its creative deformable attention module that mitigates the slow convergence and high complexity issues of DETR.

### Config
Our config file lies in [configs/sqr](configs/sqr) folder. 

### SQR-Adamixer
We provide two implementation instances of SQR-adamixer in this repo, one is in [/mmdet/models/roi_heads/adamixer_decoder_Qrecycle.py](/mmdet/models/roi_heads/adamixer_decoder_Qrecycle.py), which might be slower for training but require less GPU memory (and easy to understand the logic). Another is in [/mmdet/models/roi_heads/adamixer_decoder_Qrecycle_optimize.py](/mmdet/models/roi_heads/adamixer_decoder_Qrecycle_optimize.py), which is much faster than the former (and highly recommended for using) but has higher requirement on GPU memory. 

### SQR-Deformable DETR
Similarly, We provide two implementation instances of SQR-deformable DETR in `QRDeformableDetrTransformerDecoder` in [/mmdet/models/utils/transformer.py](/mmdet/models/utils/transformer.py). Named as `forward` and `forward_slow`, separately. Please also check [this issue](https://github.com/Fangyi-Chen/SQR/issues/9#issue-1751787799) and the `QRDeformableDETRHead` in [/mmdet/models/dense_heads/QR_deformable_detr_head.py](mmdet/models/dense_heads/QR_deformable_detr_head.py) where the recollected query is aligned with stages during training. Please note that SQR is only a training strategy that does not affect/change testing pipeline.  

### Installation

We test our models under ```python=3.7, pytorch=1.9.1, cuda=11.1, mmcv=1.3.3```. 

__NOTE:__
Please use `mmcv_full==1.3.3` and `pytorch>=1.5.0` for correct reproduction.

1. Clone this repo
```sh
git clone https://github.com/Fangyi-Chen/SQR.git
cd SQR
```

2. Create a conda env and activate it
```sh
conda create -n sqr-detr python=3.7 -y
conda activate sqr-detr
```

3. Install Pytorch and torchvision

Follow the instruction on https://pytorch.org/get-started/locally/.
```sh
# an example:
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge
```

4. Install mmcv
```sh
pip install mmcv-full=1.3.3 --no-cache-dir
```

5. Install mmdet
```sh
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

### Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.

## 🧪 Main Results

|         | #q       |AP    | AP50 | AP75 |  APs |  APm | APl  | model | cfg |
|---------|-------|------|------|------|------|------|------|-------|------|
|SQR-Adamixer-R50 | 100| 44.5  |  63.2 |  47.8 |  25.7 |  47.4 |  60.2 | [ckpt](https://drive.google.com/file/d/1-io4kMQ-6h814AMmE7wKKUcEmTjLdpIa/view?usp=share_link) |[cfg](configs/sqr/adamixer_SQR_r50_1x_coco.py)|
|SQR-Adamixer-R101-7stages| 300| 49.8  |  68.8 | 54.0 |  32.0 | 53.4 | 65.1 | [ckpt](https://drive.google.com/file/d/1alJNY8eJy-E7mURJDj5LZFHNZ6SG88CF/view?usp=share_link) | cfg |
|SQR-Deformable-DETR | 300| 45.8  |  64.7 |  49.8 |  28.2 |  49.4 |  60.0 | [ckpt](https://drive.google.com/file/d/1JTWmHXPngUJY7EaYMneVHYYlvK_5GDfV/view?usp=sharing) |[cfg](configs/sqr/deformable_detr_SQR_r50_50e_coco.py)|


## ✏️ Citation
If you find SQR useful, please use the following entry to cite us:
```
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Fangyi and Zhang, Han and Hu, Kai and Huang, Yu-Kai and Zhu, Chenchen and Savvides, Marios},
    title     = {Enhanced Training of Query-Based Object Detection via Selective Query Recollection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {23756-23765}
}
```



## Original MMDetection README.md
_The following begins the original mmdetection README.md file_
<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
</div>

**News**: We released the technical report on [ArXiv](https://arxiv.org/abs/1906.07155).

Documentation: https://mmdetection.readthedocs.io/

## Introduction

English | [简体中文](README_zh-CN.md)

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.3+**.
The old v1.x branch works with PyTorch 1.1 to 1.4, but v2.0 is strongly recommended for faster speed, higher performance, better design and more friendly usage.

![demo image](resources/coco_test_12510.jpg)

### Major features

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## License

The mmdetection project is released under the [Apache 2.0 license](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE).

## Changelog

v2.12.0 was released in 01/05/2021.
Please refer to [changelog.md](docs/changelog.md) for details and release history.
A comparison between v1.x and v2.0 codebases can be found in [compatibility.md](docs/compatibility.md).

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] VGG (ICLR'2015)
- [x] HRNet (CVPR'2019)
- [x] RegNet (CVPR'2020)
- [x] Res2Net (TPAMI'2020)
- [x] ResNeSt (ArXiv'2020)

Supported methods:

- [x] [RPN (NeurIPS'2015)](configs/rpn)
- [x] [Fast R-CNN (ICCV'2015)](configs/fast_rcnn)
- [x] [Faster R-CNN (NeurIPS'2015)](configs/faster_rcnn)
- [x] [Mask R-CNN (ICCV'2017)](configs/mask_rcnn)
- [x] [Cascade R-CNN (CVPR'2018)](configs/cascade_rcnn)
- [x] [Cascade Mask R-CNN (CVPR'2018)](configs/cascade_rcnn)
- [x] [SSD (ECCV'2016)](configs/ssd)
- [x] [RetinaNet (ICCV'2017)](configs/retinanet)
- [x] [GHM (AAAI'2019)](configs/ghm)
- [x] [Mask Scoring R-CNN (CVPR'2019)](configs/ms_rcnn)
- [x] [Double-Head R-CNN (CVPR'2020)](configs/double_heads)
- [x] [Hybrid Task Cascade (CVPR'2019)](configs/htc)
- [x] [Libra R-CNN (CVPR'2019)](configs/libra_rcnn)
- [x] [Guided Anchoring (CVPR'2019)](configs/guided_anchoring)
- [x] [FCOS (ICCV'2019)](configs/fcos)
- [x] [RepPoints (ICCV'2019)](configs/reppoints)
- [x] [Foveabox (TIP'2020)](configs/foveabox)
- [x] [FreeAnchor (NeurIPS'2019)](configs/free_anchor)
- [x] [NAS-FPN (CVPR'2019)](configs/nas_fpn)
- [x] [ATSS (CVPR'2020)](configs/atss)
- [x] [FSAF (CVPR'2019)](configs/fsaf)
- [x] [PAFPN (CVPR'2018)](configs/pafpn)
- [x] [Dynamic R-CNN (ECCV'2020)](configs/dynamic_rcnn)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [CARAFE (ICCV'2019)](configs/carafe/README.md)
- [x] [DCNv2 (CVPR'2019)](configs/dcn/README.md)
- [x] [Group Normalization (ECCV'2018)](configs/gn/README.md)
- [x] [Weight Standardization (ArXiv'2019)](configs/gn+ws/README.md)
- [x] [OHEM (CVPR'2016)](configs/faster_rcnn/faster_rcnn_r50_fpn_ohem_1x_coco.py)
- [x] [Soft-NMS (ICCV'2017)](configs/faster_rcnn/faster_rcnn_r50_fpn_soft_nms_1x_coco.py)
- [x] [Generalized Attention (ICCV'2019)](configs/empirical_attention/README.md)
- [x] [GCNet (ICCVW'2019)](configs/gcnet/README.md)
- [x] [Mixed Precision (FP16) Training (ArXiv'2017)](configs/fp16/README.md)
- [x] [InstaBoost (ICCV'2019)](configs/instaboost/README.md)
- [x] [GRoIE (ICPR'2020)](configs/groie/README.md)
- [x] [DetectoRS (ArXix'2020)](configs/detectors/README.md)
- [x] [Generalized Focal Loss (NeurIPS'2020)](configs/gfl/README.md)
- [x] [CornerNet (ECCV'2018)](configs/cornernet/README.md)
- [x] [Side-Aware Boundary Localization (ECCV'2020)](configs/sabl/README.md)
- [x] [YOLOv3 (ArXiv'2018)](configs/yolo/README.md)
- [x] [PAA (ECCV'2020)](configs/paa/README.md)
- [x] [YOLACT (ICCV'2019)](configs/yolact/README.md)
- [x] [CentripetalNet (CVPR'2020)](configs/centripetalnet/README.md)
- [x] [VFNet (ArXix'2020)](configs/vfnet/README.md)
- [x] [DETR (ECCV'2020)](configs/detr/README.md)
- [x] [Deformable DETR (ICLR'2021)](configs/deformable_detr/README.md)
- [x] [CascadeRPN (NeurIPS'2019)](configs/cascade_rpn/README.md)
- [x] [SCNet (AAAI'2021)](configs/scnet/README.md)
- [x] [AutoAssign (ArXix'2020)](configs/autoassign/README.md)
- [x] [YOLOF (CVPR'2021)](configs/yolof/README.md)


Some other methods are also supported in [projects using MMDetection](./docs/projects.md).

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Getting Started

Please see [get_started.md](docs/get_started.md) for the basic usage of MMDetection.
We provide [colab tutorial](demo/MMDet_Tutorial.ipynb), and full guidance for quick run [with existing dataset](docs/1_exist_data_model.md) and [with new dataset](docs/2_new_data_model.md) for beginners.
There are also tutorials for [finetuning models](docs/tutorials/finetune.md), [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), [customizing models](docs/tutorials/customize_models.md), [customizing runtime settings](docs/tutorials/customize_runtime.md) and [useful tools](docs/useful_tools.md).

Please refer to [FAQ](docs/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): A Comprehensive Toolbox for Text Detection, Recognition and Understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- 


