# MS-COCO Object Detection with ODConv 

We use the popular [MMDetection](https://github.com/open-mmlab/mmdetection) toolbox for experiments on the MS-COCO dataset with the pre-trained ResNet50 and MobileNetV2 (1.0×) models as the backbones for the detector. We select the mainstream Faster RCNN and Mask R-CNN detectors with Feature Pyramid Networks as the necks to build the basic object detection systems.


## Training

Please follow [MMDetection](https://github.com/open-mmlab/mmdetection) on how to prepare the environment and the dataset. Then attach our code to the origin project and modify the config files according to your own path to the pre-trained models and directories to save logs and models.

To train a detector with pre-trained models as backbone:

```shell
bash tools/dist_train.sh {path to config file} {ngpus}
```


## Evaluation

To evaluate a fine-tuned model:
```shell
bash tools/dist_test.sh {path to config file} {path to fine-tuned model} {ngpus} --eval {evaluation metrics} --show
```


## Results and Models

| Backbones | Detectors | Params | box AP | mask AP | Config | Google Drive | Baidu Netdisk |
|:--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ResNet50 | Faster R-CNN | 43.80M | 37.4 | -   | [config](configs/odconv/faster_rcnn_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1JoRu74q3qE_L3jCltN6Y_Z4M_Bvc-O9U/view?usp=sharing) | [model](https://pan.baidu.com/s/1lMWN1CAoYg4KGyfIuM1-Ng?pwd=8pgt) |
| + ODConv (4×) | Faster R-CNN | 108.91M | 39.4 | -   | [config](configs/odconv/faster_rcnn_odconv4x_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1fMHVWL6blFMw1ZzT35KPb1hnOYVy-8dH/view?usp=sharing) | [model](https://pan.baidu.com/s/1Gzg8lZ9hXICeLp6iPxQ_Yg?pwd=dt9b) |
| MobileNetV2 (1.0×) | Faster R-CNN | 21.13M | 31.8 | -   | [config](configs/odconv/faster_rcnn_mobilenetv2_100_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1XMHmOg-CclIc9XiuEOod_3bnamRCBCOa/view?usp=sharing) | [model](https://pan.baidu.com/s/1HkaaRCWthccOD9W0Gi08SA?pwd=7k2z) |
| + ODConv (4×) | Faster R-CNN | 29.14M | 35.5 | -   | [config](configs/odconv/faster_rcnn_odconv4x_mobilenetv2_100_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1vWtQpQ-KK_-Tfr9vXOyVh2POWkhCpAvi/view?usp=sharing) | [model](https://pan.baidu.com/s/1qh6o2heq4Ajg9cj5t1hG1Q?pwd=89md) |
| ResNet50 | Mask R-CNN | 46.45M | 38.2 | 34.6 | [config](configs/odconv/mask_rcnn_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1dz7HecPGQBSCYUefCPo3acmh9K7g0H9P/view?usp=sharing) | [model](https://pan.baidu.com/s/1MQOclQ3fe9_RbWmFVFU_lg?pwd=km6s) |
| + ODConv (4×) | Mask R-CNN | 111.56M | 40.2 | 36.1 | [config](configs/odconv/mask_rcnn_odconv4x_r50_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1NZKfB0dttJSmnlNbCVPc1OIIhkc2OjMe/view?usp=sharing) | [model](https://pan.baidu.com/s/1QDZc05sMWMw9F1UGFxj4VA?pwd=k4aj) |
| MobileNetV2 (1.0×) | Mask R-CNN | 23.78M | 32.4 | 29.8 | [config](configs/odconv/mask_rcnn_mobilenetv2_100_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1OtZvIP-8ktdQsmI3Pss6PSEQvRh5NAgg/view?usp=sharing) | [model](https://pan.baidu.com/s/1hoxj95GiP1FECFcv6ZGKXQ?pwd=x6c9) |
| + ODConv (4×) | Mask R-CNN | 31.80M | 36.0 | 33.0 | [config](configs/odconv/mask_rcnn_odconv4x_mobilenetv2_100_fpn_1x_coco.py) | [model](https://drive.google.com/file/d/1p_gqy6xIquEZtu1TEuWKid-0gD2Jv2Z4/view?usp=sharing) | [model](https://pan.baidu.com/s/11RqowarM_trrfdqDtBQZFg?pwd=kqar) |
