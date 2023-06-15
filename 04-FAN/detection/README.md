# Detection codebase for FAN models

Our detection code is developed on top of [MMDetection v2.11.0](https://github.com/open-mmlab/mmdetection/tree/v2.11.0).


## Dependencies

Install according to the guidelines in [MMDetection v2.11.0](https://github.com/open-mmlab/mmdetection/tree/v2.11.0).

or

```
pip install mmdet==2.13.0 --user
```

Below is the environment configuration for our codebase:

CUDA Version: 11.1

Torchvision: 0.8.1

Pytorch: 1.7.1

mmcv-full: 1.3.0

timm: 0.5.4


## Data preparation

Prepare COCO according to the guidelines in [MMDetection v2.11.0](https://github.com/open-mmlab/mmdetection/tree/v2.13.0).


## Training
To train FAN-T-Hybrid + Cascade MRCNN on COCO train2017 on 4  nodes with 32 gpus, run following commands on each node separately:

```
bash mn_train.sh configs/cascade_mask_fan_tiny_fpn_3x_mstrain_fp16.py 4 $local_rank "master node address"
```

To train FAN-T-Hybrid on a single node:
```
bash dist_train.sh configs/cascade_mask_fan_tiny_fpn_3x_mstrain_fp16.py 8
```

## COCO-C Dataset Generation
To generate COCO-C dataset:
```
python3 tools/gen_coco_c.py
```


## Test robustness on COCO-C

To test robustness on a single corruption type:
```
bash dist_test_coco_c.sh $CONFIG $CKP 8 gaussian_noise 1 --work-dir $WORKDIR
```

We also provide a script to test robustness over all corruption categories:
```
bash tools/coco_c_test_all.sh
```
