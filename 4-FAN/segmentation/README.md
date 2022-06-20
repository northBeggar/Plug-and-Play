# Segmentation codebase for FAN models

Our segmentationcode is developed on top of [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).


## Dependencies

Install according to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Below is the environment configuration for our codebase:

CUDA Version: 11.1

Pytorch: 1.7.1

Torchvision: 0.8.1

timm: 0.5.4

The rest of the dependencies can be installed via:
pip3 install -r requirements.txt


## Dataset preparation

Prepare Cityscapes according to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

To generate Cityscapes-C dataset, first install the natural image corruption lib via:

pip3 install imagecorruptions

Then, run the following command:

```
python3 tools/gen_city_c.py
```

To evaluation the robustness on Cityscape-C:
```
./tools/dist_test_city_c.sh /PATH/TO/CONFIG/FILE /PATH/TO/CHECKPOINT/FILE 1 --eval mIoU  --results-file ../output/ 
```


## Training

An example of running fan_hybrid_tiny model with SegFormer Head:
```
bash scripts/fan_tiny_hybrid_city.sh
```

