## This is the PyTorch implement of SENet (train on ImageNet dataset)

Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)


# Usage

### Prepare data

This code takes ImageNet dataset as example. You can download ImageNet dataset and put them as follows. I only provide `ILSVRC2012_dev_kit_t12` due to the restriction of memory, in other words, you need download `ILSVRC2012_img_train` and `ILSVRC2012_img_val`.

```
├── train.py # train script
├── se_resnet.py # network of se_resnet
├── se_resnext.py # network of se_resnext
├── read_ImageNetData.py # ImageNet dataset read script
├── ImageData # train and validation data
	├── ILSVRC2012_img_train
		├── n01440764
		├──    ...
		├── n15075141
	├── ILSVRC2012_img_val
	├── ILSVRC2012_dev_kit_t12
		├── data
			├── ILSVRC2012_validation_ground_truth.txt
			├── meta.mat # the map between train file name and label
```

### Train

* If you want to train from scratch, you can run as follows:

```
python train.py --network se_resnext_50 --batch-size 256 --gpus 0,1,2,3
```

parameter `--network` can be `se_resnet_18` or `se_resnet_34` or `se_resnet_50` or `se_resnet_101` or `se_resnet_152` or `se_resnext_50` or `se_resnext_101` or `se_resnext_152`.

* If you want to train from one checkpoint, you can run as follows(for example train from `epoch_4.pth.tar`, the `--start-epoch` parameter is corresponding to the epoch of the checkpoint):

```
python train.py --network se_resnext_50 --batch-size 256 --gpus 0,1,2,3 --resume output/epoch_4.pth.tar --start-epoch 4
```