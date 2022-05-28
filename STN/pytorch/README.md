# Tpatial-Transformer-Networks-pytorch
- Refer to [daviddao/spatial-transformer-tensorflow](https://github.com/daviddao/spatial-transformer-tensorflow)(Tensorflow) .
- Implementation of [**Spatial Transformer Networks**](https://arxiv.org/abs/1506.02025).  

## Statement
- Do the Experiments on the **cluttered MNIST** dataset of [daviddao](https://github.com/daviddao/spatial-transformer-tensorflow).
- The accuracy and loss records can be find in **cnn.out & stn.out**.
- The transform img can be find in **transform_img/**.
- **py35_pytorch03_version contains** the old version code

## Environment
- python3.6
- pytorch 0.4.0



## Accuracy
**CNN**

- Testing: epoch[195/200] loss:0.5264 acc:0.9211
- Testing: epoch[196/200] loss:0.5185 acc:0.9194
- Testing: epoch[197/200] loss:0.5160 acc:0.9158
- Testing: epoch[198/200] loss:0.5053 acc:0.9183
- Testing: epoch[199/200] loss:0.5057 acc:0.9153

**STN**

- Testing: epoch[195/200] loss:0.0880 acc:0.9762
- Testing: epoch[196/200] loss:0.0961 acc:0.9757
- Testing: epoch[197/200] loss:0.0893 acc:0.9742
- Testing: epoch[198/200] loss:0.1015 acc:0.9740
- Testing: epoch[199/200] loss:0.0938 acc:0.9738

## Transform Image
**(input|transform|input|transform)**
![](transform_img/0.png)
