# Tpatial-Transformer-Networks-pytorch
- Refer to [daviddao/spatial-transformer-tensorflow](https://github.com/daviddao/spatial-transformer-tensorflow)(Tensorflow) .
- Implementation of [**Spatial Transformer Networks**](https://arxiv.org/abs/1506.02025).  

## Statement
- Do the Experiments on the **cluttered MNIST** dataset of [daviddao](https://github.com/daviddao/spatial-transformer-tensorflow).
- The accuracy and loss records can be find in **cnn.out & stn.out**.
- The transform img can be find in **transform_img/**.

## Environment
- python3.5
- pytorch '0.2.0+eed323c'


## Accuracy
**CNN**

- ========= Testing: epoch[195/200] loss:0.5183 acc:0.9188
- ========= Testing: epoch[196/200] loss:0.5136 acc:0.9208
- ========= Testing: epoch[197/200] loss:0.5165 acc:0.9167
- ========= Testing: epoch[198/200] loss:0.4953 acc:0.9182
- ========= Testing: epoch[199/200] loss:0.5040 acc:0.9249

**STN**

- ========= Testing: epoch[195/200] loss:0.0899 acc:0.9743
- ========= Testing: epoch[196/200] loss:0.0889 acc:0.9745
- ========= Testing: epoch[197/200] loss:0.0920 acc:0.9703
- ========= Testing: epoch[198/200] loss:0.0871 acc:0.9765
- ========= Testing: epoch[199/200] loss:0.0900 acc:0.9748

## Transform Image
**(input|transform|input|transform)**
![](transform_img/0.png)
