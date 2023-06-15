_base_ = './adamixer_r50_300_query_crop_mstrain_480-800_3x_coco.py'

model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
