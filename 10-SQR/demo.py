from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import sys
import os.path
name = sys.argv[1]
thr = sys.argv[2]
config_file = './configs/adamixer/adamixer_r50_1x_coco.py'
checkpoint_file = name

if name == 'none':
    checkpoint_file = None


model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'data/coco/val2017/000000057597.jpg'

result = inference_detector(model, img)
show_result_pyplot(model, img, result, score_thr=float(thr),
                   out_file='demo/result.jpg')
