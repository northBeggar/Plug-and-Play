export DETECTRON2_DATASETS=~/datasets
python train_detectron.py --num-gpus 2 \
        --config-file ./detectron_configs/COCO-Keypoints/keypoint_rcnn_resnet50_triplet_attention_FPN_1x.yaml \
	#--eval-only MODEL.WEIGHTS ./output/model_final.pth
