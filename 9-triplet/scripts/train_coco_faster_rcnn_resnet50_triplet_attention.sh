export DETECTRON2_DATASETS=~/datasets
python train_detectron.py --num-gpus 4 \
        --config-file ./detectron_configs/COCO-Detection/faster_rcnn_resnet50_triplet_attention_FPN_1x.yaml \
	#--eval-only MODEL.WEIGHTS ./output/model_final.pth
