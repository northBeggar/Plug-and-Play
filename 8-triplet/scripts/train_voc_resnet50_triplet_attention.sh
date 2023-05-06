export DETECTRON2_DATASETS=~/VOCdevkit
python train_detectron.py --num-gpus 2 \
        --config-file ./detectron_configs/PascalVOC-Detection/faster_rcnn_resnet50_triplet_attention_FPN.yaml \
	#--eval-only MODEL.WEIGHTS ./output/model_final.pth
