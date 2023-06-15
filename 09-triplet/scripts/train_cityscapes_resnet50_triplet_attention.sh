export DETECTRON2_DATASETS=~/
python train_detectron.py --num-gpus 8 \
        --config-file ./detectron_configs/Cityscapes/mask_rcnn_resnet50_triplet_attention_FPN.yaml \
	#--eval-only MODEL.WEIGHTS ./output/model_final.pth
