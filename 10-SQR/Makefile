adamixer-r50:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29502 ./tools/dist_train.sh \
	configs/adamixer/adamixer_r50_1x_coco.py \
	8

adamixer-r50-3x:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh \
	configs/adamixer/adamixer_r50_300_query_crop_mstrain_480-800_3x_coco.py \
	8

adamixer-r101-3x:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh \
	configs/adamixer/adamixer_r101_300_query_crop_mstrain_480-800_3x_coco.py \
	8

adamixer-dx101-3x:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh \
	configs/adamixer/adamixer_dx101_300_query_crop_mstrain_480-800_3x_coco.py \
	8

adamixer-swin_s-3x:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29501 ./tools/dist_train.sh \
	configs/adamixer/adamixer_swin_s_300_query_crop_mstrain_480-800_3x_coco.py \
	8
