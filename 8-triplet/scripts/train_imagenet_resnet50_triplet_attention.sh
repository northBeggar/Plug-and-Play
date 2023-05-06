python train_imagenet.py \
			--ngpu 1 \
			--workers 20 \
			--arch resnet --depth 50 \
			--epochs 100 \
			--batch-size 256 \
			--lr 0.1 \
			--att-type TripletAttention \
			--prefix RESNET50_TripletAttention_IMAGENET \
			/home/shared/imagenet/raw/
