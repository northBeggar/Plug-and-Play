python train_imagenet.py \
			--ngpu 4 \
			--workers 20 \
			--arch resnet --depth 50 \
			--epochs 100 \
			--batch-size 256 \
			--lr 0.1 \
			--att-type TripletAttention \
			--prefix RESNET50_TripletAttention_IMAGENET \
			--resume checkpoints/RESNET50_IMAGENET_TripletAttention_checkpoint.pth.tar\
			/home/shared/imagenet/raw/
