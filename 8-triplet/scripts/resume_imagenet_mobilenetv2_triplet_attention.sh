python train_imagenet.py \
			--ngpu 2 \
			--workers 20 \
			--arch mobilenet \
			--epochs 400 \
			--batch-size 96 \
			--lr 0.045 \
			--weight-decay 0.00004 \
			--att-type TripletAttention \
			--prefix MOBILENET_TripletAttention_IMAGENET \
			--resume checkpoints/MOBILENET_TripletAttention_IMAGENET_checkpoint.pth.tar\
			/home/shared/imagenet/raw/
