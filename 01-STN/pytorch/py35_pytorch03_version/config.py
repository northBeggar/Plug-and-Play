cuda_num = 1
LR = 0.001
epoch = 200
show_train_result_every_batch = 100
test_every_epoch = 1
save_model_every_epoch = 1

height = 40
width = 40
channel = 1

train_batch_size = 64
test_batch_size = 128

drop_prob = 0.8

data_path = 'data/mnist_sequence1_sample_5distortions5x5.npz'
model_name = 'model_%s.pkl'
model_dir = 'models/'
transform_img_dir = 'transform_img'
mode = 'stn'



