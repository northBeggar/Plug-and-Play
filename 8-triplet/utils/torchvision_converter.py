import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

path = args.path
state = torch.load(path)

state_dict = state
new_state_dict = dict()

for k in state_dict.keys():
    if "num_batches_tracked" in k:
        continue
    new_key = k.replace("layer1", "backbone.bottom_up.res2")
    new_key = new_key.replace("layer2", "backbone.bottom_up.res3")
    new_key = new_key.replace("layer3", "backbone.bottom_up.res4")
    new_key = new_key.replace("layer4", "backbone.bottom_up.res5")
    new_key = new_key.replace("bn1", "conv1.norm")
    new_key = new_key.replace("bn2", "conv2.norm")
    new_key = new_key.replace("bn3", "conv3.norm")
    new_key = new_key.replace("downsample.0", "shortcut")
    new_key = new_key.replace("downsample.1", "shortcut.norm")
    # new_key = new_key[7:]
    if new_key.startswith("conv1"):
        print("STEM")
        new_key = "backbone.bottom_up.stem." + new_key
    new_state_dict[new_key] = state_dict[k]
    print(k + " ----> " + new_key)

# print(new_state_dict.keys())
torch.save(new_state_dict, "./checkpoints/torchvision_backbone.pth")
