import torch
import mmcv
import argparse
import os.path as osp

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('checkpoint', nargs='?', type=str, default=None)
args = parser.parse_args()

dir_name = args.checkpoint.split("/")[-2]
checkpoint = torch.load(args.checkpoint, map_location='cpu')
print(checkpoint.keys())
state_dict = checkpoint['ema']
for k, v in state_dict.items():
    print(k)
checkpoint = {'ema': state_dict}
mmcv.mkdir_or_exist("converted/")
try:
    torch.save(checkpoint, osp.join("converted", dir_name+".pth"), _use_new_zipfile_serialization=False)
except:
    torch.save(checkpoint, osp.join("converted", dir_name+".pth"))
