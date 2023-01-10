import torch
import argparse
import os
from mmcv import Config

from models import build_model
from models.utils import fuse_module, rep_model_convert


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


def main(args):
    cfg = Config.fromfile(args.config)
    model = build_model(cfg.model)
    model = model.cuda()

    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            checkpoint = torch.load(args.checkpoint)
            d = dict()
            for key, value in checkpoint['ema'].items():
                tmp = key.replace("module.", "")
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint))
            raise

    model = rep_model_convert(model)
    model = fuse_module(model)
    model_structure(model)
    new_state_dict = {}
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        if "det_head.final.conv" in k:
            print(k, v.shape)
            new_state_dict[k] = v[:1, ...] # remove auxiliary head
            # we use only one channel to predict the final results
            print(new_state_dict[k].shape)
        else:
            new_state_dict[k] = v
    torch.save(new_state_dict, args.out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('out', nargs='?', type=str, default=None)

    args = parser.parse_args()
    
    main(args)
