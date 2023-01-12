import torch
import argparse
import os
import sys
from mmcv import Config
import mmcv
from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module, rep_model_convert
from utils import ResultFormat, AverageMeter
from mmcv.cnn import get_model_complexity_info
import logging
import warnings
warnings.filterwarnings('ignore')
import json


def report_speed(model, data, speed_meters, batch_size=1, times=10):
    for _ in range(times):
        total_time = 0
        outputs = model(**data)
        for key in outputs:
            if 'time' in key:
                speed_meters[key].update(outputs[key] / batch_size)
                total_time += outputs[key] / batch_size
        speed_meters['total_time'].update(total_time)
        for k, v in speed_meters.items():
            print('%s: %.4f' % (k, v.avg))
            logging.info('%s: %.4f' % (k, v.avg))
        print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))
        logging.info('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))


def test(test_loader, model, cfg):

    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(1000 // args.batch_size),
            neck_time=AverageMeter(1000 // args.batch_size),
            det_head_time=AverageMeter(1000 // args.batch_size),
            post_time=AverageMeter(1000 // args.batch_size),
            total_time=AverageMeter(1000 // args.batch_size)
        )
    results = dict()

    for idx, data in enumerate(test_loader):
        print('Testing %d/%d\r' % (idx, len(test_loader)), flush=True, end='')
        logging.info('Testing %d/%d\r' % (idx, len(test_loader)))
        # prepare input
        if not args.cpu:
            data['imgs'] = data['imgs'].cuda(non_blocking=True)
        data.update(dict(cfg=cfg))
        # forward
        with torch.no_grad():
            outputs = model(**data)

        if cfg.report_speed:
            report_speed(model, data, speed_meters, cfg.batch_size)
            continue

        # save result
        image_names = data['img_metas']['filename']
        for index, image_name in enumerate(image_names):
            rf.write_result(image_name, outputs['results'][index])
            results[image_name] = outputs['results'][index]

    if not cfg.report_speed:
        results = json.dumps(results)
        with open('outputs/output.json', 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False)
            print("write json file success!")


def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4

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

    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed,
        ))
    if args.min_score is not None:
        cfg.test_cfg.min_score = args.min_score
    if args.min_area is not None:
        cfg.test_cfg.min_area = args.min_area

    cfg.batch_size = args.batch_size

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.worker,
        pin_memory=False
    )
    # model
    model = build_model(cfg.model)
    
    if not args.cpu:
        model = model.cuda()
    
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            logging.info("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()
            checkpoint = torch.load(args.checkpoint)
            
            if not args.ema:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint['ema']

            d = dict()
            for key, value in state_dict.items():
                tmp = key.replace("module.", "")
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint))
            raise
    
    model = rep_model_convert(model)

    # fuse conv and bn
    model = fuse_module(model)
    
    if args.print_model:
        model_structure(model)
        
    # flops, params = get_model_complexity_info(model, (3, 1280, 864))
    # flops, params = get_model_complexity_info(model, (3, 1200, 800))
    # flops, params = get_model_complexity_info(model, (3, 1344, 896))
    # flops, params = get_model_complexity_info(model, (3, 960, 640))
    # flops, params = get_model_complexity_info(model, (3, 768, 512))
    # flops, params = get_model_complexity_info(model, (3, 672, 448))
    # flops, params = get_model_complexity_info(model, (3, 480, 320))
    # print(flops, params)
    
    model.eval()
    test(test_loader, model, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report-speed', action='store_true')
    parser.add_argument('--print-model', action='store_true')
    parser.add_argument('--min-score', default=None, type=float)
    parser.add_argument('--min-area', default=None, type=int)
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--ema', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    mmcv.mkdir_or_exist("./speed_test")
    config_name = os.path.basename(args.config)
    logging.basicConfig(filename=f'./speed_test/{config_name}.txt', level=logging.INFO)

    main(args)
