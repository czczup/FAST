import torch
import numpy as np
import random
import argparse
import os
import os.path as osp
import sys
import time
import json
from mmcv import Config
import logging

from dataset import build_data_loader
from models import build_model
from utils import AverageMeter, setup_logger, EMA
from dataset.dataloader import DataLoaderX
try:
    import apex
except:
    pass


def train(train_loader, model, optimizer, ema, epoch, start_iter, cfg):
    model.train()

    # meters
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    losses_text = AverageMeter()
    losses_kernels = AverageMeter()
    losses_emb = AverageMeter()
    losses_rec = AverageMeter()

    ious_text = AverageMeter()
    ious_kernel = AverageMeter()
    accs_rec = AverageMeter()

    # start time
    start = time.time()
    for iter, data in enumerate(train_loader):
        # skip previous iterations
        if iter < start_iter:
            logging.info('Skipping iter: %d' % iter)
            sys.stdout.flush()
            continue

        # time cost of data loader
        data_time.update(time.time() - start)

        # adjust learning rate
        adjust_learning_rate(optimizer, train_loader, epoch, iter, cfg)

        # prepare input
        data.update(dict(
            cfg=cfg
        ))

        # forward
        outputs = model(**data)

        # detection loss
        loss_text = torch.mean(outputs['loss_text'])
        losses_text.update(loss_text.item())

        loss_kernels = torch.mean(outputs['loss_kernels'])
        losses_kernels.update(loss_kernels.item())

        loss_emb = torch.mean(outputs['loss_emb'])
        losses_emb.update(loss_emb.item())

        loss = loss_text + loss_kernels + loss_emb

        iou_text = torch.mean(outputs['iou_text'])
        ious_text.update(iou_text.item())
        iou_kernel = torch.mean(outputs['iou_kernel'])
        ious_kernel.update(iou_kernel.item())

        # recognition loss
        if hasattr(cfg.model, 'recognition_head'):
            loss_rec = outputs['loss_rec']
            valid = loss_rec > 0.5
            if torch.sum(valid) > 0:
                loss_rec = torch.mean(loss_rec[valid])
                losses_rec.update(loss_rec.item())
                loss = loss + loss_rec

                acc_rec = outputs['acc_rec']
                acc_rec = torch.mean(acc_rec[valid])
                accs_rec.update(acc_rec.item(), torch.sum(valid).item())

        losses.update(loss.item())

        # backward
        optimizer.zero_grad()
        if args.apex == True:
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        ema.update()
        
        batch_time.update(time.time() - start)

        # update start time
        start = time.time()

        # print log
        if iter % 10 == 0:
            output_log = '({batch}/{size}) LR: {lr:.6f} | Batch: {bt:.3f}s | Total: {total:.0f}min | ' \
                         'ETA: {eta:.0f}min | Loss: {loss:.3f} | ' \
                         'Loss(text/kernel/emb/rec): {loss_text:.3f}/{loss_kernel:.3f}/{loss_emb:.3f}/{loss_rec:.3f} ' \
                         '| IoU(text/kernel): {iou_text:.3f}/{iou_kernel:.3f} | Acc rec: {acc_rec:.3f}'.format(
                batch=iter + 1,
                size=len(train_loader),
                lr=optimizer.param_groups[0]['lr'],
                bt=batch_time.avg,
                total=batch_time.avg * iter / 60.0,
                eta=batch_time.avg * (len(train_loader) - iter) / 60.0,
                loss_text=losses_text.avg,
                loss_kernel=losses_kernels.avg,
                loss_emb=losses_emb.avg,
                loss_rec=losses_rec.avg,
                loss=losses.avg,
                iou_text=ious_text.avg,
                iou_kernel=ious_kernel.avg,
                acc_rec=accs_rec.avg,
            )
            logging.info(output_log)


def adjust_learning_rate(optimizer, dataloader, epoch, iter, cfg):
    schedule = cfg.train_cfg.schedule

    if isinstance(schedule, str):
        assert schedule == 'polylr', 'Error: schedule should be polylr!'
        cur_iter = epoch * len(dataloader) + iter
        max_iter_num = cfg.train_cfg.epoch * len(dataloader)
        lr = cfg.train_cfg.lr * (1 - float(cur_iter) / max_iter_num) ** 0.9
    elif isinstance(schedule, tuple):
        lr = cfg.train_cfg.lr
        for i in range(len(schedule)):
            if epoch < schedule[i]:
                break
            lr = lr * 0.1

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, checkpoint_path, cfg):
    file_path = osp.join(checkpoint_path, 'checkpoint.pth.tar')
    torch.save(state, file_path)
    if not hasattr(cfg.train_cfg, "save_interval"):
        cfg.train_cfg.save_interval = 10
    if cfg.data.train.type in ['synth'] or \
            (state['iter'] == 0 and state['epoch'] > cfg.train_cfg.epoch - 100 and state['epoch'] % cfg.train_cfg.save_interval == 0):
        file_name = 'checkpoint_%dep.pth.tar' % state['epoch']
        file_path = osp.join(checkpoint_path, file_name)
        torch.save(state, file_path)


def main(args):
    cfg = Config.fromfile(args.config)
    logging.info(json.dumps(cfg._cfg_dict, indent=4))

    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        cfg_name, _ = osp.splitext(osp.basename(args.config))
        checkpoint_path = osp.join('checkpoints', cfg_name)
    if not osp.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    logging.info('Checkpoint path: %s.' % checkpoint_path)

    # data loader
    data_loader = build_data_loader(cfg.data.train)
    train_loader = DataLoaderX(
        data_loader,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=args.worker,
        drop_last=True,
        pin_memory=True
    )

    # model
    model = build_model(cfg.model)
    
    # logging.info(model)

    if cfg.train_cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train_cfg.lr, momentum=0.9, weight_decay=1e-4)
    elif cfg.train_cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train_cfg.lr)
    elif cfg.train_cfg.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train_cfg.lr)

    if args.apex == True:
        model, optimizer = apex.amp.initialize(model.cuda(), optimizer, opt_level="O1")
        logging.info("Initializing mixed precision done.")


    model = torch.nn.DataParallel(model).cuda()
    
    start_epoch = 0
    start_iter = 0
    if hasattr(cfg.train_cfg, 'pretrain'):
        assert osp.isfile(cfg.train_cfg.pretrain), 'Error: no pretrained weights found!'
        logging.info('Finetuning from pretrained model %s.' % cfg.train_cfg.pretrain)
        checkpoint = torch.load(cfg.train_cfg.pretrain)
        try:
            logging.info("loading ema pretrained weights!")
            state_dict = checkpoint['ema']
        except:
            state_dict = checkpoint['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict["module.backbone."+k] = v
            state_dict = new_state_dict
        logging.info(model.load_state_dict(state_dict, strict=False))
    if args.resume:
        assert osp.isfile(args.resume), 'Error: no checkpoint directory found!'
        logging.info('Resuming from checkpoint %s.' % args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint['iter']
        logging.info(model.load_state_dict(checkpoint['state_dict']))
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    ema = EMA(model, 0.999)
    ema.register()
    
    for epoch in range(start_epoch, cfg.train_cfg.epoch):
        logging.info(cfg_name)
        logging.info('\nEpoch: [%d | %d]' % (epoch + 1, cfg.train_cfg.epoch))

        train(train_loader, model, optimizer, ema, epoch, start_iter, cfg)

        ema.apply_shadow()
        ema_state_dict = model.state_dict()
        ema.restore()
        
        state = dict(
            epoch=epoch + 1,
            iter=0,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            ema=ema_state_dict
        )
        save_checkpoint(state, checkpoint_path, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--resume', nargs='?', type=str, default=None)
    parser.add_argument('--worker', type=int, default=16)
    parser.add_argument('--apex', action='store_true', help='use apex')
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    torch.backends.cudnn.benchmark = True

    setup_logger(name=os.path.basename(args.config),
                 save_dir=os.path.join("checkpoints", os.path.basename(args.config).replace(".py", "")),
                 distributed_rank=0, mode='a+')
    main(args)