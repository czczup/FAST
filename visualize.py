
from dataset.fast.fast_msra import msra_test_data_dir, msra_test_gt_dir
from dataset.fast.fast_ctw import ctw_test_data_dir, ctw_test_gt_dir
from dataset.fast.fast_tt import tt_test_data_dir, tt_test_gt_dir
from dataset.fast.fast_ic15 import ic15_test_data_dir, ic15_test_gt_dir
from dataset.utils import get_msra_ann, get_ctw_ann, get_tt_ann, get_ic15_ann, get_img

import cv2
import mmcv
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


msra_pred_dir = 'outputs/submit_msra/'
ctw_pred_dir = 'outputs/submit_ctw/'
tt_pred_dir = 'outputs/submit_tt/'
ic15_pred_dir = 'outputs/submit_ic15/'


def get_pred(pred_path):
    lines = mmcv.list_from_file(pred_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig').replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        bbox = [int(gt[i]) for i in range(len(gt))]
        bboxes.append(bbox)
        words.append('???')
    return np.array(bboxes), words


def draw(img, boxes, words):
    
    mask = np.zeros(img.shape, dtype=np.uint8)
    
    for box in boxes:
        rand_r = random.randint(100, 255)
        rand_g = random.randint(100, 255)
        rand_b = random.randint(100, 255)
        mask = cv2.fillPoly(mask, [box], color=(rand_r, rand_g, rand_b))
    
    img[mask!=0] = (0.6 * mask + 0.4 * img).astype(np.uint8)[mask!=0]
    
    for box, word in zip(boxes, words):
        if word == '###':
            cv2.drawContours(img, [box], -1, (255, 0, 0), thickness[args.dataset])
        else:
            cv2.drawContours(img, [box], -1, (0, 255, 0), thickness[args.dataset])

    return img
    
    
def visual(get_ann, data_dir, gt_dir, pred_dir, dataset):
    
    img_names = [img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')]
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.png')])
    img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.JPG')])
    
    img_paths, gt_paths, pred_paths = [], [], []
    
    for idx, img_name in enumerate(img_names):
        img_path = data_dir + img_name
        img_paths.append(img_path)
        
        # collect paths of ground truths and predictions
        if dataset == 'msra': # MSRA-TD500
            gt_name = img_name.split('.')[0] + '.gt'
            gt_path = gt_dir + gt_name
            gt_paths.append(gt_path)
            pred_name = img_name.split('.')[0] + '.txt'
            pred_path = pred_dir + pred_name
            pred_paths.append(pred_path)
        elif dataset == 'ctw': # CTW-1500
            gt_name = img_name.split('.')[0] + '.txt'
            gt_path = gt_dir + gt_name
            gt_paths.append(gt_path)
            pred_name = img_name.split('.')[0] + '.txt'
            pred_path = pred_dir + pred_name
            pred_paths.append(pred_path)
        elif dataset == 'tt': # Total-Text
            gt_name = 'poly_gt_' + img_name.split('.')[0] + '.mat'
            gt_path = gt_dir + gt_name
            gt_paths.append(gt_path)
            pred_name = img_name.split('.')[0] + '.txt'
            pred_path = pred_dir + pred_name
            pred_paths.append(pred_path)
        elif dataset == 'ic15': # ICDAR 2015
            gt_name = 'gt_' + img_name.split('.')[0] + '.txt'
            gt_path = gt_dir + gt_name
            gt_paths.append(gt_path)
            pred_name = "res_" + img_name.split('.')[0] + '.txt'
            pred_path = pred_dir + pred_name
            pred_paths.append(pred_path)
            
    for index, (img_path, gt_path, pred_path) in tqdm(enumerate(zip(img_paths, gt_paths, pred_paths)), total=len(img_paths)):
        img = get_img(img_path) # load image
        gt, word = get_ann(img, gt_path) # load annotation

        try: # process annotations
            if dataset == 'msra':
                gt = np.reshape(gt * ([img.shape[1], img.shape[0]] * 4), (gt.shape[0], -1, 2)).astype('int32')
            elif dataset == 'ctw':
                for i in range(len(gt)):
                    gt[i] = np.reshape(gt[i] * ([img.shape[1], img.shape[0]] * (gt[i].shape[0] // 2)),
                                           (gt[i].shape[0] // 2, 2)).astype('int32')
            elif dataset == 'tt':
                for i in range(len(gt)):
                    gt[i] = np.reshape(gt[i] * ([img.shape[1], img.shape[0]] * (gt[i].shape[0] // 2)),
                                           (gt[i].shape[0] // 2, 2)).astype('int32')
            elif dataset == 'ic15':
                new_gt = gt.tolist()
                for i in range(len(gt)):
                    new_gt[i] = np.reshape(gt[i] * ([img.shape[1], img.shape[0]] * (gt[i].shape[0] // 2)),
                                           (gt[i].shape[0] // 2, 2)).astype('int32')
                gt = new_gt
        except Exception as e:
            print(e)
        
        # load predictions
        pred, _ = get_pred(pred_path)
        if dataset == 'msra': # process predictions
            if pred.shape[0] > 0:
                pred = np.reshape(pred, (pred.shape[0], -1, 2)).astype('int32')
        elif dataset == 'ctw':
            pred = pred.tolist()
            for i in range(len(pred)):
                pred[i] = np.reshape(np.flipud(pred[i]), (-1, 2)).astype('int32')
        elif dataset == 'tt':
            pred = pred.tolist()
            for i in range(len(pred)):
                pred[i] = np.reshape(np.flipud(pred[i]), (-1, 2)).astype('int32')
        elif dataset == 'ic15':
            pred = pred.tolist()
            for i in range(len(pred)):
                pred[i] = np.reshape(pred[i], (-1, 2)).astype('int32')
                
        img_ = img.copy()
        img_pred = draw(img, pred, _) # draw predictions on images
        img_gt = draw(img_, gt, word) # draw ground truths on images
        img = np.hstack((img_gt, img_pred)) # stack two images
        img = Image.fromarray(img)
        mmcv.mkdir_or_exist(f"visual/{dataset}")
        img.save(f"visual/{dataset}/{index}.png") # save images into visual/
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', nargs='?', type=str, required=True,
                        choices=['tt', 'ctw', 'msra', 'ic15'])
    parser.add_argument('--show-gt', action="store_true")
    # show the ground truths with predictions
    args = parser.parse_args()
    
    # thickness for different datasets
    thickness = {'msra': 12, 'ctw':4, 'tt':4, 'ic15': 4}
    
    if args.dataset == 'msra':
        get_ann = get_msra_ann
        test_data_dir = msra_test_data_dir
        test_gt_dir = msra_test_gt_dir
        pred_dir = msra_pred_dir
    elif args.dataset == 'ctw':
        get_ann = get_ctw_ann
        test_data_dir = ctw_test_data_dir
        test_gt_dir = ctw_test_gt_dir
        pred_dir = ctw_pred_dir
    elif args.dataset == 'tt':
        get_ann = get_tt_ann
        test_data_dir = tt_test_data_dir
        test_gt_dir = tt_test_gt_dir
        pred_dir = tt_pred_dir
    elif args.dataset == 'ic15':
        get_ann = get_ic15_ann
        test_data_dir = ic15_test_data_dir
        test_gt_dir = ic15_test_gt_dir
        pred_dir = ic15_pred_dir
        
    print(test_data_dir)
    visual(get_ann, test_data_dir, test_gt_dir, pred_dir, args.dataset)