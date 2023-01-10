import numpy as np
from PIL import Image
from torch.utils import data
import cv2
import torchvision.transforms as transforms
import torch
import mmcv
import torch.nn as nn

from dataset.utils import shrink
from dataset.utils import get_img
from dataset.utils import get_ctw_ann as get_ann
from dataset.utils import random_scale, random_horizontal_flip, random_rotate
from dataset.utils import random_crop_padding_v2 as random_crop_padding
from dataset.utils import update_word_mask, get_vocabulary
from dataset.utils import scale_aligned_short, scale_aligned_long

ctw_root_dir = './data/CTW1500/'
ctw_train_data_dir = ctw_root_dir + 'train/text_image/'
ctw_train_gt_dir = ctw_root_dir + 'train/text_label_curve/'
ctw_test_data_dir = ctw_root_dir + 'test/text_image/'
ctw_test_gt_dir = ctw_root_dir + 'test/text_label_circum/'


class FAST_CTW(data.Dataset):
    def __init__(self, split='train', is_transform=False, img_size=None, short_size=640,
                 pooling_size=9, with_rec=False, read_type='pil', repeat_times=1, report_speed=False, long_side=None):
        self.split = split
        self.is_transform = is_transform
        
        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.pooling_size = pooling_size
        self.short_size = short_size
        self.with_rec = with_rec
        self.read_type = read_type
        self.long_side = long_side
        
        self.pad = nn.ZeroPad2d(padding=(pooling_size - 1) // 2)
        self.pooling = nn.MaxPool2d(kernel_size=pooling_size, stride=1)
        self.overlap_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        if split == 'train':
            data_dirs = [ctw_train_data_dir] * repeat_times
            gt_dirs = [ctw_train_gt_dir] * repeat_times
        elif split == 'test':
            data_dirs = [ctw_test_data_dir]
            gt_dirs = [ctw_test_gt_dir]
        else:
            print('Error: split must be train or test!')
            raise
        
        self.img_paths = []
        self.gt_paths = []
        
        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names = [img_name for img_name in mmcv.utils.scandir(data_dir, '.jpg')]
            img_names.extend([img_name for img_name in mmcv.utils.scandir(data_dir, '.png')])
            
            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)
                
                gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)
            
            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)
        
        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = 200
        self.max_word_len = 32
    
    def __len__(self):
        return len(self.img_paths)
    
    def min_pooling(self, input):
        input = torch.tensor(input, dtype=torch.float)
        temp = input.sum(dim=0).to(torch.uint8)
        overlap = (temp > 1).to(torch.float32).unsqueeze(0).unsqueeze(0)
        overlap = self.overlap_pool(overlap).squeeze(0).squeeze(0)
        
        B = input.size(0)
        h_sum = input.sum(dim=2) > 0
        
        h_sum_ = h_sum.long() * torch.arange(h_sum.shape[1], 0, -1)
        h_min = torch.argmax(h_sum_, 1, keepdim=True)
        h_sum_ = h_sum.long() * torch.arange(1, h_sum.shape[1] + 1)
        h_max = torch.argmax(h_sum_, 1, keepdim=True)
        
        w_sum = input.sum(dim=1) > 0
        w_sum_ = w_sum.long() * torch.arange(w_sum.shape[1], 0, -1)
        w_min = torch.argmax(w_sum_, 1, keepdim=True)
        w_sum_ = w_sum.long() * torch.arange(1, w_sum.shape[1] + 1)
        w_max = torch.argmax(w_sum_, 1, keepdim=True)
        
        for i in range(B):
            region = input[i:i + 1, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1]
            region = self.pad(region)
            region = -self.pooling(-region)
            input[i:i + 1, h_min[i]:h_max[i] + 1, w_min[i]:w_max[i] + 1] = region
        
        x = input.sum(dim=0).to(torch.uint8)
        x[overlap > 0] = 0  # overlapping regions
        return x.numpy()
    
    def prepare_train_data(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]
        
        img = get_img(img_path, self.read_type)
        bboxes, words = get_ann(img, gt_path)
        
        if len(bboxes) > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
        
        if self.is_transform:
            img = random_scale(img, self.short_size, scales=[0.7, 1.3], aspects=None)
        
        if self.long_side is not None:
            long_side = max(img.shape[0:2])
            if long_side > self.long_side:
                img = scale_aligned_long(img, self.long_side)
        
        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if len(bboxes) > 0:
            for i in range(len(bboxes)):
                bboxes[i] = np.reshape(bboxes[i] * ([img.shape[1], img.shape[0]] * (bboxes[i].shape[0] // 2)),
                                       (bboxes[i].shape[0] // 2, 2)).astype('int32')
            for i in range(len(bboxes)):
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)
                else:
                    cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
        
        gt_kernels = []
        for i in range(len(bboxes)):
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            if words[i] != '###':
                cv2.drawContours(gt_kernel, [bboxes[i]], -1, 1, -1)
                gt_kernels.append(gt_kernel)
            else:
                if len(gt_kernels) == 0:
                    gt_kernels.append(gt_kernel)
        gt_kernels = np.array(gt_kernels)  # [instance_num, h, w]
        gt_kernel = self.min_pooling(gt_kernels)
        
        shrink_kernel_scale = 0.1
        gt_kernel_shrinked = np.zeros(img.shape[0:2], dtype='uint8')
        kernel_bboxes = shrink(bboxes, shrink_kernel_scale)
        for i in range(len(bboxes)):
            if words[i] != '###':
                cv2.drawContours(gt_kernel_shrinked, [kernel_bboxes[i]], -1, 1, -1)
        gt_kernel = np.maximum(gt_kernel, gt_kernel_shrinked)
        
        if self.is_transform:
            imgs = [img, gt_instance, training_mask, gt_kernel]
            imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs, random_angle=10)
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernel = imgs[0], imgs[1], imgs[2], imgs[3]
        
        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        
        img = Image.fromarray(img)
        img = img.convert('RGB')
        
        if self.is_transform:
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)
        
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        gt_text = torch.from_numpy(gt_text).long()
        gt_kernel = torch.from_numpy(gt_kernel).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        
        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernel,
            training_masks=training_mask,
            gt_instances=gt_instance,
        )
        
        return data
    
    def prepare_test_data(self, index):
        img_path = self.img_paths[index]
        filename = img_path.split('/')[-1][:-4]
        img = get_img(img_path, self.read_type)
        img_meta = dict(
            org_img_size=np.array(img.shape[:2])
        )
        img = scale_aligned_short(img, self.short_size)
        if self.long_side is not None:
            long_side = max(img.shape[0:2])
            if long_side > self.long_side:
                img = scale_aligned_long(img, self.long_side)
        
        img_meta.update(dict(
            img_size=np.array(img.shape[:2]),
            filename=filename
        ))
        
        img = Image.fromarray(img)
        img = img.convert('RGB')
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        
        data = dict(
            imgs=img,
            img_metas=img_meta
        )
        
        return data
    
    def __getitem__(self, index):
        if self.split == 'train':
            return self.prepare_train_data(index)
        elif self.split == 'test':
            return self.prepare_test_data(index)