import numpy as np
from PIL import Image, ImageFilter
from torch.utils import data
import cv2
import torchvision.transforms as transforms
import torch
import scipy.io as scio
import torch.nn as nn
import random

from dataset.utils import shrink
from dataset.utils import get_img
from dataset.utils import get_synth_ann as get_ann
from dataset.utils import random_scale, random_horizontal_flip, random_rotate
from dataset.utils import random_crop_padding_v2 as random_crop_padding
from dataset.utils import update_word_mask, get_vocabulary

synth_root_dir = './data/SynthText/'
synth_train_data_dir = synth_root_dir
synth_train_gt_path = synth_root_dir + 'gt.mat'


class FAST_Synth(data.Dataset):
    def __init__(self, is_transform=False, img_size=None, short_size=640,
                 pooling_size=9, with_rec=False, read_type='pil'):
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.pooling_size = pooling_size
        self.short_size = short_size
        self.with_rec = with_rec
        self.read_type = read_type
        
        self.pad = nn.ZeroPad2d(padding=(pooling_size - 1) // 2)
        self.pooling = nn.MaxPool2d(kernel_size=pooling_size, stride=1)
        self.overlap_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        data = scio.loadmat(synth_train_gt_path)

        self.img_paths = data['imnames'][0]
        self.gts = data['wordBB'][0]
        self.texts = data['txt'][0]

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
    
    def __getitem__(self, index):
        img_path = synth_train_data_dir + self.img_paths[index][0]
        img = get_img(img_path, read_type=self.read_type)
        bboxes, words = get_ann(img, self.gts, self.texts, index)

        if bboxes.shape[0] > self.max_word_num:
            bboxes = bboxes[:self.max_word_num]
            words = words[:self.max_word_num]

        gt_words = np.full((self.max_word_num, self.max_word_len), self.char2id['PAD'], dtype=np.int32)
        word_mask = np.zeros((self.max_word_num,), dtype=np.int32)
        for i, word in enumerate(words):
            if word == '###':
                continue
            word = word.lower()
            gt_word = np.full((self.max_word_len,), self.char2id['PAD'], dtype=np.int)
            for j, char in enumerate(word):
                if j > self.max_word_len - 1:
                    break
                if char in self.char2id:
                    gt_word[j] = self.char2id[char]
                else:
                    gt_word[j] = self.char2id['UNK']
            if len(word) > self.max_word_len - 1:
                gt_word[-1] = self.char2id['EOS']
            else:
                gt_word[len(word)] = self.char2id['EOS']
            gt_words[i + 1] = gt_word
            word_mask[i + 1] = 1

        if self.is_transform:
            img = random_scale(img, self.short_size, scales=[0.5, 2.0], aspects=[0.9, 1.1])

        gt_instance = np.zeros(img.shape[0:2], dtype='uint8')
        training_mask = np.ones(img.shape[0:2], dtype='uint8')
        if bboxes.shape[0] > 0:
            bboxes = np.reshape(bboxes * ([img.shape[1], img.shape[0]] * 4),
                                (bboxes.shape[0], -1, 2)).astype('int32')
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_instance, [bboxes[i]], -1, i + 1, -1)
                if words[i] == '###':
                    cv2.drawContours(training_mask, [bboxes[i]], -1, 0, -1)

        gt_kernels = []
        for i in range(len(bboxes)):
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            if words[i] != '###':
                cv2.drawContours(gt_kernel, [bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)
        gt_kernels = np.array(gt_kernels)  # [instance_num, h, w]
        gt_kernel = self.min_pooling(gt_kernels)

        shrink_kernel_scale = 0.1
        gt_kernel_shrinked = np.zeros(img.shape[0:2], dtype='uint8')
        kernel_bboxes = shrink(bboxes, shrink_kernel_scale)
        for i in range(bboxes.shape[0]):
            cv2.drawContours(gt_kernel_shrinked, [kernel_bboxes[i]], -1, 1, -1)
        gt_kernel = np.maximum(gt_kernel, gt_kernel_shrinked)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask, gt_kernel]
    
            if not self.with_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs, random_angle=30)
            gt_instance_before_crop = imgs[1].copy()
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernel = imgs[0], imgs[1], imgs[2], imgs[3]
            word_mask = update_word_mask(gt_instance, gt_instance_before_crop, word_mask)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1

        img = Image.fromarray(img)
        img = img.convert('RGB')

        if self.is_transform:
            if random.random() < 0.5:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernel = torch.from_numpy(gt_kernel).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernel,
            training_masks=training_mask,
            gt_instances=gt_instance,
        )
        if self.with_rec:
            data.update(dict(
                gt_words=gt_words,
                word_masks=word_mask
            ))

        return data
