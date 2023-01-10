import torchvision.transforms as transforms
from torch.utils import data
import scipy.io as scio
from PIL import Image
import numpy as np
import torch
import cv2

from dataset.utils import shrink
from dataset.utils import get_img
from dataset.utils import get_synth_ann as get_ann
from dataset.utils import random_scale, random_horizontal_flip, random_rotate
from dataset.utils import random_crop_padding_v1 as random_crop_padding
from dataset.utils import update_word_mask, get_vocabulary

synth_root_dir = './data/SynthText/'
synth_train_data_dir = synth_root_dir
synth_train_gt_path = synth_root_dir + 'gt.mat'


class PAN_Synth(data.Dataset):
    def __init__(self, is_transform=False, img_size=None, short_size=736,
                 kernel_scale=0.5, with_rec=False, read_type='pil'):
        self.is_transform = is_transform

        self.img_size = img_size if (img_size is None or isinstance(img_size, tuple)) else (img_size, img_size)
        self.kernel_scale = kernel_scale
        self.short_size = short_size
        self.with_rec = with_rec
        self.read_type = read_type

        data = scio.loadmat(synth_train_gt_path)

        self.img_paths = data['imnames'][0]
        self.gts = data['wordBB'][0]
        self.texts = data['txt'][0]

        self.voc, self.char2id, self.id2char = get_vocabulary('LOWERCASE')
        self.max_word_num = 200
        self.max_word_len = 32

    def __len__(self):
        return len(self.img_paths)

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
            img = random_scale(img, self.short_size, scales=[0.7, 1.3], aspects=[0.9, 1.1])

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
        for rate in [self.kernel_scale]:
            gt_kernel = np.zeros(img.shape[0:2], dtype='uint8')
            kernel_bboxes = shrink(bboxes, rate)
            for i in range(bboxes.shape[0]):
                cv2.drawContours(gt_kernel, [kernel_bboxes[i]], -1, 1, -1)
            gt_kernels.append(gt_kernel)

        if self.is_transform:
            imgs = [img, gt_instance, training_mask]
            imgs.extend(gt_kernels)

            if not self.with_rec:
                imgs = random_horizontal_flip(imgs)
            imgs = random_rotate(imgs, random_angle=10)
            gt_instance_before_crop = imgs[1].copy()
            imgs = random_crop_padding(imgs, self.img_size)
            img, gt_instance, training_mask, gt_kernels = imgs[0], imgs[1], imgs[2], imgs[3:]
            word_mask = update_word_mask(gt_instance, gt_instance_before_crop, word_mask)

        gt_text = gt_instance.copy()
        gt_text[gt_text > 0] = 1
        gt_kernels = np.array(gt_kernels)

        max_instance = np.max(gt_instance)
        gt_bboxes = np.zeros((self.max_word_num, 4), dtype=np.int32)
        for i in range(1, max_instance + 1):
            ind = gt_instance == i
            if np.sum(ind) == 0:
                continue
            points = np.array(np.where(ind)).transpose((1, 0))
            tl = np.min(points, axis=0)
            br = np.max(points, axis=0) + 1
            gt_bboxes[i] = (tl[0], tl[1], br[0], br[1])

        img = Image.fromarray(img)
        img = img.convert('RGB')

        if self.is_transform:
            img = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(img)

        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)

        gt_text = torch.from_numpy(gt_text).long()
        gt_kernels = torch.from_numpy(gt_kernels).long()
        training_mask = torch.from_numpy(training_mask).long()
        gt_instance = torch.from_numpy(gt_instance).long()
        gt_bboxes = torch.from_numpy(gt_bboxes).long()
        gt_words = torch.from_numpy(gt_words).long()
        word_mask = torch.from_numpy(word_mask).long()

        data = dict(
            imgs=img,
            gt_texts=gt_text,
            gt_kernels=gt_kernels,
            training_masks=training_mask,
            gt_instances=gt_instance,
            gt_bboxes=gt_bboxes,
        )
        if self.with_rec:
            data.update(dict(
                gt_words=gt_words,
                word_masks=word_mask
            ))

        return data
