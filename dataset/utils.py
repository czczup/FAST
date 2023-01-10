from PIL import Image
import scipy.io as scio
import Polygon as plg
import numpy as np
import pyclipper
import random
import string
import mmcv
import math
import cv2


def get_img(img_path, read_type='pil'):
    try:
        if read_type == 'cv2':
            img = cv2.imread(img_path)
            img = img[:, :, [2, 1, 0]]
        elif read_type == 'pil':
            img = np.array(Image.open(img_path))
    except Exception as e:
        print(img_path)
        raise
    return img


def get_ctw_ann_old(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        bboxes.append(bbox)
        words.append('???')
    return bboxes, words

def get_ctw_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1 * 1.0, y1 * 1.0] * 14)
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * 14)

        bboxes.append(bbox)
        words.append('???')
    return bboxes, words

def get_ic15_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        word = gt[8].replace('\r', '').replace('\n', '')
        if word[0] == '#':
            words.append('###')
        else:
            words.append(word)

        bbox = [int(gt[i]) for i in range(8)]
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words


def get_msra_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')

        gt = line.split(' ')

        w_ = np.float(gt[4])
        h_ = np.float(gt[5])
        x1 = np.float(gt[2]) + w_ / 2.0
        y1 = np.float(gt[3]) + h_ / 2.0
        theta = np.float(gt[6]) / math.pi * 180

        bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
        bbox = bbox.reshape(-1) / ([w * 1.0, h * 1.0] * 4)

        bboxes.append(bbox)
        words.append('???')
    return np.array(bboxes), words


def get_synth_ann(img, gts, texts, index):
    bboxes = np.array(gts[index])
    bboxes = np.reshape(bboxes, (bboxes.shape[0], bboxes.shape[1], -1))
    bboxes = bboxes.transpose(2, 1, 0)
    bboxes = np.reshape(bboxes, (bboxes.shape[0], -1)) / ([img.shape[1], img.shape[0]] * 4)

    words = []
    for text in texts[index]:
        text = text.replace('\n', ' ').replace('\r', ' ')
        words.extend([w for w in text.split(' ') if len(w) > 0])

    return bboxes, words


def read_mat_lindes(path):
    f = scio.loadmat(path)
    return f


def get_tt_ann(img, gt_path):
    h, w = img.shape[0:2]
    bboxes = []
    words = []
    data = read_mat_lindes(gt_path)
    data_polygt = data['polygt']
    for i, lines in enumerate(data_polygt):
        X = np.array(lines[1])
        Y = np.array(lines[3])

        point_num = len(X[0])
        word = lines[4]
        if len(word) == 0:
            word = '???'
        else:
            word = word[0]
            # word = word[0].encode("utf-8")

        if word == '#':
            word = '###'

        words.append(word)

        arr = np.concatenate([X, Y]).T
        bbox = []
        for i in range(point_num):
            bbox.append(arr[i][0])
            bbox.append(arr[i][1])
        bbox = np.asarray(bbox) / ([w * 1.0, h * 1.0] * point_num)
        bboxes.append(bbox)

    return bboxes, words


def get_ic17mlt_ann(img, gt_path):
    h, w = img.shape[0:2]
    lines = mmcv.list_from_file(gt_path)
    bboxes = []
    words = []
    for line in lines:
        line = line.encode('utf-8').decode('utf-8-sig')
        line = line.replace('\xef\xbb\xbf', '')
        gt = line.split(',')
        word = gt[9].replace('\r', '').replace('\n', '')
        if len(word) == 0 or word[0] == '#':
            words.append('###')
        else:
            words.append(word)

        bbox = [int(gt[i]) for i in range(8)]
        bbox = np.array(bbox) / ([w * 1.0, h * 1.0] * 4)
        bboxes.append(bbox)
    return np.array(bboxes), words


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs, random_angle=10):
    max_angle = random_angle
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w), flags=cv2.INTER_NEAREST)
        imgs[i] = img_rotation
    return imgs


def scale_aligned(img, h_scale, w_scale):
    h, w = img.shape[0:2]
    h = int(h * h_scale + 0.5)
    w = int(w * w_scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(img, short_size=736, scales=[0.5, 2.0], aspects=[0.9, 1.1]):
    h, w = img.shape[0:2]
    scale = np.random.uniform(scales[0], scales[1])
    scale = (scale * short_size) / min(h, w)
    if aspects is not None:
        aspect = np.random.uniform(aspects[0], aspects[1])
        h_scale = scale * math.sqrt(aspect)
        w_scale = scale / math.sqrt(aspect)
        img = scale_aligned(img, h_scale, w_scale)
    else:
        img = scale_aligned(img, scale, scale)

    return img


def scale_aligned_short(img, short_size=640):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def scale_aligned_long(img, long_size=640):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img

def random_crop_padding_v1(imgs, target_size):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    if w == t_w and h == t_h:
        return imgs

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(imgs[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(imgs[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def random_crop_padding_v2(imgs, target_size, max_tries=20):
    h, w = imgs[0].shape[0:2]
    t_w, t_h = target_size
    p_w, p_h = target_size
    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w
    gt_instance = imgs[1]
    h_array = (gt_instance.sum(axis=1) > 0).astype(np.int32)
    w_array = (gt_instance.sum(axis=0) > 0).astype(np.int32)

    # ensure the cropped area not across a text
    final_i, final_j = 0, 0
    if h > t_h:  # 图像的高度比要裁剪的大
        h_axis = np.where(h_array[:h - t_h] == 0)[0]
    if w > t_w:  # 图像的宽度比要裁剪的大
        w_axis = np.where(w_array[:w - t_w] == 0)[0]
    for idx in range(max_tries):
        try:
            i = np.random.choice(h_axis, size=1)[0] if h > t_h else 0
            j = np.random.choice(w_axis, size=1)[0] if w > t_w else 0
            flag = gt_instance[i:i + t_h, j:j + t_w].max()
            if flag > 0:  # 切分的图像中有字
                final_i, final_j = i, j
                if h_array[i + t_h - 1] or w_array[j + t_w - 1]: continue  # 如果字被切穿了, 继续搜索
                else: break
        except:
            continue
    i, j = final_i, final_j

    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            s3_length = int(imgs[idx].shape[-1])
            img = imgs[idx][i:i + t_h, j:j + t_w, :]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                       value=tuple(0 for i in range(s3_length)))
        else:
            img = imgs[idx][i:i + t_h, j:j + t_w]
            img_p = cv2.copyMakeBorder(img, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT, value=(0,))
        n_imgs.append(img_p)
    return n_imgs


def center_crop(imgs):
    h, w = imgs[0].shape[0:2]
    short_side = min(h, w)
    if short_side == h:  # 高是短边
        i, j = 0, (w - h) // 2
    else:
        i, j = (h - w) // 2, 0
    n_imgs = []
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            img = imgs[idx][i:i + short_side, j:j + short_side, :]
        else:
            img = imgs[idx][i:i + short_side, j:j + short_side]
        n_imgs.append(img)
    return n_imgs


def dist(a, b):
    return np.linalg.norm((a - b), ord=2, axis=0)


def perimeter(bbox):
    peri = 0.0
    for i in range(bbox.shape[0]):
        peri += dist(bbox[i], bbox[(i + 1) % bbox.shape[0]])
    return peri


def shrink(bboxes, rate, max_shr=20):
    rate = rate * rate
    shrinked_bboxes = []
    for bbox in bboxes:
        area = plg.Polygon(bbox).area()
        peri = perimeter(bbox)

        try:
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(bbox, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            offset = min(int(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            shrinked_bbox = pco.Execute(-offset)
            if len(shrinked_bbox) == 0:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bbox = np.array(shrinked_bbox[0])
            if shrinked_bbox.shape[0] <= 2:
                shrinked_bboxes.append(bbox)
                continue

            shrinked_bboxes.append(shrinked_bbox)
        except Exception as e:
            print(type(shrinked_bbox), shrinked_bbox)
            print('area:', area, 'peri:', peri)
            shrinked_bboxes.append(bbox)

    return shrinked_bboxes


def update_word_mask(instance, instance_before_crop, word_mask):
    labels = np.unique(instance)

    for label in labels:
        if label == 0:
            continue
        ind = instance == label
        if np.sum(ind) == 0:
            word_mask[label] = 0
            continue
        ind_before_crop = instance_before_crop == label
        # print(np.sum(ind), np.sum(ind_before_crop))
        if float(np.sum(ind)) / np.sum(ind_before_crop) > 0.9:
            continue
        word_mask[label] = 0

    return word_mask


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-5])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char