# FAST
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-searching-for-a-faster-arbitrarily/scene-text-detection-on-total-text)](https://paperswithcode.com/sota/scene-text-detection-on-total-text?p=fast-searching-for-a-faster-arbitrarily)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-searching-for-a-faster-arbitrarily/scene-text-detection-on-msra-td500)](https://paperswithcode.com/sota/scene-text-detection-on-msra-td500?p=fast-searching-for-a-faster-arbitrarily)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-searching-for-a-faster-arbitrarily/scene-text-detection-on-scut-ctw1500)](https://paperswithcode.com/sota/scene-text-detection-on-scut-ctw1500?p=fast-searching-for-a-faster-arbitrarily)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/fast-searching-for-a-faster-arbitrarily/scene-text-detection-on-icdar-2015)](https://paperswithcode.com/sota/scene-text-detection-on-icdar-2015?p=fast-searching-for-a-faster-arbitrarily)


This repository is an official implementation of the [FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation](https://arxiv.org/pdf/2111.02394).




<details open>
<summary>Text Detection</summary>

- [x] [PSENet (CVPR'2019)](config/psenet/)
- [x] [PAN (ICCV'2019)](config/pan/)
- [x] [FAST (Arxiv'2021)](config/fast/)

</details>


## News
- `Mar 11, 2024:`🚀🚀 FAST has been integrated into the [docTR](https://github.com/mindee/doctr), a seamless, high-performing & accessible library for OCR-related tasks.
- `Jan 10, 2023:`🚀 Code and models are released.
- `Dec 06, 2022:` Code and models of FAST will be released in this repository.


## Zero-shot Video Text Detection Demo

https://user-images.githubusercontent.com/23737120/212238686-64a68149-1c09-44cb-ba26-e1dfb609db9e.mp4





## Catalog

- [x] TensorRT implementation
- [x] Code and models
- [x] Initialization

## Abstract
We propose an accurate and efficient scene text detection framework, termed FAST (i.e., **F**aster **A**rbitrarily-**S**haped **T**ext detector).
Different from recent advanced text detectors that used complicated post-processing and hand-crafted network architectures, resulting in low inference speed, FAST has two new designs. (1) We design a minimalist kernel representation (only has 1-channel output) to model text with arbitrary shape, as well as a GPU-parallel post-processing to efficiently assemble text lines with a negligible time overhead. (2) We search the network architecture
tailored for text detection, leading to more powerful features than most networks that are searched for image classification. Benefiting from these two designs, FAST achieves an excellent trade-off between accuracy and efficiency on several challenging datasets, including Total Text, CTW1500, ICDAR 2015, and MSRA-TD500. For example, FAST-T yields 81.6\% F-measure at 152 FPS on Total-Text, outperforming the previous fastest method by 1.7 points and 70 FPS in terms of accuracy and speed. With TensorRT optimization, the inference speed can be further accelerated to over 600 FPS.

## Method
<img width="1382" alt="image" src="https://user-images.githubusercontent.com/23737120/206380932-c226d94e-0c07-4ffe-94fe-07e65efa6068.png">

## Usage

### Installation

First, clone the repository locally:

```shell
git clone https://github.com/czczup/FAST
```

Then, install PyTorch 1.1.0+, torchvision 0.3.0+, and other requirements:

```shell
# for python3 (training and testing)
pip install editdistance
pip install Polygon3
pip install pyclipper
pip install Cython
pip install mmcv
pip install prefetch_generator
pip install scipy
pip install yacs
pip install tqdm
pip install opencv-python==4.6.0.66

# for python2 (evaluation)
# the evaluation code is from pan_pp.pytorch
pip2 install numpy==1.10
pip2 install scipy==1.2.2
pip2 install polygon2
```

Finally, compile codes of post-processing:

```shell
# build pse, pa, and ccl algorithms
sh ./compile.sh
```

### Dataset
Please refer to [dataset/README.md](dataset/README.md) for dataset preparation.

### Training
First, please download the pretrained checkpoints:
```shell
mkdir pretrained/
cd pretrained/
wget https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic17mlt_640.pth
wget https://github.com/czczup/FAST/releases/download/release/fast_small_ic17mlt_640.pth
wget https://github.com/czczup/FAST/releases/download/release/fast_base_ic17mlt_640.pth
cd ../
```
Then, run the following command for training:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py <config>
```
For example:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py config/fast/tt/fast_base_tt_800_finetune_ic17mlt.py
```

### Testing

#### Evaluate single checkpoint

```shell
python test.py <config> <checkpoint> --ema
cd eval/
./eval_{DATASET}.sh
```
For example:
```shell
python test.py config/fast/tt/fast_base_tt_800_finetune_ic17mlt.py download/fast_base_tt_800_finetune_ic17mlt.pth --ema
cd eval/
sh eval_tt.sh
```
It should give:
```
Precision:_0.900048239267_______/Recall:_0.851633393829/Hmean:_0.875171745978
```


#### Evaluate all checkpoints in one folder
```shell
python test_all.py <config> <checkpoint-dir> --dataset [{tt/ctw/ic15/msra}] --start-ep 1 --end-ep 60 --ema
```

#### Evaluate the speed

```shell
python test.py <config> --report-speed
```
For example:
```shell
python test.py config/fast/tt/fast_base_tt_800_finetune_ic17mlt.py --report-speed
```

#### Visulization

Run the following script to visulize the prediction results:
```shell
python visualize.py --dataset [{tt/ctw/ic15/msra}] --show-gt
```
- This script will load the predictions in `outputs/` and plot them on images.

- The visulized results will be saved in `visual/`.

- Left is the ground truth and right is the prediction.

![visulization](https://user-images.githubusercontent.com/23737120/212108547-5d509a9f-75f9-46cd-b8f9-bf04be725520.png)



## Model Zoo


**IC17-MLT Pretrained FAST Models**

| Model | Backbone | Pretrain | Resolution | #Params | Config| Download |
| :---: |  :---: | :---: | :---: | :---: |  :---: | :---: | 
| FAST-T | TextNet-T | [ImageNet-1K](https://github.com/czczup/FAST/releases/download/release/fast_tiny_in1k_epoch_299.pth)  | 640x640 | 8.5M  | [config](config/fast/ic17mlt/fast_tiny_ic17mlt_640.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic17mlt_640.pth) \| [log](logs/ic17mlt/fast_tiny_ic17mlt_640.txt)   |
| FAST-S | TextNet-S | [ImageNet-1K](https://github.com/czczup/FAST/releases/download/release/fast_small_in1k_epoch_299.pth) | 640x640 | 9.7M  | [config](config/fast/ic17mlt/fast_small_ic17mlt_640.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_ic17mlt_640.pth) \| [log](logs/ic17mlt/fast_small_ic17mlt_640.txt) |
| FAST-B | TextNet-B | [ImageNet-1K](https://github.com/czczup/FAST/releases/download/release/fast_base_in1k_epoch_299.pth)  | 640x640 | 10.6M | [config](config/fast/ic17mlt/fast_base_ic17mlt_640.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic17mlt_640.pth) \| [log](logs/ic17mlt/fast_base_ic17mlt_640.txt)   |
| FAST-T | TextNet-T | -  | 640x640 | 8.5M  | - | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic17mlt_640_wo_imagenet.pth)   |
| FAST-S | TextNet-S | -  | 640x640 | 9.7M  | - | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_ic17mlt_640_wo_imagenet.pth)  |
| FAST-B | TextNet-B | -  | 640x640 | 10.6M | - | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic17mlt_640_wo_imagenet.pth)   |

- We provide the IC17-MLT pretrained weights with and without ImageNet pretraining.

**Results on Total-Text**

| Method | Backbone | Precision | Recall | F-measure | FPS | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FAST-T-448 |TextNet-T |   86.5   |  77.2  |  81.6  | 152.8 | [config](config/fast/tt/fast_tiny_tt_448_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_tt_448_finetune_ic17mlt.pth) \| [log](logs/tt/fast_tiny_tt_448_finetune_ic17mlt.txt)     |
| FAST-T-512 |TextNet-T |   87.3   |  80.0  |  83.5  | 131.1 | [config](config/fast/tt/fast_tiny_tt_512_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_tt_512_finetune_ic17mlt.pth) \| [log](logs/tt/fast_tiny_tt_512_finetune_ic17mlt.txt)     |
| FAST-T-640 |TextNet-T |   87.1   |  81.4  |  84.2  | 95.5  | [config](config/fast/tt/fast_tiny_tt_640_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_tt_640_finetune_ic17mlt.pth) \| [log](logs/tt/fast_tiny_tt_640_finetune_ic17mlt.txt)     |
| FAST-S-512 |TextNet-S |   88.3   |  81.7  |  84.9  | 115.5 | [config](config/fast/tt/fast_small_tt_512_finetune_ic17mlt.py) |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_tt_512_finetune_ic17mlt.pth) \| [log](logs/tt/fast_small_tt_512_finetune_ic17mlt.txt)   |
| FAST-S-640 |TextNet-S |   89.1   |  81.9  |  85.4  | 85.3  | [config](config/fast/tt/fast_small_tt_640_finetune_ic17mlt.py) |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_tt_640_finetune_ic17mlt.pth) \| [log](logs/tt/fast_small_tt_640_finetune_ic17mlt.txt)   |
| FAST-B-512 |TextNet-B |   89.6   |  82.4  |  85.8  | 93.2  | [config](config/fast/tt/fast_base_tt_512_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_tt_512_finetune_ic17mlt.pth) \| [log](logs/tt/fast_base_tt_512_finetune_ic17mlt.txt)     |
| FAST-B-640 |TextNet-B |   89.9   |  83.2  |  86.4  | 67.5  | [config](config/fast/tt/fast_base_tt_640_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_tt_640_finetune_ic17mlt.pth) \| [log](logs/tt/fast_base_tt_640_finetune_ic17mlt.txt)     |
| FAST-B-800 |TextNet-B |   90.0   |  85.2  |  87.5  | 46.0  | [config](config/fast/tt/fast_base_tt_800_finetune_ic17mlt.py)  |    [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_tt_800_finetune_ic17mlt.pth) \| [log](logs/tt/fast_base_tt_800_finetune_ic17mlt.txt)     |

**Results on CTW1500**

| Method | Backbone | Precision | Recall | F-measure | FPS | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FAST-T-512 | TextNet-T  |  85.5  |  77.9  |  81.5   | 129.1 | [config](config/fast/ctw/fast_tiny_ctw_512_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_ctw_512_finetune_ic17mlt.pth) \| [log](logs/ctw/fast_tiny_ctw_512_finetune_ic17mlt.txt) |
| FAST-S-512 | TextNet-S  |  85.6  | 78.7 | 82.0  | 112.9  | [config](config/fast/ctw/fast_small_ctw_512_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_ctw_512_finetune_ic17mlt.pth) \| [log](logs/ctw/fast_small_ctw_512_finetune_ic17mlt.txt) |
| FAST-B-512 | TextNet-B  |  85.7  | 80.2 | 82.9  | 92.6  | [config](config/fast/ctw/fast_base_ctw_512_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ctw_512_finetune_ic17mlt.pth) \| [log](logs/ctw/fast_base_ctw_512_finetune_ic17mlt.txt) |
| FAST-B-640 | TextNet-B  |  87.8  | 80.9 | 84.2  | 66.5  | [config](config/fast/ctw/fast_base_ctw_640_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ctw_640_finetune_ic17mlt.pth) \| [log](logs/ctw/fast_base_ctw_640_finetune_ic17mlt.txt) |

**Results on ICDAR 2015**

| Method | Backbone | Precision  | Recall  | F-measure  | FPS | Config | Download |
| :-: | :-: |:-: | :-: | :-: | :-: | :-: | :-: |
| FAST-T-736  | TextNet-T  |    86.0       |   77.9   |    81.7    | 60.9 | [config](config/fast/ic15/fast_tiny_ic15_736_finetune_ic17mlt.py)  | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic15_736_finetune_ic17mlt.pth) \| [log](logs/ic15/fast_tiny_ic15_736_finetune_ic17mlt.txt)   |
| FAST-S-736  | TextNet-S  |    86.3       |   79.8   |    82.9    | 53.9 | [config](config/fast/ic15/fast_small_ic15_736_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_ic15_736_finetune_ic17mlt.pth) \| [log](logs/ic15/fast_small_ic15_736_finetune_ic17mlt.txt) |
| FAST-B-736  | TextNet-B  |    88.0       |   81.7   |    84.7    | 42.7 | [config](config/fast/ic15/fast_base_ic15_736_finetune_ic17mlt.py)  | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic15_736_finetune_ic17mlt.pth) \| [log](logs/ic15/fast_base_ic15_736_finetune_ic17mlt.txt)   |
| FAST-B-896  | TextNet-B  |    89.2       |   83.6   |    86.3    | 31.8 | [config](config/fast/ic15/fast_base_ic15_896_finetune_ic17mlt.py)  | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic15_896_finetune_ic17mlt.pth) \| [log](logs/ic15/fast_base_ic15_896_finetune_ic17mlt.txt)   |
| FAST-B-1280 | TextNet-B  |    89.7       |   84.6   |    87.1    | 15.7 | [config](config/fast/ic15/fast_base_ic15_1280_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_ic15_1280_finetune_ic17mlt.pth) \| [log](logs/ic15/fast_base_ic15_1280_finetune_ic17mlt.txt) |


**Results on MSRA-TD500**

| Method | Backbone | Precision | Recall | F-measure | FPS | Config | Download |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| FAST-T-512 | TextNet-T  |   91.1    |  78.8  |  84.5   | 137.2 | [config](config/fast/msra/fast_tiny_msra_512_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_msra_512_finetune_ic17mlt.pth) \| [log](logs/msra/fast_tiny_msra_512_finetune_ic17mlt.txt) |
| FAST-T-736 | TextNet-T  |   88.1    |  81.9  |  84.9    | 79.6  | [config](config/fast/msra/fast_tiny_msra_736_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_tiny_msra_736_finetune_ic17mlt.pth) \| [log](logs/msra/fast_tiny_msra_736_finetune_ic17mlt.txt) |
| FAST-S-736 | TextNet-S  |   91.6    |  81.7  |  86.4   | 72.0  | [config](config/fast/msra/fast_small_msra_736_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_small_msra_736_finetune_ic17mlt.pth) \| [log](logs/msra/fast_small_msra_736_finetune_ic17mlt.txt) |
| FAST-B-736 | TextNet-B  |   92.1    |  83.0  |  87.3   | 56.8  | [config](config/fast/msra/fast_base_msra_736_finetune_ic17mlt.py) | [ckpt](https://github.com/czczup/FAST/releases/download/release/fast_base_msra_736_finetune_ic17mlt.pth) \| [log](logs/msra/fast_base_msra_736_finetune_ic17mlt.txt) |



## Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@misc{chen2021fast,
  title={FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation}, 
  author={Zhe Chen and Jiahao Wang and Wenhai Wang and Guo Chen and Enze Xie and Ping Luo and Tong Lu},
  year={2021},
  eprint={2111.02394},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
