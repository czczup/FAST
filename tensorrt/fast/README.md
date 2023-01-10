# FAST

FAST models from
"FAST: Faster Arbitrarily-Shaped Text Detector with Minimalist Kernel Representation" <https://arxiv.org/pdf/2111.02394>

# How to run

1. generate wts file.

You may convert a trained model into the inference-time structure with

```
python reparameter.py [weights file of the training-time model to load] [path to save]
```

For example,

```
cd ../../
python reparameter.py config/fast/tt/fast_base_tt_800_finetune_ic17mlt.py \
    pretrained/fast_base_tt_800_finetune_ic17mlt.pth \
    tensorrt/fast/fast-base-deploy.pth
cd tensorrt/fast/
```

Then run `gen_wts.py` to generate .wts file, for example

```
python gen_wts.py -w fast-base-deploy.pth -s fast-base-deploy.wts
```

2. build and run
```
mkdir build

cd build

cmake ..

make

sudo ./fast -s fast-base-deploy  // serialize model to plan file
sudo ./fast -d fast-base-deploy  // deserialize plan file and run inference
```

