# Transformer Knowledge Distillation for Efficient Semantic Segmentation [[arxiv](https://arxiv.org/abs/2202.13393)]
## Structure: TransKD
![TransKD](TransKDs.png)
## Introduction
We propose the structural framework, TransKD, to distill the knowledge from feature maps and patch embeddings of vision transformers.
## Requirements
Environment: create a conda environment and activate it
```
conda create -n TransKD python=3.6
conda activate TransKD
```
Additional python pachages: [poly scheduler](https://github.com/cmpark0126/pytorch-polynomial-lr-decay) and
```
pytorch == 1.7.1+cu92
torchvision == 0.8.2+cu92
mmsegmentation == 0.15.0
mmcv-full == 1.3.10
numpy
visdom
```
Datasets:
* Cityscapes: download `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` from [cityscapes official website](https://www.cityscapes-dataset.com/downloads/), then prepare the 19-class label with the `createTrainIdLabelImgs.py` from [cityscapesscripts](https://github.com/mcordts/cityscapesScripts/tree/master/cityscapesscripts/preparation).

## Usage
download [teacher checkpoints](https://1drv.ms/u/s!AlFXMOI-DJJhn3qvs5TOQlaWbbVr?e=ohlhOU) in the folder `checkpoints/`.

Example:
```
python train/train_transkd.py --datadir /path/to/data --kdtype TransKD-Base
```

## Publication
If you find this repo useful, please consider referencing the following paper [[PDF](https://arxiv.org/pdf/2202.13393)]:
```
@article{liu2022transkd,
  title={TransKD: Transformer Knowledge Distillation for Efficient Semantic Segmentation},
  author={Liu, Ruiping and Yang, Kailun and Roitberg, Alina and Zhang, Jiaming and Peng, Kunyu and Liu, Huayao and Stiefelhagen, Rainer},
  journal={arXiv preprint arXiv:2202.13393},
  year={2022}
}
```
