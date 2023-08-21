# k-Mixup-CIFAR10
By Kristjan Greenewald, Anming Gu, Mikhail Yurochkin, Justin Solomon, Edward Chien.


## Introduction

K-mixup is a generic and straightforward data augmentation principle that extends the popular mixup regularization technique. In essence, k-mixup uses optimal transport to match k-batches of training points with other k-batches of traning points. K-mixup further improves generalization and robustness over standard mixup. 

This repository contains the implementation used for the results in
our paper (https://arxiv.org/abs/2106.02933).

## Citation

If you use this method or this code in your paper, then please cite it:

```
@article{
anonymous2023kmixup,
title={\$k\$-Mixup Regularization for Deep Learning via Optimal Transport},
author={Anonymous},
journal={Submitted to Transactions on Machine Learning Research},
year={2023},
url={https://openreview.net/forum?id=lOegPKSu04},
note={Under review}
}
```

## Requirements and Installation
* A computer running macOS or Linux
* For training new models, you'll also need a NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* Python version 3.6
* A [PyTorch installation](http://pytorch.org/)

## Training
Use `python train.py` to train a new model.
Here is an example setting:
```
$ python train.py --lr=0.1 --alpha=10.0 --mixupBatch=16
```

## License

This project is CC-BY-NC-licensed.

## Acknowledgement
The CIFAR-10 reimplementation of _k-mixup_ is adapted from the [mixup-cifar10](https://github.com/facebookresearch/mixup-cifar10/tree/main) repository by [facebookresearch](https://github.com/facebookresearch).