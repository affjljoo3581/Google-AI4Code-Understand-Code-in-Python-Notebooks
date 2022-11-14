# Google AI4Code – Understand Code in Python Notebooks

## Introduction
This repository contains the code that acheived 39th place in [Google AI4Code – Understand Code in Python Notebooks](https://www.kaggle.com/competitions/AI4Code/overview).

## Requirements
* numpy
* omegaconf
* pandas
* pytorch_lightning
* scikit_learn
* torch
* tqdm
* fasttext
* sentencepiece
* transformers
* wandb

Instead of installing the above modules independently, you can simply do at once by using:
```bash
$ pip install -f requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

This repository supports [NVIDIA Apex](https://github.com/NVIDIA/apex). It will automatically detect the apex module and if it is found then some training procedures will be replaced with the highly-optimized and fused operations in the apex module. Run the below codes in the terminal to install apex and enable performance boosting:

```bash
$ git clone https://github.com/NVIDIA/apex
$ sed -i "s/or (bare_metal_minor != torch_binary_minor)//g" apex/setup.py
$ pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/
$ rm -rf apex
```

Instead, we recommend to use docker and [PyTorch NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) where apex, optimized cuda driver and faster pytorch kernel are installed:
```bash
$ docker run --gpus all -it nvcr.io/nvidia/pytorch:22.07-py3
```