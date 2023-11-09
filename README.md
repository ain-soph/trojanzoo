# TrojanZoo
![logo](https://github.com/ain-soph/trojanzoo/raw/main/docs/source/images/trojanzoo-logo-readme.svg)

[![contact](https://img.shields.io/badge/contact-rbp5354@psu.edu-yellow)](mailto:rbp5354@psu.edu)
[![License](https://img.shields.io/github/license/ain-soph/trojanzoo)](https://opensource.org/licenses/GPL-3.0)

![python>=3.11](https://img.shields.io/badge/python->=3.11-informational.svg)
[![docs](https://github.com/ain-soph/trojanzoo/workflows/docs/badge.svg)](https://ain-soph.github.io/trojanzoo/)

[![release](https://img.shields.io/github/v/release/ain-soph/trojanzoo)](https://github.com/ain-soph/trojanzoo/releases)
[![pypi](https://img.shields.io/pypi/v/trojanzoo)](https://pypi.org/project/trojanzoo/)
[![docker](https://img.shields.io/pypi/v/trojanzoo?label=docker)](https://hub.docker.com/r/local0state/trojanzoo)
<!-- [![conda](https://img.shields.io/pypi/v/trojanzoo?label=conda)](https://anaconda.org/anaconda/trojanzoo) -->

> **NOTE:** TrojanZoo requires `python>=3.11`, `pytorch>=2.0.0` and `torchvision>=0.15.0`, which must be installed manually. Recommend to use `conda` to install.

This is the code implementation (pytorch) for our paper in EuroS&P 2022:  
[TrojanZoo: Towards Unified, Holistic, and Practical Evaluation of Neural Backdoors](https://arxiv.org/abs/2012.09302)

TrojanZoo provides a universal pytorch platform to conduct security researches (especially backdoor attacks/defenses) of image classification in deep learning. It is composed of two packages: `trojanzoo` and `trojanvision`. `trojanzoo` contains abstract classes and utilities, while `trojanvision` contains abstract and concrete ones for image classification task. 

> Note: This repository is also maintained to cover the implementation of  
> our kdd 2020 paper [AdvMind: Inferring Adversary Intent of Black-Box Attacks](https://arxiv.org/abs/2006.09539)  
> and ccs 2020 paper [A Tale of Evil Twins: Adversarial Inputs versus Poisoned Models](https://arxiv.org/abs/1911.01559)

## Documentation
We have documentation available at https://ain-soph.github.io/trojanzoo.

## Screenshot
![screenshot](https://github.com/ain-soph/trojanzoo/raw/main/docs/source/images/screenshot.png)


## Features
1. **Colorful and verbose output!**
   > Note: enable with `--color` for color and `--verbose` for verbose.  
   To open an interactive window with color, use `python - --color`
2. Modular design (plug and play)
3. Good code linting support
4. Register **your own module** to the library.
5. Native Pytorch Output  
   `trojanzoo` and `trojanvision` provides API to generate raw pytorch instances, which makes it flexible to work with native `pytorch` and other 3rd party libraries.
   > `trojanzoo.datasets.DataSet` can generate `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`  
   > `trojanzoo.models.Model` attribute `_model` is `torch.nn.Module`, attribute `model` is `torch.nn.DataParallel`  
   > Specifically, `trojanvision.datasets.ImageSet` can generate `torchvision.datasets.VisionDataset`, `trojanvision.datasets.ImageFolder` can generate `torchvision.datasets.ImageFolder`
6. Enable pytorch native AMP(Automatic Mixed Precision) with `--amp` for training
7. Flexible Configuration Files
8. Good help information to check arguments. (`-h` or `--help`)
9. Detailed and well-organized `summary()` for each module.

## Installation
1. `pip install trojanzoo`  
2. `pip install --upgrade git+https://github.com/ain-soph/trojanzoo.git`
3. **(HIGHLY RECOMMEND)**
   ```
   git clone https://github.com/ain-soph/trojanzoo
   pip install -e trojanzoo
   ```
    > This could install the github repo as a package but avoid copying files to `site_packages`,
      so that you can easily keep it updated by doing `git pull`.  
4. `docker pull local0state/trojanzoo` or `docker pull ghcr.io/ain-soph/trojanzoo`  

## Quick Start

You can use the provided [example](https://github.com/ain-soph/trojanzoo/tree/main/examples) scripts to reproduce the evaluation results in our paper.  
> Note: The program won't save results without `--save`  
1. Train a model:  
    e.g. `ResNet18` on `CIFAR10` with 95% Acc
    ```python3
    python ./examples/train.py --color --verbose 1 --dataset cifar10 --model resnet18_comp --lr_scheduler --cutout --grad_clip 5.0 --save
    ```

2. Test backdoor attack (e.g., BadNet):  
    e.g. `BadNet` with `ResNet18` on `CIFAR10`
    ```python3
    python ./examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack badnet --mark_random_init --epochs 50 --lr 0.01 --save
    ```

3. Test backdoor defense (e.g., Neural Cleanse):  
    e.g. `Neural Cleanse` against `BadNet`
    ```python3
    python ./examples/backdoor_defense.py --color --verbose 1 --pretrained --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack badnet --defense neural_cleanse --mark_random_init --epochs 50 --lr 0.01
    ```
## IMC
```python3
python ./examples/backdoor_attack.py --color --verbose 1 --pretrained --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack imc --mark_random_init --epochs 50 --lr 0.01 --save
```

## AdvMind
(with `attack adaptive` and `model adaptive`)
```python3
python ./examples/adv_defense.py --color --verbose 1 --pretrained --validate_interval 1 --dataset cifar10 --model resnet18_comp --attack pgd --defense advmind --attack_adapt --defense_adapt
```
## Detailed Usage
### Configuration file structure
All arguments in the parser are able to set default values in configuration files.  
If argument values are not set in the config files, we will use the default values of `__init__()`

Parameters Config: (priority ascend order)
> The higher priority config will override lower priority ones.  
> Within each priority channel, `trojanvision` configs will overwrite `trojanzoo`
1. Package Default: `/trojanzoo/configs/`, `/trojanvision/configs/`
   > These are package default settings. Please don't modify them.  
   > You can use this as a template to set other configs.
2. User Default:  `~/.trojanzoo/configs/trojanzoo/`, `~/.trojanzoo/configs/trojanvision/`
3. Workspace Default: `/configs/trojanzoo/`, `/configs/trojanvision/`
4. Custom Config: `--config [config location]`
5. CMD parameters: `--[parameter] [value]`

### Store path of Dataset, Model, Attack & Defense Results  
Modify them in corresponding config files and command-line arguments. 
> Dataset: `--data_dir` (`./data/data`)  
> Model: `--model_dir` (`./data/model`)  
> Attack: `--attack_dir` (`./data/attack`)  
> Defense: `--defense_dir` (`./data/defense`)  


### Output Verbose Information:
1. CMD modules: `--verbose 1`
2. **Colorful output**: `--color`
3. **tqdm**: `--tqdm`
4. Check command-line argument usage: `--help`
5. AdvMind verbose information: `--output [number]`

### Use your DIY Dataset/Model/Attack/Defense
1. Follow our example to write your DIY class. (`CIFAR10`, `ResNet`, `IMC`, `Neural Cleanse`)
   > It's necessary to subclass our base class. (`Dataset`, `Model`, `Attack`, `Defense`)  
   > Optional base classes depending on your use case: (`ImageSet`, `ImageFolder`, `ImageModel`)
2. Register your DIY class in `trojanvision`
   > Example: `trojanvision.attacks.class_dict[attack_name]=AttackClass`
3. Create your config files if necessary.  
   No need to modify any codes. Just directly add `{attack_name}.yml` (`.json`) in the config directory.
4. Good to go!

## Todo List
1. Sphinx Docs  
2. **Unit test**

## License
TrojanZoo has a GPL-style license, as found in the [LICENSE](https://github.com/ain-soph/trojanzoo/blob/main/LICENSE) file.
## Cite our paper
```
@InProceedings{pang:2022:eurosp,
      title={TrojanZoo: Towards Unified, Holistic, and Practical Evaluation of Neural Backdoors}, 
      author={Ren Pang and Zheng Zhang and Xiangshan Gao and Zhaohan Xi and Shouling Ji and Peng Cheng and Ting Wang},
      year={2022},
      booktitle={Proceedings of IEEE European Symposium on Security and Privacy (Euro S\&P)},
}

@inproceedings{pang:2020:ccs, 
    title = "{A Tale of Evil Twins: Adversarial Inputs versus Poisoned Models}", 
    author = {Ren Pang and Hua Shen and Xinyang Zhang and Shouling Ji and Yevgeniy Vorobeychik and Xiapu Luo and Alex Liu and Ting Wang}, 
    year = {2020}, 
    booktitle = {Proceedings of ACM SAC Conference on Computer and Communications (CCS)},
}

@inproceedings{pang:2020:kdd, 
    title = "{AdvMind: Inferring Adversary Intent of Black-Box Attacks}", 
    author = {Ren Pang and Xinyang Zhang and Shouling Ji and Xiapu Luo and Ting Wang}, 
    year = {2020}, 
    booktitle = {Proceedings of ACM International Conference on Knowledge Discovery and Data Mining (KDD)},
}
```
