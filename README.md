# Trojan-Zoo

This is the code implementation (pytorch) for our paper: 
[TROJANZOO: Everything you ever wanted to know about neural backdoors (but were afraid to ask)](https://arxiv.org/abs/2012.09302)

> Note: This repository is also maintained to cover the implementation of  
> our kdd 2020 paper [AdvMind: Inferring Adversary Intent of Black-Box Attacks](https://arxiv.org/abs/2006.09539)  
> and ccs 2020 paper [A Tale of Evil Twins: Adversarial Inputs versus Poisoned Models](https://arxiv.org/abs/1911.01559).

## Screenshot

Add the screen shot

## Quick Start:

> Note: The program won't save results without `--save`

1. Train a model:  
    e.g. `ResNetComp18` for `CIFAR10` with 95% Acc
    ```python3
    python train.py --verbose 1 --amp --dataset cifar10 --model resnetcomp18 --epoch 150 --lr 0.1 --lr_scheduler --lr_decay_step 50 --save
    ```

2. Test backdoor attack (e.g., BadNet):  
    e.g. `BadNet`
    ```python3
    python backdoor_attack.py --verbose 1 --pretrain --validate_interval 1 --amp --dataset cifar10 --model resnetcomp18 --attack badnet --random_init --epoch 50 --lr 0.01 --save
    ```

3. Test backdoor defense (e.g., Neural Cleanse):  
    e.g. `Neural Cleanse` for `BadNet`
    ```python3
    python backdoor_attack.py --verbose 1 --pretrain --validate_interval 1 --dataset cifar10 --model resnetcomp18 --attack badnet --defense neural_cleanse --random_init --epoch 50 --lr 0.01
    ```
## IMC
```python3
python backdoor_attack.py --verbose --pretrain --validate_interval 1 --amp --dataset cifar10 --model resnetcomp18 --attack imc --random_init --epoch 50 --lr 0.01 --save
```

## AdvMind
(with `attack adaptive` and `model adaptive`)
```python3
python adv_defense.py --verbose --pretrain --validate_interval 1 --dataset cifar10 --model resnetcomp18 --attack pgd --defense advmind --attack_adapt --defense_adapt
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
2. User Default: Not decided yet. 
   > (Enable it in the code `trojanzoo/configs/__init__.py`, `trojanvision/configs/__init__.py`)
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
1. CMD modules: `--verbose`
2. **Colorful output**: `--color`
3. **tqdm progress bar**: `--tqdm`
4. Check command-line argument usage: `--help`
5. AdvMind verbose information: `--output [number]`

### Use your DIY Dataset/Model/Attack/Defense
1. Follow our example to write your DIY class. (`CIFAR10`, `ResNet`, `IMC`, `Neural Cleanse`)
   > It's necessary to subset our base class. (`Dataset`, `Model`, `Attack`, `Defense`)  
   > Optional base classes depending on your use case: (`ImageSet`, `ImageFolder`, `ImageModel`)
2. Register your DIY class in `trojanvision`
   > Example: `trojanvision.attacks.class_dict[attack_name]=AttackClass`
3. Create your config files if necessary.  
   No need to modify any codes. Just directly add `{attack_name}.yml` in the config directory.
4. Good to go!

## Todo List

1. Travis-CI support
   1. Build Test
   2. Pypi package release
   3. Docker Hub publish
2. Sphinx Docs
3. Test and find bugs
## Cite our paper:
```
@InProceedings{pang2020trojanzoo,
      title={TROJANZOO: Everything you ever wanted to know about neural backdoors (but were afraid to ask)}, 
      author={Ren Pang and Zheng Zhang and Xiangshan Gao and Zhaohan Xi and Shouling Ji and Peng Cheng and Ting Wang},
      year={2020},
      booktitle={arXiv Preprint},
}
```