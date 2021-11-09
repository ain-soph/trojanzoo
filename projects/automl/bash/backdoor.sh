# CUDA_VISIBLE_DEVICES=0 bash projects/automl/bash/backdoor.sh > results/backdoor.txt 2>&1

declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "sgas" "snas_mild" "random")
declare -a models=("bit_comp" "densenet_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")

attack="badnet"
dataset="cifar10"
args=$1

for model in "${models[@]}"; do
    echo $model
    python examples/backdoor_attack.py --poison_percent 0.02 --pretrain --mark_alpha 0.0 --epoch 20 --batch_size 96 --lr 0.01 --validate_interval 1 --attack $attack --dataset $dataset --model $model $args
done

for arch in "${archs[@]}"; do
    echo $arch
    python examples/backdoor_attack.py --poison_percent 0.02 --pretrain --mark_alpha 0.0 --epoch 20 --batch_size 96 --lr 0.01 --validate_interval 1 --attack $attack --dataset $dataset --model darts --model_arch $arch $args
done
