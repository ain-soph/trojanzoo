# CUDA_VISIBLE_DEVICES=0 bash projects/automl/bash/neuron_number.sh > results/neuron_number.txt 2>&1

declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "sgas" "snas_mild" "random")
declare -a models=("bit_comp" "densenet_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")
declare -a neuron_numbers=("1" "2" "4" "8")

attack="trojannn"
dataset="cifar10"
args=$1

for model in "${models[@]}"; do
    echo $model
    for neuron_number in "${neuron_numbers[@]}"; do
        echo $neuron_number
        python examples/backdoor_attack.py --pretrained --epochs 20 --batch_size 96 --lr 0.01 --validate_interval 1 --attack $attack --dataset $dataset --model $model $args
    done
done

for arch in "${archs[@]}"; do
    echo $arch
    for neuron_number in "${neuron_numbers[@]}"; do
        echo $neuron_number
        python examples/backdoor_attack.py --pretrained --epochs 20 --batch_size 96 --lr 0.01 --validate_interval 1 --attack $attack --dataset $dataset --model darts --model_arch $arch $args
    done
done
