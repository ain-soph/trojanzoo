# CUDA_VISIBLE_DEVICES=0 bash projects/automl/bash/grad_var.sh > results/grad_var.txt 2>&1

declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "sgas" "snas_mild")
declare -a models=("bit_comp" "densenet_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")

dataset="cifar10"
args=$1

for model in "${models[@]}"
do
    python projects/automl/grad_var.py --dataset $dataset --model $model $args
done

for arch in "${archs[@]}"
do
    python projects/automl/grad_var.py --dataset $dataset --model darts --model_arch $arch $args
done
