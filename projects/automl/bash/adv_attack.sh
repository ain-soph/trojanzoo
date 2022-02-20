# CUDA_VISIBLE_DEVICES=0 bash projects/automl/bash/adv_attack.sh > results/adv_pgd.txt 2>&1

declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "sgas" "snas_mild" "random")
declare -a models=("bit_comp" "densenet_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")

attack="pgd"
dataset="cifar10"
args=$1

for model in "${models[@]}"; do
    echo $model
    python examples/adv_attack.py --pretrained --attack $attack --dataset $dataset --model $model --pretrained --require_class $args
    echo ""
done

for arch in "${archs[@]}"; do
    echo $arch
    python examples/adv_attack.py --pretrained --attack $attack --dataset $dataset --model darts --model_arch $arch --pretrained --require_class $args
    echo ""
done
