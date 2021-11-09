# CUDA_VISIBLE_DEVICES=0 bash projects/automl/bash/train.sh > results/train.txt 2>&1

declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "sgas" "snas_mild" "random")
declare -a models=("bit_comp" "densenet_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")

dataset="cifar10"
args=$1

for model in "${models[@]}"; do
    echo $model
    python examples/train.py --verbose 1 --epoch 200 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --save --dataset $dataset --model $model $args
    echo ""
done

for arch in "${archs[@]}"; do
    echo $arch
    python examples/train.py --verbose 1 --epoch 200 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --save --dataset $dataset --model darts --model_arch $arch $args
    echo ""
done
