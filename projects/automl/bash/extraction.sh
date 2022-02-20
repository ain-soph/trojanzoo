# CUDA_VISIBLE_DEVICES=0 bash projects/automl/bash/extraction.sh > results/extraction.txt 2>&1

declare -a rates=("1000" "2000" "4000" "8000" "16000")
declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "sgas" "snas_mild" "random")
declare -a models=("bit_comp" "densenet_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")

dataset="cifar10"
args=$1

for model in "${models[@]}"; do
    echo $model
    for rate in "${rates[@]}"; do
        echo $rate
        python projects/automl/extraction_attack.py --pretrained --dataset $dataset --model $model --nb_stolen $rate $args
    done
done

for arch in "${archs[@]}"; do
    echo $arch
    for rate in "${rates[@]}"; do
        echo $rate
        python projects/automl/extraction_attack.py --pretrained --dataset $dataset --model darts --model_arch $arch --nb_stolen $rate $args
    done
done
