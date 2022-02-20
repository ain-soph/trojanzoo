# CUDA_VISIBLE_DEVICES=0 bash projects/automl/bash/mitigation_backdoor.sh > results/mitigation_backdoor.txt 2>&1

declare -a archs=("diy_deep" "diy_noskip" "diy_deep_noskip")

dataset="cifar10"
args=$1

attack="trojannn"
dataset="cifar10"

for arch in "${archs[@]}"; do
    echo $arch
    python examples/backdoor_attack.py --mark_alpha 0.7 --epochs 20 --batch_size 96 --lr 0.01 --pretrained --save --validate_interval 1 --attack $attack --dataset $dataset --model darts --model_arch $arch $args
done
