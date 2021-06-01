declare -a archs=("diy_deep" "diy_noskip" "diy_deep_noskip")

attack="trojannn"
dataset="cifar10"

for arch in "${archs[@]}"; do
    echo $arch
    python examples/backdoor_attack.py --mark_alpha 0.7 --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset $dataset --model darts --model_arch $arch
done
