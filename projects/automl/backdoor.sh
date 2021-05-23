declare -a archs=("enas" "pc_darts" "amoebanet" "snas_mild" "darts" "nasnet" "pdarts" "sgas" "drnas")

declare -a models=("mobilenet_v2_comp" "vgg13_bn_comp" "densenet_comp" "resnet18_comp" "wide_resnet50_2_comp" "resnext50_32x4d_comp" "dla34_comp")

attack="trojannn"

for model in "${models[@]}"
do
    echo $model
    python examples/backdoor_attack.py --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset cifar100 --model $model
done

for arch in "${archs[@]}"
do
    echo $arch
    python examples/backdoor_attack.py --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset cifar100 --model darts --model_arch $arch
done
