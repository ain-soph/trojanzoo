declare -a archs=("enas" "pc_darts" "amoebanet" "snas_mild" "darts" "nasnet" "pdarts" "sgas" "drnas")

declare -a models=("mobilenet_v2_comp" "vgg13_bn_comp" "densenet_comp" "resnet18_comp" "wide_resnet50_2_comp" "resnext50_32x4d_comp" "dla34_comp" "proxylessnas")

for model in "${models[@]}"
do
    echo $model
    python examples/membership_distance.py  --dataset cifar10 --pretrain --model $model
done

for arch in "${archs[@]}"
do
    echo $arch
    python examples/membership_distance.py  --dataset cifar10 --pretrain --model darts --model_arch $arch
done
