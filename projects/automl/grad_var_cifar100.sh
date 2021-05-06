declare -a archs=("enas" "darts" "pc_darts" "amoebanet" "snas_mild" "nasnet" "pdarts" "sgas" "drnas")
declare -a models=("mobilenet_v2_comp" "vgg13_bn_comp" "resnet18_comp" "densenet_comp" "wide_resnet50_2_comp" "resnext50_32x4d_comp" "dla34_comp" "bit_comp" "proxylessnas")

for model in "${models[@]}"
do
    echo $model
    python projects/automl/grad_var.py --pretrain --dataset cifar100 --model $model
done

for arch in "${archs[@]}"
do
    echo $arch
    python projects/automl/grad_var.py --pretrain --dataset cifar100 --model darts --model_arch $arch
done
