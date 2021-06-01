declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "snas_mild" "sgas" "random")
declare -a models=("bit_comp" "densenet121_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")

attack="trojannn"

for model in "${models[@]}"; do
    echo $model
    python examples/backdoor_attack.py --mark_alpha 0.7 --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset imagenet32 --num_classes 60 --model $model
done

for arch in "${archs[@]}"; do
    echo $arch
    python examples/backdoor_attack.py --mark_alpha 0.7 --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset imagenet32 --num_classes 60 --model darts --model_arch $arch
done
