declare -a archs=("enas" "darts" "pc_darts" "amoebanet" "snas_mild" "nasnet" "pdarts" "sgas" "drnas")
declare -a models=("mobilenet_v2_comp" "resnet50_comp" "densenet161_comp" "wide_resnet50_2_comp" "resnext50_32x4d_comp" "dla60_comp" "bit_comp")

attack="trojannn"

for model in "${models[@]}"
do
    echo $model
    python examples/backdoor_attack.py --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset imagenet16 --num_classes 120 --model $model
done

for arch in "${archs[@]}"
do
    echo $arch
    python examples/backdoor_attack.py --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset imagenet16 --num_classes 120 --model darts --model_arch $arch
done
