declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "snas_mild" "sgas" "random")
declare -a models=("bit_comp" "densenet121_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")
declare -a rates=("0.0" "0.025" "0.05" "0.1" "0.2" "0.4")
# "mobilenet_v2_comp"

for model in "${models[@]}"; do
    echo $model
    for rate in "${rates[@]}"; do
        echo $rate
        python examples/adv_attack.py --verbose 1 --epoch 50 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --validate_interval 1 --save \
            --attack poison_random --dataset imagenet32 --num_classes 60 --poison_percent $rate --model $model --train_mode dataset
    done
done

for arch in "${archs[@]}"; do
    echo $arch
    for rate in "${rates[@]}"; do
        echo $rate
        python examples/adv_attack.py --verbose 1 --epoch 50 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --validate_interval 1 --save \
            --attack poison_random --dataset imagenet32 --num_classes 60 --poison_percent $rate --model darts --model_arch $arch --train_mode dataset
    done
done
