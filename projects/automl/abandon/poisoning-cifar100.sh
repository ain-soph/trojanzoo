declare -a archs=("enas" "darts" "pc_darts" "amoebanet" "snas_mild" "nasnet" "pdarts" "sgas" "drnas")
declare -a models=("mobilenet_v2_comp" "vgg13_bn_comp" "resnet50_comp" "densenet_comp" "wide_resnet50_2_comp" "resnext50_32x4d_comp" "dla34_comp" "bit_comp" "proxylessnas")
declare -a rates=("0.0" "0.025" "0.05" "0.1" "0.2" "0.4")

for model in "${models[@]}"
do
    echo $model
    for rate in "${rates[@]}"
    do 
        echo $rate
        python examples/adv_attack.py --verbose 1 --epoch 50 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --validate_interval 1 --attack poison_random --dataset cifar100 --poison_percent $rate --model $model --train_mode dataset --save
    done  
done

for arch in "${archs[@]}"
do
    echo $arch
    for rate in "${rates[@]}"
    do
        echo $rate
        python examples/adv_attack.py --verbose 1 --epoch 50 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --validate_interval 1 --attack poison_random --dataset cifar100 --poison_percent $rate --model darts --model_arch $arch --train_mode dataset --save
    done
done
