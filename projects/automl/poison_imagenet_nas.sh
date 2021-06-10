declare -a archs=("sgas")
declare -a rates=("0.05" "0.1" "0.2" "0.4")

for arch in "${archs[@]}"; do
    echo $arch
    for rate in "${rates[@]}"; do
        echo $rate
        python examples/adv_attack.py --verbose 1 --epoch 50 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --validate_interval 1 --save \
            --attack poison_random --dataset imagenet32 --num_classes 60 --poison_percent $rate --model darts --model_arch $arch --train_mode dataset
    done
done
