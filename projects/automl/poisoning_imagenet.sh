declare -a rates=("0.0" "0.025" "0.05" "0.1" "0.2" "0.4")
dataset="imagenet16"
model="darts"

for rate in "${rates[@]}"; do
    echo $rate
    python examples/adv_attack.py --epoch 50 --batch_size 96 --cutout --grad_clip 5.0 --lr 0.025 --lr_scheduler --save --attack poison_random --train_mode dataset --poison_percent $rate --dataset $dataset --num_classes 120 --model $model
done
