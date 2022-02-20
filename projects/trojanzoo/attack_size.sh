# CUDA_VISIBLE_DEVICES=0 bash projects/trojanzoo/attack_size.sh 2 > results/attack_2.txt 2>&1
# CUDA_VISIBLE_DEVICES=1 bash projects/trojanzoo/attack_size.sh 3 > results/attack_3.txt 2>&1
# CUDA_VISIBLE_DEVICES=2 bash projects/trojanzoo/attack_size.sh 4 > results/attack_4.txt 2>&1
# CUDA_VISIBLE_DEVICES=3 bash projects/trojanzoo/attack_size.sh 5 > results/attack_5.txt 2>&1

declare -a attacks=("badnet" "trojannn" "reflection_backdoor" "badnet --mark_random_pos" "latent_backdoor" "trojannet" "bypass_embed" "imc")

mark_size=$1
mark_alpha=0.8

for attack in "${attacks[@]}"; do
    echo $attack
    python examples/backdoor_attack.py --attack $attack --pretrained --validate_interval 1 --lr_scheduler --epoch 50 --lr 1e-2 --mark_alpha $mark_alpha --mark_height $mark_size --mark_width $mark_size --lr_scheduler --lr_scheduler_type StepLR --lr_step_size 10 --batch_size 96
done
