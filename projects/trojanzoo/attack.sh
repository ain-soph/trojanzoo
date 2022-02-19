# CUDA_VISIBLE_DEVICES=0 bash projects/trojanzoo/attack.sh > results/attack.txt 2>&1

declare -a attacks=("badnet" "trojannn" "reflection_backdoor" "badnet --mark_random_pos" "latent_backdoor" "trojannet" "bypass_embed" "imc")

mark_size=3
mark_alpha=0.0

for attack in "${attacks[@]}"; do
    echo $attack
    python examples/backdoor_attack.py --attack $attack --pretrain --validate_interval 1 --lr_scheduler --epoch 50 --lr 1e-2 --mark_alpha $mark_alpha --mark_height $mark_size --mark_width $mark_size --lr_scheduler --lr_scheduler_type StepLR --lr_step_size 10 --save
done
