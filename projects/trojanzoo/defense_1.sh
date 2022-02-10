# CUDA_VISIBLE_DEVICES=1 bash projects/trojanzoo/defense_1.sh > results/defense_1.txt 2>&1

declare -a attacks=("badnet" "trojannn" "reflection_backdoor" "badnet --random_pos" "latent_backdoor" "trojannet" "bypass_embed" "imc")
declare -a defenses=("neural_cleanse" "deep_inspect" "neuron_inspect" "tabor" "abs")

mark_size=3
mark_alpha=0.0

for defense in "${defenses[@]}"; do
    for attack in "${attacks[@]}"; do
        echo $defense $attack
        python examples/backdoor_defense.py --attack $attack --defense $defense --pretrain --validate_interval 1 --epoch 50 --lr 1e-2 --mark_alpha $mark_alpha --mark_height $mark_size --mark_width $mark_size
    done
done
