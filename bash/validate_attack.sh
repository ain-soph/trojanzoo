work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'

CUDA_VISIBLE_DEVICES=$1

size=3
alpha=8
for attack in 'badnet' 'trojannn' 'reflection_backdoor' 'latent_backdoor' 'trojannet' 'bypass_embed' 'imc'; do
    echo $attack
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/validate_backdoor.py \
    --dataset $dataset --model $model \
    --attack $attack --mark_alpha 0.$alpha --height $size --width $size
done

attack='badnet'
echo $attack
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/validate_backdoor.py \
--dataset $dataset --model $model \
--attack $attack --random_pos --mark_alpha 0.$alpha --height $size --width $size
