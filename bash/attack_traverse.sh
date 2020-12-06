work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='sample_imagenet'
model='resnetcomp18'

CUDA_VISIBLE_DEVICES=$1

dirname=${work_dir}/result/${dataset}/${model}/traverse
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

alpha=8
size=7
for attack in 'badnet' 'trojannn' 'latent_backdoor' 'reflection_backdoor' 'bypass_embed' 'imc'
do
    echo $attack
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_attack.py \
    --dataset $dataset --model $model --attack $attack --mark_alpha 0.$alpha --height $size --width $size \
    --lr 1e-2 --epoch 50 --lr_scheduler --step_size 10 --validate_interval 1 --pretrain --amp \
    --percent 0.01 --verbose --save \
    > $dirname/${attack}.txt 2>&1
done

attack='badnet'
echo $attack
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_attack.py --random_pos \
--dataset $dataset --model $model --attack $attack --mark_alpha 0.$alpha --height $size --width $size \
--lr 1e-2 --epoch 50 --lr_scheduler --step_size 10 --validate_interval 1 --pretrain --amp \
--percent 0.01 --verbose --save \
> $dirname/targeted.txt 2>&1
