# CUDA_VISIBLE_DEVICES=0 python backdoor_defense.py --defense tabor --attack badnet --mark_alpha 0.0 --height 3 --width 3

work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
defense=$2

CUDA_VISIBLE_DEVICES=$1

dirname=${work_dir}/result/${dataset}/${model}/${defense}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

size=3
alpha=0.0


for attack in 'badnet' 'latent_backdoor' 'trojannn' 'imc' 'reflection_backdoor' 'bypass_embed' 'trojannet'
do
    echo $attack
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py --verbose \
    --dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size --pretrain \
    > $dirname/${attack}.txt 2>&1
done

attack='badnet'
echo "targeted"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py --verbose \
--dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size \
--random_pos \
> $dirname/targeted.txt 2>&1

attack='badnet'
echo "clean"
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py --verbose \
--dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size --original --pretrain \
> $dirname/clean.txt 2>&1