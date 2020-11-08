# CUDA_VISIBLE_DEVICES=0 python backdoor_defense.py --defense tabor --attack badnet --mark_alpha 0.0 --height 3 --width 3
set -e

work_dir='/home/rbp5354/trojanzoo'
save_dir='/home/zxz147/git_clones/Trojan-Zoo/log/defense_grouped'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
defense=$2

CUDA_VISIBLE_DEVICES=$1

dirname=${save_dir}/$defense
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

size_ary=(3 16)
alpha_ary=(0 8)

for size in ${size_ary[@]}
do
    for alpha in ${alpha_ary[@]}
    do
        for attack in 'badnet' 'latent_backdoor' 'trojannn' 'imc' 'reflection_backdoor' 'bypass_embed' 'trojannet'
        do
            echo "Settings: "
            echo "Size - [${size}], Alpha - [${alpha}], Attack - [${attack}], Defense - [${defense}]"
            CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py --verbose \
            --dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha 0.$alpha --height $size --width $size --pretrain \
            2>&1 | tee $dirname/${attack}_s${size}_a${alpha}.txt
        done
    done
done

#attack='badnet'
#echo "targeted"
#CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py --verbose \
#--dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size \
#--random_pos \
#> $dirname/targeted.txt 2>&1
#
#attack='badnet'
#echo "clean"
#CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py --verbose \
#--dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size --original --pretrain \
#> $dirname/clean.txt 2>&1