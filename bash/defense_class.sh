work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
defense=$2
attack=$3

CUDA_VISIBLE_DEVICES=$1

dirname=${work_dir}/result/${dataset}/${model}/${defense}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

ext=''

name=${attack}

if [ $attack = targeted_backdoor ];then
    attack=badnet
    ext=' --random_pos'
fi

if [ $attack = clean_label ];then
    ext=' --poison_generation_method pgd'
fi


size=3
alpha=0.0

for target_class in {0..9}
do
    echo $target_class
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py \
    --dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size $ext\
    --target_class $target_class \
    > $dirname/${name}_class_$target_class.txt 2>&1
done