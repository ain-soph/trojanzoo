work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack='clean_label'
poison_generation_method=$2

CUDA_VISIBLE_DEVICES=$1

dirname=${work_dir}/result/${dataset}/${model}/${attack}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

alpha=0
size=3
for target_class in {1..9}
do
    echo $target_class
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_attack.py \
    --dataset $dataset --model $model --attack $attack --mark_alpha 0.$alpha --height $size --width $size \
    --lr 1e-2 --epoch 50 --lr_scheduler--step_size 10 --validate_interval 1 --pretrain --amp \
    --percent 0.1 --random_init --verbose --save \
    --poison_generation_method $poison_generation_method \
    --target_class $target_class \
    > $dirname/${poison_generation_method}_class_${target_class}.txt 2>&1
done
