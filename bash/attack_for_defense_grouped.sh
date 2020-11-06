set -e

work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack=$2

CUDA_VISIBLE_DEVICES=$1

dirname=${work_dir}/result/${dataset}/${model}/${attack}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

size_ary=(6)
alpha_ary=(0 8)

for size in ${size_ary[@]}
do
    for alpha in ${alpha_ary[@]}
    do
        echo 0.$alpha
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_attack.py \
        --dataset $dataset --model $model --attack $attack --mark_alpha 0.$alpha --height $size --width $size \
        --lr 1e-2 --epoch 50 --lr_scheduler --step_size 10 --validate_interval 1 --pretrain \
        --percent 0.01 --verbose --save \
        > $dirname/alpha0.${alpha}.txt 2>&1
    done
done