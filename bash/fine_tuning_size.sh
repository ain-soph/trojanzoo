work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack='badnet'
parameters=$1

CUDA_VISIBLE_DEVICES=0

dirname=${work_dir}/result/${dataset}/${model}/${attack}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

alpha=0.0
for size in {1..7}
do
    echo $size
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/fine_tuning.py --attack $attack --mark_alpha $alpha --height $size --width $size \
    --verbose --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr 1e-2 \
    > $dirname/fine_tuning_${parameters}_size${size}.txt 2>&1
done