work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack=$1

CUDA_VISIBLE_DEVICES=0

dirname=${work_dir}/result/${dataset}/${model}/${attack}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

size=3
for alpha in {1..9}
do
    echo 0.$alpha
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/transfer_attack.py \
    --attack $attack --mark_alpha 0.$alpha --height $size --width $size \
    --parameters partial --verbose --validate_interval 1 --lr_scheduler --step_size 10 --epoch 5 --lr 1e-2 \
    > $dirname/transfer_partial_alpha0.${alpha}.txt 2>&1
done