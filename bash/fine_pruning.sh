work_dir='/home/panrusheng/Trojan-Zoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack='badnet'
defense='fine_pruning'

CUDA_VISIBLE_DEVICES=0,1,2,3

dirname=${work_dir}/result/${dataset}/${model}/${defense}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

alpha=0
size=3
clean_image_num=50
prune_ratio=0.02
lr=0.01



CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py \
--dataset $dataset --model $model --attack $attack --defense $defense --clean_image_num $clean_image_num --prune_ratio $prune_ratio \
--verbose --pretrain --mark_alpha 0.$alpha --height $size --width $size --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr $lr --save \
> $dirname/${attack}_${dataset}_${model}_clean${clean_image_num}_pr_${prune_ratio}_lr${lr}.txt 2>&1