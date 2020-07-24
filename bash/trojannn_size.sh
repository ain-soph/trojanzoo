work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack='trojannn'

CUDA_VISIBLE_DEVICES=2

alpha=0.0
for size in {1..7}
do
    echo $size
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_attack.py --attack $attack --mark_alpha $alpha --height $size --width $size \
    --verbose --pretrain --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr 1e-2 --save \
    > ${work_dir}/result/${attack}_${dataset}_${model}_size${size}.txt 2>&1
done