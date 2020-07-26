work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack='badnet'

CUDA_VISIBLE_DEVICES=2

size=2
for alpha in {1..9}
do
    echo 0.$alpha
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_attack.py --attack $attack --random_pos --mark_alpha 0.$alpha --height $size --width $size \
    --percent 0.3 --verbose --pretrain --validate_interval 1 --lr_scheduler --step_size 10 --epoch 50 --lr 1e-2 --save \
    > ${work_dir}/result/random_pos_${attack}_${dataset}_${model}_alpha0.${alpha}.txt 2>&1
done