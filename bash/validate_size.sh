work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack=$2

CUDA_VISIBLE_DEVICES=$1

alpha=8
for size in {1..7}
do
    echo $size
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/validate_backdoor.py \
    --attack $attack --mark_alpha 0.$alpha --height $size --width $size
done