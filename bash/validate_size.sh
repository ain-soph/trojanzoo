work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset=$2
model=$3
attack=$4

CUDA_VISIBLE_DEVICES=$1

alpha=0.0
for size in {1..7}
do
    echo $size
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/validate_backdoor.py \
    --attack $attack --mark_alpha $alpha --height $size --width $size
done