work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset=$2
model=$3
attack=$4

CUDA_VISIBLE_DEVICES=$1

size=3
for alpha in {1..9}
do
    echo 0.$alpha
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/validate_backdoor.py \
    --attack $attack --mark_alpha 0.$alpha --height $size --width $size
done