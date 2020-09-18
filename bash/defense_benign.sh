work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
defense=$2
attack=badnet

CUDA_VISIBLE_DEVICES=$1

dirname=${work_dir}/result/${dataset}/${model}/${defense}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

size=3
alpha=0.0

for suffix in {0..9}
do
    echo $suffix
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py \
    --dataset $dataset --model $model --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size\
    --original --suffix _$suffix --pretrain \
    > $dirname/benign_$suffix.txt 2>&1
done