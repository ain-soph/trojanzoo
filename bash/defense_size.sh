work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack='latent_backdoor'
defense='neural_cleanse'

CUDA_VISIBLE_DEVICES=1

dirname=${work_dir}/result/${dataset}/${model}/${defense}/${attack}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

alpha=0.0
for size in {1..7}
do
    echo $size
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py \
    --defense $defense --attack $attack --mark_alpha $alpha --height $size --width $size \
    > $dirname/size${size}.txt 2>&1
done