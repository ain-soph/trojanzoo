work_dir='/home/rbp5354/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack='latent_backdoor'
defense='neural_cleanse'

CUDA_VISIBLE_DEVICES=0

dirname=${work_dir}/result/${dataset}/${model}/${defense}/${attack}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

dirname=${work_dir}/result/${dataset}/${model}/${defense}/${attack}
if [ ! -d $dirname  ];then
    mkdir -p $dirname
fi

size=2
for alpha in {1..9}
do
    echo 0.$alpha
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${work_dir}/backdoor_defense.py \
    --defense $defense --attack $attack --mark_alpha 0.$alpha --height $size --width $size \
    > $dirname/alpha0.${alpha}.txt 2>&1
done