work_dir='/home/wilsonzhang/Documents/Research/Trojan-Zoo'
cd $work_dir/trojanzoo

dataset='cifar10'
model='resnetcomp18'
attack='trojannet'

dirname=${work_dir}/trojanzoo/result/${dataset}/${model}/${attack}
if [ ! -d $dirname ]; then
    mkdir -p $dirname
fi

python $work_dir/adv_attack.py \
  --attack $attack \
  --syn_backdoor_map 16 5 \
  --download True \
  --model_save_path "${dirname}/saved_model"