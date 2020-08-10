work_dir='/Users/wilsonzhang/Documents/PROJECTS/Research/ALPS-Lab/Trojan-Zoo/trojanzoo'
cd $work_dir

dataset='cifar10'
model='resnetcomp18'
attack='trojannet'

dirname=${work_dir}/result/${dataset}/${model}/${attack}
if [ ! -d $dirname ]; then
    mkdir -p $dirname
fi

python /Users/wilsonzhang/Documents/PROJECTS/Research/ALPS-Lab/Trojan-Zoo/adv_attack.py --attack $attack --syn_backdoor_map "(16, 5)"