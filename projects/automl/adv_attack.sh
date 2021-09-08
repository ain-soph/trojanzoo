# CUDA_VISIBLE_DEVICES=0 bash projects/automl/adv_attack.sh "--grad_method nes --target_idx 1" > results/nes.txt 2>&1
# CUDA_VISIBLE_DEVICES=2 bash projects/automl/adv_attack.sh "--target_idx 1 --pgd_alpha 1.0 --pgd_eps 4.0" > results/pgd_4.txt 2>&1

declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "sgas" "snas_mild" "random")
declare -a models=("bit_comp" "densenet_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")

attack="pgd"
dataset="cifar10"
args=$1

for model in "${models[@]}"; do
    echo $model
    python examples/adv_attack.py --pretrain --attack $attack --dataset $dataset --model $model --pretrain --require_class $args --verbose 1
    echo ""
done

for arch in "${archs[@]}"; do
    echo $arch
    python examples/backdoor_attack.py --pretrain --attack $attack --dataset $dataset --model darts --model_arch $arch --pretrain --require_class $args
    echo ""
done
