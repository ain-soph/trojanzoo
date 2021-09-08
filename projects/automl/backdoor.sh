# CUDA_VISIBLE_DEVICES=0 bash projects/automl/backdoor.sh "--neuron_num 2" > results/backdoor_2.txt 2>&1

# declare -a archs=("amoebanet" "darts" "drnas" "enas" "nasnet" "pc_darts" "pdarts" "sgas" "snas_mild" "random")
# declare -a models=("bit_comp" "densenet_comp" "dla34_comp" "resnet18_comp" "resnext50_32x4d_comp" "vgg13_bn_comp" "wide_resnet50_2_comp")

# declare -a archs=("random")
declare -a models=("bit_comp")

attack="trojannn"
dataset="cifar10"
args=$1

for model in "${models[@]}"; do
    echo $model
    python examples/backdoor_attack.py --verbose 1 --mark_alpha 0.7 --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset $dataset --model $model $args
done

# for arch in "${archs[@]}"; do
#     echo $arch
#     python examples/backdoor_attack.py --verbose 1 --mark_alpha 0.7 --epoch 20 --batch_size 96 --lr 0.01 --pretrain --save --validate_interval 1 --attack $attack --dataset $dataset --model darts --model_arch $arch $args
# done
