declare -a models=("resnet18_comp" "densenet_comp" "darts" "darts --model_arch enas")
declare -a tmodels=("resnet18_comp" "densenet_comp" "darts" "darts --tmodel_arch enas")

rate="16000"
dataset="cifar10"

for model in "${models[@]}"; do
    for tmodel in "${tmodels[@]}"; do
        echo $model "->" $tmodel
        python projects/automl/extraction_attack2.py --pretrain --dataset $dataset --model $model --tmodel $tmodel --nb_stolen $rate
    done
done
