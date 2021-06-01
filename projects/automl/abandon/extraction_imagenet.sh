declare -a rates=("1000" "2000" "4000" "8000" "16000")
declare -a archs=("enas" "darts" "pc_darts" "amoebanet" "snas_mild" "nasnet" "pdarts" "sgas" "drnas")
declare -a models=("mobilenet_v2_comp" "resnet50_comp" "densenet161_comp" "wide_resnet50_2_comp" "resnext50_32x4d_comp" "dla60_comp" "bit_comp")

dataset="imagenet16"

for model in "${models[@]}"; do
    echo $model
    for rate in "${rates[@]}"; do
        echo $rate
        python projects/automl/extraction_attack2.py --pretrain --dataset $dataset --num_classes 120 --model $model --nb_stolen $rate
    done
done

for arch in "${archs[@]}"; do
    echo $arch
    for rate in "${rates[@]}"; do
        echo $rate
        python projects/automl/extraction_attack2.py --pretrain --dataset $dataset --num_classes 120 --model darts --model_arch $arch --nb_stolen $rate
    done
done
