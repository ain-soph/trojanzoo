declare -a archs=("enas" "darts" "pc_darts" "amoebanet" "snas_mild" "nasnet" "pdarts" "sgas" "drnas")
declare -a models=("mobilenet_v2_comp" "resnet50_comp" "densenet161_comp" "wide_resnet50_2_comp" "resnext50_32x4d_comp" "dla60_comp" "bit_comp")

for model in "${models[@]}"
do
    echo $model
    python projects/automl/grad_var.py --pretrain --dataset imagenet16 --num_classes 120 --model $model
done

for arch in "${archs[@]}"
do
    echo $arch
    python projects/automl/grad_var.py --pretrain --dataset imagenet16 --num_classes 120 --model darts--model_arch $arch
done
