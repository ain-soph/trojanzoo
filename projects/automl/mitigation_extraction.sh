declare -a rates=("1000" "2000" "4000" "8000" "16000")
declare -a archs=("diy_deep" "diy_noskip" "diy_deep_noskip")

dataset="cifar10"

for arch in "${archs[@]}"; do
    echo $arch
    for rate in "${rates[@]}"; do
        echo $rate
        python projects/automl/extraction_attack2.py --pretrain --dataset $dataset --model darts --model_arch $arch --nb_stolen $rate
    done
done
