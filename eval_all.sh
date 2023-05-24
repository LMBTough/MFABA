attr_method=("big" "mfaba_cos" "mfaba_norm" "mfaba_sharp" "mfaba_smooth" "agi" "ig" "sm" "sg")
models=("resnet50" "inception_v3" "vgg16")
for method in ${attr_method[@]}
do
    for model in ${models[@]}
    do
        python eval.py --model $model --attr_method $method
    done
done