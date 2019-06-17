
GPU=0
for param in 0.1 0.01 0.001 0.0001 0.00001 0.000001
do
    for model in cnn smallresnet largeresnet
    do
        screen -dmS g bash -c "export CUDA_VISIBLE_DEVICES=$GPU;
        python quickstart_classification.py --model $model --dataset cifar10 --parameter $param";
        GPU=$((GPU+1))
    done
done







