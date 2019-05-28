
GPU=0
for param in 1 0.1 0.01 0.001 0.0001 0
do
    screen -dmS g bash -c "export CUDA_VISIBLE_DEVICES=$GPU;
    python quickstart_classification.py --model cnn --dataset cifar100 --data_augmentation 1 --parameter $param";
    GPU=$((GPU+1))
done







