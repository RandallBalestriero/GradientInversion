
screen -dmS z bash -c "export CUDA_VISIBLE_DEVICES=6;
python quickstart_classification.py --model cnn --dataset cifar10 --data_augmentation 1 --parameter 0.01 --samples 2000";

screen -dmS x bash -c "export CUDA_VISIBLE_DEVICES=6;
python quickstart_classification.py --model cnn --dataset cifar10 --data_augmentation 1 --parameter 0.1 --samples 2000";

screen -dmS y bash -c "export CUDA_VISIBLE_DEVICES=7;
python quickstart_classification.py --model cnn --dataset cifar10 --data_augmentation 1 --parameter 1 --samples 2000";

screen -dmS w bash -c "export CUDA_VISIBLE_DEVICES=7;
python quickstart_classification.py --model cnn --dataset cifar10 --data_augmentation 1 --parameter 0.001 --samples 2000";








