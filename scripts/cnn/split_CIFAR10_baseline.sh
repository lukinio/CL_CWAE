GPUID=$1
OUTDIR=outputs/cnn/split_CIFAR10
REPEAT=10
mkdir -p $OUTDIR

# INCREMENTAL TASK

#python -u iBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --optimizer Adam --force_out_dim 0 --first_split_size 2 --other_split_size 2 --schedule 12 \
#       --batch_size 128 --model_name cnn --model_type cnn --agent_type customization \
#       --agent_name EWC_online --lr 0.001 --reg_coef 500 \
#       | tee ${OUTDIR}/IT_EWC_online.log
#
#python -u iBatchLearn.py --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT \
#       --optimizer Adam --force_out_dim 0 --first_split_size 2 --other_split_size 2 --schedule 12 \
#       --batch_size 128 --model_name cnn --model_type cnn --agent_type customization \
#       --agent_name EWC --lr 0.001 --reg_coef 1000 \
#       | tee ${OUTDIR}/IT_EWC.log

# INCREMENTAL DOMAIN

#python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --dataset CIFAR10 --train_aug \
#       --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --batch_size 128 --model_name cnn --model_type cnn \
#       --agent_type customization --agent_name EWC --lr 0.001 --reg_coef 100 \
#       | tee ${OUTDIR}/ID_EWC.log
#
#python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --dataset CIFAR10 --train_aug \
#       --gpuid $GPUID --repeat $REPEAT --optimizer Adam --force_out_dim 2 --first_split_size 2 \
#       --other_split_size 2 --schedule 12 --batch_size 128 --model_name cnn --model_type cnn \
#       --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 500 \
#       | tee ${OUTDIR}/ID_EWC_online.log




# INCREMENTAL CLASS

python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 12 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 2            | tee ${OUTDIR}/IC_EWC.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim 10 --no_class_remap --first_split_size 2 --other_split_size 2 --schedule 12 --batch_size 128 --model_name cnn --model_type cnn --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 2            | tee ${OUTDIR}/IC_EWC_online.log

