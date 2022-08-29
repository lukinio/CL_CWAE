GPUID=$1

REPEAT=5
OUTDIR=outputs/split_CIFAR100_5t_resnet50_baseline
mkdir -p $OUTDIR

# COMMON
dataset=CIFAR100
epoch_per_task=12
batch=128
model_name=resnet50_pretrained
model_type=resnet2



# INCREMENTAL TASK
force_out_dim=0
first_split_size=20
other_split_size=20


python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.001 --offline_training                                       | tee ${OUTDIR}/IT_Offline.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.001                                                          | tee ${OUTDIR}/IT_Adam.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.01                                                           | tee ${OUTDIR}/IT_SGD.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.01                                                           | tee ${OUTDIR}/IT_Adagrad.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IT_EWC_online.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IT_EWC.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name SI         --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IT_SI.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name L2         --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IT_L2.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name MAS        --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IT_MAS.log




# INCREMENTAL DOMAIN
force_out_dim=20
first_split_size=20
other_split_size=20

python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.001 --offline_training                                       | tee ${OUTDIR}/ID_Offline.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.001                                                          | tee ${OUTDIR}/ID_Adam.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.01                                                           | tee ${OUTDIR}/ID_SGD.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.01                                                           | tee ${OUTDIR}/ID_Adagrad.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/ID_EWC_online.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/ID_EWC.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name SI         --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/ID_SI.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name L2         --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/ID_L2.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name MAS        --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/ID_MAS.log




# INCREMENTAL CLASS
force_out_dim=100
first_split_size=20
other_split_size=20

python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.001 --offline_training                                       | tee ${OUTDIR}/IC_Offline.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.001                                                          | tee ${OUTDIR}/IC_Adam.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer SGD     --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.01                                                           | tee ${OUTDIR}/IC_SGD.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adagrad --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type                                                     --lr 0.01                                                           | tee ${OUTDIR}/IC_Adagrad.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type customization  --agent_name EWC_online --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IC_EWC_online.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule 1 --batch_size $batch --model_name $model_name --model_type $model_type --agent_type customization  --agent_name EWC        --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IC_EWC.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name SI         --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IC_SI.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name L2         --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IC_L2.log
python -u iBatchLearn.py --dataset $dataset --dataroot /shared/sets/datasets/vision --train_aug --gpuid $GPUID --repeat $REPEAT --incremental_class --optimizer Adam    --force_out_dim $force_out_dim --first_split_size $first_split_size --other_split_size $other_split_size --schedule $epoch_per_task --batch_size $batch --model_name $model_name --model_type $model_type --agent_type regularization --agent_name MAS        --lr 0.001 --reg_coef 0.001 0.01 0.1 1 10 100 200 500 1000 10000    | tee ${OUTDIR}/IC_MAS.log

