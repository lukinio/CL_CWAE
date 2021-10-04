GPUID=$1
OUTDIR=outputs/mlp2/permuted_MNIST
REPEAT=10
mkdir -p $OUTDIR

#### INCREMENTAL DOMAIN
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2                                                     --lr 0.0001  --offline_training  | tee ${OUTDIR}/ID_Offline.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2                                                     --lr 0.0001                      | tee ${OUTDIR}/ID_Adam.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer SGD     --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2                                                     --lr 0.001                       | tee ${OUTDIR}/ID_SGD.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer Adagrad --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2                                                     --lr 0.001                       | tee ${OUTDIR}/ID_Adagrad.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2 --agent_type customization  --agent_name EWC_online --lr 0.0001 --reg_coef 250       | tee ${OUTDIR}/ID_EWC_online.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2 --agent_type customization  --agent_name EWC        --lr 0.0001 --reg_coef 150       | tee ${OUTDIR}/ID_EWC.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2 --agent_type regularization --agent_name SI         --lr 0.0001 --reg_coef 10        | tee ${OUTDIR}/ID_SI.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2 --agent_type regularization --agent_name L2         --lr 0.0001 --reg_coef 0.02      | tee ${OUTDIR}/ID_L2.log
python -u iBatchLearn.py --dataroot /shared/sets/datasets/vision --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --n_permutation 10 --no_class_remap --force_out_dim 10 --schedule 10 --batch_size 128 --model_name MLP2 --agent_type regularization --agent_name MAS        --lr 0.0001 --reg_coef 0.1       | tee ${OUTDIR}/ID_MAS.log

