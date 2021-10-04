GPUID=$1
OUTDIR=outputs/mlp2/split_MNIST
REPEAT=10
mkdir -p $OUTDIR


#### =================================================================================
#### INCREMENTAL TASK
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
       --exp_tag IT_latent8_best \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 0.01 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_CW_TaLaR.log


#### INCREMENTAL DOMAIN
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID_latent8_best \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 10 --reg_coef_2 0 \
       | tee ${OUTDIR}/ID_CW_TaLaR.log


#### INCREMENTAL CLASS
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
       --incremental_class --no_class_remap \
       --exp_tag IC_latent8_best \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 10 --reg_coef_2 0.0001 \
       | tee ${OUTDIR}/IC_CW_TaLaR.log

