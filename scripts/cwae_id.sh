GPUID=$1
OUTDIR=outputs/cwae_id_reg
REPEAT=3
mkdir -p $OUTDIR


#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 2 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.1 \
#       | tee ${OUTDIR}/cwae_01.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 2 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/cwae_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 2 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/cwae_001.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 2 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/cwae_0001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 2 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.00001 \
#       | tee ${OUTDIR}/cwae_00001.log



# katalog cwae_id_reg
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8_reg_0_000001 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.000001 \
#       | tee ${OUTDIR}/cwae_reg_0_000001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8_reg_0_00001 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.00001 \
#       | tee ${OUTDIR}/cwae_reg_0_00001.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8_reg_0_00001 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/cwae_reg_0_0001.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8_reg_0_001 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/cwae_reg_0_001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8_reg_0_01 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/cwae_reg_0_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8_reg_0_1 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.1 \
#       | tee ${OUTDIR}/cwae_reg_0_1.log



# MODEL MLP2
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID_latent8_best_reg1_100000_reg2_0_100001 \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 1 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 250 --reg_coef_2 0.0001 \
       | tee ${OUTDIR}/cwae_reg_test.log