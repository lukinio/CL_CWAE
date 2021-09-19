GPUID=$1
OUTDIR=outputs/cwae_ic_reg
REPEAT=3
mkdir -p $OUTDIR


#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent8 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 \
#       | tee ${OUTDIR}/cwae_latent8.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent16 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 16 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 \
#       | tee ${OUTDIR}/cwae_latent16.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent4 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 4 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 \
#       | tee ${OUTDIR}/cwae_latent4.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent2 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 2 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 \
#       | tee ${OUTDIR}/cwae_latent2.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent12 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 12 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 \
#       | tee ${OUTDIR}/cwae_latent12.log

#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent8_reg_0_000001 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.000001 \
#       | tee ${OUTDIR}/cwae_reg_0_000001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent8_reg_0_00001 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.00001 \
#       | tee ${OUTDIR}/cwae_reg_0_00001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent8_reg_0_0001 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/cwae_reg_0_0001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent8_reg_0_001 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/cwae_reg_0_001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent8_reg_0_01 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/cwae_reg_0_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap --exp_tag IC_latent8_reg_0_1 --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
#       --generator_epoch 1 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 100000 1000000 --reg_coef_2 0.1 \
#       | tee ${OUTDIR}/cwae_reg_0_1.log



python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
       --incremental_class --no_class_remap \
       --exp_tag IC_latent8_reg_0_1 \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 1 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 100 --reg_coef_2 0.0001 \
       | tee ${OUTDIR}/cwae_reg_test.log