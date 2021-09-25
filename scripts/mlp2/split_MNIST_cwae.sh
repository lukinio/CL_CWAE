GPUID=$1
OUTDIR=outputs/mlp2/split_MNIST_test
REPEAT=3
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
       | tee ${OUTDIR}/IT_CWAE.log


#### INCREMENTAL DOMAIN
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID_latent8_best \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 100 --reg_coef_2 0 \
       | tee ${OUTDIR}/ID_CWAE.log


#### INCREMENTAL CLASS
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
       --incremental_class --no_class_remap \
       --exp_tag IC_latent8_best \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 10 --reg_coef_2 0.0001 \
       | tee ${OUTDIR}/IC_CWAE.log

# =================================================================================

## INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8_test --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-3 --latent_size 8 \
#       --reg_coef 100 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_CWAE_test.log


## INCREMENTAL CLASS
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap \
#       --exp_tag IC_latent8_test --wandb_logger \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 10 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/IC_CWAE_test.log





# ======================================================================================
# ======================================== GRID ========================================
# ======================================================================================
#
### INCREMENTAL TASK
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
#       --exp_tag IT_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_CWAE_0.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
#       --exp_tag IT_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/IT_CWAE_0_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
#       --exp_tag IT_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/IT_CWAE_0_001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
#       --exp_tag IT_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/IT_CWAE_0_0001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
#       --exp_tag IT_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.00001 \
#       | tee ${OUTDIR}/IT_CWAE_0_00001.log
#
#
### INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_CWAE_0.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/ID_CWAE_0_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/ID_CWAE_0_001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/ID_CWAE_0_0001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.00001 \
#       | tee ${OUTDIR}/ID_CWAE_0_00001.log
#
### INCREMENTAL CLASS
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap \
#       --exp_tag IC_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IC_CWAE_0.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap \
#       --exp_tag IC_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/IC_CWAE_0_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap \
#       --exp_tag IC_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/IC_CWAE_0_001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap \
#       --exp_tag IC_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/IC_CWAE_0_0001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap \
#       --exp_tag IC_latent8 \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0.00001 \
#       | tee ${OUTDIR}/IC_CWAE_0_00001.log
#
## ======================================================================================
## ======================================== GRID ========================================
## ======================================================================================