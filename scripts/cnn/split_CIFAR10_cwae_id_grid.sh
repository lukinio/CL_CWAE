GPUID=$1
OUTDIR=outputs/cnn/split_CIFAR10_grid_id
REPEAT=3
mkdir -p $OUTDIR


# INCREMENTAL DOMAIN

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_CWAE_0.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/ID_CWAE_0_001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/ID_CWAE_0_0001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 --reg_coef_2 0.00001 \
#       | tee ${OUTDIR}/ID_CWAE_0_00001.log


python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
       --model_type cnn --model_name cnn --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID_latent8 \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 50 100 150 200 250 --reg_coef_2 0.00001 \
       | tee ${OUTDIR}/ID_CWAE_0_00001_more.log