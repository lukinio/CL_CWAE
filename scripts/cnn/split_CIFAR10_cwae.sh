GPUID=$1
OUTDIR=outputs/cnn/split_CIFAR10_test
REPEAT=10
mkdir -p $OUTDIR

# INCREMENTAL TASK

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
       --model_type cnn --model_name cnn --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
       --exp_tag IT_latent8 \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 100 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 0.1 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_CWAE.log



# INCREMENTAL DOMAIN

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
       --model_type cnn --model_name cnn --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID_latent8 \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 0.001 --reg_coef_2 0.00001 \
       | tee ${OUTDIR}/ID_CWAE.log


# INCREMENTAL CLASS

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
       --model_type cnn --model_name cnn --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
       --incremental_class --no_class_remap \
       --exp_tag ID_latent8  \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-3 --latent_size 8 \
       --reg_coef 0.1 --reg_coef_2 0 \
       | tee ${OUTDIR}/IC_CWAE.log
