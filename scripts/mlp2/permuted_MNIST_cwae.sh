GPUID=$1
OUTDIR=outputs/mlp2/permuted_MNIST_grid
REPEAT=5
mkdir -p $OUTDIR


# INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_test \
#       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 50 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.01 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/ID_CWAE.log

#       --reg_coef 0.01 --reg_coef_2 0.01 \

##=============================================================
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_test \
#       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_CWAE_0.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_test \
#       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/ID_CWAE_0_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_test \
#       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/ID_CWAE_0_001.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_test \
#       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 20 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/ID_CWAE_0_0001.log
#

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --n_permutation 10 --no_class_remap --force_out_dim 10 \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE \
       --exp_tag permuted_test \
       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 0.01 --reg_coef_2 0.01 \
       | tee ${OUTDIR}/ID_CWAE_test.log

