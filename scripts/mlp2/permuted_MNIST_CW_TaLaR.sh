GPUID=$1
OUTDIR=outputs/mlp2/permuted_MNIST
REPEAT=10
mkdir -p $OUTDIR


# INCREMENTAL DOMAIN
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --n_permutation 10 --no_class_remap --force_out_dim 10 \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
       --agent_type cwae --agent_name CWAE \
       --exp_tag permuted_mnist \
       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
       --reg_coef 0.01 --reg_coef_2 0.01 \
       | tee ${OUTDIR}/ID_CWAE.log
