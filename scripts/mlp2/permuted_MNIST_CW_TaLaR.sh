GPUID=$1
OUTDIR=outputs/mlp2/permuted_MNIST_test_cw
REPEAT=10
mkdir -p $OUTDIR


## INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation/ 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name Generator2 \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_mnist \
#       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 8 \
#       --reg_coef 0.01 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/ID_CWAE.log



#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_mnist --mode ENCODER \
#       --schedule 10 --batch_size 128 --optimizer SGD --lr 1e-2 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
#       --reg_coef 0.01 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_CWAE_enc_SGD_best.log


## TETS
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae1 --agent_name CWAE1 \
#       --exp_tag permuted_mnist --mode CW \
#       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_CWAE_Adam.log

### CW
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --n_permutation 10 --no_class_remap --force_out_dim 10 \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 \
       --exp_tag permuted_mnist --mode CW \
       --schedule 10 --batch_size 128 --optimizer SGD --lr 1e-2 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 0.1 --reg_coef_2 0 \
       | tee ${OUTDIR}/ID_CWAE_SGD_best.log


## INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_mnist --mode ENCODER \
#       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
#       --reg_coef 10 0.001 0.01 0.1 1 100 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/ID_CWAE_enc.log
#
## INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_mnist --mode ENCODER \
#       --schedule 10 --batch_size 128 --optimizer SGD --lr 1e-2 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
#       --reg_coef 10 0.001 0.01 0.1 1 100 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/ID_CWAE_enc_SGD.log


## INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_mnist --mode ENCODER \
#       --schedule 10 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
#       --reg_coef 10 0.001 0.01 0.1 1 100 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_CWAE_enc0.log
#
## INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#       --n_permutation 10 --no_class_remap --force_out_dim 10 \
#       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE \
#       --exp_tag permuted_mnist --mode ENCODER \
#       --schedule 10 --batch_size 128 --optimizer SGD --lr 1e-2 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
#       --reg_coef 10 0.001 0.01 0.1 1 100 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_CWAE_enc_SGD0.log