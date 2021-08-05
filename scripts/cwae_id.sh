GPUID=$1
OUTDIR=outputs/cwae_id
REPEAT=5
mkdir -p $OUTDIR

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
       --generator_epoch 10 --generator_lr 1e-3 --latent_size 8 \
       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 \
       | tee ${OUTDIR}/cwae_latent8.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
       --generator_epoch 10 --generator_lr 1e-3 --latent_size 16 \
       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 \
       | tee ${OUTDIR}/cwae_latent16.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
       --generator_epoch 10 --generator_lr 1e-3 --latent_size 4 \
       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 \
       | tee ${OUTDIR}/cwae_latent4.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
       --generator_epoch 10 --generator_lr 1e-3 --latent_size 2 \
       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 \
       | tee ${OUTDIR}/cwae_latent2.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP400 --generator_type generator --generator_name Generator \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 0.001 \
       --generator_epoch 10 --generator_lr 1e-3 --latent_size 12 \
       --reg_coef 0.0001 0.001 0.01 0.1 1 10 100 1000 10000 \
       | tee ${OUTDIR}/cwae_latent12.log
