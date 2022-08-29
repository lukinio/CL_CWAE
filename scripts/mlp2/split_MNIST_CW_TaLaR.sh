GPUID=$1
OUTDIR=outputs/mlp2/split_MNIST_test_cw
REPEAT=10
mkdir -p $OUTDIR


#### GENERATOR
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



### ENCODER

#### INCREMENTAL TASK
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
       --exp_tag IT_latent8_best --mode ENCODER \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 0.001 0.01 0.1 1 10 100 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_CW_TaLaR_enc.log


#### INCREMENTAL DOMAIN
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID_latent8_best --mode ENCODER \
       --schedule 4 --batch_size 128 --optimizer SGD --lr 1e-2 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 100 --reg_coef_2 0 \
       | tee ${OUTDIR}/ID_CW_TaLaR_enc1.log


#### INCREMENTAL CLASS
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
       --incremental_class --no_class_remap \
       --exp_tag IC_latent8_best --mode ENCODER \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 2000 1000 500 100 10 1 --reg_coef_2 0 \
       | tee ${OUTDIR}/IC_CW_TaLaR_enc.log



### CW

#### INCREMENTAL TASK
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
       --exp_tag IT_latent8_best --mode CW \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_CW_TaLaR_Adam.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
       --exp_tag IT_latent8_best --mode CW \
       --schedule 4 --batch_size 128 --optimizer SGD --lr 1e-2 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_CW_TaLaR_SGD.log


#### INCREMENTAL DOMAIN
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID_latent8_best --mode CW \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 --reg_coef_2 0 \
       | tee ${OUTDIR}/ID_CW_TaLaR_Adam.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID_latent8_best --mode CW \
       --schedule 4 --batch_size 128 --optimizer SGD --lr 1e-2 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 --reg_coef_2 0 \
       | tee ${OUTDIR}/ID_CW_TaLaR_SGD.log


#### INCREMENTAL CLASS
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
       --model_type mlp --model_name MLP2 --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
       --incremental_class --no_class_remap \
       --exp_tag IC_latent8_best --mode CW \
       --schedule 4 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 32 \
       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 --reg_coef_2 0 \
       | tee ${OUTDIR}/IC_CW_TaLaR.log
