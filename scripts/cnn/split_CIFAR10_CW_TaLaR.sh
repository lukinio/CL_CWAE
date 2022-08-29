GPUID=$1
OUTDIR=outputs/cnn/split_CIFAR10_test_cw
REPEAT=10
mkdir -p $OUTDIR

## INCREMENTAL TASK
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
#       --exp_tag IT_latent8 \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 0.1 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_CW_TaLaR.log
#
#
#
### INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name cnn --model_type cnn --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 10 --reg_coef_2 0.00001 \
#       | tee ${OUTDIR}/ID_CW_TaLaR_test.log
#
#
## INCREMENTAL CLASS
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap \
#       --exp_tag ID_latent8  \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-3 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 1 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IC_CW_TaLaR_test.log





# INCREMENTAL TASK

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
#       --exp_tag IT --mode ENCODER \
#       --schedule 2 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 1 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_cw_talar_enc_best1.log

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
#       --exp_tag IT --mode ENCODER \
#       --schedule 12 --batch_size 128 --optimizer SGD --lr 1e-2 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 1 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_cw_talar_enc_SGD_best.log


#
### INCREMENTAL DOMAIN
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name cnn --model_type cnn --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID --mode ENCODER \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 100 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar_enc_best.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name cnn --model_type cnn --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID --mode ENCODER \
#       --schedule 12 --batch_size 128 --optimizer SGD --lr 1e-2 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 10 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar_enc_SGD_best.log
#
#
## INCREMENTAL CLASS
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_type cnn --model_name cnn --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
#       --incremental_class --no_class_remap \
#       --exp_tag IC  --mode ENCODER \
#       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-3 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 0.01 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IC_cw_talar_enc_best.log




###########################################
###########################################

## INCREMENTAL TASK

REPEAT=5
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
       --model_type cnn --model_name cnn --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
       --exp_tag IT --mode CW \
       --schedule 12 --batch_size 128 --optimizer SGD --lr 1e-2 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_cw_talar_SGD.log

REPEAT=10
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
       --model_type cnn --model_name cnn --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 0 --first_split_size 2 --other_split_size 2 \
       --exp_tag IT --mode CW \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
       --reg_coef 100 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_cw_talar_best.log



## INCREMENTAL DOMAIN
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
       --model_name cnn --model_type cnn --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
       --exp_tag ID --mode CW \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-4 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
       --reg_coef 100 --reg_coef_2 0 \
       | tee ${OUTDIR}/ID_cw_talar_best.log

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name cnn --model_type cnn --generator_type generator --generator_name EncoderCIFAR \
#       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID --mode CW \
#       --schedule 12 --batch_size 128 --optimizer SGD --lr 1e-2 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar_SGD.log

## INCREMENTAL CLASS

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset CIFAR10 --train_aug --dataroot /shared/sets/datasets/vision \
       --model_type cnn --model_name cnn --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae1 --agent_name CWAE1 --force_out_dim 10 --first_split_size 2 --other_split_size 2 \
       --incremental_class --no_class_remap \
       --exp_tag IC  --mode CW \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-3 \
       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
       --reg_coef 0.1 --reg_coef_2 0 \
       | tee ${OUTDIR}/IC_cw_talar_best.log
