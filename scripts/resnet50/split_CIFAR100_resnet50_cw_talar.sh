GPUID=$1
OUTDIR=outputs/split_CIFAR100_resnet50_cw_talar
REPEAT=3
mkdir -p $OUTDIR

dataset=CIFAR100

# Incremental task
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
       --model_name resnet50_pretrained --model_type resnet2 \
       --mode ENCODER --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 20 --other_split_size 20 \
       --exp_tag ID_latent8 \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-3 \
       --generator_epoch 10 --generator_lr 1e-3 --latent_size 64 \
       --reg_coef 10000 1000 100 10 1 0.1 0.01 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_cw_talar_enc.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
       --model_name resnet50_pretrained --model_type resnet2 \
       --mode GENERATOR --generator_type generator --generator_name GeneratorCIFAR \
       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 20 --other_split_size 20 \
       --exp_tag ID_latent8 \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-3 \
       --generator_epoch 12 --generator_lr 1e-3 --latent_size 64 \
       --reg_coef 10000 1000 100 10 1 0.1 0.01 --reg_coef_2 0 \
       | tee ${OUTDIR}/IT_cw_talar_gen.log


# Incremental class
python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
       --model_name resnet50_pretrained --model_type resnet2 \
       --mode ENCODER --generator_type generator --generator_name EncoderCIFAR \
       --agent_type cwae --agent_name CWAE --force_out_dim 20 --first_split_size 20 --other_split_size 20 \
       --exp_tag ID_latent8 --incremental_class \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-3 \
       --generator_epoch 10 --generator_lr 1e-3 --latent_size 64 \
       --reg_coef 10000 1000 100 10 1 0.1 0.01 --reg_coef_2 0 \
       | tee ${OUTDIR}/IC_cw_talar_enc.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
       --model_name resnet50_pretrained --model_type resnet2 \
       --mode GENERATOR --generator_type generator --generator_name GeneratorCIFAR \
       --agent_type cwae --agent_name CWAE --force_out_dim 0 --first_split_size 20 --other_split_size 20 \
       --exp_tag ID_latent8 --incremental_class \
       --schedule 12 --batch_size 128 --optimizer Adam --lr 1e-3 \
       --generator_epoch 12 --generator_lr 1e-3 --latent_size 64 \
       --reg_coef 10000 1000 100 10 1 0.1 0.01 --reg_coef_2 0 \
       | tee ${OUTDIR}/IC_cw_talar_gen.log


## 5 task
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 20 --first_split_size 20 --other_split_size 20 \
#       --exp_tag ID_latent8 \
#       --schedule 20 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar45.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 20 --first_split_size 20 --other_split_size 20 \
#       --exp_tag ID_latent8 \
#       --schedule 20 --batch_size 128 --optimizer Adam --lr 1e-3 \
#       --generator_epoch 10 --generator_lr 1e-3 --latent_size 128 \
#       --reg_coef 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar35.log


# 10 task
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 10 --other_split_size 10 \
#       --exp_tag ID_latent8 \
#       --schedule 20 --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar4.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 10 --first_split_size 10 --other_split_size 10 \
#       --exp_tag ID_latent8 \
#       --schedule 20 --batch_size 128 --optimizer Adam --lr 1e-3 \
#       --generator_epoch 10 --generator_lr 1e-3 --latent_size 128 \
#       --reg_coef 0.01 0.1 1 10 100 250 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar3.log

########### EPOCH = 4 ###############################
#epoch=4
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule $epoch --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/e4_pre_ID_test.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule $epoch --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 10000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/e4_pre_ID_test_0_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule $epoch --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 10000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/e4_pre_ID_test_0_001.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule $epoch --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 10000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/e4_pre_ID_test_0_0001.log
#
#
#
#
#
#
############ EPOCH = 8 ###############################
#epoch=8
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule $epoch --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 10000 --reg_coef_2 0 \
#       | tee ${OUTDIR}/e8_pre_ID_test.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule $epoch --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 10000 --reg_coef_2 0.01 \
#       | tee ${OUTDIR}/e8_pre_ID_test_0_01.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule $epoch --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 10000 --reg_coef_2 0.001 \
#       | tee ${OUTDIR}/e8_pre_ID_test_0_001.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset $dataset --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name resnet50_pretrained --model_type resnet2 --generator_type generator --generator_name GeneratorCIFAR \
#       --agent_type cwae --agent_name CWAE --force_out_dim 2 --first_split_size 2 --other_split_size 2 \
#       --exp_tag ID_latent8 \
#       --schedule $epoch --batch_size 128 --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001 0.01 0.1 1 10 100 500 1000 10000 --reg_coef_2 0.0001 \
#       | tee ${OUTDIR}/e8_pre_ID_test_0_0001.log
