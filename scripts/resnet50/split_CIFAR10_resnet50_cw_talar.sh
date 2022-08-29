GPUID=$1

REPEAT=2
OUTDIR=outputs/split_CIFAR10_resnet50_cw_talar_best_enc_test_cw
mkdir -p $OUTDIR

# COMMON
dataset=CIFAR10
epoch_per_task=4
batch=128
model_name=resnet50_pretrained
model_type=resnet2


## INCREMENTAL TASK
#force_out_dim=0
#first_split_size=2
#other_split_size=2
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode ENCODER \
#       --agent_type cwae --agent_name CWAE \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IT  \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 0.1 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_cw_talar_enc_best.log



#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode ENCODER \
#       --agent_type cwae --agent_name CWAE \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IT  \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 0.01 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_cw_talar_enc.log

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name GeneratorCIFAR --mode GENERATOR \
#       --agent_type cwae --agent_name CWAE \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IT \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.001  --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_cw_talar_gen.log


## INCREMENTAL DOMAIN
#force_out_dim=2
#first_split_size=2
#other_split_size=2
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode ENCODER \
#       --agent_type cwae --agent_name CWAE \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag ID \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 10 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar_enc_best.log


#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode ENCODER \
#       --agent_type cwae --agent_name CWAE \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag ID \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 10 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar_enc.log

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name GeneratorCIFAR --mode GENERATOR \
#       --agent_type cwae --agent_name CWAE \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag ID \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 100 --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar_gen.log



## INCREMENTAL CLASS
#force_out_dim=10
#first_split_size=2
#other_split_size=2
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode ENCODER \
#       --agent_type cwae --agent_name CWAE --incremental_class --no_class_remap \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IC \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-3 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 1 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IC_cw_talar_enc_best.log


#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode ENCODER \
#       --agent_type cwae --agent_name CWAE --incremental_class --no_class_remap \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IC \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-3 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 0.0001 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IC_cw_talar_enc1.log


#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name GeneratorCIFAR --mode GENERATOR \
#       --agent_type cwae --agent_name CWAE --incremental_class --no_class_remap \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IT \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-3 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 128 \
#       --reg_coef 0.0001 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IC_cw_talar_gen1.log



##################################################
##################################################
## CW
##################################################
##################################################

## INCREMENTAL TASK
#force_out_dim=0
#first_split_size=2
#other_split_size=2
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode CW \
#       --agent_type cwae1 --agent_name CWAE1 \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IT \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 10 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_cw_talar_Adam_best.log

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode CW \
#       --agent_type cwae1 --agent_name CWAE1 \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IT \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer SGD --lr 1e-2 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 1 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IT_cw_talar_SGD_best.log


# INCREMENTAL DOMAIN
force_out_dim=2
first_split_size=2
other_split_size=2

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
       --model_name ${model_name} --model_type ${model_type} \
       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
       --exp_tag ID \
       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
       --reg_coef 100 \
       | tee ${OUTDIR}/ID_cw_talar_test.log

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode CW \
#       --agent_type cwae1 --agent_name CWAE1 \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag ID \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-4 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 100  --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar_Adam_best.log
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode CW \
#       --agent_type cwae1 --agent_name CWAE1 \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag ID \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer SGD --lr 1e-2 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 0.1  --reg_coef_2 0 \
#       | tee ${OUTDIR}/ID_cw_talar_SGD_best.log
#
### INCREMENTAL CLASS
#force_out_dim=10
#first_split_size=2
#other_split_size=2
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" \
#       --dataset ${dataset} --train_aug --dataroot /shared/sets/datasets/vision \
#       --model_name ${model_name} --model_type ${model_type} \
#       --generator_type generator --generator_name EncoderCIFAR --mode CW \
#       --agent_type cwae1 --agent_name CWAE1 --incremental_class --no_class_remap \
#       --force_out_dim ${force_out_dim} --first_split_size ${first_split_size} --other_split_size ${other_split_size} \
#       --exp_tag IT  \
#       --schedule ${epoch_per_task} --batch_size ${batch} --optimizer Adam --lr 1e-3 \
#       --generator_epoch 10 --generator_lr 1e-4 --latent_size 64 \
#       --reg_coef 0.01 --reg_coef_2 0 \
#       | tee ${OUTDIR}/IC_cw_talar_Adam_best.log

