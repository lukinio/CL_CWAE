GPUID=$1
OUTDIR=outputs/ewc
REPEAT=1
mkdir -p $OUTDIR

#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#      --model_name MLP400 --agent_type customization  --agent_name EWC_online_mnist \
#      --force_out_dim 2 --first_split_size 2 --other_split_size 2 --optimizer Adam \
#      --schedule 4 --batch_size 128 --lr 0.001 --reg_coef 700 \
#      | tee ${OUTDIR}/EWC_online_id.log
#
#
#python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
#      --model_name MLP400 --agent_type customization  --agent_name EWC_mnist \
#      --force_out_dim 2 --first_split_size 2 --other_split_size 2 --optimizer Adam \
#      --schedule 4 --batch_size 128 --lr 0.001 --reg_coef 700 \
#      | tee ${OUTDIR}/EWC_id.log




python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
      --model_name MLP400 --agent_type customization  --agent_name EWC_mnist \
      --force_out_dim 2 --first_split_size 2 --other_split_size 2 --optimizer Adam \
      --schedule 4 --batch_size 128 --lr 0.001 --reg_coef 700 \
      | tee ${OUTDIR}/EWC_id_test.log

python -u iBatchLearn.py --gpuid "${GPUID}" --repeat "${REPEAT}" --dataroot /shared/sets/datasets/vision \
      --model_name MLP400 --agent_type customization  --agent_name EWC_online_mnist \
      --force_out_dim 2 --first_split_size 2 --other_split_size 2 --optimizer Adam \
      --schedule 4 --batch_size 128 --lr 0.001 --reg_coef 700 \
      | tee ${OUTDIR}/EWC_online_test.log

