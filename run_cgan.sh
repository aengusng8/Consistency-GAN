# bash run_cm.sh imagenet_64 train

CURDIR=$(
    cd $(dirname $0)
    pwd
)
echo 'The work dir is: ' $CURDIR

DATASET=$1
MODE=$2
GPUS=$3

if [ -z "$1" ]; then
    GPUS=1
fi

echo $DATASET $MODE $GPUS

export DATASET_DIR=../datasets/$DATASET

# -----------------  Consistency GAN -----------------
if [[ $MODE == train ]]; then
    echo "==> Training Consistency Without Distillation"

    if [[ $DATASET == imagenet_64 ]]; then
        python -m scripts.cm_train --training_mode consistency_training --data_dir $DATASET_DIR \
            --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 --total_training_steps 1 --loss_norm lpips --lr_anneal_steps 0 \
            --attention_resolutions 32,16,8 --use_scale_shift_norm True --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 \
            --schedule_sampler uniform --use_fp16 False --weight_decay 0.0 --weight_schedule uniform \
            --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True \
            --global_batch_size 1 --image_size 64 --lr 0.0001 --class_cond False

    fi
fi

# python cm_train.py --training_mode consistency_training
# --target_ema_mode adaptive
# --start_ema 0.95 --scale_mode progressive
# --start_scales 2 --end_scales 200
# --total_training_steps 800000
# --loss_norm lpips
# --lr_anneal_steps 0
# --teacher_model_path /path/to/edm_imagenet64_ema.pt
# --attention_resolutions 32,16,8
# --class_cond True --use_scale_shift_norm True
# --dropout 0.0 --teacher_dropout 0.1
# --ema_rate 0.999,0.9999,0.9999432189950708
# --global_batch_size 2048 --image_size 64
# --lr 0.0001 --num_channels 192
# --num_head_channels 64 --num_res_blocks 3
# --resblock_updown True --schedule_sampler uniform
# --use_fp16 True --weight_decay 0.0
# --weight_schedule uniform --data_dir /path/to/imagenet64
