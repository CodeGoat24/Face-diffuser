export WANDB_NAME=text-only-ffhq-1_5-1e-5
export WANDB_DISABLE_SERVICE=true
export CUDA_VISIBLE_DEVICES="0,1,2,3,"

DATASET_PATH="data/ffhq_wild_files"

DATASET_NAME="ffhq"
FAMILY=runwayml
MODEL=stable-diffusion-v1-5

accelerate launch \
    --mixed_precision=bf16 \
    --machine_rank 0 \
    --num_machines 1 \
    --main_process_port 11135 \
    --num_processes 4 \
    --multi_gpu \
    facediffuser/train_text_only.py \
    --pretrained_model_name_or_path ${FAMILY}/${MODEL} \
    --dataset_name ${DATASET_PATH} \
    --logging_dir logs/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --output_dir models/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --max_train_steps 150000 \
    --num_train_epochs 150000 \
    --train_batch_size 4 \
    --learning_rate 1e-5 \
    --unet_lr_scale 1.0 \
    --checkpointing_steps 2000 \
    --mixed_precision bf16 \
    --allow_tf32 \
    --keep_only_last_checkpoint \
    --keep_interval 10000 \
    --seed 42 \
    --num_image_tokens 1 \
    --max_num_objects 4 \
    --train_resolution 512 \
    --uncondition_prob 0.1 \
    --disable_flashattention \
    --object_types person \
    --resume_from_checkpoint latest \
    --report_to wandb
