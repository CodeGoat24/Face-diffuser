DATASET_PATH="/data/CodeGoat24/ffhq_wild_files"
DEMO_NAME="ours"
CUDA_VISIBLE_DEVICES=3 accelerate launch \
    --mixed_precision=fp16 \
    facediffuser/test.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --finetuned_model_path model/SDM \
    --finetuned_model_text_only_path model/TDM \
    --test_reference_folder data/${DEMO_NAME} \
    --test_caption "${CAPTION}" \
    --output_dir outputs/${DEMO_NAME} \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --max_num_objects 4 \
    --object_resolution 256 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 1 \
    --num_rows 1 \
    --seed 47 \
    --guidance_scale 5 \
    --inference_steps 50 \
    --start_merge_step 10 \
    --final_step 30 \
    --no_object_augmentation \
    --test_batch_size 8 \
    --dataset_name ${DATASET_PATH} \
