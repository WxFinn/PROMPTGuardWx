
SAFE_EMBEDDINGS="./safe_embeddings/YOUR_SAFE_EMBEDDINGS_FOLDER"
CUDA_DEVICE=0

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" accelerate launch --main_process_port 29501 ./scripts/our_pipeline_sd_sdedit_v2.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --train_data_dir="./PromptGuardDataset/Disturbing/disturbing_filter&gpt4_train_sdv14" \
  --edited_data_dir="./PromptGuardDataset/Disturbing/disturbing_filter&gpt4_train_sdv14_edit" \
  --train_data_csv="./PromptGuardDataset/Disturbing/disturbing_filter&gpt4_train.csv" \
  --placeholder_token="<a>" \
  --initializer_token="safe" \
  --position="end" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=10 \
  --learning_rate=5.0e-04 \
  --scale_lr \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="$SAFE_EMBEDDINGS" \
  --num_vectors=1 \
  --coefficient=0.1 \
#  --resume_from_checkpoint= "latest" \