export MODEL_NAME="/workspace/Project/stable-diffusion-2-1-base"
export TRAIN_DATA_PATH="/workspace/dataset/train"
export VAL_DATA_PATH="/workspace/dataset/val"
export TRAIN_JSON_PATH="/workspace/dataset/preprocess/train.json"
export OUTPUT_DIR="./runs/vars_for_pds"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

accelerate launch --mixed_precision="no" train_rec.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_data_path=$TRAIN_DATA_PATH \
  --val_data_path=$VAL_DATA_PATH \
  --train_json_path=$TRAIN_JSON_PATH \
  --resolution=512 \
  --random_flip \
  --train_batch_size=4 \
  --num_train_epochs=1 \
  --validation_epochs=1 \
  --checkpointing_steps=2000 \
  --learning_rate=1e-04 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --seed=42 \
  --checkpoints_total_limit 2 \
  --validation_prompt="a cute bear" \
  --lighting_layers=8 \
    # --val_json_path=$VAL_JSON_PATH \