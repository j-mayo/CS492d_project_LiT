export WORKDIR="$HOME/diffusion/Project"
export MODEL_NAME="$WORKDIR/Relight/models"
export TRAIN_DATA_PATH="$WORKDIR/dataset/train"
export VAL_DATA_PATH="$WORKDIR/dataset/train"
export TRAIN_LAT_PATH="$WORKDIR/dataset_vae/train"
export VAL_LAT_PATH="$WORKDIR/dataset_vae/train"
export TRAIN_JSON_PATH="$WORKDIR/dataset/preprocess/train.json"
export VAL_JSON_PATH="$WORKDIR/dataset/preprocess/eval.json"
export POSE_PATH="$WORKDIR/dataset/light_pos.npy"
export OUTPUT_DIR="./runs/diffrelight"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

accelerate launch --mixed_precision="no" train_from_scratch.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --train_img_path=$TRAIN_DATA_PATH \
  --val_img_path=$TRAIN_DATA_PATH \
  --train_lat_path=$TRAIN_LAT_PATH \
  --val_lat_path=$TRAIN_LAT_PATH \
  --train_json_path=$TRAIN_JSON_PATH \
  --pose_data_path=$POSE_PATH \
  --resolution=512 \
  --train_batch_size=8 \
  --val_batch_size=8 \
  --num_train_epochs=20 \
  --snr_gamma=5 \
  --validation_epochs=1 \
  --prediction_type="epsilon" \
  --checkpointing_steps=500 \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --gradient_accumulation_steps=5 \
  --lr_warmup_steps=0 \
  --seed=42 \
  --checkpoints_total_limit 5 \
  --val_json_path=$VAL_JSON_PATH \
  --dataloader_num_workers=4 \
  --loss_type="dream" \
  --dream_detail_preservation=1. \
  --train_encoder \
  --resume_from_checkpoint='latest'
