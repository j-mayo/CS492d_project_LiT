export WORKDIR="$HOME/diffusion/Project"
export MODEL_NAME="$WORKDIR/Relight/models"
export EVAL_DATA_PATH="$WORKDIR/dataset/train"
export EVAL_JSON_PATH="$WORKDIR/dataset/preprocess/eval.json"
export POSE_PATH="$WORKDIR/dataset/light_pos.npy"
export LIGHT_ENC_PATH="$WORKDIR/Relight/runs/encoder"
export OUTPUT_DIR="$WORKDIR/Relight/eval_results/pndm"
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

accelerate launch --mixed_precision="no" eval_model.py \
  --model_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --eval_img_path=$EVAL_DATA_PATH \
  --eval_json_path=$EVAL_JSON_PATH \
  --pose_data_path=$POSE_PATH \
  --resolution=512 \
  --eval_batch_size=8 \
  --seed=4 \
  --lighting_layers=4 \
  --latent_layers=2 \
  --logging_dir="logs" \
  --dataloader_num_workers=4