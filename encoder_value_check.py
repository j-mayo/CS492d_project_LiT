import torch
from transformers import CLIPTextModel, CLIPTokenizer

# 모델 및 토크나이저 로드
pretrained_model_name_or_path = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path)
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path)

# 예시 프롬프트
prompts = ["A photo of a cat", "An illustration of a futuristic city", "Abstract art with vibrant colors"]

# 토큰화
inputs = tokenizer(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt")

# 텍스트 인코딩
with torch.no_grad():
    prompt_embeds = text_encoder(input_ids=inputs.input_ids).last_hidden_state  # shape: (batch_size, max_length, embedding_dim)

# 임베딩 벡터의 값 범위 및 통계 확인
print("Embedding shape:", prompt_embeds.shape)
print("Min value:", prompt_embeds.min().item())
print("Max value:", prompt_embeds.max().item())
print("Mean value:", prompt_embeds.mean().item())
print("Standard deviation:", prompt_embeds.std().item())

import torch
from transformers import CLIPTextModel, CLIPTokenizer


# 텍스트 인코더와 토크나이저 삭제
del text_encoder
del tokenizer

# GPU 메모리 캐시 정리
torch.cuda.empty_cache()