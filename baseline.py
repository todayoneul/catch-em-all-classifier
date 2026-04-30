import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os

# 1. 모델 및 프로세서 로드 (ImageNet 사전 학습 가중치)
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# 2. 테스트할 포켓몬 이미지 경로 설정 (예: 피카츄 이미지)
# 다운로드 받으신 데이터셋 내의 임의의 이미지 경로를 지정하세요.
image_path = "./PokemonData/Pikachu/00000000.jpg"
image = Image.open(image_path).convert("RGB")

# 3. 이미지 전처리 및 추론
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 4. Top-5 예측 결과 출력
probs = torch.nn.functional.softmax(logits, dim=-1)[0]
top5_prob, top5_catid = torch.topk(probs, 5)

print("=== 0단계: Base Model 예측 결과 (Fine-tuning X) ===")
for i in range(top5_prob.size(0)):
    class_name = model.config.id2label[top5_catid[i].item()]
    print(f"{i+1}. {class_name} ({top5_prob[i].item() * 100:.2f}%)")