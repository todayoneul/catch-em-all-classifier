import os
import torch
import pandas as pd
import json
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification, 
    TrainingArguments, 
    Trainer
)
from torchvision.transforms import (
    Compose,
    Resize,
    RandomCrop,
    CenterCrop,
    RandomHorizontalFlip,
    ColorJitter,
    ToTensor,
    Normalize,
)

# ==========================================
# 1. 환경 및 데이터셋 설정
# ==========================================
# Kaggle 데이터셋이 있는 폴더 경로 (현재 디렉토리에 맞게 수정하세요)
dataset_path = "./PokemonData" 

print("데이터셋을 로드하는 중...")
dataset = load_dataset("imagefolder", data_dir=dataset_path)

# Train / Test 분할 (8:2 비율, 재현성을 위해 seed 고정)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

# 클래스 라벨 매핑 정보 추출 (포켓몬 이름들)
labels = train_ds.features["label"].names
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}
num_labels = len(labels)
print(f"총 {num_labels}개의 포켓몬 클래스를 학습합니다.")

# ==========================================
# 2. 이미지 전처리 및 증강 (Data Augmentation)
# ==========================================
# Base Model (분류 헤드가 없는 in21k 모델 사용)
model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)

# ViT 모델이 요구하는 기본 정규화 수치
image_mean = processor.image_mean
image_std = processor.image_std

# 학습용 데이터 증강 파이프라인
train_transforms = Compose([
    Resize((256, 256)),               
    RandomCrop((224, 224)),           
    RandomHorizontalFlip(p=0.5),      
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(),                       
    Normalize(mean=image_mean, std=image_std),
])

# 평가/테스트용 파이프라인 (증강 미적용)
val_transforms = Compose([
    Resize((256, 256)),
    CenterCrop((224, 224)),           
    ToTensor(),
    Normalize(mean=image_mean, std=image_std),
])

def apply_train_transforms(example_batch):
    example_batch["pixel_values"] = [
        train_transforms(img.convert("RGB")) for img in example_batch["image"]
    ]
    return example_batch

def apply_val_transforms(example_batch):
    example_batch["pixel_values"] = [
        val_transforms(img.convert("RGB")) for img in example_batch["image"]
    ]
    return example_batch

# 데이터셋에 전처리 함수 적용
train_ds.set_transform(apply_train_transforms)
test_ds.set_transform(apply_val_transforms)

# ==========================================
# 3. 모델 로드 및 평가 지표 설정
# ==========================================
print("모델을 로드하는 중...")
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# 평가 지표 (Accuracy, Precision, Recall)
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    acc = accuracy.compute(predictions=predictions, references=labels)
    prec = precision.compute(predictions=predictions, references=labels, average="macro", zero_division=0)
    rec = recall.compute(predictions=predictions, references=labels, average="macro", zero_division=0)
    
    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"]
    }

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch])
    }

# 실험 이름을 지정하세요. (예: "vit_lora", "vit_full", "resnet50" 등)
EXPERIMENT_NAME = "vit_full" 

training_args = TrainingArguments(
    output_dir=f"./results_{EXPERIMENT_NAME}",
    remove_unused_columns=False,
    eval_strategy="epoch",       # 매 에폭마다 평가 수행
    save_strategy="epoch",       # 매 에폭마다 모델 체크포인트 저장
    learning_rate=5e-5,          # LoRA/QLoRA는 5e-4, Full/ResNet은 5e-5 권장
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=50,         # 총 50 Epoch까지 논스톱 학습
    weight_decay=0.01,
    load_best_model_at_end=True, # 50번 중 가장 성능이 좋았던 모델을 최종적으로 불러옴
    metric_for_best_model="accuracy",
    logging_dir=f'./logs_{EXPERIMENT_NAME}',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=processor,
    compute_metrics=compute_metrics,
)

print(f"=== [{EXPERIMENT_NAME}] 50 Epoch 자동 학습 및 평가 시작 ===")
trainer.train()

# ==========================================
# 5. 실험 결과 추출 및 CSV 자동 저장
# ==========================================
print("=== 학습 내역(History) 추출 및 저장 중 ===")

# Trainer의 log_history에는 학습/평가 때마다 기록된 모든 지표가 리스트 형태로 들어있습니다.
history = trainer.state.log_history

train_logs = []
eval_logs = []

# 로그 분리 작업
for log in history:
    if 'eval_loss' in log:
        eval_logs.append(log)
    elif 'loss' in log: # train_loss
        train_logs.append(log)

# 평가(Eval) 로그를 데이터프레임으로 변환
df_eval = pd.DataFrame(eval_logs)

# 5, 10, 15 ... 50 에폭의 데이터만 필터링 (원하신다면 모든 에폭을 저장해도 무방합니다)
# Epoch 값이 소수점으로 떨어질 수 있으므로 반올림 후 필터링
df_eval['epoch'] = df_eval['epoch'].round(0)
df_eval_filtered = df_eval[df_eval['epoch'] % 5 == 0].copy()

# 보기 좋게 컬럼 정리
columns_to_save = ['epoch', 'eval_loss', 'eval_accuracy', 'eval_precision', 'eval_recall']
df_eval_filtered = df_eval_filtered[columns_to_save]

# CSV 파일로 저장
csv_filename = f"experiment_results_{EXPERIMENT_NAME}.csv"
df_eval_filtered.to_csv(csv_filename, index=False)

print(f"✅ 결과가 성공적으로 저장되었습니다: {csv_filename}")
print(df_eval_filtered.to_string(index=False))

# ==========================================
# 6. 최종 베스트 모델 저장
# ==========================================
save_directory = f"./saved_models/best_{EXPERIMENT_NAME}"
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f"🏆 가장 성능이 좋았던 베스트 모델이 저장되었습니다: {save_directory}")