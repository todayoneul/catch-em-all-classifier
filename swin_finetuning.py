import os
import torch
import pandas as pd
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    SwinForImageClassification, 
    TrainingArguments, 
    Trainer
)
from torchvision.transforms import (
    Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip,
    ColorJitter, ToTensor, Normalize,
)

# 1. 환경 및 데이터셋 설정
dataset_path = "./PokemonData" 
print("데이터셋을 로드하는 중...")
dataset = load_dataset("imagefolder", data_dir=dataset_path)

# Train / Test 분할 (8:2 비율, 공정한 비교를 위해 기존과 동일한 seed=42 고정)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

# 클래스 라벨 매핑 정보 추출
labels = train_ds.features["label"].names
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}
num_labels = len(labels)
print(f"총 {num_labels}개의 포켓몬 클래스를 학습합니다.")

# 2. 이미지 전처리 및 증강 (Data Augmentation)
# 계층적 트랜스포머 아키텍처인 Swin-Tiny 사용
model_name = "microsoft/swin-tiny-patch4-window7-224"
processor = AutoImageProcessor.from_pretrained(model_name)

# Swin 모델이 요구하는 기본 정규화 수치
image_mean = processor.image_mean
image_std = processor.image_std

# 학습용 데이터 증강 파이프라인 (기존 실험들과 100% 동일)
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

# 3. Swin 모델 로드
print("Swin 모델 로드 중...")
model = SwinForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True # 기존 1000개 클래스 헤드를 무시하고 새로운 헤드(150개)를 초기화
)

# 4. 평가 지표 및 Trainer 설정
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

EXPERIMENT_NAME = "swin" 
training_args = TrainingArguments(
    output_dir=f"./results_{EXPERIMENT_NAME}",
    remove_unused_columns=False,
    eval_strategy="epoch",       # 매 에폭마다 평가 수행
    save_strategy="epoch",       # 매 에폭마다 모델 체크포인트 저장
    learning_rate=5e-5,          # Full Fine-tuning 기준 LR
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=50,         # 총 50 Epoch
    weight_decay=0.01,
    load_best_model_at_end=True, # 50번 중 가장 성능이 좋았던 모델을 최종 로드
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

# 5. 실험 결과 추출 및 CSV 자동 저장
print("=== 학습 내역(History) 추출 및 저장 중 ===")

history = trainer.state.log_history

train_logs = []
eval_logs = []

for log in history:
    if 'eval_loss' in log:
        eval_logs.append(log)
    elif 'loss' in log:
        train_logs.append(log)

# 평가(Eval) 로그 데이터프레임 변환
df_eval = pd.DataFrame(eval_logs)

# 5, 10, 15 ... 50 에폭의 데이터 필터링
df_eval['epoch'] = df_eval['epoch'].round(0)
df_eval_filtered = df_eval[df_eval['epoch'] % 5 == 0].copy()

# 컬럼 정리
columns_to_save = ['epoch', 'eval_loss', 'eval_accuracy', 'eval_precision', 'eval_recall']
df_eval_filtered = df_eval_filtered[columns_to_save]

# CSV 파일 저장 (reports 폴더에 통합)
os.makedirs("reports", exist_ok=True)
csv_filename = os.path.join("reports", f"experiment_results_{EXPERIMENT_NAME}.csv")
df_eval_filtered.to_csv(csv_filename, index=False)

print(f"결과가 성공적으로 저장되었습니다: {csv_filename}")
print(df_eval_filtered.to_string(index=False))

# 6. 최종 베스트 모델 저장
save_directory = f"./saved_model/best_{EXPERIMENT_NAME}"
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f"가장 성능이 좋았던 베스트 모델이 저장되었습니다: {save_directory}")
