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
    Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip,
    ColorJitter, ToTensor, Normalize,
)
from peft import LoraConfig, get_peft_model


# 1. 데이터셋 로드 및 전처리 (1단계와 동일)
dataset_path = "./PokemonData" 
dataset = load_dataset("imagefolder", data_dir=dataset_path)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

labels = train_ds.features["label"].names
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}
num_labels = len(labels)

model_name = "google/vit-base-patch16-224-in21k"
processor = ViTImageProcessor.from_pretrained(model_name)
image_mean, image_std = processor.image_mean, processor.image_std

train_transforms = Compose([
    Resize((256, 256)), RandomCrop((224, 224)), RandomHorizontalFlip(p=0.5),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ToTensor(), Normalize(mean=image_mean, std=image_std),
])
val_transforms = Compose([
    Resize((256, 256)), CenterCrop((224, 224)), ToTensor(), Normalize(mean=image_mean, std=image_std),
])

def apply_train_transforms(example_batch):
    example_batch["pixel_values"] = [train_transforms(img.convert("RGB")) for img in example_batch["image"]]
    return example_batch

def apply_val_transforms(example_batch):
    example_batch["pixel_values"] = [val_transforms(img.convert("RGB")) for img in example_batch["image"]]
    return example_batch

train_ds.set_transform(apply_train_transforms)
test_ds.set_transform(apply_val_transforms)


# 2. Base 모델 로드
print("Base 모델 로드 중...")
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)


# 3. LoRA 설정 및 PEFT 모델 래핑
lora_config = LoraConfig(
    r=16,                       # LoRA Rank
    lora_alpha=16,              # 스케일링 팩터 (보통 r과 같게 하거나 2배로 설정)
    target_modules=["query", "value"], # ViT 어텐션 레이어의 타겟 모듈 이름
    lora_dropout=0.1,           # 과적합 방지용 드롭아웃
    bias="none",
    modules_to_save=["classifier"], # 최종 분류기(헤드)는 새롭게 학습해야 하므로 저장 대상에 포함
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 


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
    return {"accuracy": acc["accuracy"], "precision": prec["precision"], "recall": rec["recall"]}

def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["label"] for x in batch])
    }

EXPERIMENT_NAME = "vit_lora" 
training_args = TrainingArguments(
    output_dir=f"./results_{EXPERIMENT_NAME}",
    remove_unused_columns=False,
    eval_strategy="epoch",       # 매 에폭마다 평가 수행
    save_strategy="epoch",       # 매 에폭마다 모델 체크포인트 저장
    learning_rate=5e-4,          # LoRA/QLoRA는 5e-4, Full/ResNet은 5e-5 권장
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

 
# 5. 실험 결과 추출 및 CSV 자동 저장
 
print("=== 학습 내역(History) 추출 및 저장 중 ===")

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

# 5, 10, 15 ... 50 에폭의 데이터만 필터링
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

 
# 6. 최종 베스트 모델 저장
 
save_directory = f"./saved_model/best_{EXPERIMENT_NAME}"
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)
print(f"🏆 가장 성능이 좋았던 베스트 모델이 저장되었습니다: {save_directory}")