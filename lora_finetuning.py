import os
import torch
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
    r=16,                       # LoRA Rank (보통 8이나 16을 많이 씁니다)
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


training_args = TrainingArguments(
    output_dir="./vit-pokemon-lora",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-4,            
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs_lora',
    logging_steps=50,
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

print("=== 2단계: LoRA 학습 시작 ===")
trainer.train()

print("=== 2단계: Test Set 최종 평가 ===")
metrics = trainer.evaluate(test_ds)
print(metrics)


# 5. 파인튜닝 모델(LoRA 가중치) 최종 저장
save_directory = "./saved_model/exp2_vit_lora"
os.makedirs(save_directory, exist_ok=True)


model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print(f"=== 2단계 실험 모델 저장 완료 ===")
print(f"저장 경로: {save_directory} (용량이 매우 작은지 확인해 보세요!)")