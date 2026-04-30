import os
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    ResNetForImageClassification, 
    TrainingArguments, 
    Trainer
)
from torchvision.transforms import (
    Compose, Resize, RandomCrop, CenterCrop, RandomHorizontalFlip,
    ColorJitter, ToTensor, Normalize,
)


# 1. 데이터셋 로드 및 전처리
dataset_path = "./PokemonData" # 실제 데이터 폴더 경로로 확인 후 사용하세요
dataset = load_dataset("imagefolder", data_dir=dataset_path)
dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]

labels = train_ds.features["label"].names
id2label = {str(i): c for i, c in enumerate(labels)}
label2id = {c: str(i) for i, c in enumerate(labels)}
num_labels = len(labels)

# ResNet50 프로세서 로드
model_name = "microsoft/resnet-50"
processor = AutoImageProcessor.from_pretrained(model_name)
image_mean, image_std = processor.image_mean, processor.image_std

# 동일한 Data Augmentation 적용 (공정한 비교를 위해)
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


# 2. ResNet50 모델 로드
print("ResNet50 모델 로드 중...")
# 기존 1000개 클래스용 헤드를 버리고 새로운 헤드를 달기 위해 ignore_mismatched_sizes=True 필수
model = ResNetForImageClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True 
)


# 3. 평가 지표 및 Trainer 설정
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
    output_dir="./resnet50-pokemon-finetuned",
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5, 
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs_resnet',
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

print("=== 4단계: ResNet50 학습 시작 ===")
trainer.train()

print("=== 4단계: Test Set 최종 평가 ===")
metrics = trainer.evaluate(test_ds)
print(metrics)

# 4. 파인튜닝 모델 최종 저장
save_directory = "./saved_models/exp4_resnet50_finetuned"
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory)
processor.save_pretrained(save_directory)

print(f"=== 4단계 실험 모델 저장 완료 ===")
print(f"저장 경로: {save_directory}")