import os
import gc
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel, PeftConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm


# 1. 평가할 모델 목록 및 설정
models_to_evaluate = {
    "ResNet50": "./saved_model/best_resnet50_pokemon",
    "ViT_Full": "./saved_model/best_vit_full",
    "ViT_LoRA": "./saved_model/best_vit_lora",
    "ViT_QLoRA": "./saved_model/best_vit_qlora"
}

dataset_path = "./PokemonData"
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("./assets", exist_ok=True)

print(f"총 {len(models_to_evaluate)}개 모델의 일괄 평가를 시작합니다. (Device: {device})")


# 2. 데이터셋 로드
print("\n데이터셋을 로드하고 분할하는 중... (seed=42)")
raw_dataset = load_dataset("imagefolder", data_dir=dataset_path)
split_dataset = raw_dataset["train"].train_test_split(test_size=0.2, seed=42)
test_ds = split_dataset["test"]
class_names = test_ds.features["label"].names

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


# 3. 모델 순회하며 평가 진행
for model_name, model_path in models_to_evaluate.items():
    print(f"\n" + "="*50)
    print(f"[{model_name}] 평가 시작! (경로: {model_path})")
    
    if not os.path.exists(model_path):
        print(f"폴더를 찾을 수 없어 건너뜁니다: {model_path}")
        continue

    # --- A. 모델 및 전처리기 로드 (LoRA 대응) ---
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    try:
        if is_peft:
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path
            
            try:
                processor = AutoImageProcessor.from_pretrained(model_path)
            except:
                processor = AutoImageProcessor.from_pretrained(base_model_name)
                
            # Full 모델의 이름표 빌려오기
            config_path = "./saved_model/best_vit_full/config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = json.load(f)
                id2label = full_config["id2label"]
                label2id = full_config["label2id"]

            base_model = AutoModelForImageClassification.from_pretrained(
                base_model_name, num_labels=150, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
            )
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModelForImageClassification.from_pretrained(model_path)

        model.to(device)
        model.eval()
    except Exception as e:
        print(f"{model_name} 로드 실패: {e}")
        continue

    # --- B. 데이터 전처리 적용 (모델별로 image_mean/std가 다를 수 있음) ---
    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    size = processor.size.get("height", 224) if isinstance(processor.size, dict) else processor.size
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize
    ])

    def apply_transforms(examples):
        examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
        return examples

    test_ds.set_transform(apply_transforms)
    test_loader = DataLoader(test_ds, collate_fn=collate_fn, batch_size=32, shuffle=False)

    # --- C. 추론 (Inference) ---
    y_true = []
    y_pred = []

    print(f"🔍 {model_name} 모델로 테스트 이미지를 예측 중입니다...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{model_name} Inference"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]
            
            outputs = model(pixel_values=pixel_values)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            y_true.extend(labels.numpy())
            y_pred.extend(predictions.cpu().numpy())

    # --- D. 통계 지표 CSV 저장 ---
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    csv_filename = f"classification_report_{model_name.lower()}.csv"
    report_df.to_csv(csv_filename)
    print(f"✅ {csv_filename} 저장 완료!")

    # --- E. 혼동 행렬 시각화 및 PNG 저장 ---
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0) # 정답을 맞춘 경우는 0으로 치환

    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm[i, j] > 0:
                confused_pairs.append({
                    'True Label': class_names[i],
                    'Predicted Label': class_names[j],
                    'Count': cm[i, j]
                })

    confused_df = pd.DataFrame(confused_pairs)
    if not confused_df.empty:
        confused_df = confused_df.sort_values(by='Count', ascending=False).head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Count', 
            y=confused_df.apply(lambda row: f"True: {row['True Label']} \nPred: {row['Predicted Label']}", axis=1),
            data=confused_df,
            palette="Reds_r"
        )
        plt.title(f"Top 10 Most Confused Pairs ({model_name})", fontsize=16, fontweight='bold')
        plt.xlabel("Number of Misclassifications", fontsize=12)
        plt.ylabel("Confused Pair", fontsize=12)
        plt.tight_layout()

        png_filename = f"./assets/top10_confusions_{model_name.lower()}.png"
        plt.savefig(png_filename, dpi=300)
        plt.close() # 그래프가 겹치지 않게 닫아주기
        print(f"{png_filename} 시각화 완료")
    else:
        print(f"{model_name} 모델은 오분류가 전혀 없습니다!")

    # 한 모델의 평가가 끝나면 메모리에서 삭제하여 다음 모델이 로드될 수 있는 공간 확보
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

print("모든 모델의 평가 및 결과 저장이 완료")