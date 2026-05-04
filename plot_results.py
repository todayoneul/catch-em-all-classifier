import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. 설정
save_dir = "./assets"
os.makedirs(save_dir, exist_ok=True)

experiments = {
    "ViT Full": "reports/experiment_results_vit_full.csv",
    "ViT LoRA": "reports/experiment_results_vit_lora.csv",
    "ViT QLoRA": "reports/experiment_results_vit_qlora.csv",
    "ResNet50": "reports/experiment_results_resnet50.csv"
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 파, 주, 초, 빨

# 2. 그래프 그리기 (1행 2열 구조)
# 2개의 그래프가 넉넉히 들어가도록 가로 길이를 20으로 늘립니다.
fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Model Performance & Training Analytics', fontsize=18, fontweight='bold')

data_found = False

for i, (label, filename) in enumerate(experiments.items()):
    if os.path.exists(filename):
        data_found = True
        df = pd.read_csv(filename)
        epochs = df['epoch']
        
        # 1. Validation Accuracy 그래프 (마커 추가)
        if 'eval_accuracy' in df.columns:
            ax_acc.plot(epochs, df['eval_accuracy'], marker='o', color=colors[i], linewidth=2, markersize=5, label=label)
            
        # 2. Train & Val Loss 그래프 (실선: Train, 점선: Val)
        if 'train_loss' in df.columns:
            ax_loss.plot(epochs, df['train_loss'], color=colors[i], linestyle='-', linewidth=2, alpha=0.6, label=f'{label} (Train)')
        if 'eval_loss' in df.columns:
            ax_loss.plot(epochs, df['eval_loss'], color=colors[i], linestyle='--', linewidth=2.5, label=f'{label} (Val)')
        


if not data_found:
    print("오류: 그릴 수 있는 CSV 데이터를 찾을 수 없습니다.")
    exit()

# 3. 그래프 세부 설정 및 꾸미기
# Accuracy 차트 설정
ax_acc.set_title('Validation Accuracy', fontsize=14)
ax_acc.set_xlabel('Epoch', fontsize=12)
ax_acc.set_ylabel('Accuracy', fontsize=12)
ax_acc.grid(True, linestyle=':', alpha=0.7)
ax_acc.legend(fontsize=10, loc='lower right')

# Loss 차트 설정
ax_loss.set_title('Training & Validation Loss', fontsize=14)
ax_loss.set_xlabel('Epoch', fontsize=12)
ax_loss.set_ylabel('Loss', fontsize=12)
ax_loss.grid(True, linestyle=':', alpha=0.7)
ax_loss.legend(fontsize=9, loc='upper right', ncol=2)


plt.tight_layout()

# 4. 이미지 저장
output_path = os.path.join(save_dir, "training_analytics.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"그래프가 성공적으로 저장되었습니다: {output_path}")

# 화면에 띄워서 확인
plt.show()