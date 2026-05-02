import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. 폴더 및 파일 설정
# 결과 이미지를 저장할 assets 폴더 생성
save_dir = "./assets"
os.makedirs(save_dir, exist_ok=True)

# 실험별 라벨 이름과 실제 생성된 CSV 파일 이름 매핑
experiments = {
    "ViT Full": "experiment_results_vit_full.csv",
    "ViT LoRA": "experiment_results_vit_lora.csv",
    "ViT QLoRA": "experiment_results_vit_qlora.csv",
    "ResNet50": "experiment_results_resnet50.csv"
}

# 그래프 스타일 설정 (선 색상과 마커 모양)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 파, 주, 초, 빨
markers = ['o', 's', '^', 'D']


# 2. 그래프 그리기 준비 (1행 2열 구조)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance Comparison (5~50 Epochs)', fontsize=16, fontweight='bold')

data_found = False

for i, (label, filename) in enumerate(experiments.items()):
    # 파일이 존재하는지 확인 (아직 돌리지 않은 실험은 건너뜀)
    if os.path.exists(filename):
        data_found = True
        df = pd.read_csv(filename)
        
        # X축 데이터 (Epoch)
        epochs = df['epoch']
        
        # 첫 번째 그래프: Accuracy (정확도)
        ax1.plot(epochs, df['eval_accuracy'], marker=markers[i], color=colors[i], 
                 linewidth=2, label=label, markersize=6)
        
        # 두 번째 그래프: Loss (손실 - 낮을수록 정답에 대한 확신이 높음)
        ax2.plot(epochs, df['eval_loss'], marker=markers[i], color=colors[i], 
                 linewidth=2, label=label, markersize=6, linestyle='--')
    else:
        print(f"경고: {filename} 파일을 찾을 수 없어 그래프에서 제외합니다.")

if not data_found:
    print("❌ 오류: 그릴 수 있는 CSV 데이터가 하나도 없습니다. 학습을 먼저 진행해 주세요.")
    exit()


# 3. 그래프 세부 설정 및 꾸미기

# Accuracy 그래프 설정
ax1.set_title('Validation Accuracy over Epochs', fontsize=14)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='lower right')

# Loss 그래프 설정
ax2.set_title('Validation Loss over Epochs', fontsize=14)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper right')

plt.tight_layout()

# 4. assets 폴더에 이미지 저장
output_path = os.path.join(save_dir, "performance_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"그래프가 성공적으로 저장되었습니다: {output_path}")

# 화면에 띄워서 바로 확인
plt.show()