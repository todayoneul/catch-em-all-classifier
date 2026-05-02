import streamlit as st
import torch
import requests
import os
import json
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel, PeftConfig
from PIL import Image
import torch.nn.functional as F

# 페이지 기본 설정
st.set_page_config(page_title="포켓몬 분류기 데모", page_icon="🐾", layout="wide")

st.title("🐾 Pokemon Classifier")
st.markdown("학습된 모델(ViT, ResNet, LoRA 등)을 사용하여 포켓몬의 이름을 예측하고 모습을 확인합니다.")

# 1. 만능 모델 로드 함수 (캐싱 & 에러 방어 적용)
@st.cache_resource
def load_model(model_path):
    try:
        is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        if is_peft:
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path
            
            try:
                processor = AutoImageProcessor.from_pretrained(model_path)
            except:
                processor = AutoImageProcessor.from_pretrained(base_model_name)
            
            # LoRA 모델을 위해 Full 모델의 이름표(config.json) 빌려오기
            try:
                # 사용자의 실제 Full 파인튜닝 폴더 경로
                config_path = "./saved_model/best_vit_full/config.json"
                with open(config_path, "r", encoding="utf-8") as f:
                    full_config = json.load(f)
                    id2label = full_config["id2label"]
                    label2id = full_config["label2id"]
            except Exception as e:
                print(f"이름표 로드 실패: {e}. 임시 라벨을 사용합니다.")
                id2label = {str(i): f"LABEL_{i}" for i in range(150)}
                label2id = {f"LABEL_{i}": str(i) for i in range(150)}

            base_model = AutoModelForImageClassification.from_pretrained(
                base_model_name,
                num_labels=150,     
                id2label=id2label,  
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            
        else:
            # 일반 파인튜닝 모델 (ViT Full, ResNet50 등)
            processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModelForImageClassification.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        return processor, model, device, None
        
    except Exception as e:
        return None, None, None, str(e)


# 2. PokeAPI 이미지 호출 함수

@st.cache_data
def get_pokemon_image_url(pokemon_name):
    try:
        name_lower = pokemon_name.lower().replace(" ", "-").replace(".", "").replace("'", "")
        url = f"https://pokeapi.co/api/v2/pokemon/{name_lower}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["sprites"]["other"]["official-artwork"]["front_default"]
        return None
    except:
        return None


# 3. 사이드바 UI - 모델 드롭다운 선택

st.sidebar.header("⚙️ 실험 설정")


MODEL_PATHS = {
    "ViT Full Fine-tuning": "./saved_model/best_vit_full",
    "ViT + LoRA": "./saved_model/best_vit_lora",
    "ViT + QLoRA (4-bit)": "./saved_model/best_vit_qlora",
    "ResNet50": "./saved_model/best_resnet50_pokemon"
}

# 텍스트 입력 대신 마우스로 선택하는 드롭다운 박스 생성
selected_model_name = st.sidebar.selectbox(
    "테스트할 모델을 선택하세요:",
    list(MODEL_PATHS.keys())
)

# 선택한 모델의 경로를 가져옴
model_path = MODEL_PATHS[selected_model_name]

st.sidebar.info(f"현재 장치: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

# 모델 로드
processor, model, device, error_msg = load_model(model_path)

if error_msg:
    st.sidebar.error(f"모델 로드 실패: {error_msg}\n\n경로({model_path})에 모델이 존재하는지 확인하세요.")
elif model is not None:
    is_peft_ui = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_peft_ui:
        st.sidebar.success(f"⚡ {selected_model_name} 로드 완료!")
    else:
        st.sidebar.success(f"📦 {selected_model_name} 로드 완료!")

# 4. 메인 화면 - 이미지 업로드 및 추론
uploaded_file = st.file_uploader("포켓몬 이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner(f'{selected_model_name} 모델이 포켓몬을 식별하는 중...'):
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Softmax를 거쳐 확률 계산
        probs = F.softmax(logits, dim=-1)[0]
        top5_prob, top5_catid = torch.topk(probs, 5)
        
        # 정수/문자열 모두 유연하게 대응하는 .get() 방식 사용
        cat_id_0 = top5_catid[0].item()
        top1_label = model.config.id2label.get(cat_id_0, model.config.id2label.get(str(cat_id_0), "Unknown"))
        
        top1_score = top5_prob[0].item()
        artwork_url = get_pokemon_image_url(top1_label)
        
        st.markdown("---")
        
        # 레이아웃 구성: 좌측(입력 이미지) / 우측(예측 결과)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 입력된 이미지")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader(f"✨ 예측 결과: 1. {top1_label} ({top1_score * 100:.1f}%)")
            if artwork_url:
                st.image(artwork_url, use_container_width=True)
            else:
                st.info("해당 포켓몬의 공식 이미지를 불러올 수 없습니다.")

        # Top-5 예측 확률 바
        st.markdown("---")
        st.subheader("📊 Top-5 예측 확률")
        
        for i in range(top5_prob.size(0)):
            cat_id_i = top5_catid[i].item()
            label = model.config.id2label.get(cat_id_i, model.config.id2label.get(str(cat_id_i), "Unknown"))
            score = top5_prob[i].item()
            
            c1, c2 = st.columns([1, 4])
            with c1:
                st.write(f"**{i+1}. {label}**")
            with c2:
                st.progress(score)
                st.write(f"{score * 100:.2f}%")