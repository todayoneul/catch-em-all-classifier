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
            # 1. LoRA 설정 파일 읽기
            config = PeftConfig.from_pretrained(model_path)
            base_model_name = config.base_model_name_or_path
            
            # 2. 전처리기 로드 (로컬 실패 시 원본 베이스 모델에서 가져옴)
            try:
                processor = AutoImageProcessor.from_pretrained(model_path)
            except:
                processor = AutoImageProcessor.from_pretrained(base_model_name)
            
            # 3. 포켓몬 이름표(id2label) 가져오기
            try:
                # 1단계에서 학습했던 Full 파인튜닝 모델의 config 파일 경로 (자신의 환경에 맞게 확인)
                config_path = "./saved_model/exp1_vit_full_finetuned/config.json"
                with open(config_path, "r", encoding="utf-8") as f:
                    full_config = json.load(f)
                    id2label = full_config["id2label"]
                    label2id = full_config["label2id"]
            except Exception as e:
                # 터미널에만 경고 출력 (UI 캐싱 에러 방지)
                print("⚠️ 이름표(config.json)를 찾지 못했습니다. 임시 라벨을 사용합니다.")
                id2label = {str(i): f"LABEL_{i}" for i in range(150)}
                label2id = {f"LABEL_{i}": str(i) for i in range(150)}

            # 4. Base 모델 로드 (150개 클래스와 이름표 주입)
            base_model = AutoModelForImageClassification.from_pretrained(
                base_model_name,
                num_labels=150,     
                id2label=id2label,  
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
            
            # 5. Base 모델 + LoRA 어댑터 결합
            model = PeftModel.from_pretrained(base_model, model_path)
            
        else:
            # 일반 파인튜닝 모델 (ViT Full, ResNet50 등)
            processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModelForImageClassification.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        # 정상 로드 시 에러 메시지(None) 반환
        return processor, model, device, None
        
    except Exception as e:
        # 에러 발생 시 UI를 호출하지 않고 텍스트로 에러 반환
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

 
# 3. 사이드바 UI - 실험 설정
 
st.sidebar.header("⚙️ 실험 설정")
# 기본 경로 설정
default_path = "./saved_model/best_vit_qlora" 
model_path = st.sidebar.text_input("모델 저장 경로", value=default_path)

st.sidebar.info(f"현재 장치: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}")

# 모델 로드 (4개의 반환값 받기)
processor, model, device, error_msg = load_model(model_path)

# 에러 메시지 UI 처리
if error_msg:
    st.sidebar.error(f"모델 로드 실패: {error_msg}")
elif model is not None:
    is_peft_ui = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    if is_peft_ui:
        st.sidebar.success("⚡ PEFT (LoRA/QLoRA) 모델 로드 완료!")
    else:
        st.sidebar.success("📦 일반 (Full Weights) 모델 로드 완료!")

 
# 4. 메인 화면 - 이미지 업로드 및 추론
uploaded_file = st.file_uploader("포켓몬 이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner('야생의 포켓몬을 식별하는 중...'):
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Softmax를 거쳐 확률 계산
        probs = F.softmax(logits, dim=-1)[0]
        top5_prob, top5_catid = torch.topk(probs, 5)
        
        # 1위 포켓몬 정보
        top1_label = model.config.id2label[str(top5_catid[0].item())]
        top1_score = top5_prob[0].item()
        artwork_url = get_pokemon_image_url(top1_label)
        
        st.markdown("---")
        
        # 레이아웃 구성: 좌측(입력 이미지) / 우측(예측 결과)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader(f"✨ Expected : {top1_label} ({top1_score * 100:.1f}%)")
            if artwork_url:
                st.image(artwork_url, use_container_width=True)
            else:
                st.info("해당 포켓몬의 공식 이미지를 불러올 수 없습니다.")

        # Top-5 예측 확률 바
        st.markdown("---")
        st.subheader("📊 Top-5 예측 확률")
        
        for i in range(top5_prob.size(0)):
            label = model.config.id2label[str(top5_catid[i].item())]
            score = top5_prob[i].item()
            
            c1, c2 = st.columns([1, 4])
            with c1:
                st.write(f"**{i+1}. {label}**")
            with c2:
                st.progress(score)
                st.write(f"{score * 100:.2f}%")

elif model is None and not error_msg:
    st.warning("유효한 모델 경로를 왼쪽 사이드바에 입력해주세요.")