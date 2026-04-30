import streamlit as st
import torch
import requests
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch.nn.functional as F

# 페이지 설정 (전체 화면을 넓게 쓰도록 layout="wide" 추가)
st.set_page_config(page_title="포켓몬 분류기 데모", page_icon="🐾", layout="wide")

st.title("🐾 Pokemon Classifier")
st.markdown("학습된 ViT 모델을 사용하여 포켓몬의 이름을 예측하고 모습을 확인합니다.")

# ==========================================
# 1. 모델 로드 함수 (캐싱)
# ==========================================
@st.cache_resource
def load_model(model_path):
    try:
        processor = ViTImageProcessor.from_pretrained(model_path)
        model = ViTForImageClassification.from_pretrained(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return processor, model, device
    except Exception as e:
        st.error(f"모델을 불러오는데 실패했습니다: {e}")
        return None, None, None

# ==========================================
# 2. PokeAPI 이미지 호출 함수
# ==========================================
@st.cache_data
def get_pokemon_image_url(pokemon_name):
    try:
        # PokeAPI는 소문자를 사용하며, 띄어쓰기 등을 하이픈(-)으로 처리합니다.
        name_lower = pokemon_name.lower().replace(" ", "-").replace(".", "").replace("'", "")
        url = f"https://pokeapi.co/api/v2/pokemon/{name_lower}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # 고화질 공식 일러스트(official-artwork) URL 반환
            return data["sprites"]["other"]["official-artwork"]["front_default"]
        else:
            return None
    except:
        return None

# ==========================================
# 3. 사이드바 - 실험 설정
# ==========================================
st.sidebar.header("실험 설정")
default_path = "./saved_models/exp1_vit_full_finetuned"
model_path = st.sidebar.text_input("모델 저장 경로", value=default_path)

st.sidebar.info(f"현재 장치: {'GPU (RTX 5070)' if torch.cuda.is_available() else 'CPU'}")

processor, model, device = load_model(model_path)

# ==========================================
# 4. 메인 화면 - 이미지 업로드 및 추론
# ==========================================
uploaded_file = st.file_uploader("포켓몬 이미지를 업로드하세요...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with st.spinner('포켓몬을 식별하는 중...'):
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probs = F.softmax(logits, dim=-1)[0]
        top5_prob, top5_catid = torch.topk(probs, 5)
        
        # 1위 포켓몬 정보 추출
        top1_label = model.config.id2label[top5_catid[0].item()]
        top1_score = top5_prob[0].item()
        artwork_url = get_pokemon_image_url(top1_label)
        
        st.markdown("---")
        
        # 레이아웃: 좌측(입력 이미지) / 우측(예측된 포켓몬 이미지)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 입력된 이미지")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader(f"✨ 예측 결과: 1. {top1_label} ({top1_score * 100:.2f}%)")
            if artwork_url:
                # PokeAPI에서 가져온 공식 이미지 렌더링
                st.image(artwork_url, use_column_width=True)
            else:
                st.info("해당 포켓몬의 공식 이미지를 불러올 수 없습니다.")

        st.markdown("---")
        st.subheader("📊 Top-5 예측 확률")
        
        for i in range(top5_prob.size(0)):
            label = model.config.id2label[top5_catid[i].item()]
            score = top5_prob[i].item()
            
            # 진행률 바를 이용한 깔끔한 UI
            c1, c2 = st.columns([1, 4])
            with c1:
                st.write(f"**{i+1}. {label}**")
            with c2:
                st.progress(score)
                st.write(f"{score * 100:.2f}%")

elif model is None:
    st.warning("유효한 모델 경로를 입력해주세요.")