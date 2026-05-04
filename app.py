import streamlit as st
import torch
import requests
import os
import json
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel, PeftConfig
from PIL import Image
import torch.nn.functional as F
import io

# 1. 페이지 기본 설정 (가장 먼저 호출해야 함)
st.set_page_config(page_title="Pokemon Classifier", page_icon="🐾", layout="wide", initial_sidebar_state="collapsed")

# 2. 커스텀 CSS 적용 (더 서비스스러운 느낌을 위해)
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 3rem 0;
    }
    .main-header h1 {
        font-size: 3.5rem;
        color: #ffcb05;
        -webkit-text-stroke: 2px #3c5aa6;
        text-shadow: 4px 4px 0px #3c5aa6;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.2rem;
        color: #555;
        font-weight: 500;
    }
    /* 카드 느낌의 컨테이너를 위해 여백 조정 */
    div[data-testid="stVerticalBlock"] {
        gap: 1.2rem;
    }
</style>
<div class="main-header">
    <h1>Pokemon Classifier</h1>
    <p>AI(ViT, ResNet)를 통해 포켓몬을 식별하고 도감 정보를 확인하세요 🐾</p>
</div>
""", unsafe_allow_html=True)

# 3. 만능 모델 로드 함수 (캐싱 & 에러 방어 적용)
@st.cache_resource(show_spinner=False)
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
                config_path = "./saved_model/best_vit_full/config.json"
                with open(config_path, "r", encoding="utf-8") as f:
                    full_config = json.load(f)
                    id2label = full_config["id2label"]
                    label2id = full_config["label2id"]
            except Exception as e:
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
            processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModelForImageClassification.from_pretrained(model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        
        return processor, model, device, None
        
    except Exception as e:
        return None, None, None, str(e)


# 4. PokeAPI 데이터 호출 함수 (이미지, 타입, 키, 몸무게)
@st.cache_data(show_spinner=False)
def get_pokemon_data(pokemon_name):
    try:
        name_lower = pokemon_name.lower().replace(" ", "-").replace(".", "").replace("'", "")
        url = f"https://pokeapi.co/api/v2/pokemon/{name_lower}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "artwork_url": data["sprites"]["other"]["official-artwork"]["front_default"],
                "types": [t["type"]["name"] for t in data["types"]],
                "height": data["height"] / 10.0, # meters
                "weight": data["weight"] / 10.0  # kg
            }
        return None
    except:
        return None


MODEL_PATHS = {
    "ViT Full Fine-tuning": "./saved_model/best_vit_full",
    "ViT + LoRA": "./saved_model/best_vit_lora",
    "ViT + QLoRA (4-bit)": "./saved_model/best_vit_qlora",
    "ResNet50": "./saved_model/best_resnet50_pokemon",
    "ConvNeXt": "./saved_model/best_convnext",
    "Swin Transformer": "./saved_model/best_swin"
}

# 5. 설정 및 입력 영역 (깔끔한 컨테이너 UI)
with st.container(border=True):
    st.markdown("### ⚙️ 분석 설정")
    setting_col1, setting_col2 = st.columns([1, 2])

    with setting_col1:
        compare_models = st.toggle("모델 비교 모드 활성화", value=False)
        st.caption(f"🚀 현재 가속 장치: **{'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}**")

    with setting_col2:
        if compare_models:
            col_a, col_b = st.columns(2)
            with col_a:
                model_name_a = st.selectbox("Model A", list(MODEL_PATHS.keys()), index=0)
            with col_b:
                model_name_b = st.selectbox("Model B", list(MODEL_PATHS.keys()), index=1)
            
            with st.spinner("모델 A 로딩 중..."):
                processor_a, model_a, device_a, err_a = load_model(MODEL_PATHS[model_name_a])
            with st.spinner("모델 B 로딩 중..."):
                processor_b, model_b, device_b, err_b = load_model(MODEL_PATHS[model_name_b])
            
            if err_a: st.error(f"Model A 로드 실패: {err_a}")
            if err_b: st.error(f"Model B 로드 실패: {err_b}")
            
            active_models = [
                ("Model A", model_name_a, processor_a, model_a, device_a), 
                ("Model B", model_name_b, processor_b, model_b, device_b)
            ]
        else:
            model_name = st.selectbox("분석에 사용할 모델", list(MODEL_PATHS.keys()))
            with st.spinner("모델 로딩 중..."):
                processor, model, device, err = load_model(MODEL_PATHS[model_name])
            
            if err: st.error(f"모델 로드 실패: {err}")
            
            active_models = [("Result", model_name, processor, model, device)]

with st.container(border=True):
    st.markdown("### 🖼️ 이미지 선택")
    EXAMPLE_IMAGES = {
        "직접 파일 업로드하기": None,
        "예시: 피카츄 (Pikachu)": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png",
        "예시: 이상해씨 (Bulbasaur)": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/1.png",
        "예시: 꼬부기 (Squirtle)": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/7.png",
        "예시: 파이리 (Charmander)": "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/4.png"
    }
    
    # 라디오 버튼을 가로로 배치하여 모던하게
    selected_example = st.radio("테스트 방식을 선택하세요", list(EXAMPLE_IMAGES.keys()), horizontal=True, label_visibility="collapsed")
    
    image_to_process = None
    if selected_example == "직접 파일 업로드하기":
        uploaded_file = st.file_uploader("포켓몬 이미지를 드래그 앤 드롭하세요", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
    else:
        example_url = EXAMPLE_IMAGES[selected_example]
        try:
            response = requests.get(example_url)
            if response.status_code == 200:
                image_to_process = Image.open(io.BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"예시 이미지를 불러오는 중 오류가 발생했습니다: {e}")


# 6. 추론 및 결과 출력 컴포넌트
def predict_and_display(name_prefix, model_name, processor, model, device, image):
    if model is None:
        st.warning(f"모델({model_name})이 정상적으로 로드되지 않았습니다.")
        return
        
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probs = F.softmax(logits, dim=-1)[0]
    top5_prob, top5_catid = torch.topk(probs, 5)
    
    cat_id_0 = top5_catid[0].item()
    top1_label = model.config.id2label.get(cat_id_0, model.config.id2label.get(str(cat_id_0), "Unknown"))
    top1_score = top5_prob[0].item()
    
    poke_data = get_pokemon_data(top1_label)
    
    st.markdown(f"<h5 style='text-align: center; color: #777; margin-bottom: 0;'>{name_prefix}: {model_name}</h5>", unsafe_allow_html=True)
    
    # 메트릭
    st.metric(label="🏆 Top-1 Prediction", value=top1_label.title(), delta=f"{top1_score*100:.1f}% Confidence", delta_color="normal")
    
    tab1, tab2, tab3 = st.tabs(["📝 Overview", "📊 Stats", "🔍 Top-5 Details"])
    
    with tab1:
        if poke_data and poke_data["artwork_url"]:
            st.image(poke_data["artwork_url"], use_container_width=True)
        else:
            st.info("공식 일러스트가 없습니다.")
            
    with tab2:
        if poke_data:
            st.markdown(f"**속성(Types)**: {', '.join(poke_data['types']).title()}")
            st.markdown(f"**신장(Height)**: {poke_data['height']} m")
            st.markdown(f"**체중(Weight)**: {poke_data['weight']} kg")
        else:
            st.warning("도감 정보를 가져올 수 없습니다.")
            
    with tab3:
        for i in range(top5_prob.size(0)):
            c_id = top5_catid[i].item()
            lbl = model.config.id2label.get(c_id, model.config.id2label.get(str(c_id), "Unknown"))
            scr = top5_prob[i].item()
            
            st.caption(f"**{i+1}. {lbl.title()}**")
            st.progress(scr, text=f"{scr * 100:.1f}%")


# 7. 메인 실행 부
if image_to_process is not None:
    st.markdown("---")
    
    with st.container():
        if compare_models:
            col_img, col_res = st.columns([1, 2.5])
            with col_img:
                st.markdown("### 📷 입력 이미지")
                st.image(image_to_process, use_container_width=True)
                
            with col_res:
                st.markdown("### ✨ 분석 결과")
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container(border=True):
                        m_prefix, m_name, m_proc, m_model, m_dev = active_models[0]
                        with st.spinner('분석 중...'):
                            predict_and_display(m_prefix, m_name, m_proc, m_model, m_dev, image_to_process)
                with col2:
                    with st.container(border=True):
                        m_prefix, m_name, m_proc, m_model, m_dev = active_models[1]
                        with st.spinner('분석 중...'):
                            predict_and_display(m_prefix, m_name, m_proc, m_model, m_dev, image_to_process)
        else:
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.markdown("### 📷 입력 이미지")
                st.image(image_to_process, use_container_width=True)
                
            with col2:
                st.markdown("### ✨ 분석 결과")
                with st.container(border=True):
                    m_prefix, m_name, m_proc, m_model, m_dev = active_models[0]
                    with st.spinner('분석 중...'):
                        predict_and_display(m_prefix, m_name, m_proc, m_model, m_dev, image_to_process)