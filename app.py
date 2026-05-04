import streamlit as st
import torch
import requests
import os
import json
from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
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
</div>
""", unsafe_allow_html=True)

# 3. 만능 모델 로드 함수 (캐싱 & 에러 방어 적용)
@st.cache_resource(show_spinner=False)
def load_model(model_path):
    try:
        # 허깅페이스 리포지토리 또는 로컬 경로에서 PEFT 설정 파일 확인
        try:
            config = PeftConfig.from_pretrained(model_path)
            is_peft = True
        except:
            is_peft = False
        
        if is_peft:
            base_model_name = config.base_model_name_or_path
            
            try:
                processor = AutoImageProcessor.from_pretrained(model_path)
            except:
                processor = AutoImageProcessor.from_pretrained(base_model_name)
            
            # 1. 포켓몬 분류는 무조건 150개의 클래스입니다.
            num_labels = 150
            
            # 2. 라벨 매핑 딕셔너리 구성 (직접 생성하여 충돌 방지)
            try:
                # 로컬에 저장된 Full FT 모델의 config가 있다면 가장 완벽한 포켓몬 이름 딕셔너리 사용
                config_path = "./saved_model/best_vit_full/config.json"
                with open(config_path, "r", encoding="utf-8") as f:
                    full_config = json.load(f)
                    id2label = {int(k): v for k, v in full_config["id2label"].items()}
                    label2id = full_config["label2id"]
            except Exception:
                try:
                    # 로컬 파일이 없으면 배포된 허깅페이스 저장소에서 올바른 매핑을 강제로 가져옵니다.
                    reference_config = AutoConfig.from_pretrained("gyann/pokemon-vit-full")
                    id2label = {int(k): v for k, v in reference_config.id2label.items()}
                    label2id = reference_config.label2id
                except Exception:
                    # 최후의 수단으로 더미 생성
                    id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
                    label2id = {f"LABEL_{i}": i for i in range(num_labels)}

            # 3. Base 모델을 로드할 때 반드시 num_labels를 명시해야 classifier 헤드 사이즈가 150으로 초기화됩니다.
            base_model = AutoModelForImageClassification.from_pretrained(
                base_model_name,
                num_labels=num_labels,     
                id2label=id2label,  
                label2id=label2id,
                ignore_mismatched_sizes=True
            )
            # 4. 150 사이즈로 맞춰진 Base 모델에 LoRA 가중치 결합
            model = PeftModel.from_pretrained(base_model, model_path)
            
        else:
            processor = AutoImageProcessor.from_pretrained(model_path)
            model = AutoModelForImageClassification.from_pretrained(model_path)
            
            # 비-PEFT 모델도 config에 이름이 없는 경우(Label_15 등)를 대비해 매핑을 덮어씌웁니다.
            if getattr(model.config, "id2label", {}).get(0, "") == "LABEL_0" or getattr(model.config, "id2label", {}).get("0", "") == "LABEL_0":
                try:
                    reference_config = AutoConfig.from_pretrained("gyann/pokemon-vit-full")
                    model.config.id2label = {int(k): v for k, v in reference_config.id2label.items()}
                    model.config.label2id = reference_config.label2id
                except Exception:
                    pass

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
    "ViT Full Fine-tuning": "gyann/pokemon-vit-full",
    "ViT + LoRA": "gyann/pokemon-vit-lora",
    "ViT + QLoRA (4-bit)": "gyann/pokemon-vit-qlora",
    "ResNet50": "gyann/pokemon-resnet50",
    "ConvNeXt": "gyann/pokemon-convnext",
    "Swin Transformer": "gyann/pokemon-swin"
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

# 8. 모델 아키텍처 정보 UI (최하단 배치)
ARCHITECTURE_INFO = {
    "ViT Full Fine-tuning": {
        "title": "Vision Transformer (ViT) - Full FT",
        "desc": "이미지를 16x16 픽셀 패치(Patch)로 나누어 처리합니다. 이미지 전체의 **전역적 문맥(Global Context)** 을 파악하는 데 뛰어나며, 모든 가중치를 재학습하여 최고 성능을 도출하지만 연산 비용이 가장 큽니다."
    },
    "ViT + LoRA": {
        "title": "ViT with LoRA (Low-Rank Adaptation)",
        "desc": "거대한 원본 모델을 얼려두고(Freeze), 핵심 연산층에 아주 얇은 **학습 가능한 우회로(저랭크 행렬)** 를 덧붙입니다. 단 1%의 파라미터만 학습하여 비용을 극적으로 낮추면서도 Full FT와 유사한 성능을 냅니다."
    },
    "ViT + QLoRA (4-bit)": {
        "title": "ViT with QLoRA (Quantized LoRA)",
        "desc": "LoRA에서 한 발 더 나아가, 원본 모델을 **4-bit 정밀도** 로 압축(Quantization)하여 메모리에 적재합니다. VRAM이 매우 부족한 환경에서도 대규모 모델을 튜닝할 수 있게 하는 최적화 기법입니다."
    },
    "ResNet50": {
        "title": "ResNet50 (Baseline CNN)",
        "desc": "합성곱(Convolution) 필터를 겹쳐 이미지의 **국소적 패턴(Local Feature)** 을 찾아내는 전통의 강자입니다. 잔차 연결(Residual Connection)로 깊은 신경망의 학습 안정성을 보장합니다."
    },
    "ConvNeXt": {
        "title": "ConvNeXt (Modernized CNN)",
        "desc": "트랜스포머의 설계 철학(큰 커널, LayerNorm, GELU 등)을 역으로 CNN에 도입한 **'모던 CNN'** 입니다. CNN의 지역적 귀납적 편향을 유지하면서도 트랜스포머급 성능을 냅니다."
    },
    "Swin Transformer": {
        "title": "Swin Transformer (Hierarchical ViT)",
        "desc": "CNN처럼 작은 영역(Window)부터 점점 넓은 영역으로 **계층적(Hierarchical)** 으로 병합하며 학습하는 트랜스포머입니다. ViT가 놓치기 쉬운 미세한 디테일 구분에 강합니다."
    }
}

st.markdown("---")
st.markdown("#### 🧠 선택된 아키텍처 알아보기")
if compare_models:
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        with st.expander(f"{model_name_a} 구조", expanded=False):
            info = ARCHITECTURE_INFO.get(model_name_a, {"title": model_name_a, "desc": "설명이 준비되지 않았습니다."})
            st.markdown(f"**{info['title']}**\n\n{info['desc']}")
    with col_info2:
        with st.expander(f"{model_name_b} 구조", expanded=False):
            info = ARCHITECTURE_INFO.get(model_name_b, {"title": model_name_b, "desc": "설명이 준비되지 않았습니다."})
            st.markdown(f"**{info['title']}**\n\n{info['desc']}")
else:
    with st.expander(f"{model_name} 구조", expanded=False):
        info = ARCHITECTURE_INFO.get(model_name, {"title": model_name, "desc": "설명이 준비되지 않았습니다."})
        st.markdown(f"**{info['title']}**\n\n{info['desc']}")