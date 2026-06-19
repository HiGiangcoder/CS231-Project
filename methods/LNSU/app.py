import streamlit as st
import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from backbones import get_model
from expression.models import SwinTransFER
import torch.nn.functional as F
import os
import sys

# Add current directory to path for local imports
sys.path.append(os.path.dirname(__file__))

# Load model (tương tự như trong notebook)
@st.cache_resource
def load_model():
    swin = get_model('swin_t')
    net = SwinTransFER(swin=swin, swin_num_features=768, num_classes=7, cam=True)
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'results', 'checkpoint_step_59999_gpu_0.pt')
    dict_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    net.load_state_dict(dict_checkpoint["state_dict_model"])
    net.eval()
    return net

import pandas as pd

# Emotion labels (giả sử theo thứ tự, bạn có thể điều chỉnh)
emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

# Transform cho ảnh
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.set_page_config(page_title="Facial Expression Recognition Demo", page_icon="😊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fbff 0%, #ffffff 100%);
    }
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .big-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #0f172a;
    }
    .secondary {
        color: #475569;
        font-size: 1rem;
    }
    .result-card {
        background: #ffffff;
        border-radius: 18px;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
        padding: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("# Facial Expression Recognition Demo")
st.markdown("### Chụp ảnh bằng camera, để model nhận diện cảm xúc và hiển thị phân phối xác suất.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📷 Camera")
    camera_input = st.camera_input("Nhấn chụp để demo cảm xúc")

    if camera_input is not None:
        image = cv2.imdecode(np.frombuffer(camera_input.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (112, 112))
        
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.image(image_resized, caption="Ảnh đã chụp", width=360)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### 🧠 Kết quả dự đoán")
    if camera_input is not None:
        model = load_model()
        image_tensor = transform(image_resized).unsqueeze(0)
        with torch.no_grad():
            outputs, _ = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy() * 100
            predicted_class = np.argmax(probabilities)
            predicted_emotion = emotion_labels[predicted_class]

        top_score = probabilities[predicted_class]
        st.metric(label="Cảm xúc chính", value=predicted_emotion, delta=f"{top_score:.2f}%")

        data = pd.DataFrame({
            "Emotion": emotion_labels,
            "Probability": probabilities,
        })
        data = data.sort_values("Probability", ascending=False).reset_index(drop=True)

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("**Phân phối xác suất (%)**")
        st.dataframe(data, use_container_width=True)
        st.bar_chart(data.set_index("Emotion"))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Ghi chú:** Ảnh đã được resize về 112x112 để phù hợp với mô hình.")
    else:
        st.info("Hãy chụp một ảnh để nhận diện cảm xúc.")