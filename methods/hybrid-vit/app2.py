import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt

from former import HybridVisionFormer

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Upload FER",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Nhận diện Cảm xúc từ Ảnh Tải Lên")

CLASS_NAMES = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral"
]

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridVisionFormer(num_classes=7).to(device)

    try:
        checkpoint = torch.load("best_model.pth", map_location=device)

        # Nếu là checkpoint dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Remove "module." nếu train DataParallel
        state_dict = {
            k.replace("module.", ""): v
            for k, v in state_dict.items()
        }

        model.load_state_dict(state_dict)
        model.eval()

        return model, device

    except Exception as e:
        st.error(f"Lỗi tải mô hình: {e}")
        return None, None


model, device = load_model()

# ==============================
# TRANSFORM
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==============================
# UPLOAD
# ==============================
st.write("Vui lòng tải lên ảnh có chứa khuôn mặt.")

uploaded_file = st.file_uploader(
    "Chọn file ảnh",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# PREDICT
# ==============================
if uploaded_file and model is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ảnh đầu vào")
            st.image(image)

        with col2:
            st.subheader("Kết quả phân tích")

            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                if device.type == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = model(img_tensor)
                else:
                    outputs = model(img_tensor)

                probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()

            pred_idx = np.argmax(probs)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx] * 100

            st.success(
                f"### Cảm xúc: **{pred_class}** ({confidence:.2f}%)"
            )

            # Chart
            df_probs = pd.DataFrame({
                "Emotion": CLASS_NAMES,
                "Probability": probs * 100
            })

            chart = (
                alt.Chart(df_probs)
                .mark_bar()
                .encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("Emotion:N", sort="-x"),
                    color=alt.condition(
                        alt.datum.Emotion == pred_class,
                        alt.value("orange"),
                        alt.value("steelblue")
                    )
                )
                .properties(height=350)
            )

            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {e}")