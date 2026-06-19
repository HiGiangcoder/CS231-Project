import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt
import os

# Import kiến trúc model từ file former.py của bạn
from former import HybridVisionFormer

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Realtime FER Demo", page_icon="🎭", layout="wide")
st.title("🎭 Nhận diện Cảm xúc Khuôn mặt (Real-time)")

# --- 1. TẢI MÔ HÌNH (Sử dụng Cache để tránh load lại gây lag) ---
@st.cache_resource
def load_model_inference():
    # Tự động chọn GPU nếu có, không thì dùng CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Khởi tạo kiến trúc model (phải khớp với file former.py)
    model = HybridVisionFormer(num_classes=7)
    
    model_path = 'best_model.pth'
    if not os.path.exists(model_path):
        st.error(f"Không tìm thấy file trọng số '{model_path}'. Hãy đảm bảo file nằm cùng thư mục!")
        return None, device

    try:
        # Load weights và xử lý mapping thiết bị
        state_dict = torch.load(model_path, map_location=device)
        
        # Xử lý nếu model được lưu bằng DataParallel (loại bỏ tiền tố 'module.')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval() # Chuyển sang chế độ dự đoán
        return model, device
    except Exception as e:
        st.error(f"Lỗi khi nạp mô hình: {e}")
        return None, device

model, device = load_model_inference()

# Danh sách nhãn cảm xúc
class_names = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

# Bộ tiền xử lý ảnh (phải giống hệt lúc bạn training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. GIAO DIỆN NGƯỜI DÙNG ---
st.info("Hướng dẫn: Cho phép trình duyệt truy cập Camera, sau đó nhấn 'Take Photo' để dự đoán.")

# Widget chụp ảnh từ Webcam
camera_image = st.camera_input("Chụp ảnh khuôn mặt của bạn")

# --- 3. XỬ LÝ DỰ ĐOÁN ---
if camera_image is not None:
    if model is not None:
        # Chia layout thành 2 cột: Bên trái ảnh chụp, bên phải kết quả
        col1, col2 = st.columns(2)
        
        # Chuyển dữ liệu camera sang PIL Image
        img = Image.open(camera_image).convert('RGB')
        
        with col1:
            st.subheader("🖼️ Ảnh gốc")
            st.image(img, use_container_width=True)
            
        with col2:
            st.subheader("📊 Kết quả phân tích")
            
            # Tiền xử lý ảnh cho mô hình
            input_tensor = transform(img).unsqueeze(0).to(device)
            
            # Thực hiện dự đoán (Tắt tính toán gradient để tăng tốc)
            with torch.inference_mode():
                outputs = model(input_tensor)
                probabilities = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            
            # Lấy cảm xúc có xác suất cao nhất
            pred_idx = np.argmax(probabilities)
            label = class_names[pred_idx]
            score = probabilities[pred_idx] * 100
            
            # Hiển thị nhãn chính
            st.success(f"Dự đoán: **{label}** ({score:.2f}%)")
            
            # Vẽ biểu đồ cột hiển thị xác suất của tất cả cảm xúc
            df_res = pd.DataFrame({
                'Cảm xúc': class_names,
                'Xác suất (%)': probabilities * 100
            })
            
            chart = alt.Chart(df_res).mark_bar().encode(
                x=alt.X('Xác suất (%):Q', title='Tỷ lệ (%)', scale=alt.Scale(domain=[0, 100])),
                y=alt.Y('Cảm xúc:N', sort='-x', title='Loại cảm xúc'),
                color=alt.condition(
                    alt.datum['Cảm xúc'] == label,
                    alt.value('orange'), # Màu nổi cho kết quả cao nhất
                    alt.value('steelblue')
                )
            ).properties(height=350)
            
            st.altair_chart(chart, use_container_width=True)
    else:
        st.error("Không thể dự đoán vì chưa tải được model.")