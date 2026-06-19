import cv2
import time
import numpy as np
import streamlit as st
import tensorflow as tf
import mediapipe as mp
from collections import Counter; 
from tensorflow.keras.models import load_model

# Cấu hình trang
st.set_page_config(page_title="FER Realtime CPU", layout="wide")
st.title("Nhận diện Cảm xúc")

# Khai báo biến toàn cục & MediaPipe Tasks API
CLASSES = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
TASK_PATH = 'face_landmarker.task' 
MODEL_PATH = 'best_cnn_model.h5'

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# 1. Loaders (Cache)
@st.cache_resource
def load_keras_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_landmarker():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=TASK_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5
    )
    return FaceLandmarker.create_from_options(options)

with st.spinner('Đang khởi tạo mô hình...'):
    model = load_keras_model()
    try:
        landmarker = load_landmarker()
    except Exception as e:
        st.error(f"Lỗi tải file .task: {e}")
        st.stop()

# 2. Giao diện điều khiển
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Bảng điều khiển")
    run_camera = st.checkbox("Bật Camera", value=False)
    padding_ratio = st.slider("Padding khuôn mặt (%)", 0.0, 0.5, 0.1, 0.05)
    smoothing_factor = st.slider("Độ mượt khung hình (EMA)", 0.1, 1.0, 0.6, 0.1, help="Nhỏ = mượt hơn nhưng trễ, Lớn = nhanh nhưng dễ rung.")
    st.markdown("---")
    st.info("**Tính năng:**\n- Chống rung khung hình (EMA)\n- Tính cảm xúc chủ đạo mỗi 5 giây.")

with col2:
    frame_window = st.image([])

# 3. Pipeline Realtime Tối ưu
if run_camera:
    cap = cv2.VideoCapture(0)
    
    # Biến phục vụ tính toán cảm xúc trung bình 5s
    start_time = time.time()
    emotion_buffer = []
    avg_emotion_text = "Dang phan tich..." # Hạn chế dùng tiếng Việt có dấu khi vẽ bằng cv2
    
    # Biến phục vụ chống rung (EMA Bounding Box)
    prev_box = None

    if not cap.isOpened():
        st.error("Không thể kết nối Webcam.")
    else:
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                break

            # Lật ảnh và lấy thông số
            frame = cv2.flip(frame, 1)
            ih, iw, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = landmarker.detect(mp_image)

            if results.face_landmarks:
                face_landmarks = results.face_landmarks[0]
                
                # Trích xuất tọa độ
                x_coords = [int(landmark.x * iw) for landmark in face_landmarks]
                y_coords = [int(landmark.y * ih) for landmark in face_landmarks]
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                box_width = x_max - x_min
                box_height = y_max - y_min
                
                # Bounding box mục tiêu (Target Box) sau khi tính padding
                tx = max(0, int(x_min - box_width * padding_ratio))
                ty = max(0, int(y_min - box_height * padding_ratio))
                tw = min(iw - tx, int(box_width * (1 + 2 * padding_ratio)))
                th = min(ih - ty, int(box_height * (1 + 2 * padding_ratio)))

                # --- BƯỚC TỐI ƯU 1: CHỐNG RUNG BOUNDING BOX (EMA) ---
                if prev_box is None:
                    prev_box = (tx, ty, tw, th)
                    x, y, w, h = tx, ty, tw, th
                else:
                    px, py, pw, ph = prev_box
                    alpha = smoothing_factor
                    # Công thức mượt: Gia trị mới = (alpha * Target) + (1-alpha) * Cũ
                    x = int(alpha * tx + (1 - alpha) * px)
                    y = int(alpha * ty + (1 - alpha) * py)
                    w = int(alpha * tw + (1 - alpha) * pw)
                    h = int(alpha * th + (1 - alpha) * ph)
                    prev_box = (x, y, w, h)

                roi_rgb = frame_rgb[y:y+h, x:x+w]

                if roi_rgb.shape[0] > 10 and roi_rgb.shape[1] > 10:
                    # --- BƯỚC TỐI ƯU 2: TIỀN XỬ LÝ VÀ PREDICT ---
                    roi_resized = cv2.resize(roi_rgb, (100, 100))
                    roi_normalized = roi_resized / 255.0
                    roi_input = np.expand_dims(roi_normalized, axis=0)

                    preds = model.predict(roi_input, verbose=0)[0]
                    label_idx = np.argmax(preds)
                    label = CLASSES[label_idx]
                    confidence = preds[label_idx]

                    # Đưa cảm xúc hiện tại vào buffer
                    emotion_buffer.append(label)

                    # --- BƯỚC TỐI ƯU 3: LOGIC TRUNG BÌNH 5 GIÂY ---
                    current_time = time.time()
                    if current_time - start_time >= 5.0:
                        if emotion_buffer:
                            # Đếm tần suất xuất hiện nhiều nhất trong 5 giây qua
                            most_common = Counter(emotion_buffer).most_common(1)[0][0]
                            avg_emotion_text = f"Chu dao (5s): {most_common.upper()}"
                        
                        # Reset chu kỳ
                        emotion_buffer.clear()
                        start_time = current_time

                    # --- VẼ GIAO DIỆN LÊN FRAME ---
                    color = (0, 255, 0)
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), color, 2)
                    
                    # Vẽ Label thời gian thực (nhỏ, gắn kèm bounding box)
                    cv2.rectangle(frame_rgb, (x, y-25), (x + w, y), color, -1)
                    cv2.putText(frame_rgb, f"{label}: {confidence*100:.1f}%", (x + 5, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Vẽ Label Cảm xúc chủ đạo 5s (To, rõ ràng ở góc trái màn hình)
            cv2.rectangle(frame_rgb, (10, 10), (350, 50), (0, 0, 0), -1)
            cv2.putText(frame_rgb, avg_emotion_text, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Cập nhật UI
            frame_window.image(frame_rgb)
            
        cap.release()