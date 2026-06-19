import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm.notebook import tqdm

from sklearn.utils import shuffle

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.backend import clear_session
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

DATA_ROOT = os.environ["DATA_ROOT"]
LABEL_FILE = os.environ["LABEL_FILE"]

def load_raf_dataset(data_root, label_file):
    images = []
    labels = []
    with open(label_file) as f:
        for line in f:
            img_name, label = line.strip().split()
            img_path = os.path.join(
                data_root,
                img_name
            )
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )
            img = cv2.resize(
                img,
                (100, 100),
                interpolation=cv2.INTER_AREA
            )
            images.append(img)
            labels.append(int(label))

    return np.array(images), np.array(labels)

classes = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
label_map = {label: (idx+1) for idx, label in enumerate(classes)}

X_all, y_all = load_raf_dataset(
    DATA_ROOT,
    LABEL_FILE
)

from sklearn.model_selection import train_test_split

X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(
    X_all,
    y_all,
    test_size=0.15,
    stratify=y_all,
    random_state=42
)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

# Undersample majority class ('happy' = 4)
def reduce_class(X, y, target_class, target_size):
    class_indices = np.where(y == target_class)[0]
    non_class_indices = np.where(y != target_class)[0]
    # Select exactly target_size instances
    reduced_class_indices = np.random.choice(class_indices, target_size, replace=False)
    final_indices = np.concatenate([reduced_class_indices, non_class_indices])
    return X[final_indices], y[final_indices]

X_reduced, y_reduced = reduce_class(
    X_train_raw,
    Y_train_raw,
    target_class=4,
    target_size=3500
)

# Augment minority classes (Optimized for Memory & Speed)
def augment_classes_optimized(images, labels, target_counts):
    datagen = ImageDataGenerator(
        rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,  
        zoom_range=0.1, horizontal_flip=True, channel_shift_range=50.0, fill_mode='nearest'
    )

    # Khởi tạo list chứa các mảng numpy để nối 1 lần duy nhất ở cuối
    augmented_images_list = [images]
    augmented_labels_list = [labels]

    for target_class, target_count in target_counts.items():
        class_images = images[labels == target_class]
        current_count = len(class_images)
        augment_count = target_count - current_count

        if augment_count > 0:
            print(f'Class {target_class}: Generating {augment_count} augmented samples...')

            # Tối ưu Generator: Tăng batch_size để tận dụng vector hóa
            batch_size = min(128, current_count) 
            generator = datagen.flow(class_images, batch_size=batch_size, seed=42)

            generated_count = 0
            class_augmented_imgs = []

            while generated_count < augment_count:
                batch_imgs = next(generator)

                # Tính toán số lượng ảnh cần lấy từ batch hiện tại để không bị lố target
                take_count = min(len(batch_imgs), augment_count - generated_count)
                class_augmented_imgs.append(batch_imgs[:take_count].astype(np.uint8))
                generated_count += take_count

            # Gộp ảnh ảo của class hiện tại và đưa vào list tổng
            augmented_images_list.append(np.concatenate(class_augmented_imgs, axis=0))
            augmented_labels_list.append(np.full(augment_count, target_class))

    # Concatenate toàn bộ 1 lần duy nhất trên RAM
    print("Đang tổng hợp dữ liệu vào bộ nhớ (Concatenation)...")
    final_images = np.concatenate(augmented_images_list, axis=0)
    final_labels = np.concatenate(augmented_labels_list, axis=0)

    return final_images, final_labels

target_counts = {1: 3500, 2: 3500, 3: 3500, 5: 3500, 6: 3500, 7: 3500}
X_train_balanced, Y_train_balanced = augment_classes_optimized(X_reduced, y_reduced, target_counts)

# Shuffle the balanced train set to prevent learning ordering patterns
X_train_balanced, Y_train_balanced = shuffle(X_train_balanced, Y_train_balanced, random_state=42)

# Normalize [0, 1]
X_train_norm = X_train_balanced / 255.0
X_test_norm = X_test_raw / 255.0

# Reshape for CNN explicitly (height, width, channels)
X_train_cnn = X_train_norm.reshape((X_train_norm.shape[0], 100, 100, 3))
X_test_cnn = X_test_norm.reshape((X_test_norm.shape[0], 100, 100, 3))

# One-hot encoding labels (subtract 1 because RAF-DB labels are 1-7)
Y_train_cat = to_categorical(Y_train_balanced - 1, num_classes=len(classes))
Y_test_cat = to_categorical(Y_test_raw - 1, num_classes=len(classes))

# Train Data Generator (only applied during fit process on Train Data)
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,  
    horizontal_flip=True, fill_mode='nearest'
)
train_generator = datagen.flow(X_train_cnn, Y_train_cat, batch_size=32)

from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Activation, add, Dense, Dropout, Multiply, Reshape, SpatialDropout2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# Tối ưu lại L2 Regularization
regularization = l2(1e-4)

# --- Squeeze and Excitation Block (Attention) ---
def se_block(tensor, ratio=8):
    """
    Squeeze-and-Excitation block: Giúp model focus vào các feature map quan trọng.
    """
    filters = tensor.shape[-1]
    se = GlobalAveragePooling2D()(tensor)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = Multiply()([tensor, se])
    return x

inputs = Input(shape=(100, 100, 3))

# --- Base / Stem ---
x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularization, use_bias=False)(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

# --- Các Module được tăng cường Filter (32 -> 64 -> 128 -> 256) ---
filter_blocks = [32, 64, 128, 256]

for filters in filter_blocks:
    # Nhánh Skip Connection
    residual = Conv2D(filters, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Nhánh Separable Convolution
    x = SeparableConv2D(filters, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters, (3, 3), padding='same', depthwise_regularizer=regularization, pointwise_regularizer=regularization, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Đưa qua SE Block
    x = se_block(x)

    # [TỐI ƯU MỚI] Áp dụng SpatialDropout2D với rate tăng dần theo độ sâu của mạng (tùy chọn 0.1 - 0.2)
    x = SpatialDropout2D(0.15)(x)

    x = add([x, residual])

# --- Classifier Head Nâng cấp ---
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu', kernel_regularizer=regularization)(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x) 

# Số lượng class (Chờ thông tin từ bạn)
num_classes = len(classes) 
outputs = Dense(num_classes, activation='softmax')(x)

# Khởi tạo model
cnn_model = Model(inputs=inputs, outputs=outputs, name="Advanced_FER_Model")

# [TỐI ƯU MỚI] Giảm learning rate và thêm label smoothing
optimizer = Adam(learning_rate=0.0005)
loss_fn = CategoricalCrossentropy(label_smoothing=0.1)

cnn_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
cnn_model.summary()

# Training Callbacks
# Patience của ReduceLR nên nhỏ hơn EarlyStopping để nó có cơ hội giảm LR trước khi bị ngắt.
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=6, min_delta=0.0001, verbose=1, min_lr=1e-6) 
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
checkpoint = ModelCheckpoint(filepath='best_CNNModel.keras', monitor='val_accuracy', save_best_only=True, verbose=1) 

# Train Model
# LƯU Ý: Nếu train_generator là một object của ImageDataGenerator.flow() hoặc tf.data.Dataset, 
# tham số batch_size ở đây có thể gây thừa hoặc báo lỗi tùy phiên bản TF. 
CNN_History = cnn_model.fit(
    train_generator,
    epochs=120, 
    validation_data=(X_test_cnn, Y_test_cat), 
    callbacks=[reduce_lr, early_stop, checkpoint]
)

# Loss & Accuracy Curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].plot(CNN_History.history['loss'], label='Train Loss', color='red')
ax[0].plot(CNN_History.history['val_loss'], label='Validation Loss', color='green')
ax[0].set_title('Loss Curve')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(CNN_History.history['accuracy'], label='Train Accuracy', color='red')
ax[1].plot(CNN_History.history['val_accuracy'], label='Validation Accuracy', color='green')
ax[1].set_title('Accuracy Curve')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Evaluation Results
train_result = cnn_model.evaluate(X_train_cnn, Y_train_cat, verbose=0)
test_result = cnn_model.evaluate(X_test_cnn, Y_test_cat, verbose=0)

print(f"Train Loss: {train_result[0]:.4f} | Train Accuracy: {train_result[1]*100:.2f}%")
print(f"Test Loss: {test_result[0]:.4f}  | Test Accuracy: {test_result[1]*100:.2f}%")


# Predictions on untouched Test Set
y_pred_prob = cnn_model.predict(X_test_cnn)
y_pred_classes = np.argmax(y_pred_prob, axis=1)
y_true_classes = np.argmax(Y_test_cat, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title("CNN Confusion Matrix (Test Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


final_model_path = 'best_cnn_model.h5'
cnn_model.save(final_model_path)
print(f"Model saved successfully to {final_model_path}")

# Giải phóng bộ nhớ
del CNN_History, train_result, test_result
clear_session()
gc.collect()

