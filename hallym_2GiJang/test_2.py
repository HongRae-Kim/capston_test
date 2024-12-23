import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

directory1 = "/Users/hongrae/Downloads/Hallym_project/apg 파일"
data = []

# for i in range(5):
#     min_data = i * 200  # 0, 200, 400, 600, 800
#     max_data = min_data + 200  # 200, 400, 600, 800, 1000
for root, dirs, files in os.walk(directory1):
    #     files.sort()
    for file in files:
        if file.endswith(".csv"):
            full_path = os.path.join(root, file)
            df = pd.read_csv(full_path, usecols=['APG Wave'])
            series = df.values.flatten()[:200]  # max data
            series = series.reshape(-1, 1)  # 1D -> 2D
            scaler = MinMaxScaler()
            series = scaler.fit_transform(series)
            data.append(series)

data_array = np.array(data)

# y를 적절하게 수정 필요 (클래스 불균형 고려)

print("Shape of data_array:", data_array.shape)


import pandas as pd

# 엑셀 파일 불러오기 (엑셀 파일의 경로를 넣어야 합니다)
# df = pd.read_csv("2024-10-11 (11-00-32)-APG【 이기장 】.csv")
df = pd.read_csv("/Users/hongrae/Downloads/Hallym_project/hallym_2GiJang/2024-10-11 (11-00-32)-APG【 이기장 】_3 (1).csv")
df_sorted = df.sort_values(by='TestDate')

# 정렬된 데이터프레임을 확인
# VasType 열의 모든 값에 3을 곱하기
df_sorted['VasType'] = df_sorted['VasType'] * 3
pd.set_option('display.max_rows', None)
# 결과 확인
import pandas as pd

# 조건에 따른 VasType 값 조정 함수 정의
def adjust_vastype(row):
    if row['TypeLebel'] == '+++':
        return row['VasType'] - 3
    elif row['TypeLebel'] == '++':
        return row['VasType'] - 2
    elif row['TypeLebel'] == '+':
        return row['VasType'] - 1
    else:
        return row['VasType']  # 조건에 맞지 않으면 변경하지 않음

# apply 함수로 각 행에 대해 VasType 값을 조정
df_sorted['VasType'] = df_sorted.apply(adjust_vastype, axis=1)

# 결과 확인
pd.set_option('display.max_rows', None)
df_sorted['VasType'].reset_index(drop=True,inplace=True)
df_sorted['VasType'].value_counts()
y = df_sorted['VasType'].values
# y = np.tile(df_sorted['VasType'].values, 5)


X_flipped = np.flip(data_array, axis=0)

def adjust_amplitude(signal, factor=0.05):
    # signal과 동일한 모양의 랜덤 값을 생성하여 진폭 조정
    return signal * (1 + factor * np.random.uniform(-1, 1, signal.shape))

adjusted_signal = adjust_amplitude(data_array, factor=0.05)
adjusted_signal.shape

X_adjusted=np.vstack((data_array, adjusted_signal))

# 반전 데이터와 원본 데이터 결합
X_augmented = np.vstack((data_array, X_flipped))  # X와 반전 데이터를 위아래로 결합
y_augmented = np.concatenate((y, y))

print("Shape of data_array:",X_adjusted.shape)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Conv1D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling1D

smote = SMOTE(random_state=42, k_neighbors=2)
X = X_adjusted.reshape(X_adjusted.shape[0], -1)  # (650, 200 * 1) 형태로 변환

X_res, y_res = smote.fit_resample(X, y_augmented)
print(X_res.shape, y_res.shape)
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

from sklearn.model_selection import StratifiedKFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold 교차 검증
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 교차 검증 반복
fold_val_accuracies = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"Fold {fold}/{skf.get_n_splits()}")

    # 학습과 검증 데이터 분할
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # 모델 정의
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(32, kernel_size=8, activation='relu', kernel_regularizer=regularizers.l2(0.2)),
        BatchNormalization(),
        Dropout(0.1),
        Conv1D(32, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.3)),
        BatchNormalization(),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(18, activation='softmax')
    ])

    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # 모델 학습
    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        batch_size=4,
        epochs=10,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler],
        verbose=1
    )

    # 최종 검증 정확도 저장
    val_accuracy = history.history['val_accuracy'][-1]
    fold_val_accuracies.append(val_accuracy)
    print(f"Fold {fold} validation accuracy: {val_accuracy:.2%}")

# 평균 검증 정확도 출력
mean_val_accuracy = np.mean(fold_val_accuracies)
print(f"평균 검증 정확도: {mean_val_accuracy:.2%}")

# 최종 모델 저장 (최고 성능의 모델을 저장)
best_model = model
best_model.save('final_model-3.keras')
print("최종 모델이 'final_model.keras'로 저장되었습니다.")

from sklearn.model_selection import StratifiedKFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold 교차 검증
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 교차 검증 반복
fold_val_accuracies = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"Fold {fold}/{skf.get_n_splits()}")

    # 학습과 검증 데이터 분할
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # 모델 정의
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        Dropout(0.2),
        Conv1D(32, kernel_size=8, activation='relu', kernel_regularizer=regularizers.l2(0.2)),
        BatchNormalization(),
        Dropout(0.1),
        Conv1D(32, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.3)),
        BatchNormalization(),
        Dropout(0.2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(18, activation='softmax')
    ])

    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # 모델 학습
    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        batch_size=4,
        epochs=20,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler],
        verbose=1
    )

    # 최종 검증 정확도 저장
    val_accuracy = history.history['val_accuracy'][-1]
    fold_val_accuracies.append(val_accuracy)
    print(f"Fold {fold} validation accuracy: {val_accuracy:.2%}")

# 평균 검증 정확도 출력
mean_val_accuracy = np.mean(fold_val_accuracies)
print(f"평균 검증 정확도: {mean_val_accuracy:.2%}")

# 최종 모델 저장 (최고 성능의 모델을 저장)
best_model = model
best_model.save('final_model-epoch20.keras')
print("최종 모델이 'final_model.keras'로 저장되었습니다.")

import matplotlib.pyplot as plt
import numpy as np

# Mock data for demonstration purposes (replace these with actual history.history values)
epochs = 50
mock_loss = np.random.uniform(0.2, 1.0, epochs).cumsum() / np.arange(1, epochs + 1)
mock_val_loss = np.random.uniform(0.2, 1.0, epochs).cumsum() / np.arange(1, epochs + 1)
mock_accuracy = np.linspace(0.5, 0.9, epochs) + np.random.uniform(-0.05, 0.05, epochs)
mock_val_accuracy = np.linspace(0.4, 0.8, epochs) + np.random.uniform(-0.05, 0.05, epochs)

# Plot Loss
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), mock_loss, label='Train Loss')
plt.plot(range(1, epochs + 1), mock_val_loss, label='Validation Loss', linestyle='--')
plt.title('Model Loss and Accuracy')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), mock_accuracy, label='Train Accuracy')
plt.plot(range(1, epochs + 1), mock_val_accuracy, label='Validation Accuracy', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.show()


from sklearn.model_selection import StratifiedKFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold 교차 검증
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# 교차 검증 반복
fold_val_accuracies = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    print(f"Fold {fold}/{skf.get_n_splits()}")

    # 학습과 검증 데이터 분할
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    # 모델 정의
    model = Sequential([
        Conv1D(32, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(32, kernel_size=8, activation='relu', kernel_regularizer=regularizers.l2(0.2)),
        BatchNormalization(),
        Dropout(0.3),
        Conv1D(32, kernel_size=10, activation='relu', kernel_regularizer=regularizers.l2(0.3)),
        BatchNormalization(),
        Dropout(0.3),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(18, activation='softmax')
    ])

    # Learning Rate Scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    # 모델 컴파일
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # 모델 학습
    history = model.fit(
        X_train_fold, y_train_fold,
        validation_data=(X_val_fold, y_val_fold),
        batch_size=6,
        epochs=20,
        class_weight=class_weight_dict,
        callbacks=[lr_scheduler],
        verbose=1
    )

    # 최종 검증 정확도 저장
    val_accuracy = history.history['val_accuracy'][-1]
    fold_val_accuracies.append(val_accuracy)
    print(f"Fold {fold} validation accuracy: {val_accuracy:.2%}")

# 평균 검증 정확도 출력
mean_val_accuracy = np.mean(fold_val_accuracies)
print(f"평균 검증 정확도: {mean_val_accuracy:.2%}")

# 최종 모델 저장 (최고 성능의 모델을 저장)
best_model = model
best_model.save('final_model_epoch_20241117_1.keras')
print("최종 모델이 'final_model_epoch_20241117.keras'로 저장되었습니다.")

import matplotlib.pyplot as plt
import numpy as np

# Mock data for demonstration purposes (replace these with actual history.history values)
epochs = 20
mock_loss = np.random.uniform(0.2, 1.0, epochs).cumsum() / np.arange(1, epochs + 1)
mock_val_loss = np.random.uniform(0.2, 1.0, epochs).cumsum() / np.arange(1, epochs + 1)
mock_accuracy = np.linspace(0.5, 0.9, epochs) + np.random.uniform(-0.05, 0.05, epochs)
mock_val_accuracy = np.linspace(0.4, 0.8, epochs) + np.random.uniform(-0.05, 0.05, epochs)

# Plot Loss
plt.subplot(2, 1, 1)
plt.plot(range(1, epochs + 1), mock_loss, label='Train Loss')
plt.plot(range(1, epochs + 1), mock_val_loss, label='Validation Loss', linestyle='--')
plt.title('Model Loss and Accuracy')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(2, 1, 2)
plt.plot(range(1, epochs + 1), mock_accuracy, label='Train Accuracy')
plt.plot(range(1, epochs + 1), mock_val_accuracy, label='Validation Accuracy', linestyle='--')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# directory1 = "test_apg/"
directory1 = "/Users/hongrae/Downloads/Hallym_project/apg 파일"
data = []

# for i in range(5):
#     min_data = i * 200  # 0, 200, 400, 600, 800
#     max_data = min_data + 200  # 200, 400, 600, 800, 1000
for root, dirs, files in os.walk(directory1):
    #     files.sort()
    i=0
    for file in files:

        if file.endswith(".csv"):

            full_path = os.path.join(root, file)
            df = pd.read_csv(full_path, usecols=['APG Wave'])
            series = df.values.flatten()[:200]  # max data
            series = series.reshape(-1, 1)  # 1D -> 2D
            scaler = MinMaxScaler()
            series = scaler.fit_transform(series)
            data.append(series)
            print(file)
            print(i,"\n")
            i+=1


data_array = np.array(data)

import numpy as np

# 숫자를 문자열로 변환하여 각 자리수를 리스트로 만듦
number = 544444444443344
labels = [int(digit) for digit in str(number)]

# Numpy 배열로 변환 (필요하면)
labels_array = np.array(labels)

# 결과 출력
print("Labels as list:", labels)
print("Labels as numpy array:", labels_array)

