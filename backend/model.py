import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model as keras_load_model
import logging
from dotenv import load_dotenv
import os
from typing import Dict, Union, List, Any

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_func(model_path: str) -> Any:
    """
    모델 로드 함수

    Args:
        model_path (str): 모델 파일 경로

    Returns:
        model: 로드된 keras 모델

    Raises:
        FileNotFoundError: 모델 파일이 존재하지 않을 경우
        SystemExit: 모델 로드 실패 시
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    try:
        model = keras_load_model(model_path)
        logger.info("모델 로드 성공")
        return model
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {e}")
        raise SystemExit("모델 로드에 실패하였습니다. 애플리케이션을 종료합니다.")

def preprocess_input_data(
    timeseries_data: Union[np.ndarray, List[float]],
    expected_length: int = 200
) -> np.ndarray:
    """
    입력 데이터 전처리 함수

    Args:
        timeseries_data: 시계열 데이터 (numpy 배열 또는 리스트)
        expected_length: 목표 데이터 길이 (기본값: 200)

    Returns:
        np.ndarray: 전처리된 데이터

    Raises:
        ValueError: 입력 데이터 검증 실패 시
    """
    try:
        from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler 임포트

        # 입력 데이터 타입 및 형태 검증
        if not isinstance(timeseries_data, (np.ndarray, list)):
            raise ValueError("입력 데이터는 numpy 배열 또는 리스트여야 합니다.")

        # 빈 데이터 체크
        if len(timeseries_data) == 0:
            raise ValueError("입력 데이터가 비어있습니다.")

        # 리스트인 경우 numpy 배열로 변환
        if isinstance(timeseries_data, list):
            timeseries_data = np.array(timeseries_data, dtype=np.float32)

        # NaN 값 체크
        if np.isnan(timeseries_data).any():
            raise ValueError("입력 데이터에 NaN 값이 포함되어 있습니다.")

        # 데이터 길이 조정 (최대 expected_length개의 데이터 포인트 사용)
        timeseries_data = timeseries_data[:expected_length]

        # 데이터 길이가 expected_length보다 작으면 패딩 추가
        current_length = len(timeseries_data)
        if current_length < expected_length:
            # 패딩 추가 (뒤쪽에 0으로 채움)
            timeseries_data = np.pad(
                timeseries_data,
                (0, expected_length - current_length),
                mode='constant',
                constant_values=0
            )

        # 2D 형태로 변환
        processed_data = timeseries_data.reshape(-1, 1)  # (expected_length, 1)

        # MinMaxScaler 적용
        scaler = MinMaxScaler()
        processed_data = scaler.fit_transform(processed_data)

        logger.info(f"전처리 완료. 데이터 형태: {processed_data.shape}")
        return processed_data

    except Exception as e:
        logger.error(f"전처리 중 오류 발생: {e}")
        raise

def predict(
    model: Any,
    processed_data: np.ndarray
) -> Dict[str, Union[int, float, str, None]]:
    """
    예측 수행 함수

    Args:
        model: 로드된 keras 모델
        processed_data: 전처리된 입력 데이터 (np.ndarray)

    Returns:
        Dict: 예측 결과를 포함하는 딕셔너리
    """
    try:
        # 모델 검증
        if model is None:
            raise ValueError("모델이 로드되지 않았습니다.")

        # 입력 데이터 검증
        if processed_data is None or not isinstance(processed_data, np.ndarray):
            raise ValueError("유효하지 않은 입력 데이터입니다.")

        # 모델 입력 형태로 변환
        expected_length = model.input_shape[1]
        model_input = processed_data.reshape(1, expected_length, 1)
        logger.info(f"모델 입력 데이터 형태: {model_input.shape}")

        # 예측 수행
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            prediction = model.predict(model_input, verbose=0)

        logger.info(f"예측 결과 형태: {prediction.shape}")

        # 예측 결과 처리
        if not isinstance(prediction, np.ndarray) or prediction.size == 0:
            raise ValueError("유효하지 않은 예측 결과")

        predicted_class = int(np.argmax(prediction))
        predicted_probability = float(prediction[0][predicted_class])

        # 결과 검증
        if not (0 <= predicted_class <= 17):
            raise ValueError(f"예측 클래스가 유효 범위를 벗어남: {predicted_class}")
        if not (0 <= predicted_probability <= 1):
            raise ValueError(f"예측 확률이 유효 범위를 벗어남: {predicted_probability}")

        # 건강 지표 계산
        vascular_score = int(np.clip(predicted_probability * 100, 0, 100))
        aging_speed = float(np.clip(
            predicted_class * 0.5 + (1 - predicted_probability) * 2,
            0, 5
        ))

        # 혈관 나이 매핑
        vascular_age_mapping = {
            0: '20-25', 1: '26-30', 2: '31-35', 3: '36-40',
            4: '41-45', 5: '46-50', 6: '51-55', 7: '56-60',
            8: '61-65', 9: '66-70', 10: '71-75', 11: '76-80',
            12: '81-85', 13: '86-90', 14: '91-95', 15: '96-100',
            16: '101-105', 17: '106 이상'
        }
        vascular_age = vascular_age_mapping.get(predicted_class, '알 수 없음')

        result = {
            "predicted_class": predicted_class,
            "predicted_probability": predicted_probability,
            "vascular_score": vascular_score,
            "aging_speed": aging_speed,
            "vascular_age": vascular_age
        }

        logger.info(f"예측 결과: {result}")
        return result

    except Exception as e:
        logger.error(f"예측 과정 중 오류 발생: {e}")
        return {
            "error": str(e),
            "predicted_class": None,
            "predicted_probability": None,
            "vascular_score": None,
            "aging_speed": None,
            "vascular_age": None
        }
