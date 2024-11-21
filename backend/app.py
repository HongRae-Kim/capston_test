import os
import mysql.connector
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
from model import predict, load_model_func, preprocess_input_data
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from dotenv import load_dotenv
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

# Flask 설정
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')  # .env에서 로드
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER')  # .env에서 로드

# JWTManager 초기화
jwt = JWTManager(app)

# 모델 경로 설정 및 로드
model_path = os.getenv('MODEL_PATH')  # .env에서 로드

try:
    model = load_model_func(model_path)  # 모델 로드 함수 호출
    logger.info("모델 로드에 성공했습니다.")
except Exception as e:
    logger.error(f"모델 로드에 실패했습니다: {e}")
    exit()

# 데이터베이스 설정
DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME')
}

# MySQL 데이터베이스 연결 함수
def get_db_connection():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to MySQL: {err}")
        return None

# 데이터베이스 작업 함수 (재사용을 위한 헬퍼 함수)
def execute_db_query(query, params=(), commit=False):
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        if conn is None:
            return None

        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, params)
        if commit:
            conn.commit()
            return True
        else:
            return cursor.fetchall()
    except mysql.connector.Error as err:
        logger.error(f"Database error occurred: {err}")
        return None
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

# 아이디 중복 체크 API
@app.route('/check-username', methods=['POST'])
def check_username():
    data = request.json
    user_id = data.get('id')

    if not user_id:
        return jsonify({"error": "아이디를 입력하세요"}), 400

    result = execute_db_query("SELECT COUNT(*) FROM member WHERE id = %s", (user_id,))
    if result is not None:
        count = result[0]['COUNT(*)']
        return jsonify({"available": count == 0}), 200
    else:
        return jsonify({"error": "Database error occurred"}), 500

# 회원가입 API
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    user_id = data.get('id')
    password = data.get('pass')

    if not user_id or not password:
        return jsonify({"error": "ID와 비밀번호를 입력하세요"}), 400

    # 아이디 중복 체크
    result = execute_db_query("SELECT COUNT(*) FROM member WHERE id = %s", (user_id,))
    if result is not None and result[0]['COUNT(*)'] > 0:
        return jsonify({"error": "이미 존재하는 아이디입니다"}), 409

    # 회원 정보 저장
    hashed_password = generate_password_hash(password)
    save_result = execute_db_query(
        "INSERT INTO member (id, pass) VALUES (%s, %s)", (user_id, hashed_password), commit=True
    )
    if save_result:
        return jsonify({"message": "회원가입 성공"}), 201
    else:
        return jsonify({"error": "Database error occurred"}), 500

# 로그인 API
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user_id = data.get('id')
    password = data.get('pass')

    if not user_id or not password:
        return jsonify({"error": "ID와 비밀번호를 입력하세요"}), 400

    result = execute_db_query("SELECT * FROM member WHERE id = %s", (user_id,))
    if result is not None and len(result) > 0:
        user = result[0]
        if user and check_password_hash(user['pass'], password):
            access_token = create_access_token(identity=user_id)
            return jsonify({"message": "로그인 성공", "access_token": access_token}), 200
        else:
            return jsonify({"error": "ID 또는 비밀번호가 잘못되었습니다"}), 401
    else:
        return jsonify({"error": "로그인에 실패했습니다"}), 401
    
# 예측 API
@app.route('/predict', methods=['POST'])
@jwt_required()
def predict_route():
    if 'file' not in request.files:
        return jsonify({"error": "파일을 제공하지 않았습니다."}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "올바른 형식의 CSV 파일을 업로드해주세요."}), 400

    try:
        # CSV 파일 읽기 및 데이터 검증
        df = pd.read_csv(file)
        if 'APG Wave' not in df.columns:
            return jsonify({"error": "CSV 파일에 'APG Wave' 열이 없습니다."}), 400

        timeseries_data = df['APG Wave'].values

        # 데이터 전처리 수행
        expected_length = model.input_shape[1]  # 모델이 기대하는 입력 길이
        processed_data = preprocess_input_data(timeseries_data, expected_length)
        if processed_data is None:
            return jsonify({"error": "데이터 전처리 실패"}), 400

        # 모델 예측 수행
        prediction_result = predict(model, processed_data)

        # 예측 결과 응답 반환
        if prediction_result.get('error'):
            return jsonify({"error": prediction_result['error']}), 500
        else:
            return jsonify(prediction_result), 200

    except pd.errors.EmptyDataError:
        logger.error("빈 CSV 파일이 업로드되었습니다.")
        return jsonify({"error": "빈 CSV 파일입니다. 유효한 데이터가 필요합니다."}), 400
    except Exception as e:
        logger.error(f"전체 처리 과정 중 오류 발생: {e}")
        return jsonify({"error": f"처리 중 오류가 발생했습니다: {str(e)}"}), 500



# 건강 팁 API
@app.route('/lifestyle_tips', methods=['POST'])
def lifestyle_tips():
    data = request.json
    vascular_age = data.get('vascular_age')

    tips = []
    if vascular_age and vascular_age.startswith('5') or vascular_age.startswith('6'):
        tips.append("혈관 나이가 높습니다. 식단에서 포화지방과 트랜스지방을 줄이세요.")
        tips.append("규칙적인 유산소 운동을 매일 30분 이상 수행하는 것이 좋습니다.")
    else:
        tips.append("혈관 나이가 양호한 상태입니다. 건강한 상태를 유지하세요.")

    tips.append("금연을 통해 혈관 건강을 보호하세요.")
    tips.append("과일과 채소를 매일 다섯 번 이상 섭취하여 항산화 효과를 높이세요.")

    return jsonify({"lifestyle_tips": tips}), 200

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')  # 기본값 0.0.0.0
    port = int(os.getenv('FLASK_PORT', 5080))  # 기본값 5080
    debug_mode = os.getenv('FLASK_DEBUG', 'True')

    app.run(host, port, debug=True)
