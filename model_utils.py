

# model_utils.py

import subprocess
import os
import sys

def install_libraries(libraries):
    for lib in libraries:
        try:
            # 라이브러리 임포트를 시도
            __import__(lib)
            print(f"{lib} 라이브러리가 이미 설치되어 있습니다.")
        except ImportError:
            # 설치되지 않은 경우 설치 진행
            print(f"{lib} 라이브러리를 설치합니다...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"{lib} 설치가 완료되었습니다.")

# 필수 라이브러리 목록
required_libraries = [ 'numpy', 'optuna', 'pandas','import-ipynb'] # scikit-learn

# 동적 설치 함수 실행
# install_libraries(required_libraries)

#   4model_utils.py

from sklearn.metrics import accuracy_score
from sklearn.metrics import root_mean_squared_error, r2_score

import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import numpy as np
import optuna

# 모델 경로는 act_path 기반으로 설정
def get_model_path(filename, act_path, version="v1"):
    return os.path.join(act_path, f"{filename}_{version}.pkl")

def train_individual_models(X_train, y_train):
    """개별 모델 (RandomForest, GradientBoosting) 학습"""
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    print("개별 모델 학습 완료")
    return rf_model, gb_model

def train_meta_model(individual_models, X_train, y_train):
    """메타 모델 학습 (개별 모델의 예측을 바탕으로 학습)"""
    rf_preds = individual_models[0].predict(X_train)
    gb_preds = individual_models[1].predict(X_train)

    meta_features = np.column_stack((rf_preds, gb_preds))

    meta_model = LinearRegression()
    meta_model.fit(meta_features, y_train)

    print("메타 모델 학습 완료")
    return meta_model

def save_model(model, filename, act_path):
    version = get_next_version(act_path, filename)  # 자동 버전 관리
    model_path = get_model_path(filename, act_path, version)
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"모델 저장 완료: {model_path}")
    except Exception as e:
        print(f"모델 저장 실패: {e}")

def load_model(filename, act_path, version=None):
    """모델을 불러오는 함수 (최신 버전 자동 로드 지원)"""
    if version is None:
        version = get_latest_version(act_path, filename)  # 최신 버전 탐색
        if version is None:
            print(f"모델 파일이 없습니다: {filename}")
            return None

    model_path = get_model_path(filename, act_path, version)
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"모델 로드 성공: {model_path}")
        return model
    except FileNotFoundError:
        print(f"모델 파일이 없습니다: {model_path}")
        return None

def optimize_hyperparameters(X_train, y_train):
    """Optuna를 사용한 하이퍼파라미터 튜닝"""
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 3, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_train)
        mse = root_mean_squared_error(y_train, preds)
        return mse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    return study.best_params


def evaluate_model(models, X, y):
    """모델 성능 평가 (RMSE 및 R²)"""
    predictions = sum([model.predict(X) for model in models]) / len(models)
    rmse = root_mean_squared_error(y, predictions, squared=False)
    r2 = r2_score(y, predictions)
    print(f"모델 RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return r2  # R² 값을 기준으로 평가


# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import SelectFromModel

# 데이터 불러오기 (예시)
## data = pd.read_csv(ACT_PATH + 'lotto_data.csv')
# X = data.drop(columns=['target'])  # Feature
# y = data['target']  # Label
#


import os
import pickle
import datetime

# 모델을 저장할 때 버전 관리
def save_model_with_version(model, act_path, base_filename, version, epoch, best_score):
    # 버전 번호와 변경 사항 기록
    model_info = {
        'version': version,
        'epoch': epoch,
        'best_score': best_score,
        'date': str(datetime.datetime.now())
    }

    model_path = os.path.join(act_path, f"{base_filename}_v{version}.pkl")

    # 모델과 메타 정보를 함께 저장
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'model_info': model_info}, f)

    print(f"모델 저장 완료: {model_path}, 버전: {version}, 변경사항: {model_info}")

# 모델을 로드할 때 버전 관리
def load_model_with_version(act_path, base_filename, version):
    model_path = os.path.join(act_path, f"{base_filename}_v{version}.pkl")

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        print(f"모델 로드 성공: {model_path}, 버전: {version}")
        return model_data['model'], model_data['model_info']
    else:
        print(f"모델이 존재하지 않습니다. 버전: {version}")
        return None, None

def get_next_version(act_path, filename):
    """기존 버전 번호를 확인하고, 새로운 버전 번호를 반환"""
    existing_versions = []
    for file in os.listdir(act_path):
        if file.startswith(filename):
            # 파일 이름에서 버전 번호 추출 (예: trained_model_individual_rf_v1.pkl)
            base_name, version = file.split("_v")
            if version.endswith(".pkl"):
                existing_versions.append(int(version.split(".")[0]))
    if existing_versions:
        return max(existing_versions) + 1
    return 1  # 버전 1부터 시작

def get_latest_version(act_path, filename):
    """파일 디렉토리를 검색하여 가장 최신 버전을 반환"""
    existing_versions = []
    for file in os.listdir(act_path):
        if file.startswith(filename):
            # 버전 번호 추출 (예: trained_model_individual_rf_v1.pkl)
            try:
                base_name, version = file.split("_v")
                if version.endswith(".pkl"):
                    version_number = int(version.split(".")[0])
                    existing_versions.append(version_number)
            except Exception:
                continue
    if existing_versions:
        return max(existing_versions)  # 가장 큰 버전 반환
    return None  # 해당 파일이 없으면 None 반환

def update_version(current_version, performance_improved):
    """성능 개선 시 버전 업데이트"""
    major, minor, patch = map(int, current_version.split('.'))
    if performance_improved:
        minor += 1  # 성능 개선 시 minor 업데이트
    else:
        patch += 1  # 사소한 변경 시 patch 업데이트

    new_version = f"{major}.{minor}.{patch}"
    return new_version

#  end
