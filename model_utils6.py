import subprocess
import os
import sys
import time
import pickle
import datetime
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

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
required_libraries = ['numpy', 'optuna', 'pandas', 'import-ipynb', 'scikit-learn']

# 동적 설치 함수 실행
# install_libraries(required_libraries)

# 모델 경로는 act_path 기반으로 설정
def get_model_path(filename, act_path, version=None):
    if version is None:
        version = get_next_version(act_path, filename)  # 자동 버전 관리
    return os.path.join(act_path, f"{filename}_v{version}.pkl")

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

# 하이퍼파라미터 최적화 함수 개선
def optimize_hyperparameters(X_train, y_train, model_class, param_space, n_trials=50):
    """동적 모델 및 하이퍼파라미터 최적화 지원"""
    def objective(trial):
        params = {
            key: trial.suggest_categorical(key, values) if isinstance(values, list) else
            trial.suggest_float(key, values[0], values[1]) for key, values in param_space.items()
        }
        model = model_class(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_train)
        return mean_squared_error(y_train, preds)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

from sklearn.model_selection import GridSearchCV

def tune_model_hyperparameters(model, X_train, y_train):
    """
    모델의 하이퍼파라미터를 튜닝하는 함수.
    GridSearchCV를 사용하여 튜닝합니다.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'learning_rate': [0.01, 0.1, 0.2]  # XGBoost나 다른 모델에서 사용하는 파라미터들
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1_weighted', cv=3)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    logging.info(f"Best hyperparameters for {type(model).__name__}: {grid_search.best_params_}")

    return best_model


# evamuate
def evaluate_model(models, X, y):
    """모델 성능 평가 (RMSE 및 R²)"""
    predictions = sum([model.predict(X) for model in models]) / len(models)
    rmse = mean_squared_error(y, predictions, squared=False)
    r2 = r2_score(y, predictions)
    print(f"모델 RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return r2  # R² 값을 기준으로 평가

# 모델을 저장할 때 버전 관리
def save_model_with_version(model, act_path, base_filename, version, epoch, best_score):
    # 버전 번호와 변경 사항 기록
    model_info = {
        'version': version,
        'epoch': epoch,
        'best_score': best_score,
        'date': str(datetime.datetime.now())
    }

    model_path = os.path.join(act_path, f"{base_filename}_v{version}_{int(time.time())}.pkl")

    # 모델과 메타 정보를 함께 저장
    try:
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'model_info': model_info}, f)
        print(f"모델 저장 완료: {model_path}, 버전: {version}, 변경사항: {model_info}")
    except Exception as e:
        print(f"모델 저장 실패: {e}")

# 모델을 로드할 때 버전 관리
def load_model_with_version(act_path, base_filename, version):
    model_path = os.path.join(act_path, f"{base_filename}_v{version}.pkl")

    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            print(f"모델 로드 성공: {model_path}, 버전: {version}")
            return model_data['model'], model_data['model_info']
        except Exception as e:
            print(f"모델 로드 오류: {e}")
            return None, None
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

import datetime

class TrainingInfo:
    def __init__(self, model_changes="No changes", best_score=0.0, previous_best_score=0.0, hyperparameter_tuning=False, feature_changes=False):
        self.model_changes = model_changes  # 모델 변경 사항
        self.best_score = best_score  # 최고 성능
        self.previous_best_score = previous_best_score  # 이전 최고 성능
        self.hyperparameter_tuning = hyperparameter_tuning  # 하이퍼파라미터 튜닝 여부
        self.feature_changes = feature_changes  # 특징 변경 여부

def update_version(current_version, training_info):
    """모델 변경 사항, 하이퍼파라미터 튜닝, 특징 변경에 따른 자동 버전 업데이트"""
    major, minor, patch = map(int, current_version.split('.'))

    # 모델 변경이 이루어졌을 때 (예: 모델 아키텍처 변경)
    if training_info.model_changes != "No changes":
        major += 1, minor = 0, patch = 0  # major 버전 증가 시 minor와 patch 초기화
        print(f"Model changes detected, incrementing major version to {major}")

    # 하이퍼파라미터 튜닝이 이루어진 경우 (예: 성능 향상 시)
    elif training_info.hyperparameter_tuning:
        minor += 1, patch = 0  # minor 버전 증가 시 patch 초기화
        print(f"Hyperparameter tuning detected, incrementing minor version to {minor}")

    # 특징 엔지니어링 또는 특징 변경이 있을 경우 (예: 데이터 또는 feature 변화)
    elif training_info.feature_changes:
        major += 1, minor = 0, patch = 0  # feature 변경 시 major 버전 증가
        print(f"Feature changes detected, incrementing major version to {major}")

    # 성능이 향상되었을 경우 patch 버전 증가
    elif training_info.best_score > training_info.previous_best_score and training_info.best_score <= 0.9:
        patch += 1
        print(f"Performance improvement detected, incrementing patch version to {patch}")

    # 새로운 버전 문자열 생성
    new_version = f"{major}.{minor}.{patch}"
    return new_version

def update_version_based_on_changes(
    current_version, performance_improved=False, algorithm_changed=False, feature_changed=False
):
    """
    버전을 변경하는 함수.
    
    Args:
        current_version (str): 현재 버전 문자열 (예: "1.0.0").
        performance_improved (bool): 성능이 개선된 경우 True.
        algorithm_changed (bool): 알고리즘 또는 모델 구조가 변경된 경우 True.
        feature_changed (bool): 피처 엔지니어링이 변경된 경우 True.
        
    Returns:
        str: 업데이트된 버전 문자열.
    """
    # 현재 버전 숫자를 분리
    major, minor, patch = map(int, current_version.split('.'))

    # 알고리즘 변경 또는 주요 피처 변경은 major 버전을 증가
    if algorithm_changed or feature_changed:
        major += 1, minor = 0, patch = 0
    # 성능 개선은 minor 버전을 증가
    elif performance_improved:
        minor += 1, patch = 0
    # 나머지 사소한 변경 사항은 patch 버전을 증가
    else:
        patch += 1

    # 새로운 버전 문자열 생성
    new_version = f"{major}.{minor}.{patch}"
    return new_version

"""    model_info = {
        'version': version,
        'epoch': training_info['epoch'],
        'best_score': training_info['best_score'],
        'training_loss': training_info['training_loss'],
        'validation_loss': training_info['validation_loss'],
        'validation_score': training_info['validation_score'],
        'learning_rate': training_info['learning_rate'],
        'batch_size': training_info['batch_size'],
        'optimizer': training_info['optimizer'],
        'input_data_info': training_info.get('input_data_info', "No info"),
        'model_changes': training_info.get('model_changes', "No changes"),
        'feature_changes': training_info.get('feature_changes', "No changes"),
        'date': str(datetime.datetime.now())
    } """

import os
import re

import datetime
import pickle

def get_latest_version(model_path):
    """모델 디렉토리에서 최신 버전 및 타임스탬프의 모델을 찾는 함수"""
    model_files = [f for f in os.listdir(model_path) if f.endswith(".pkl")]
    if not model_files:
        return None

    # 파일명에서 버전과 타임스탬프 추출
    pattern = r"_(v\d+\.\d+\.\d+)_(\d{8}T\d{6})\.pkl$"
    versioned_files = [
        (f, re.search(pattern, f).groups()) for f in model_files if re.search(pattern, f)
    ]

    if not versioned_files:
        return None

    # 버전과 타임스탬프를 기준으로 정렬 (최신 버전 및 타임스탬프 순)
    versioned_files.sort(
        key=lambda x: (
            tuple(map(int, x[1][0][1:].split('.'))),  # 버전
            x[1][1]  # 타임스탬프
        ),
        reverse=True
    )

    # 가장 최신 파일 반환
    return versioned_files[0][0]

def save_model(model, model_path, base_filename, version, epoch, best_score):
    """모델과 관련 정보를 저장하는 함수"""
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    model_info = {
        'model': model,
        'version': version,
        'epoch': epoch,
        'best_score': best_score,
        'timestamp': timestamp,
        'date': str(datetime.datetime.now())
    }
    filename = f"{base_filename}_v{version}_{timestamp}.pkl"
    full_path = os.path.join(model_path, filename)
    
    os.makedirs(model_path, exist_ok=True)
    
    try:
        with open(full_path, "wb") as f:
            pickle.dump(model_info, f)
        print(f"모델 저장 완료: {full_path}")
    except Exception as e:
        print(f"모델 저장 실패: {e}")

def load_model(model_path):
    """모델을 불러오는 함수, 최신 버전 및 타임스탬프의 모델을 로드"""
    latest_model_filename = get_latest_version(model_path)
    if latest_model_filename is None:
        print(f"모델 파일이 없습니다. {model_path}에 모델을 새로 학습시켜야 합니다.")
        return None, None

    try:
        with open(os.path.join(model_path, latest_model_filename), "rb") as f:
            model_info = pickle.load(f)
        print(f"최신 모델을 로드했습니다: {latest_model_filename}")
        return model_info['model'], model_info
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        return None, None

def get_current_version():
    """현재 시스템의 요구 버전 정보를 반환"""
    # 이 부분은 시스템 또는 설정에 맞는 버전 규칙을 반환해야 합니다.
    return "1.0.0"  # 예시 버전

# end 
