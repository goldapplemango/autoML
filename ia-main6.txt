
# ia-main6.py

import subprocess
import os
import sys
import pandas as pd
import time
import numpy as np
from google.colab import drive
from utils4 import conditional_tuning, log_progress, provide_recommendations
from model_utils4 import (get_model_path, train_individual_models, train_meta_model,
                          save_model, load_model, evaluate_model, optimize_hyperparameters)
from feature_utils4 import generate_features10

def mount_drive():
    try:
        if not os.path.isdir('/content/drive'):
            print("Google Drive를 마운트 중입니다...")
            drive.mount('/content/drive')
            print("Google Drive 마운트 완료!")
        else:
            print("Google Drive가 이미 마운트되어 있습니다.")
    except Exception as e:
        print(f"Google Drive 마운트 실패: {e}")
        sys.exit(1)

def install_libraries(libraries):
    for lib in libraries:
        try:
            __import__(lib)
            print(f"{lib} 라이브러리가 이미 설치되어 있습니다.")
        except ImportError:
            print(f"{lib} 라이브러리를 설치합니다...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"{lib} 설치가 완료되었습니다.")

def load_and_preprocess_data(data_path):
    try:
        print(f"데이터 경로: {data_path}")
        data = pd.read_csv(data_path)
        print("데이터 로드 성공")
        if data.isnull().any().any():
            raise ValueError("데이터에 결측치가 있습니다.")

        required_columns = ['번호1', '번호2', '번호3', '번호4', '번호5', '번호6', '보너스']
        if not set(required_columns).issubset(data.columns):
            raise ValueError(f"필요한 컬럼이 없습니다: {required_columns}")
        data = data[required_columns]
        data = generate_features10(data)

        return data
    except Exception as e:
        print(f"데이터 처리 오류: {e}")
        return None

def train_and_evaluate_models(X, y, act_path, individual_model_path_gb, meta_model_path):
    individual_models = load_model(individual_model_path_gb, act_path)
    meta_model = load_model(meta_model_path, act_path)

    if individual_models is None or meta_model is None:
        print("기존 모델이 없어 새로 학습을 시작합니다...")
        individual_models = train_individual_models(X, y)
        meta_model = train_meta_model(individual_models, X, y)
        save_model(individual_models, individual_model_path_gb, act_path)
        save_model(meta_model, meta_model_path, act_path)

    return individual_models, meta_model

def conditional_tuning_and_hyperparameters(epoch, eval_accuracy, best_eval_accuracy, last_tuning_time, X, y, tuning_delay):
    best_params, last_tuning_time = conditional_tuning(epoch, eval_accuracy, best_eval_accuracy, last_tuning_time, X, y)
    if best_params:
        current_model.set_params(**best_params)
        current_model.fit(X, y)
        print("튜닝 후 모델 재학습 완료.")

    if time.time() - last_tuning_time > tuning_delay:
        print("조건 만족 및 타이머 경과: 기간 기반 하이퍼파라미터 튜닝 실행...")
        best_params = optimize_hyperparameters(X, y)
        last_tuning_time = time.time()  # 타이머 리셋
        return best_params, last_tuning_time
    return None, last_tuning_time

def handle_user_input(max_wait_time=60):
    print("\n결과를 사용하시겠습니까? (Enter로 확인, 일정 시간 후 자동 학습 계속)")
    start_time = time.time()
    while True:
        user_input = input(f"입력 대기중... (최대 대기 시간: {max_wait_time}초): ").strip()
        if user_input.lower() == 'y':
            print("결과를 사용합니다.")
            break
        elif time.time() - start_time > max_wait_time:
            print("시간 초과로 학습을 계속합니다.")
            break
        time.sleep(1)

import threading

def timeout_input(prompt, timeout=30):
    """
    일정 시간 동안 사용자 입력을 대기하고,
    시간 초과 시 None 반환.
    """
    user_input = [None]

    def input_thread():
        user_input[0] = input(prompt)

    thread = threading.Thread(target=input_thread)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    return user_input[0]


def main():
    """로또 분석 메인 프로그램"""
    # 경로 설정 및 데이터 준비
    act_path = "/content/drive/MyDrive/lotto4"
    data_path = f"{act_path}/lotto_data11.csv"
    print(f"데이터 경로: {data_path} 입니다.")

    mount_drive()
    data = load_and_preprocess_data(data_path)
    if data is None:
        print("데이터 로드 또는 전처리에 실패했습니다. 프로그램을 종료합니다.")
        return

    print(f"데이터 크기: {data.shape}")
    print(f"데이터 컬럼: {data.columns.tolist()}")

    X = data.drop(columns=['보너스']).values
    y = data['보너스'].values

    # 모델 로드
    individual_model_path_rf = get_model_path("trained_model_individual_rf", act_path)
    meta_model_path = get_model_path("trained_model_meta", act_path)
    rf_model = load_model(individual_model_path_rf, act_path)
    meta_model = load_model(meta_model_path, act_path)

    if rf_model is None or meta_model is None:
        print("기존 모델이 없어 새로 학습을 시작합니다...")
        rf_model, meta_model = train_individual_models(X, y), train_meta_model([rf_model], X, y)
        save_model(rf_model, "trained_model_individual_rf", act_path)
        save_model(meta_model, "trained_model_meta", act_path)

    # 학습 루프
    best_eval_accuracy = 0
    last_prediction_time = time.time()
    prediction_interval = 300  # 예측 결과 출력 간격 (초 단위)
    improvement_threshold = 0.01  # 성능 개선 임계값 (1%)

    for epoch in range(1, 1001):
        print(f"\n[{epoch}] 학습 시작...")

        eval_accuracy = evaluate_model([rf_model], X, y)
        print(f"현재 평가 정확도: {eval_accuracy:.4f}, 최고 평가 정확도: {best_eval_accuracy:.4f}")

        # 성능이 개선되었을 경우
        if eval_accuracy > best_eval_accuracy * (1 + improvement_threshold):
            best_eval_accuracy = eval_accuracy
            save_model(meta_model, "best_model", act_path)
            print(f"최고 성능 갱신! (정확도: {best_eval_accuracy:.4f})")

            # 예측 결과 제공
            if time.time() - last_prediction_time > prediction_interval:
                print("\n[예측 결과]")
                provide_recommendations(meta_model, X)
                last_prediction_time = time.time()

                # 사용자 입력 대기
                user_decision = timeout_input("학습을 종료하시겠습니까? (yes/no): ", timeout=30)
                if user_decision and user_decision.lower() == "yes":
                    print("학습을 종료합니다.")
                    break
                elif user_decision is None:
                    print("시간 초과! 학습을 계속 진행합니다.")
                else:
                    print("학습을 계속 진행합니다.")

        # 조건부 하이퍼파라미터 튜닝
        best_params, _ = conditional_tuning(epoch, eval_accuracy, best_eval_accuracy, X, y)
        if best_params:
            print("튜닝된 하이퍼파라미터 적용 중...")
            rf_model.set_params(**best_params)
            rf_model.fit(X, y)
            meta_model = train_meta_model([rf_model], X, y)
            save_model(meta_model, "trained_model_meta", act_path)

        log_progress(epoch, best_eval_accuracy)
        time.sleep(1)

        # 종료 조건: 목표 성능 도달
        if best_eval_accuracy >= 0.95:
            print("목표 정확도에 도달하여 학습을 종료합니다.")
            break

    print("프로그램 종료.")

if __name__ == "__main__":
    main()
