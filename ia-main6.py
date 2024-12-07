#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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


#ia-main7.py

import time
import json
import os

# 튜닝 이력 파일 경로
tuning_history_file = "/content/drive/MyDrive/lotto4/tuning_history.json"
initial_data_size_file = "/content/drive/MyDrive/lotto4/initial_data_size.json"

def load_tuning_history():
    """
    저장된 튜닝 이력을 불러오는 함수.
    """
    if os.path.exists(tuning_history_file):
        with open(tuning_history_file, "r") as file:
            history = json.load(file)
        return history
    else:
        return []  # 이력이 없으면 빈 리스트 반환

def load_last_data_size():
    """
    저장된 초기 데이터 크기를 불러오는 함수.
    """
    if os.path.exists(last_data_size_file):
        with open(last_data_size_file, "r") as file:
            data_size = json.load(file)
        return data_size
    else:
        return None  # 초기 데이터 크기가 없으면 None 반환

def save_last_data_size(data_size):
    """
    초기 데이터 크기를 저장하는 함수.
    """
    with open(last_data_size_file, "w") as file:
        json.dump(data_size, file, indent=4)

def save_tuning_history(epoch, accuracy, best_params, model_path):
    """
    튜닝 이력을 저장하는 함수.
    """
    tuning_history = load_tuning_history()
    new_entry = {
        "epoch": epoch,
        "accuracy": accuracy,
        "best_params": best_params,
        "model_path": model_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    tuning_history.append(new_entry)
    with open(tuning_history_file, "w") as file:
        json.dump(tuning_history, file, indent=4)

def conditional_tuning(current_accuracy, previous_accuracy, X, y):
    """
    조건 기반 하이퍼파라미터 튜닝 함수.
    """
    # 마지막 튜닝 시간 불러오기
    tuning_history = load_tuning_history()
    if tuning_history:
        last_tuning_time = time.strptime(tuning_history[-1]["timestamp"], "%Y-%m-%d %H:%M:%S")
        last_tuning_time = time.mktime(last_tuning_time)
    else:
        last_tuning_time = time.time()  # 첫 실행 시 현재 시간 설정

    performance_drop = (current_accuracy - previous_accuracy) < 0.01  # 성능 저하 조건

    # 데이터 크기 증가 여부 판단
    initial_data_size = load_initial_data_size()  # 초기 데이터 크기 불러오기
    current_data_size = len(X)  # 현재 데이터 크기

    if initial_data_size and current_data_size > initial_data_size * 1.1:
        data_growth = True  # 데이터 크기 10% 이상 증가
    else:
        data_growth = False  # 데이터 크기 변화 없음

    # 조건 튜닝 실행 및 타이머 초기화
    if performance_drop or data_growth:
        print("조건 변화 감지: 조건 기반 하이퍼파라미터 튜닝 실행...")
        best_hyperparameters = optimize_hyperparameters(X, y)  # 조건 튜닝 실행
        last_tuning_time = time.time()  # 타이머 리셋

        # 튜닝 이력 저장
        save_tuning_history(epoch, current_accuracy, best_hyperparameters, "/content/drive/MyDrive/lotto4/best_model.pkl")

        # 데이터 크기 갱신
        save_initial_data_size(current_data_size)  # 갱신된 데이터 크기 저장

        return best_hyperparameters, last_tuning_time  # 최적화된 파라미터 반환

    # 기간 튜닝 실행 조건 확인
    tuning_delay = 7 * 24 * 3600  # 7일 (단위: 초)
    if time.time() - last_tuning_time > tuning_delay:
        print("조건 만족 및 타이머 경과: 기간 기반 하이퍼파라미터 튜닝 실행...")
        best_hyperparameters = optimize_hyperparameters(X, y)  # 기간 튜닝 실행
        last_tuning_time = time.time()  # 타이머 리셋

        # 튜닝 이력 저장
        save_tuning_history(epoch, current_accuracy, best_hyperparameters, "/content/drive/MyDrive/lotto4/best_model.pkl")

        # 데이터 크기 갱신
        save_initial_data_size(current_data_size)  # 갱신된 데이터 크기 저장

        return best_hyperparameters, last_tuning_time  # 최적화된 파라미터 반환

    return None  # 튜닝 조건 미충족 시 None 반환

def main():
    print("로또 분석 프로그램 시작")

    # 경로 설정
    act_path = "/content/drive/MyDrive/lotto4"
    data_path = f"{act_path}/lotto_data11.csv"
    print(f"데이터 경로: {data_path} 입니다.")

    # 모델 경로 설정
    individual_model_path_rf = get_model_path("trained_model_individual_rf", act_path)
    individual_model_path_gb = get_model_path("trained_model_individual_gb", act_path)
    meta_model_path = get_model_path("trained_model_meta", act_path)

    # 데이터 로드 및 피처 생성
    data = load_and_preprocess_data(data_path)
    print(data.shape)  # 데이터 크기 출력

    if data is None:
        print("데이터 로딩에 실패했습니다. 프로그램을 종료합니다.")
        return

    # 초기 데이터 크기 저장 (첫 실행 시)
    last_data_size = load_last_data_size()
    if not last_data_size:
        save_last_data_size(len(data))  # 초기 데이터 크기 저

    train_and_evaluate(data, data_path)
  
    # 추천 번호 제공
    provide_recommendations(meta_model, X, top_n=5)
    print("프로그램을 종료합니다.")

# end main

import os
import time
from sklearn.metrics import accuracy_score
from feature_utils6 import generate_features
from model_utils6 import load_model, save_model, train_individual_models, train_meta_model, feature_engineering
from recommendations import provide_recommendations
from tuning_conditional import period_tuning. #conditional_tuning

def train_and_evaluate(data, paths):
    """학습 루프 및 모델 평가"""
    X = data.drop(columns=['보너스']).values
    y = data['보너스'].values

    # 모델 로드 또는 초기화
    rf_model = load_model(paths["model_rf_path"])
    meta_model = load_model(paths["model_meta_path"])

    if rf_model is None or meta_model is None:
        print("기존 모델이 없어 새로 학습을 시작합니다...")
        rf_model = train_individual_models(X, y)
        meta_model = train_meta_model([rf_model], X, y)
        save_model(rf_model, paths["model_rf_path"])
        save_model(meta_model, paths["model_meta_path"])

    # 학습 루프 초기화
    best_eval_accuracy = 0
    no_improvement_count = 0
    max_no_improvement_epochs = 5
    performance_log = []

    while True:
        print("\n[학습 루프 시작]")
        # 1. 피처 엔지니어링
        X = feature_engineering(X)

        # 2. 모델 평가
        eval_accuracy = evaluate_model([rf_model], X, y)
        print(f"현재 평가 정확도: {eval_accuracy:.4f}, 최고 평가 정확도: {best_eval_accuracy:.4f}")

        # 성능 향상 여부 확인
        if eval_accuracy > best_eval_accuracy:
            best_eval_accuracy = eval_accuracy
            save_model(meta_model, paths["best_model_path"])
            print(f"최고 성능 갱신! (정확도: {best_eval_accuracy:.4f})")
            provide_recommendations(meta_model, X)
            no_improvement_count = 0  # 개선되었으므로 카운트 초기화
        else:
            no_improvement_count += 1

        # 2-1. 최적의 하이퍼파라미터 튜닝
        best_params, improvement = conditional_tuning(eval_accuracy, best_eval_accuracy, X, y)
        if improvement:
            print("하이퍼파라미터가 업데이트되었습니다.")
            rf_model.set_params(**best_params)
            rf_model.fit(X, y)
            meta_model = train_meta_model([rf_model], X, y)
            save_model(meta_model, paths["model_meta_path"])
        elif no_improvement_count >= max_no_improvement_epochs:
            # 2-2. 기간 기반 튜닝
            print("성능 개선이 일정 시간 동안 이루어지지 않아 기간 기반 튜닝을 실행합니다.")
            best_params = period_tuning(rf_model, meta_model, X, y)
            rf_model.set_params(**best_params)
            rf_model.fit(X, y)
            meta_model = train_meta_model([rf_model], X, y)
            save_model(meta_model, paths["model_meta_path"])

        # 3. 일정 성능 향상 시 결과 저장
        if best_eval_accuracy >= 0.95:
            print("목표 성능에 도달하였습니다. 5세트의 추천을 저장합니다.")
            recommendations = provide_recommendations(meta_model, X, top_n=5)
            with open(os.path.join(paths["output_path"], "recommendations.txt"), "w") as f:
                for rec in recommendations:
                    f.write(f"{rec}\n")

        # 4. 사용자 입력 처리
        user_input = input("예측 결과를 5~10세트 출력할까요? 학습을 계속 진행할까요? (y/Enter: 계속, n: 종료): ")
        if user_input.lower() == "n":
            print("학습을 종료합니다.")
            break

        performance_log.append((time.time(), eval_accuracy))

        # 5. 중단 시점 저장
        with open(os.path.join(paths["output_path"], "training_progress.log"), "a") as log_file:
            log_file.write(f"Time: {time.time()}, Accuracy: {eval_accuracy:.4f}\n")

    return best_eval_accuracy

# end main
