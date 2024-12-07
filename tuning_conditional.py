#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
import json
import os

# 이전 데이터 크기 파일 경로
previous_data_size_file = "/content/drive/MyDrive/lotto4/previous_data_size.json"

# 튜닝 이력 파일 경로
tuning_history_file = "/content/drive/MyDrive/lotto4/tuning_history.json"

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

def load_previous_data_size():
    """
    저장된 이전 데이터 크기를 불러오는 함수.
    """
    if os.path.exists(previous_data_size_file):
        with open(previous_data_size_file, "r") as file:
            data_size = json.load(file)
        return data_size
    else:
        return None  # 이전 데이터 크기가 없으면 None 반환

def save_previous_data_size(data_size):
    """
    이전 데이터 크기를 저장하는 함수.
    """
    with open(previous_data_size_file, "w") as file:
        json.dump(data_size, file, indent=4)

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
    previous_data_size = load_previous_data_size()  # 이전 데이터 크기 불러오기
    current_data_size = X.shape  # 현재 데이터 크기

    if previous_data_size and current_data_size > previous_data_size * 1.1:
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

        # 이전 데이터 크기 갱신
        save_previous_data_size(current_data_size)  # 갱신된 이전 데이터 크기 저장

        return best_hyperparameters, last_tuning_time  # 최적화된 파라미터 반환

    # 기간 튜닝 실행 조건 확인
    tuning_delay = 7 * 24 * 3600  # 7일 (단위: 초)
    if time.time() - last_tuning_time > tuning_delay:
        print("조건 만족 및 타이머 경과: 기간 기반 하이퍼파라미터 튜닝 실행...")
        best_hyperparameters = optimize_hyperparameters(X, y)  # 기간 튜닝 실행
        last_tuning_time = time.time()  # 타이머 리셋

        # 튜닝 이력 저장
        save_tuning_history(epoch, current_accuracy, best_hyperparameters, "/content/drive/MyDrive/lotto4/best_model.pkl")

        # 이전 데이터 크기 갱신
        save_previous_data_size(current_data_size)  # 갱신된 이전 데이터 크기 저장

        return best_hyperparameters, last_tuning_time  # 최적화된 파라미터 반환

    return None
    # 튜닝 조건 미충족 시 None 반환

# end

