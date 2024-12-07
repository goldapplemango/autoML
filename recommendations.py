#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import json
import numpy as np
import time

class IncompatibleCombinationsManager:
    def __init__(self, explicit_file, dynamic_file):
        self.explicit_file = explicit_file
        self.dynamic_file = dynamic_file
        self.explicit_combinations = self.load_explicit_combinations()
        self.dynamic_combinations = self.load_dynamic_combinations()

    def load_explicit_combinations(self):
        try:
            df = pd.read_csv(self.explicit_file)
            return [list(row) for row in df.values]
        except FileNotFoundError:
            return []

    def load_dynamic_combinations(self):
        try:
            with open(self.dynamic_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_explicit_combinations(self):
        df = pd.DataFrame(self.explicit_combinations)
        df.to_csv(self.explicit_file, index=False)
        print(f"명시적 부적합 조합 저장 완료: {self.explicit_file}")

    def save_dynamic_combinations(self):
        with open(self.dynamic_file, 'w') as f:
            json.dump(self.dynamic_combinations, f)
        print(f"동적 부적합 조합 저장 완료: {self.dynamic_file}")

    def update_dynamic_combinations(self, new_combinations):
        today = "2024-12-06"  # 오늘 날짜로 가정
        if today not in self.dynamic_combinations:
            self.dynamic_combinations[today] = []
        self.dynamic_combinations[today].extend(new_combinations)
        self.save_dynamic_combinations()

    def filter_invalid_combinations(self, predictions):
        valid_predictions = []
        for pred in predictions:
            if pred in self.explicit_combinations:
                continue
            for date, combs in self.dynamic_combinations.items():
                if pred in combs:
                    continue
            valid_predictions.append(pred)
        return valid_predictions

# 예측 로직과 학습 루프 결합

def provide_recommendations(models, data, num_sets=5, comb_manager=None):
    """로또 번호 추천 (1~45 범위로 조정)"""
    recommendations = []
    for _ in range(num_sets):
        sample = data.sample(n=1)
        predictions = sum([model.predict(sample) for model in models]) / len(models)
        numbers = np.round(predictions).astype(int)
        numbers = np.clip(numbers, 1, 45)  # 1~45 범위로 조정
        numbers = np.unique(numbers)  # 중복 제거
        if len(numbers) < 6:  # 6개 번호가 안되면 추가
            additional_numbers = np.random.choice(range(1, 46), 6 - len(numbers), replace=False)
            numbers = np.concatenate([numbers, additional_numbers])

        # 부적합 조합 필터링
        valid_numbers = comb_manager.filter_invalid_combinations([numbers])
        recommendations.append(sorted(valid_numbers[0]))  # 6개 번호 추천

    for idx, rec in enumerate(recommendations, 1):
        print(f"세트 {idx}: 번호: {rec}")

# 프로그램 시작 시 IncompatibleCombinationsManager 인스턴스 생성
comb_manager = IncompatibleCombinationsManager("explicit_incompatible_combinations.csv", "dynamic_incompatible_combinations.json")

# 예시로 사용하는 모델들
models = [model1, model2]  # 예시 모델들

# 로또 번호 추천
data = ...  # 예시 데이터
provide_recommendations(models, data, num_sets=5, comb_manager=comb_manager)

# 새로운 예측 후 동적 부적합 조합 갱신
new_combinations = [[2, 4, 6, 8, 10, 12]]
comb_manager.update_dynamic_combinations(new_combinations)


# In[1]:


from google.colab import drive
drive.mount('/content/drive')

