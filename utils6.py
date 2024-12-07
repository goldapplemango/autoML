#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# utils6.py

import json
import time
import os
import sys #sys

from google.colab import drive
drive.mount("/content/drive")

import numpy as np

def remove_special(act_path, filename):

    # 코드 내 모든 non-breaking space를 일반 공백으로 교체
    with open(f"{act_path}/{filename}", 'r', encoding='utf-8') as file:
        content = file.read()
        content = content.replace('\u00A0', ' ')  # 불연속 공백을 일반 공백으로 변환

    with open(f"{act_path}/{filename}", 'w', encoding='utf-8') as file:
        file.write(content)
    return None

# !jupyter nbconvert --to python /content/drive/MyDrive/lotto4/model_utils.ipynb

# act_path = "/content/drive/MyDrive/lotto4"

# filename = "feature_utils4.py"
# filename1 = "model_utils4.py"
# filename2 = "utils4.py"
# filename3 = "ai-main1202.ipynb"

# remove_special(act_path, filename)
# remove_special(act_path, filename1)
# remove_special(act_path, filename2)
# remove_special(act_path, filename3)

def log_progress(epoch, best_score):
    """학습 진행 상황 로그 기록"""
    with open("progress.log", "a") as log_file:
        log_file.write(f"Epoch {epoch}: Best Accuracy: {best_score}\n")



# end


# In[ ]:




