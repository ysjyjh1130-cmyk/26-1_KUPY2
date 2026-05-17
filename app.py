#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

# --------------------------
# 모델 정의 # 1. 모델 아키텍처 재정의 (저장된 가중치를 씌우기 위해 똑같은 뼈대가 필요함)
# --------------------------
class PPANet(nn.Module):
    def __init__(self):
        super(PPANet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.model(x)

# --------------------------
# 2. 저장된 파일 로드
# --------------------------
model = PPANet()
model.load_state_dict(torch.load("ppa_model.pth", map_location=torch.device("cpu")))
model.eval()

x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# --------------------------
# 3. Streamlit UI 구성 및 동작 로직
# --------------------------
st.title("AI-based PPA Predictor")

# 사용자로부터 8개의 변수를 입력받는 UI
# (인자: "표시될 텍스트", 최소값, 최대값, 기본설정값)
gate_count = st.number_input("Gate Count", 1000, 50000, 10000)
wire_length = st.number_input("Wire Length (um)", 100.0, 10000.0, 3000.0)
fanout = st.number_input("Fanout", 1, 20, 5)
logic_depth = st.number_input("Logic Depth", 5, 80, 20)
frequency = st.number_input("Frequency (Hz)", 1e8, 2e9, 1e9)
vdd = st.number_input("VDD (V)", 0.7, 1.2, 1.0)
toggle_rate = st.number_input("Toggle Rate", 0.01, 0.5, 0.2)
temperature = st.number_input("Temperature (K)", 250.0, 400.0, 300.0)

# "Predict" 버튼이 눌렸을 때 실행될 내용
if st.button("Predict"):
    # 1. 입력받은 값들을 딥러닝에 넣기 위해 하나의 배열(Matrix)로 묶어줌
    x = np.array([[
        gate_count,
        wire_length,
        fanout,
        logic_depth,
        frequency,
        vdd,
        toggle_rate,
        temperature
    ]])

    # 2. 학습 때 사용했던 x_scaler를 이용해 동일한 비율로 데이터 스케일링
    x_scaled = x_scaler.transform(x)

    # 3. 모델에 넣고 예측 (tensor 변환 필수)
    with torch.no_grad():
        pred = model(torch.tensor(x_scaled, dtype=torch.float32))

    # 4. 스케일링되어 있는 예측 결과(0~1 사이 값)를 원래 단위(mW, ns)로 복원
    pred = y_scaler.inverse_transform(pred.numpy())

    # 5. 최종 결과 웹 화면에 출력
    st.subheader("Prediction Result")
    st.write(f"Power: {pred[0][0]:.3f} mW")
    st.write(f"Delay: {pred[0][1]:.3f} ns")
    st.write(f"Area: {pred[0][2]:.3f} um^2")

