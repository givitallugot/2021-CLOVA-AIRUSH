# 1-2: 실시간 버스 도착 소요시간 예측 정확도 향상

- 1라운드 참여 과제로 특정 서울 시내버스 (143번)의 실시간 정보가 주어졌을 때 현재 위치의 정류장에서 다음 정류장까지의 소요시간을 예측하는 문제입니다. 데이터셋은 버스의 10주 동안의 실시간 로그를 학습 데이터로 제공하며 평가를 위해 2주 동안의 실시간 로그를 사용합니다.

<br>

## 리더보드 점수

No. | 모델 | RMSE
---- | ----- | ----- 
**x** | **Baseline** | 65.xxxx
**377** | **LSTM+Windowing** | 72.7681
**399** | **Hybrid+Windowing** | 63.24296
**539** | **MLP (BEST)** | 62.20957

<br>

## 1. LSTM + Windowing
Window size를 7로 설정하여 7번째 전 도착 정보를 함께 훈련한다. 다음과 같은 피처를 함께 사용한다.
- prev_velocity = prev_station_distance/prev_duration: 트래픽을 반영하여 이전 정류장의 속도로 다음 정류장의 속도를 가늠한다.
- diff_station_id = next_station_id - station_id: 건너뛰는 정류장이 있는 경우 1이 아니라 더 큰 값이 된다.

<br>

## 2. Hybrid + (LSTM) Windowing
앞의 LSTM + Windowing에 Multi-Layer를 추가한다. 앞과 다르게 LSTM에는 시계열적 특성을 가진 변수만 사용하고 MLP에 다른 피처들을 사용한다. 각각 피처를 추출한 뒤 concatenate 후 사용한다. 다음과 같은 피처를 추가로 함께 사용한다.
- next_curvature = next_station_distance/next_direct_distance*100: 다음 정류장까지 거리를 직선 거리로 나눠 곡률을 예상한다.
- next_duration_pred = next_station_distance*prev_velocity: 이전 정류장의 속도를 이용하여 다음 정류장까지의 시간을 가늠한다.

<br>

## 3. MLP
베이스라인 MLP 모델에 다양한 피처를 추가한다. 추가한 피처는 다음과 같다.
- prev_station_distance2 = prev_station_distance*2
- next_station_distance2 = next_station_distance*2
- dows: 주말과 주중으로 구분
- hours: 시간대 구분 설정
- dowhour = dows*hours
- prev_duration_bin, next_curvature_bin: 샘플 데이터에서 벗어나는 이상치를 구분하는 지시변수

<br>

## 1라운드를 마치며

1. 여러 주제에 시간을 분배해서 투자하는 것이 더욱 효율적이었을 것 같다.
2. 피처와 모델링 고민을 많이 했으나, 성능을 높이기 위해서는 더욱 획기적인 아이디어가 필요했던 것 같다.
