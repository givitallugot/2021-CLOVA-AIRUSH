# 1-2: 실시간 버스 도착 소요시간 예측 정확도 향상
- 버스의 실시간 정보를 사용하여 도착 소요시간을 예측합니다.

## Introduction
이 문제는 특정 서울 시내버스 (143번)의 실시간 정보가 주어졌을때 현재 위치의 정류장에서 다음 정류장까지의 소요시간을 예측하는 문제입니다. 데이터셋은 버스의 10주 동안의 실시간 로그를 학습 데이터로 제공하며 평가를 위해 2주 동안의 실시간 로그를 사용합니다.

## 학습 데이터
- ```train/train_data/info```: 데이터를 이해하기 위해 필요한 추가 정보 파일입니다. 
  - ```stations.csv```: 정류장 정보를 담고 있습니다.
    - station_id: 정류장 id
    - station_name: 정류장 이름
    - lat: 정류장 위도
    - lng: 정류장 경도
  - ```routes.csv```: 노선 정보를 담고 있습니다.
    - route_id: 노선 id
    - route_name: 노선 이름
    - start_point_name: 노선 기점 정류장 이름
    - end_point_name: 노선 종점 정류장 이름
    - main_bus_stop: 노선 주요 정류장 
    - turning_point_sequence: 노선 회차 정류장 시퀀스
  - ```shapes.csv```: 각 노선의 정류장 순서 및 정보를 담고 있습니다.
    - route_id: 노선 id
    - station_seq: 노선에서 정류장의 시퀀스
    - station_id: 정류장 id
    - next_station_id: 다음 정류장의 id
    - distance: 다음 정류장까지의 거리
- ```train/train_data/data```: 학습 데이터 (약 1,200,000 로그).
  - schema:
    - route_id
    - plate_no
    - operation_id
    - ts 
    - dow 
    - hour
    - station_id
    - station_seq
    - station_lng
    - station_lat
    - prev_station_id
    - prev_station_seq
    - prev_station_lng
    - prev_station_lat
    - prev_station_distance
    - prev_duration
    - next_station_id
    - next_station_seq
    - next_station_lng
    - next_station_lat
    - next_station_distance
- ```train/train_label```: 학습 데이터 label (약 1,200,000 로그).
  - schema:
    - route_id
    - plate_no
    - operation_id
    - station_seq
    - next_duration: 다음 정류장까지의 소요시간을 예측.
    
## 테스트 데이터
- ```test/test_data```: 테스트 데이터
  - 스키마는 위 train_data와 동일합니다.
  - 각 차량의 운행 (각 <plate_no, operation_id> 페어)마다 주어지는 로그 데이터는 다르며 주어진 로그 데이터의 마지막 정류장에서 다음 정류장까지 소요시간을 예측해야됩니다.
  - 예를 들어, 143번 버스의 <가> 차량의 100번째 운행에 대해서 1번째 ~ 40번째 정류장까지의 로그 데이터가 주어졌으면 40번째에서 41번째 정류장까지의 소요시간을 예측해야됩니다. 

## 베이스라인
- 베이스라인 모델로 전 정류장의 소요시간만 사용해 다음 정류장의 소요시간을 예측하게 학습 데이터에서 필요한 피쳐들을 추출해내고 3개의 dense layer를 사용합니다. 
- 주어진 학습 데이터를 어떻게 가공할지 (예. 더 많은 전 정류장의 데이터를 사용하기 위해 Data Windowing 적용)는 자유입니다. 

## 평가
- RMSE를 사용하여 평가합니다. 

## 실행
- `nsml run -d airush2021-1-2`

## Q&A
- 운행 시간이 날씨에 민감할텐데, 날씨 정보를 활용해도 되는가? (이번 범위에서는 벗어나는 것 같다)
- 동일한 번호(plate_no)는 동일인인가? 아닐수 있다.
- TF code인데, Pytorch로 다시 만들어도 되나? 됩니다.
