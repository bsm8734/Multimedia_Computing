# Real-time hand-tracking

> 2019-1 학기 인하대학교 멀티미디어컴퓨팅 과제(프로젝트) 수행을 기록한 레포입니다.  
> 추후, 이 코드를 활용하여 `2019 Problem Solving 경진대회`에서 `드론을 활용한 익수자 추적 소프트웨어`에 기여하여 `🏅 금상 🏅`을 수상하였습니다.

## 수행일자

- `2019.03.01.` ~ `2019.06.10.`

## 프로젝트 설명

- 실시간으로 물체(손)을 추적하여, 프레임 내 일정 구간에 진입했을 때, 타악기 소리를 내는 프로그램
- 딥러닝 방식인 SSD(Shot Multi Box Detector)와 openCV의 CSRT Tracker를 사용하여 구현
- 실시간성의 기준을 20 FPS로 설정

## 아키텍처

### SSD (Single Shot Multi Box Detector)

- [[github] Real-time Hand-Detection using Neural Networks (SSD) on Tensorflow - SSD pretrained model](https://github.com/victordibia/handtracking)
- [[paper] SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

### CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability)

- [[paper] Discriminative Correlation Filter with Channel and Spatial Reliability](https://arxiv.org/abs/1611.08461)

- // 작성중
- 지정된 박스 좌표를 넘겨받아, 박스 안의 픽셀 이동을 파악하는 방법으로 물체를 추적하는 역할을 합니다.
- 기존의 Single-Channel CF tracker에서 channel reliability와 spatial reliability를 도입한 방법입니다.
- Single-Channel CF tracker
- Channel reliability
- Spatial reliability
- OpenCV에서 제공하는 tracker 중 하나
  - OpenCV에서 제공하는 tracker인 KCF, CSRT, MOSSE 등의 트레커를 사용하였는데, 각 트레커 마다 성능과 속도의 차이가 존재했습니다.
  - 특징
    - CSRT tracker는 상대적으로 높은 정확도를 보였으나, 프레임이 16 프레임 안팎 정도로 트레킹하는데 어려움을 겪음
    - MOSSE tracker는 평균적으로 100프레임 이상의 높은 프레임을 자랑했으나, 저조한 정확도로 인하여 트레킹 하는 물체를 자주 놓침
     ![image](https://user-images.githubusercontent.com/35002768/136603480-aa12952a-aa6e-44db-b5d3-a1ebad199211.png)


## 수행환경

> 모든 실험은 아래의 환경에서 수행되었으며, 아래와 같이 좋지 않은 사양이었습니다.  
> 최종적으로 사용한 방법은 이러한 저사양 기기에서도 수월하게 실시간 연주가 가능함을 보였습니다.  

- CPU: Intel I3-5005U (2.00GHz)
- Ram 4GB
- Graphic Intel HD graphics 5500 (노트북 내장 그래픽카드)

