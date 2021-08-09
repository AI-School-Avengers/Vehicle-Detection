# Vehicle-Detection
차량종류(버스, 오토바이, 승용차, 트럭), 차량번호판 인식 후 Azure OCR API 이용한 차량번호 추출

<br/>

#### 목표 
차량(버스, 오토바이, 승용차, 트럭), 차량번호판 인식 모델 구현

#### 프로젝트 기간 
10일 (07.26 - 08.06)

---

## 진행
#### 라벨 종류
Car (승용차), Truck (트럭), Bus (버스), Etc vehicle (기타 차량-덤프트럭, 레미콘 등 건설용차량), Bike( 이륜
차) License(번호판)

#### 최종 학습 데이터 수 
2606장

#### 선정 모델 
fast-RCNN
  - YOLO는 Train속도가 빠르다는 장점이 있었지만, 
				결론적으로 정확도 면에서 Fast-RCNN이 더 정교함을 확인할 수 있었음
- 추후 작업하는 과정에서 YOLO는 모델 자체에 NMS 등 데이터 후처리 기능이 잘 되어있다는 이점이 있었지만, 후처리를 따로 구현하기로 하고 정확도가 더 높은 fast-RCNN을 선정  

#### Annotation 작업
[cvat](https://cvat.org/)에서 직접 bounding box를 그려서 생성 

<br/>

 #### 최종 테스트에선 model_83.pth를 사용 ([model 파일](https://drive.google.com/drive/folders/1uPNTmPwH0yoMDWR3hRPaWWebjgmB2Q1w?usp=sharing))

|file_name|epoch|optmizer|lr|weight decay|preset|
|:---:|:---:|:------:|:---:|:---:|:---:|
|model_85|83|SGD|0.0025|0.0001|hflip|
|model_ssd41|41|SGD|0.0025|0.0001|ssd|

<br/>

---
## 결과
- 차량은 대부분 잘 검출하고 분류함
- 번호판 글자 인식은 서툴다

---
## 아쉬운 점

#### 데이터 후처리(mAP, nms)
  - nms 함수가 iou가 임계값 이상이더라도 높은 conf로 다른 label로 예상 시, bounding box를 걸러내지 못한다.
  - mAP가 제대로 동작하지 않는다.
#### 데이터 전처리
  - 다양한 전처리를 해보지 않고 기존 제공된 전처리 방식을 사용했다.
#### 데이터 학습
  - 특정 차량에 대한 데이터 수집이 부족해서 해당 차량에 대한 detection을 잘 수행하지 못한다.
<br/>


![학습 데이터 부족](https://user-images.githubusercontent.com/37794363/128749755-783e5a6d-8564-42e0-932d-2c9b4282017a.png)

![데이터 후처리](https://user-images.githubusercontent.com/37794363/128749857-5ce2b321-db56-45d3-aedf-f3f8e1f89e70.png)
<br/>
<br/>

---

## 다음 프로젝트 진행 시 개선 방향
- Dataset 만드는 것에 너무 많은 시간을 빼앗기지 않기 
  - 미숙함의 원인 : 스케줄을 생각하지않고 진행하다가 후반에 여러 기능을 구현하는데 시간이 많이 부족했다.
- 후처리를 제공하는 모델을 사용해서 train 결과를 먼저 보는 것도 나쁘지 않은 선택
- nms 후에도 걸러내지 못했던 bounding box는 임계값을 75% 이상으로 하고 conf 스코어가 높은 박스만을 살리는 방식으로 했으면 해결 가능

<br/>

---

## 참고 자료
[nms](https://dyndy.tistory.com/275) <br/>
[nms 코드 설명](https://naknaklee.github.io/etc/2021/03/08/NMS/) <br/>
[nms 구현](https://deep-learning-study.tistory.com/403) <br/>
[iou](https://ballentain.tistory.com/12) <br/>
[nms 영문](https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c) <br/>
[Azure OCR 지원언어](https://docs.microsoft.com/ko-kr/azure/cognitive-services/computer-vision/language-support#optical-character-recognition-ocr) <br/>
[Azure OCR API](https://docs.microsoft.com/ko-kr/azure/cognitive-services/computer-vision/overview-ocr) <br/>
[성능지표 관련 자료 (mAP)](https://eehoeskrap.tistory.com/546) <br/>
[성능지표 관련 자료 (mAP)](https://glee1228.tistory.com/5) <br/>
[xml](http://egloos.zum.com/sweeper/v/3045370) <br/>
