# Emotion Recognition for PONITOR

PONITOR에서 보이스피싱 피해자를 탐지하는데 감정 인식은 중요한 기술이다. 
하지만 코로나19로 인해 마스크 착용이 일상화된 지금, 표정만으로 감정을 탐지하는 것은 정확도가 낮았다.
따라서 표정뿐만아니라 몸짓, 장면 맥락까지 고려한 emotion recognition 모델을 사용하게 되었다. 
<br></br>
![image](https://user-images.githubusercontent.com/90603399/206726985-3e302299-12dd-4b08-ace1-7b2e5e3a2414.png) <br>
관련 기술블로그 참고 <br>
*[두두의 벨로그_PONITOR](https://velog.io/@kang1221/Fonitor-대면편취형-보이스피싱-을-예방하기-위한-모니터링-서비스)*
<br></br>
## Pipeline

다음의 논문에서 사용하는 CNN 모델을 사용하였다. 
그 구조와 해당 논문은 다음과 같다. 
![Pipeline](https://raw.githubusercontent.com/Tandon-A/emotic/master/assets/pipeline%20model.jpg "Model Pipeline") 
###### Model Pipeline ([Image source](https://arxiv.org/pdf/2003.13401.pdf))

첫번째 모듈에서는 YOLO를 이용하여 Body를 detect하고 여기서 body feature를 추출한다.
두번째 모듈에서는 이미지 전체의 image(context) feature를 추출한다.
Fusion network에서는 앞서 추출한 두가지의 feature를 combine하여 최종 결과값인 vad값과 카테고리를 예측한다.  
<br></br>
## Emotic Dataset 
다음의 EMOTIC dataset을 사용하였다.  
*[EMOTIC dataset'](https://paperswithcode.com/dataset/emotic)*
<br></br>
![image](https://user-images.githubusercontent.com/90603399/206726105-30f3b099-dea3-4f4f-9548-a507ba2efafd.png)


<br></br>


## Usage
pre 파일에는 EMOTIC dataset을 다운로드받아 전처리한 데이터셋들이 npy형식으로 저장되어있어 
별도로 데이터를 다운로드 받고 전처리할 필요가 없다. 
또한 학습이 완료된 모델은 다음에서 다운로드 받을 수 있다.
<br></br>
*['Trained model and thresholds'](https://drive.google.com/drive/folders/1e-JLA7V73CQD5pjTFCSWnKCmB0gCpV1D)*
<br></br>
다운로드 받은 후 model 폴더 안에 저장해 놓고 yolo_inference.py를 실행시킬 때 
experiment_path에서 경로를 입력하면 별도로 학습시키지 않아도 저장된 모델을 불러와 바로 test할 수 있다. 
<br></br>

## Test video
Peacebite의 김세림, 강연수(본인) 팀원이 직접 촬영한 테스트 영상과 보이스피싱 피해자 이미지는 아래에서 다운로드 받을 수 있다. 
<br></br>
*[Voice Phishing Test Inputs](https://drive.google.com/drive/folders/1s5REQo49n3526jeE71j-qjrYSyLm7Aj0?usp=sharing)*

<br></br>

## To perform inference: 
1. 비디오 파일을 inference하는 코드이다. 
```python
>  python yolo_inference.py --experiment_path proj/debug_exp --video_file C:\emotic-master\assets\video_file.mp4
```
experiment_path : experiment directory의 경로명으로 학습된 모델이 저장되어 있음<br></br>
video_file: 입력 비디오 파일의 경로  

실행 결과 비디오는 
\model\results 에서 result_vid.mp4 형식으로 확인할 수 있다. 

<br></br>

2. 이미지 파일을 inference하고 싶다면 다음의 코드를 실행시킨다. 
```python
>  python yolo_inference.py --experiment_path C:\emotic-master\model  --inference_file C:\emotic-master\assets\friends.jpg
```
experiment_path : experiment directory의 경로명으로 학습된 모델이 저장되어 있음<br></br>
inference_file: 입력 이미지 파일의 경로 정보가 적혀있는 txt파일의 경로 
(assets/inference_file.txt 참고) 
<br></br>

## Results 

![Result GIF 1](https://github.com/Ponitor/Ponitor_DL/blob/main/EmotionRecognition/assets/test_result.gif "Result GIF 1")

<br></br>
## Acknowledgements

* [Places365-CNN](https://github.com/CSAILVision/places365) 
* [Pytorch-Yolo](https://github.com/eriklindernoren/PyTorch-YOLOv3)
* 
<br></br>
### Context Based Emotion Recognition using Emotic Dataset 
_Ronak Kosti, Jose Alvarez, Adria Recasens, Agata Lapedriza_ <br>
[[Paper]](https://arxiv.org/pdf/2003.13401.pdf) [[Project Webpage]](http://sunai.uoc.edu/emotic/) [[Authors' Implementation]](https://github.com/rkosti/emotic)

```
@article{kosti2020context,
  title={Context based emotion recognition using emotic dataset},
  author={Kosti, Ronak and Alvarez, Jose M and Recasens, Adria and Lapedriza, Agata},
  journal={arXiv preprint arXiv:2003.13401},
  year={2020}
}
```
<br></br>

## Reference
[Context Based Emotion Recognition using EMOTIC Dataset]([https://github.com/Tandon-A](https://paperswithcode.com/paper/context-based-emotion-recognition-using))

# Phone Detection
<br/> 
코랩에서 실행
https://colab.research.google.com/drive/1Mkf10zWwA1xrKG2m34_5f4gsSUUDgHFy?usp=sharing

YOLOv5
<br/> 
![image](https://user-images.githubusercontent.com/84585914/206726165-498595e8-e5f0-459c-8411-b74ba87c3dbb.png)


<br/> 
Dataset 

<br/> 
Roboflow(https://roboflow.com/)

# AI Model Serving

## FastAPI
실시간 서비스인 포니터의 특성상 연산 속도가 중요하기 때문에 속도측면에서 가장 빠르다고 평가받는 FastAPI를 이용해 모델을 서빙했다.  
https://fastapi.tiangolo.com/![image](https://github.com/Ponitor/AI_serving/assets/90603399/af2effb8-faca-418e-8207-c30be46ee5b6)
