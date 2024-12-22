# 프로젝트 개요
한국어로 작성된 PaddleOCR의 Recognition 모델에 대한 세부 문서가 부족하여, 본 프로젝트를 추진하게 되었습니다. 본 프로젝트의 주요 목적은 사용자 맞춤형 전이 학습을 통해 Text Detection 및 Text Recognition 모델을 최적화하여, 특정 목적에 부합하는 활용성을 높이는 데 있습니다.<br/>

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [설치 지침](#설치-지침)
- [데이터 주석](#데이터-주석)
- [구성 파일 예시](#구성-파일-예시)
- [모델 학습](#모델-학습)
- [모델 내보내기](#모델-내보내기)

## 설치 지침

### 환경 요구 사항
- PaddlePaddle >= 2.1.0
- Python 3.5 <= Python < 3.9
- PaddleOCR >= 2.1

### 설치 명령어
```sh
# 프로젝트 클론
!git clone https://gitee.com/paddlepaddle/PaddleOCR.git

# PaddleOCR 설치
!pip install fasttext==0.8.3
!pip install paddleocr --no-deps -r requirements.txt

# 디렉토리 이동
%cd PaddleOCR/
```

## 데이터 주석
ppocrlabel 프로그램으로 라벨링을 진행한 뒤 개별 문자를 학습하기 위해 croping을 수행합니다.

```bash
with open('./M2021/Label.txt','r',encoding='utf8')as fp:
    s = [i[:-1].split('\t') for i in fp.readlines()]
    f1 = open('M2021_crop/rec_gt_train.txt', 'w', encoding='utf-8')
    f2 = open('M2021_crop/rec_gt_eval.txt', 'w', encoding='utf-8')
    for i in enumerate(s):
        path = i[1][0]
        anno = json.loads(i[1][1])
        filename = i[1][0][6:-4]
        image = Image.open(path)
        for j in range(len(anno)): 
            label = anno[j]['transcription']
            roi = anno[j]['points']
            coordinate = {'left_top': anno[j]['points'][0], 'right_top': anno[j]['points'][1], 'right_bottom': anno[j]['points'][2], 'left_bottom': anno[j]['points'][3]}
            print(roi, label)
            rotate = Rotate(image, coordinate)
            # 把图片放到目录下
            crop_path = 'M2021_crop' + path[5:-4:] + '_' + str(j) + '.jpg'
            rotate.run().convert('RGB').save(crop_path)
            # label文件不写入图片目录
            crop_path = path[6:-4:] + '_' + str(j) + '.jpg'
            if i[0] % 5 != 0:
                f1.writelines(crop_path + '\t' + label + '\n')
            else:
                f2.writelines(crop_path + '\t' + label + '\n')
    f1.close()
    f2.close()
```


## 구성 파일 예시
이 프로젝트는 en_number_mobile_v2.0_rec_train/config.yml구성 파일을 사용합니다. 파일의 변경된 지점은 다음과 같습니다.

```yaml
Global:
  debug: false
  use_gpu: true
  epoch_num: 200
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: ./output
  save_epoch_step: 20
  eval_batch_step:
  - 0
  - 200
  cal_metric_during_train: true
  checkpoints: null
  save_inference_dir: null
  use_visualdl: true
  infer_img: null
  character_dict_path: ppocr/utils/en_dict.txt
  character_type: EN
  max_text_length: 25
  infer_mode: false
  use_space_char: false
  distributed: false
  pretrained_model: en_number_mobile_v2.0_rec_train/best_accuracy
Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
  regularizer:
    name: L2
    factor: 1.0e-05
Architecture:
  model_type: rec
  algorithm: CRNN
  Transform: null
  Backbone:
    name: MobileNetV3
    scale: 0.5
    model_name: small
    small_stride:
    - 1
    - 2
    - 2
    - 2
  Neck:
    name: SequenceEncoder
    encoder_type: rnn
    hidden_size: 48
  Head:
    name: CTCHead
    fc_decay: 1.0e-05
Loss:
  name: CTCLoss
PostProcess:
  name: CTCLabelDecode
Metric:
  name: RecMetric
  main_indicator: acc
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ../M2021_crop
    label_file_list:
    - ../M2021_crop/rec_gt_train.txt
    ratio_list:
    - 1.0
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - RecAug: null
    - CTCLabelEncode: null
    - RecResizeImg:
        image_shape:
        - 3
        - 32
        - 320
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
  loader:
    shuffle: true
    batch_size_per_card: 256
    drop_last: true
    num_workers: 8
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ../M2021_crop
    label_file_list:
    - ../M2021_crop/rec_gt_eval.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - CTCLabelEncode: null
    - RecResizeImg:
        image_shape:
        - 3
        - 32
        - 320
    - KeepKeys:
        keep_keys:
        - image
        - label
        - length
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 64
    num_workers: 4
```

## 모델 학습
```sh
!python tools/train.py -c en_PP-OCRv3_rec_train/config.yml
```

## 모델 내보내기
```sh
!python tools/export_model.py -c en_PP-OCRv3_rec_train/config.yml -o Global.pretrained_model=output/v3_en_mobile2/best_accuracy Global.save_inference_dir=./transrec0620/
```

참고 출처
https://aistudio.baidu.com/projectdetail/3495816
