# DSHCF
 Learning Dynamic-Sensitivity Enhanced Correlation Filter with Adaptive Second-order Difference Spatial Regularization for UAV Object Tracking
# LDEFC_Jungwoo
# ResNet-18 Feature Integration Guide

## 개요
이 가이드는 DSHCF 추적기에 ResNet-18 pre-trained 모델의 medium layer 특징을 통합하는 방법을 설명합니다.

## 요구사항
- MATLAB R2018b 이상
- Deep Learning Toolbox
- (선택) Parallel Computing Toolbox (GPU 사용 시)

## 사용 방법

### 1. 기본 사용법

`run_DSHCF.m` 파일에서 ResNet-18 특징을 활성화하려면:

```matlab
% ResNet-18 feature parameters
use_resnet = true;  % false에서 true로 변경
resnet_params.layer_name = 'res4b_relu';  % Medium layer
resnet_params.use_gpu = false;  % GPU 사용 시 true
```

### 2. 레이어 선택

ResNet-18의 다양한 레이어에서 특징을 추출할 수 있습니다:

- **res4b_relu** (기본값): Medium layer, 좋은 성능과 속도의 균형
- **res3b_relu**: 더 낮은 레이어, 더 많은 공간 정보
- **res5b_relu**: 더 높은 레이어, 더 많은 의미론적 정보

레이어 이름 확인:
```matlab
net = resnet18;
layer_names = arrayfun(@(x) x.Name, net.Layers, 'UniformOutput', false);
```

### 3. GPU 사용

GPU를 사용하려면:
```matlab
resnet_params.use_gpu = true;  % GPU 사용
```

GPU 사용 가능 여부 확인:
```matlab
canUseGPU  % true면 GPU 사용 가능
```

### 4. 특징 차원

ResNet-18 특징의 차원은 선택한 레이어에 따라 자동으로 결정됩니다:
- res4b_relu: 약 256 채널
- res5b_relu: 약 512 채널

특징 차원이 너무 크면 성능에 영향을 줄 수 있으므로, 필요시 PCA를 사용하여 차원을 축소할 수 있습니다.

## 성능 고려사항

### 장점
- **강력한 특징 표현**: ImageNet으로 pre-trained된 딥러닝 특징은 복잡한 시나리오에서 우수한 성능
- **가림/간섭 대응**: 의미론적 특징으로 유사 객체와의 구분 향상
- **조명 변화 강건성**: 딥러닝 특징의 조명 불변성

### 단점
- **계산 비용**: 특징 추출 시간이 증가 (GPU 사용 시 개선)
- **메모리 사용**: 특징 차원이 커서 메모리 사용량 증가
- **초기화 시간**: 첫 프레임에서 모델 로딩 시간 소요

## 최적화 팁

1. **GPU 사용**: 가능하면 GPU를 사용하여 특징 추출 속도 향상
2. **레이어 선택**: 성능과 속도의 균형을 위해 medium layer 사용
3. **특징 융합**: ResNet-18 특징과 기존 특징(HOG, ColorNames)을 함께 사용
4. **차원 축소**: 필요시 PCA를 사용하여 특징 차원 축소

## 문제 해결

### "ResNet-18 requires Deep Learning Toolbox" 오류
- Deep Learning Toolbox가 설치되어 있는지 확인
- MATLAB 버전이 R2018b 이상인지 확인

### GPU 관련 오류
- GPU가 지원되는지 확인: `canUseGPU`
- GPU가 없으면 `use_gpu = false`로 설정

### 메모리 부족 오류
- 특징 차원이 너무 큰 경우, 더 낮은 레이어 사용 고려
- 또는 PCA를 사용하여 차원 축소

## 예제

```matlab
% ResNet-18 특징만 사용
params.t_features = {
    struct('getFeature',@get_resnet18,'fparams',resnet_params),...
};

% ResNet-18 + HOG 특징 융합
params.t_features = {
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_resnet18,'fparams',resnet_params),...
};
```

