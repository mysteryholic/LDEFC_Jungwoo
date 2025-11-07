# DSHCF 추적기 개선 사항 가이드

## 개요
이 문서는 DSHCF 추적기에 추가된 개선 사항들을 설명합니다. 주요 목표는 **가림(occlusion) 문제**와 **유사 주파수 간섭(distractor) 문제**를 해결하는 것입니다.

## 주요 개선 사항

### 1. 신뢰도 기반 업데이트 스케줄링

**문제**: 가림이나 배경 간섭 시 모델이 잘못 업데이트되어 추적이 실패함

**해결책**:
- **PSR (Peak-to-Sidelobe Ratio)**: 피크와 주변 응답의 비율
- **APCE (Average Peak-to-Correlation Energy)**: 피크와 평균 응답의 비율
- **엔트로피**: 응답 분포의 균일성 측정
- **Max/Mean Ratio**: 최대값과 평균값의 비율

**구현 파일**: `utils/calculate_confidence.m`

**사용법**:
```matlab
[psr, apce, entropy_val, max_mean_ratio] = calculate_confidence(response);
confidence = (psr/10.0 + apce/5.0 + max_mean_ratio/3.0) / 3.0;
```

### 2. 다중 피크 처리 및 Distractor 억제

**문제**: 유사한 객체가 있을 때 주파수 도메인에서 간섭이 발생하여 잘못된 위치를 선택

**해결책**:
- **다중 피크 탐지**: 상위 3개 피크를 찾아 평가
- **Inhibition Map**: 이전 프레임의 distractor 위치에 가우시안 억제 맵 생성
- **Distractor 히스토리**: 지속적인 distractor를 추적하여 억제

**구현 파일**:
- `utils/find_multiple_peaks.m`: 다중 피크 탐지
- `utils/build_inhibition_map.m`: 억제 맵 생성

**사용법**:
```matlab
[peaks, peak_values, num_peaks] = find_multiple_peaks(response, 3, 5);
inhibition_map = build_inhibition_map(peaks, peak_values, use_sz, 2.0);
response = response .* inhibition_map; % 억제 적용
```

### 3. 재검출(Re-detection) 파이프라인

**문제**: 가림 후 객체를 다시 찾지 못함

**해결책**:
- **Coarse-to-fine 검색**: 다운샘플된 이미지에서 시작하여 점진적으로 정밀화
- **확대된 탐색 영역**: 기존 탐색 영역의 3배까지 확대
- **신뢰도 기반 트리거**: 연속 5프레임 이상 낮은 신뢰도 시 재검출 시도

**구현 파일**: `utils/re_detection.m`

**사용법**:
```matlab
if low_confidence_frames >= 5
    [new_pos, re_conf, found] = re_detection(im, pos, target_sz, 3.0, features, params);
    if found && re_conf > confidence
        pos = new_pos;
    end
end
```

### 4. 장단기 모델 이중화

**문제**: 빠른 적응과 안정성 사이의 트레이드오프

**해결책**:
- **장기 모델 (Long-term)**: 느린 업데이트 (η=0.01), 안정적, 가림 시 사용
- **단기 모델 (Short-term)**: 빠른 업데이트 (η=0.0365), 적응적, 정상 추적 시 사용
- **신뢰도 기반 선택**: 낮은 신뢰도 시 장기 모델 사용

**구현 파일**: `implementation/training_phase/long_short_term_models.m`

**사용법**:
```matlab
[wf_selected, model_xf_selected, lr_actual] = long_short_term_models(...
    wf_long, wf_short, model_xf_long, model_xf_short, ...
    confidence, 0.3, 0.01, 0.0365);
```

### 5. 하드 네거티브 학습 (Hard Negative Mining)

**문제**: Distractor 위치에서도 높은 응답이 발생

**해결책**:
- **음성 라벨 생성**: Distractor 위치에 음성 가우시안 라벨 생성
- **ADMM 학습에 통합**: 음성 라벨과의 상관관계를 빼서 학습
- **가중치 조절**: 음성 예제의 가중치를 조절하여 효과 제어

**구현 파일**:
- `implementation/training_phase/prepare_negative_labels.m`: 음성 라벨 생성
- `implementation/training_phase/run_training.m`: 하드 네거티브 학습 통합

**사용법**:
```matlab
[yf_negative, negative_mask] = prepare_negative_labels(use_sz, distractor_peaks, peak_values);
% run_training에서 자동으로 사용됨
```

### 6. Spatial Reliability Mask 개선

**문제**: 배경 영역에서도 강한 정규화가 필요하지만, 전경 영역은 약한 정규화 필요

**해결책**:
- **응답 기반 적응형 마스크**: 응답 맵의 피크 위치를 기반으로 전경/배경 구분
- **신뢰도 기반 강도 조절**: 낮은 신뢰도 시 더 강한 정규화

**구현 파일**: `implementation/training_phase/context_mask.m` (업데이트됨)

**사용법**:
```matlab
% 기존 방법 (단순)
context_m = context_mask(pixels, target_sz);

% 개선된 방법 (적응형)
context_m = context_mask(pixels, target_sz, response_map, confidence);
```

### 7. ResNet-18 특징 통합

**문제**: 전통적인 특징(HOG, ColorNames)만으로는 복잡한 시나리오에서 한계

**해결책**:
- **Pre-trained ResNet-18**: ImageNet으로 학습된 딥러닝 특징 사용
- **Medium layer 추출**: conv4 레이어에서 의미론적 특징 추출
- **기존 특징과 융합**: HOG, ColorNames와 함께 사용

**구현 파일**: `feature/get_resnet18.m`

**사용법**: `run_DSHCF.m`에서 `use_resnet = true`로 설정

## 통합된 파일 구조

### 새로 추가된 파일
1. `utils/calculate_confidence.m` - 신뢰도 계산
2. `utils/find_multiple_peaks.m` - 다중 피크 탐지
3. `utils/build_inhibition_map.m` - 억제 맵 생성
4. `utils/re_detection.m` - 재검출 파이프라인
5. `implementation/training_phase/long_short_term_models.m` - 장단기 모델 관리
6. `implementation/training_phase/prepare_negative_labels.m` - 음성 라벨 생성
7. `feature/get_resnet18.m` - ResNet-18 특징 추출
8. `feature/RESNET18_INTEGRATION.md` - ResNet-18 사용 가이드

### 수정된 파일
1. `implementation/detection_phase/run_detection.m` - 신뢰도 계산, 다중 피크, distractor 억제 통합
2. `implementation/tracker.m` - 장단기 모델, 재검출, 신뢰도 기반 업데이트 통합
3. `implementation/training_phase/run_training.m` - 하드 네거티브 학습 추가
4. `implementation/training_phase/context_mask.m` - 적응형 spatial reliability mask
5. `run_DSHCF.m` - 새로운 파라미터 추가

## 파라미터 설정

`run_DSHCF.m`에서 다음 파라미터를 조절할 수 있습니다:

```matlab
% Enhanced tracking parameters
params.use_enhanced_tracking = true;     % 전체 개선 기능 활성화
params.confidence_threshold = 0.3;       % 낮은 신뢰도 임계값
params.re_detection_threshold = 5;       % 재검출 트리거 프레임 수
params.learning_rate_long = 0.01;        % 장기 모델 학습률
params.use_hard_negative_mining = true;   % 하드 네거티브 학습
params.use_distractor_suppression = true; % Distractor 억제
```

## 성능 최적화 팁

1. **GPU 사용**: ResNet-18 특징 추출 시 GPU 사용 권장
2. **파라미터 튜닝**: 데이터셋에 따라 `confidence_threshold` 조절
3. **기능 선택적 사용**: 필요에 따라 개별 기능 활성화/비활성화
4. **메모리 관리**: Distractor 히스토리는 최근 5프레임만 유지

## 문제 해결

### 추적이 너무 보수적임
- `confidence_threshold`를 낮춤 (예: 0.2)
- `re_detection_threshold`를 높임 (예: 10)

### Distractor 억제가 너무 강함
- `build_inhibition_map`의 `sigma_factor`를 조절
- 하드 네거티브 학습의 `negative_weight`를 낮춤

### 재검출이 너무 자주 발생
- `re_detection_threshold`를 높임
- `confidence_threshold`를 낮춤

## 참고 문헌

개선 사항들은 다음 논문들의 아이디어를 참고했습니다:
- CSR-DCF: Spatial reliability mask
- ECO: Long-short term models
- MDNet: Hard negative mining
- SiamRPN: Re-detection strategies

## 향후 개선 방향

1. **칼만 필터 통합**: 모션 예측으로 탐색 영역 최적화
2. **템플릿 풀 관리**: 다양한 포즈/조명 템플릿 유지
3. **Adaptive ADMM**: ρ 파라미터 자동 조절
4. **채널 가중치 학습**: 특징 채널별 중요도 학습

