%   This script runs the implementation of EMCF, which is borrowed from
%   BACF.

clear; 
clc;
close all;
setup_paths;

%   Load video information
% base_path  = 'F:/tracking/datasets/UAVTrack112';

base_path = 'C:/Users/user/LDECF-master-edit/seq';
video      = 'truck2';
video_path = [base_path '/' video];
[seq, ground_truth] = load_video_info(video,base_path,video_path);
seq.path = video_path;
seq.name = video;
seq.startFrame = 1;
seq.endFrame = seq.len;

% multiple object initial value -> 복잡해지면 파일 하나 만들어서 모듈처럼 가져와서 써도 됨. 이후,
init_rects = [
     1152, 503, 45, 17;  % truck2 오른쪽 트럭 -> 정확하게 tracking
    % 670,530,30,20   % truck2 가운데 하얀 승용차 -> 건물에 가려졌을 때 건물과 차의 색상(feature)이 비슷하여 tracking을 놓침
    % 485,250,35,70 % person12 사람 -> 처마 밑에서 나오면서 옷 무늬와 그림자의 feature가 비슷하여 tracking을 놓침
    % 772.68, 455.43, 41.871, 127.61 % MOT16-01 -> 사람이 겹치면 tracking 실패
    % 571.03,402.13,104.56,315.68 % MOT16-02 -> 빨간색 옷을 tracking하다가 보라색 옷으로 변경됨. 보라색 옷은 occulsion이 발생해도 정상적으로 tracking 성공
];

% number of objects
num_objects = size(init_rects, 1);
results_all = cell(num_objects, 1);

% 실제 object의 location
gt_boxes = [ground_truth(:,1:2), ground_truth(:,1:2) + ground_truth(:,3:4) - ones(size(ground_truth,1), 2)]; 

% 각 object의 location prediction
for i = 1:num_objects
    % 각 객체마다 새로운 시퀀스 구조 생성
    seq_i = seq;
    seq_i.init_rect = init_rects(i, :);

    % 추적 실행
    results = run_DSHCF(seq_i);
    results_all{i} = results;

end


% Run DSHCF
%(modified) results = run_DSHCF(seq);

%   compute the OP
pd_boxes = results.res;
pd_boxes = [pd_boxes(:,1:2), pd_boxes(:,1:2) + pd_boxes(:,3:4) - ones(size(pd_boxes,1), 2)  ];
OP = zeros(size(gt_boxes,1),1);  % overlap rate
CE = zeros(size(gt_boxes,1),1);  % ceter location error
for i=1:size(gt_boxes,1)
    b_gt = gt_boxes(i,:);
    b_pd = pd_boxes(i,:);
    OP(i) = computePascalScore(b_gt,b_pd);
    centerGT = [b_gt(1) + (b_gt(3) - 1)/2, b_gt(2) + (b_gt(4) - 1)/2];
    centerPD = [b_pd(1) + (b_pd(3) - 1)/2, b_pd(2) + (b_pd(4) - 1)/2];
    CE(i) = sqrt((centerPD(1) - centerGT(1))^2 + (centerPD(2) - centerGT(2))^2);
end
OP_vid = sum(OP >= 0.5) / numel(OP);
CE_vid = sum(CE <= 20) / numel(CE);
FPS_vid = results.fps;
display([video  '---->' '   FPS:   ' num2str(FPS_vid)   '    op:   '   num2str(OP_vid)  '    ce:   '   num2str(CE_vid)]);