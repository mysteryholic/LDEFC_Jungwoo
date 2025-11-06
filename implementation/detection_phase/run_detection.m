% function [xtf,xtcf,pos,translation_vec,response,disp_row,disp_col,forwardR] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame,...
%     xw,wf_minus,delta_f,s,delta_s,yf,xtf_set)
function [xtf,xtcf,pos,translation_vec,response,disp_row,disp_col,forwardR] = run_detection(im,pos,sz,target_sz,currentScaleFactor,features,cos_window,g_f,global_feat_params,use_sz,ky,kx,newton_iterations,featureRatio,Vy,Vx,frame,...
    delta_f)

    center=pos+[Vy Vx];
%    center=pos;
    pixel_template=get_pixels(im, center, round(sz*currentScaleFactor), sz);

%     savedir='D:\ML\MA\experiment\\DSHCF\Ablation\non-denoising\features_pixel\uav3\';
%     if ~exist(savedir, 'dir')
%         mkdir(savedir)
%     end
%     f = figure(10);
%     imshow(pixel_template);
%     saveas(f, [savedir,num2str(frame),'.png']);

%         figure(10);
%         imshow(pixel_template);
% %         
%              figure(11);
%              imshow(m);
%     % 应用双边滤波进行去噪
%     ksize = 1;  % 定义滤波核大小
%     sigmac = 2;  % 定义空间域的标准差
%     sigmas = 25;  % 定义值域的标准差
%     
%     % 调用双边滤波函数
%     denoised_template = bifilter(ksize, sigmac, sigmas, pixel_template);
%     denoised_template = imgaussfilt(pixel_template, 1);
%     denoised_template = fastBilateralFilter(pixel_template, sigmac, sigmas, 4);

%     xt_noised = get_features(noisyImage, features, global_feat_params);
%     xtf_noised = fft2(bsxfun(@times,xt_noised,cos_window));

    
        % 设置去噪参数
        edist = sqrt(sum(pixel_template,3));
        patchSigma = sqrt(var(edist(:)));
        degreeOfSmoothing = 0.05 * patchSigma; % 根据图像的噪声水平调整
        searchWindowSize = 11;  % 搜索窗口大小
        comparisonWindowSize = 5;  % 比较窗口大小
        
        % 应用非局部均值去噪
        denoised_img = imnlmfilt(pixel_template, 'DegreeOfSmoothing', degreeOfSmoothing, ...
                                            'SearchWindowSize', searchWindowSize, ...
                                            'ComparisonWindowSize', comparisonWindowSize);

% %     savedir='D:\ML\MA\experiment\DSHCF\Ablation\denoising\features_pixel\uav3\';
% %     if ~exist(savedir, 'dir')
% %         mkdir(savedir)
% %     end
% %     f = figure(11);
% %     imshow(denoised_img);
% %     saveas(f, [savedir,num2str(frame),'.png']);
%     以下代码继续使用去噪后的图像
    m = context_mask(denoised_img, round(target_sz/currentScaleFactor));
    xt = get_features(denoised_img, features, global_feat_params);

%     m = context_mask(noisyImage,round(target_sz/currentScaleFactor));
%     xt = get_features(noisyImage,features,global_feat_params);
    inverse_m = mexResize(m,[size(xt,1) size(xt,2)],'auto');
    
%      figure(11);
%      imshow(m);
     
    xtc = xt .* inverse_m;
    xtf = fft2(bsxfun(@times,xt,cos_window));    
    xtcf = fft2(bsxfun(@times,xtc,cos_window));
    
    forwardR = xtf .* delta_f;

    
%     delta_f_R = s .* g_f;
%     savedir='D:\ML\MA\experiment\DSHCF\Ablation\s\Sheep1\';
%     if ~exist(savedir, 'dir')
%         mkdir(savedir)
%     end
%     if frame==32
%         set(gcf,'visible','on'); 
% %         delta_f_R_real=ifft2(delta_f_R,'symmetric');
% % %         delta_f_R_real_t=fftshift(delta_f_R_real);
% %         delta_f_R_real_show=sum(delta_f_R_real,3);
%         colormap("jet");
%         surf(delta_s);
%         shading interp;
%         axis ij;
%         axis off;
%         view([137,50]);
%         saveas(gcf,[savedir,num2str(frame),'.png']);
%         savefig(gcf,[savedir,num2str(frame),'.fig']);
%     end  


%     numFrames = 5;
%     framesToFeature = zeros(1, numFrames);
%     startTime = 55;  % 开始时间（秒）
%     for t = 1:numFrames
%         framesToFeature(t) = startTime + t - 1;
%     end
% 
%     savedir='D:\ML\MA\experiment\DSHCF\Ablation\noised\features\uav3\';
%     if ~exist(savedir, 'dir')
%         mkdir(savedir)
%     end
%     if ismember(frame, framesToFeature)
%         set(gcf,'visible','on'); 
%         xt_f=ifft2(xtf,'symmetric');
%         Xt=sum(xt_f,3);
%         colormap(jet);
%         surf(Xt);
% %         shading interp;
%         axis ij;
%         axis off;
%         view([137,50]);
%         saveas(gcf,[savedir,num2str(frame),'.png']);
%         savefig(gcf,[savedir,num2str(frame),'.fig']);
%     end

%             savedir='D:\ML\MA\experiment\DSHCF\Ablation\denoised\features_channel\uav3_57th\';
%             if ~exist(savedir, 'dir')
%                 mkdir(savedir)
%             end
%             if frame==57
%                 for i=1:42
%                     set(gcf,'visible','on'); 
%                     Q = surf(ifft2(xtf(:,:,i),'symmetric'));
%                     colormap(parula);
%                     axis ij;
%                     axis off;
%                     view([0,90]);
%                     set(Q,'edgecolor','none');
% %                     shading interp
%                     saveas(gcf,[savedir,num2str(i),'.png']);
% %                     savefig(gcf,[savedir,num2str(i),'.fig']);
%                 end
%             end

%             if frame==32
%                 for i=1:42
%             set(gcf,'visible','off'); 
%             colormap(parula);
%             Q=surf(ifft(g_f(:,:,i),'symmetric'));
%             axis ij;
%             axis off;
%             view([0,90]);
%             set(Q,'edgecolor','none');
%             shading interp
%             saveas(gcf,[savedir,num2str(i),'.png']);
%                 end
%             end
    responsef = permute(sum(bsxfun(@times, conj(g_f), xtf), 3), [1 2 4 3]);
    % if we undersampled features, we want to interpolate the
    % response so it has the same size as the image patch

    responsef_padded = resizeDFT2(responsef, use_sz);
    % response in the spatial domain
    response = ifft2(responsef_padded, 'symmetric');
%         figure(15),surf(fftshift(response));

%     disp(class(fftshift(response)));
%     disp(size(fftshift(response)));
%     disp(isnumeric(fftshift(response)));

% %     get detection response
%     numFrames = 5;
%     framesToFeature = zeros(1, numFrames);
%     startTime = 55;  % 开始时间（秒）
%     for t = 1:numFrames
%         framesToFeature(t) = startTime + t - 1;
%     end
% 
%     savedir='D:\ML\MA\experiment\DSHCF\Ablation\noised\resposes\uav3\';
%         if ~exist(savedir, 'dir')
%             mkdir(savedir)
%         end
%     if ismember(frame, framesToFeature)
%         set(gcf,'visible','on'); 
%         Xt_r=fftshift(response);
%         colormap(jet);
% %         [Gx, Gy] =  gradient(Xt_r);
% %         G = sqrt(Gx.^2 + Gy.^2);  % 梯度的大小
% %         % 归一化梯度
% %         maxG = max(G(:));
% %         normalizedG = G / maxG;
% %         
% %         r_surf = surf(Xt_r);
% %         r_surf.AlphaData = normalizedG .^ 2; 
% %         r_surf.FaceAlpha = 'flat';
% %         shading interp;
% %         Xt_s=sum(Xt_r,3); % 非必要
%         surf(Xt_r);
% %         shading interp;
%         axis ij;
%         axis off;
%         view([-46.576,54.4418]);
%         saveas(gcf, [savedir,num2str(frame),'.png']);
%         savefig(gcf, [savedir,num2str(frame),'.fig']);
%     end

    % find maximum peak
    [disp_row, disp_col] = resp_newton(response, responsef_padded,newton_iterations, ky, kx, use_sz);
    % calculate translation
    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
    %update position
    pos = center + translation_vec;
end

