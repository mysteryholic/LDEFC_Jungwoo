function out = bifilter(ksize, sigmac, sigmas, in)
    dims = size(in);
    h = dims(1);
    w = dims(2);
    if length(dims) < 3
        d = 1;
    else
        d = dims(3);
    end

    padSize = floor(ksize/2);

    % 조건문 수정 (any 추가)
    if any(h < ksize) || any(w < ksize)
        warning('입력 이미지가 너무 작아서 필터를 적용할 수 없습니다.');
        out = in;
        return;
    end


    
    [X, Y] = meshgrid(-padSize:padSize, -padSize:padSize);
    dist = X.^2 + Y.^2;
    Gc = exp(-dist/(2*sigmac^2));

    out = zeros(size(in), 'like', in);
    padim = padarray(in, [padSize padSize], 'replicate', 'both');

    for i = 1:h
        for j = 1:w
            localRegion = double(padim(i:i+ksize-1, j:j+ksize-1, :));
            centerPixel = localRegion(padSize+1, padSize+1, :);

            for k = 1:d
                tempDiff = localRegion(:,:,k) - centerPixel(:,:,k);
                Gs = exp(-tempDiff.^2 / (2 * sigmas^2));
                Wp = sum(sum(Gc .* Gs));
                out(i,j,k) = sum(sum(Gc .* Gs .* localRegion(:,:,k))) / Wp;
            end
        end
    end

    out = uint8(out);
end
