function [wf,hf,wf_minus] = run_training(xf, xcf, xf_p, wf_p, wf_p_1, use_sz, params, yf, small_filter_sz, s, model_s)


    % feature size
    sz = size(xf);
    T = prod(use_sz);
    
    Sxy = bsxfun(@times, xf, conj(yf));
    Sxx = bsxfun(@times, xf, conj(xf));
    x_fuse = xf + xf_p;
    Sxx_fuse = bsxfun(@times, x_fuse, conj(x_fuse));
    wf_minus = 2 * wf_p - wf_p_1;

    % initialize h
    hf = single(zeros(sz));

    % initialize Lagrangian multiplier
    zetaf = single(zeros(sz));

    % penalty
    mu = params.mu;
    beta = params.beta;
    mu_max = 10000;
    lambda = params.t_lambda;
    gamma = params.gamma;


    i = 1;
    % ADMM iterations
    while (i <= params.admm_iterations)
        % filter에 대한 보조변수
        wf = (Sxy + gamma * (Sxx_fuse .* wf_minus) + mu * hf - zetaf) ./ (Sxx + gamma * Sxx_fuse + mu);

        % Solve for H (filter)
        h = (ifft2(mu * wf + zetaf)) ./ (lambda/T * s.^2 + mu);
%         hf = fft2(bsxfun(@rdivide, ifft2((mu*g_f) + zetaf), (T/((mu*T)+ params.admm_lambda))));
        [sx, sy, h] = get_subwindow_no_window(h, floor(use_sz/2), small_filter_sz);
        t = single(zeros(use_sz(1), use_sz(2), size(h, 3)));
        t(sx, sy, :) = h;
        hf = fft2(t);

        % Update L
        zetaf = zetaf + (mu * (wf - hf));

        % Update mu - betha = 10
        mu = min(beta * mu, mu_max);
        i = i + 1;
    end

end
