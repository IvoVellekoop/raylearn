% Experiment Adaptive Optics on tube
% Tested Zernike modes are astigmatism (aligned with tube) and defocus

%% === Settings ===
% Flags
office_mode = 1;
do_save = 1;
do_plot_final = 1;
do_plot_scan = 0;
do_save_plot_scan = 0;
force_reset = 0;

% Other settings
p.samplename = 'tube500nL';

p.slm_rotation_deg = -5.9;                  % Can be found with an SLM coordinate calibration
p.truncate = false;                         % Truncate Zernike modes at circle edge
p.pattern_patch_id = 1;
p.feedback_func = @(frames)(var(frames, [], [1 2 3]));
p.upscale_factor = 10;                      % How much the final scan should be bicubically upscaled

% GIF animation settings
filename_gif = "tube_adaptive_optics.gif";  % Specify the output file name of gif animation
delaytime_gif = 0.05;

% Define test range astigmatism
p.min_cycles_2_2 = 3;                       % Minimum phase cycles (1 => 2pi phase)
p.max_cycles_2_2 = 11;                      % Max phase cycles (1 => 2pi phase)
p.num_patterns_2_2 = 8;                    % Number of patterns
p.phase_range_2_2 = pi * linspace(p.min_cycles_2_2, p.max_cycles_2_2, p.num_patterns_2_2);

% Define test range spherical aberrations
p.min_cycles_4_2 = -1;                      % Minimum phase cycles (1 => 2pi phase)
p.max_cycles_4_2 = 1;                       % Max phase cycles (1 => 2pi phase)
p.num_patterns_4_2 = 5;                     % Number of patterns
p.phase_range_4_2 = pi * linspace(p.min_cycles_4_2, p.max_cycles_4_2, p.num_patterns_4_2);

%% === Initialize fake/real hardware ===
if office_mode
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Grand parent dir
    dirconfig_raylearn

    % Fake hardware handles
    slm = struct('Height', 400);
    hSI = struct();
    hSICtl = struct();
    grabSIFrame = @(hSI, hSICtl)(rand(128));

    % Fake aberrations
    fake_amplitude_2_2 = 8.2.* pi;
    fake_amplitude_4_2 = 0.4.* pi;
else
    if force_reset || ~exist('slm', 'var')
        close all; clear; clc
        setup_raylearn_exptpm
    end
end

if do_save
    % Script path, revision and date&time
    p.script_name = mfilename('fullpath');
    [~, p.script_version] = system(['git --git-dir="' dirs.repo '/.git" rev-parse HEAD']);
    [~, p.git_diff] = system(['git --no-pager --git-dir="' dirs.repo '/.git" diff']);
    p.save_time = now;

    % Create save directory
    filenameprefix = 'tube_ao';
    p.savename = sprintf('%s_%f_%s', filenameprefix, now, p.samplename);
    p.subdir = ['/raylearn-data/TPM/adaptive-optics/' date '-' p.samplename];
    p.savedir = fullfile([dirs.localdata p.subdir], p.savename);
    try mkdir(p.savedir); catch 'MATLAB:MKDIR:DirectoryExists'; end
    fprintf('\nSave Directory: %s\n', p.savedir)
end

% Initialize coords
coord_x = linspace(-1, 1, slm.Height);
coord_y = coord_x';

% Generate aligned astigmatism and z
p.Zcart_2_2 = imrotate(zernfun_cart(coord_x, coord_y, 2, 2, p.truncate), p.slm_rotation_deg, "bilinear", "crop");
p.Zcart_4_2 = imrotate(zernfun_cart(coord_x, coord_y, 4, 2, p.truncate), p.slm_rotation_deg, "bilinear", "crop");

% Initialize feedback
all_feedback = zeros(p.num_patterns_2_2, p.num_patterns_4_2);

if do_plot_scan
    figure
    fig_resize(800, 1);
    movegui('center')
    colormap magma
end


%% === Scan Zernike modes ===
starttime = now;
count = 1;
for i_4_2 = 1:p.num_patterns_4_2                  % Scan spherical aberrations
    phase_amp_4_2 = p.phase_range_4_2(i_4_2);
    for i_2_2 = 1:p.num_patterns_2_2              % Scan astigmatism
        phase_amp_2_2 = p.phase_range_2_2(i_2_2);

        slm_pattern_2pi = phase_amp_2_2.*p.Zcart_2_2 + phase_amp_4_2.*p.Zcart_4_2;
        slm_pattern_gv = (128/pi) * slm_pattern_2pi;

        if office_mode
            secret_pattern = fake_amplitude_2_2 .* p.Zcart_2_2 + fake_amplitude_4_2 .* p.Zcart_4_2;
            fake_wavefront = exp(1i .* (secret_pattern - slm_pattern_2pi)) ...
                .* ((coord_x.^2) + (coord_y.^2) < 1);
            frames = abs(fftshift(fft2(fake_wavefront))).^2;

        else
            slm.setData(p.pattern_patch_id, slm_pattern_gv);
            slm.update;
    
            frames = grabSIFrame(hSI, hSICtl);  % Find a way to get multiple slices
        end

        feedback = p.feedback_func(frames);
        all_feedback(i_2_2, i_4_2) = feedback;

        if do_plot_scan
            % Plot pattern on SLM
            subplot(2,2,1)
            imagesc(angle(exp(1i .* slm_pattern_2pi)))
            title('SLM pattern')
            xticklabels([])
            yticklabels([])
            colorbar
            axis image

            if office_mode
                % Plot secret pattern (office mode only)
                subplot(2,2,2)
                imagesc(angle(exp(1i .* secret_pattern)))
                title('Secret aberration pattern')
                xticklabels([])
                yticklabels([])
                axis image
                colorbar
            end
    
            % Plot aberrated image
            subplot(2,2,3)
            imagesc(max(frames(end/2-20:end/2+20, end/2-20:end/2+20), [], 3))
            title('Aberrated image (zoomed)')
            xlabel('x (pix)')
            ylabel('y (pix)')
            axis image
            colorbar

            % Feedback signal
            subplot(2,2,4)
            imagesc(p.phase_range_4_2, p.phase_range_2_2, log(all_feedback))
            title('Log of feedback signals')
            xlabel('Spherical Aberration phase amplitude')
            ylabel('Aligned astigmatism phase amplitude')
            hold on
            plot(fake_amplitude_4_2, fake_amplitude_2_2, '+r')
            legend('Secret aberration')
            hold off
            colorbar

            drawnow
            
            % Save gif animation
            if do_save_plot_scan
                im = frame2im(getframe(gcf));
                [A,map] = rgb2ind(im,256);
                if i_4_2 == 1 && i_2_2 == 1
                    imwrite(A,map,filename_gif,"gif","LoopCount",Inf,"DelayTime",delaytime_gif);
                else
                    imwrite(A,map,filename_gif,"gif","WriteMode","append","DelayTime",delaytime_gif);
                end
            end
        end
    
        if do_save
            % Save intermediate frames
            savepath = fullfile(p.savedir, sprintf('%s_%03i_%03i.mat', p.savename, i_4_2, i_2_2));
            disp('Saving...')
            save(savepath, '-v7.3', 'p', 'frames', 'i_4_2', 'i_2_2', ...
                'slm_pattern_2pi', 'slm_pattern_gv')
        end

        eta(count, p.num_patterns_4_2.*p.num_patterns_2_2,  starttime, 'cmd', 'Scanning Zernike modes', 0);
        count = count+1;
    end
end

% Upscale image and find coordinates of maximum
all_feedback_upscaled = imresize(all_feedback, p.upscale_factor);
[~, imax] = max(all_feedback_upscaled(:));
[peak_index_2_2_upscaled, peak_index_4_2_upscaled] = ind2sub(size(all_feedback_upscaled), imax);
peak_index_2_2 = downscale(peak_index_2_2_upscaled, p.upscale_factor);
peak_index_4_2 = downscale(peak_index_4_2_upscaled, p.upscale_factor);

% Compute optimal coordinates
phase_amp_2_2_optimal = interp1(1:p.num_patterns_2_2, p.phase_range_2_2, peak_index_2_2);
phase_amp_4_2_optimal = interp1(1:p.num_patterns_4_2, p.phase_range_4_2, peak_index_4_2);

% Compute optimal pattern
slm_pattern_2pi_optimal = phase_amp_2_2_optimal.*p.Zcart_2_2 + phase_amp_4_2_optimal.*p.Zcart_4_2;
slm_pattern_gv_optimal = (128/pi) * slm_pattern_2pi_optimal;

if do_plot_final
    figure;
    fig_resize(800, 1);
    movegui('center')
    colormap magma

    % Plot parameter landscape
    subplot(2,2,1)
    imagesc(p.phase_range_4_2, p.phase_range_2_2, all_feedback); hold on
    plot(phase_amp_4_2_optimal, phase_amp_2_2_optimal, '+r'); hold off
    title(sprintf('Zernike mode\namplitude landscape'))
    xlabel('Spherical Aberration phase amplitude')
    ylabel('Aligned astigmatism phase amplitude')
    colorbar

    % Plot upscaled parameter landscape
    subplot(2,2,2)
    imagesc(all_feedback_upscaled); hold on
    plot(peak_index_4_2_upscaled, peak_index_2_2_upscaled, '+r'); hold off
    title(sprintf('Zernike mode\namplitude landscape upscaled'))
    xlabel('Index Spherical Aberration')
    ylabel('Index Aligned astigmatism')
    colorbar

    % Plot optimal SLM pattern
    subplot(2,2,3)
    imagesc(angle(exp(1i .* slm_pattern_2pi_optimal)))
    title('Phase pattern optimized by AO')
    xticklabels([])
    yticklabels([])
    axis image
    colorbar

    if office_mode
        % Plot secret pattern (office mode only)
        subplot(2,2,4)
        imagesc(angle(exp(1i .* secret_pattern)))
        corr_with_secret = field_corr(exp(1i.*secret_pattern) .* ((coord_x.^2) + (coord_y.^2) < 1), ...
            exp(1i.*slm_pattern_2pi_optimal) .* ((coord_x.^2) + (coord_y.^2) < 1));
        title(sprintf('Secret aberration pattern\nField correlation = %.3f', abs(corr_with_secret)))
        xticklabels([])
        yticklabels([])
        axis image
        colorbar
    end
end


if do_save
    % Save intermediate frames
    savepath = fullfile(p.savedir, sprintf('%s_optimal_pattern.mat', p.savename));
    disp('Saving...')
    save(savepath, '-v7.3', 'p', 'all_feedback', 'all_feedback_upscaled', ...
        'phase_amp_2_2_optimal', 'phase_amp_4_2_optimal', ...
        'slm_pattern_2pi_optimal', 'slm_pattern_gv_optimal')
end

% Compute downscaled image coordinates
function x_down = downscale(x_up, upscale_factor)
    x_down = x_up ./ upscale_factor + 0.5*(1 - 1./upscale_factor);
end
