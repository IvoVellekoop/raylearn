% Experiment Adaptive Optics on tube
% Tested Zernike modes are astigmatism (aligned with tube) and defocus

%% === Settings ===
% Flags
office_mode = 0;
do_save = 1;
do_plot_final = 1;
do_plot_scan = 0;
do_save_plot_scan = 0;
force_reset = 0;

% Other settings
p.samplename = 'tube-500nL-bottom';
p.sampleid = 'DCOX-2024-8-A';

p.slm_rotation_deg = 3.4;                   % Can be found with a calibration
p.scale_slm_x = 0.9827;                     % Scale SLM 'canvas' by this amount in x
p.scale_slm_y = 0.9557;                     % Scale SLM 'canvas' by this amount in y
p.truncate = false;                         % Truncate Zernike modes at circle edge
p.pattern_patch_id = 1;
p.feedback_func = @(frames)(var(frames, [], [1 2 3]));
p.upscale_factor = 40;                      % How much the final scan should be bicubically upscaled

% Image acquisition
p.zstep_um = 1.5;                           % Size of a z-step for the volume acquisition
p.num_zslices = 10;                         % Number of z-slices per volume acquisition
p.z_backlash_distance_um = -10;             % Backlash distance piezo (must be negative!)
assert(p.z_backlash_distance_um < 0, 'Z backlash distance must be negative.')

% GIF animation settings
filename_gif = "tube_adaptive_optics.gif";  % Specify the output file name of gif animation
delaytime_gif = 0.05;

% Define test range mode 1
p.min_amp_mode1 = 18;                       % Min radians (0 to edge max)
p.max_amp_mode1 = 30;                       % Max radians (0 to edge max)
p.num_patterns_mode1 = 13;                  % Number of amplitudes to test
p.phase_range_mode1 = linspace(p.min_amp_mode1, p.max_amp_mode1, p.num_patterns_mode1);

% Define test range mode 2
p.min_amp_mode2 = -7;                       % Min radians (0 to edge max)
p.max_amp_mode2 = 7;                        % Max radians (0 to edge max)
p.num_patterns_mode2 = 10;                  % Number of amplitudes to test
p.phase_range_mode2 = linspace(p.min_amp_mode2, p.max_amp_mode2, p.num_patterns_mode2);

%% === Initialize fake/real hardware ===
if office_mode
    addpath(fileparts(fileparts(mfilename('fullpath'))))   % Grand parent dir
    dirconfig_raylearn

    % Fake hardware handles
    slm = struct('getSize', [1152 1920]);
    hSI = struct();
    hSI.hMotors = struct();
    hSI.hMotors.motorPosition = [0 0 100];
    hSICtl = struct();
    grabSIFrame = @(hSI, hSICtl)(rand(128));

    p.system_aberration_pattern = 0;

    % Fake aberrations
    fake_amplitude_mode1 = 22;
    fake_amplitude_mode2 = 2;
else
    if force_reset || ~exist('slm', 'var')
        close all; clear; clc
        setup_raylearn_exptpm
    end
    p.calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
    offset_center_slm = p.calibrationdata.sopt.offset_center_slm;

    p.system_aberration_data = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\28-Sep-2023-system-aberration\tube_ao_739157.585029_system-aberration\tube_ao_739157.585029_system-aberration_optimal_pattern.mat");
    p.system_aberration_pattern = p.system_aberration_data.slm_pattern_gv_optimal;

    % Random pattern on SLM
    slm.setData(p.pattern_patch_id, 255*rand(300));
    slm.update

    abort_if_required(hSI, hSICtl)
    hSI.hStackManager.numSlices = 1;        % One slice per grab
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
slm_size = slm.getSize;
coord_x = linspace(-1, 1, slm_size(1));
coord_y = coord_x';

% Generate aligned astigmatism and z
p.Zcart_mode1_unscaled = imrotate(zernfun_cart(coord_x, coord_y, 2, 2, p.truncate), p.slm_rotation_deg, "bilinear", "crop");
p.Zcart_mode2_unscaled = imrotate(zernfun_cart(coord_x, coord_y, 4, 4, p.truncate), p.slm_rotation_deg, "bilinear", "crop");

% Prepare scaled SLM pattern (slm pattern must be square!)
x_slmpattern = linspace(-1, 1, slm_size(1));
y_slmpattern = x_slmpattern';
xq = x_slmpattern * p.scale_slm_x;
yq = y_slmpattern * p.scale_slm_y;

extrapval = 0;
p.Zcart_mode1 = interp2(x_slmpattern, y_slmpattern, p.Zcart_mode1_unscaled, xq, yq, 'bilinear', extrapval);
p.Zcart_mode2 = interp2(x_slmpattern, y_slmpattern, p.Zcart_mode2_unscaled, xq, yq, 'bilinear', extrapval);



% Initialize feedback
all_feedback = zeros(p.num_patterns_mode1, p.num_patterns_mode2);

if do_plot_scan
    fig_scan = figure;
    fig_resize(800, 1);
    movegui('center')
    colormap magma
end

if ~office_mode
    % Random pattern to get dark frame
    slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);
    slm.setData(p.pattern_patch_id, 255*rand(300));
    slm.update
    frames_dark = grabSIFrame(hSI, hSICtl);
    frames_dark_mean = mean(frames_dark(:));
else
    frames_dark_mean = zeros(slm_size(1));
end

p.piezo_center_um = hSI.hMotors.motorPosition;
p.piezo_start_um = p.piezo_center_um(3) - p.zstep_um * p.num_zslices/2;
p.piezo_range_um = p.piezo_start_um + (0:p.num_zslices-1) * p.zstep_um;

frames_flatslm = [];
frames_ao = [];

%% === Scan Zernike modes ===
starttime = now;
count = 1;
for i_mode2 = 1:p.num_patterns_mode2                  % Scan mode 2
    phase_amp_mode2 = p.phase_range_mode2(i_mode2);
    for i_mode1 = 1:p.num_patterns_mode1              % Scan mode 1
        phase_amp_mode1 = p.phase_range_mode1(i_mode1);

        slm_pattern_2pi = phase_amp_mode1.*p.Zcart_mode1 + phase_amp_mode2.*p.Zcart_mode2;
        slm_pattern_gv = (128/pi) * slm_pattern_2pi + p.system_aberration_pattern;


        if office_mode
            secret_pattern = fake_amplitude_mode1 .* p.Zcart_mode1 + fake_amplitude_mode2 .* p.Zcart_mode2;
            fake_wavefront = exp(1i .* (secret_pattern - slm_pattern_2pi)) ...
                .* ((coord_x.^2) + (coord_y.^2) < 1);
            fake_wavefront_flatslm = exp(1i .* (secret_pattern)) ...
                .* ((coord_x.^2) + (coord_y.^2) < 1);
            frames_ao = abs(fftshift(fft2(fake_wavefront))).^2;
            frames_flatslm = abs(fftshift(fft2(fake_wavefront_flatslm))).^2;

        else
            slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);

            % === Flat pattern ===
            slm.setData(p.pattern_patch_id, 0);
            slm.update;

            % To start position
            hSI.hMotors.motorPosition = [0 0 p.piezo_start_um + p.z_backlash_distance_um];
            current_piezo_z = p.piezo_start_um;
            hSI.hMotors.motorPosition = [0 0 current_piezo_z];

            % Volume acquisition
            for iz = 1:length(p.num_zslices)
                frames_flatslm(:, :, iz) = grabSIFrame(hSI, hSICtl);
                current_piezo_z = current_piezo_z + p.zstep_um;
                hSI.hMotors.motorPosition = [0 0 current_piezo_z];
            end
            
            % === Zernike Modes pattern ===
            slm.setData(p.pattern_patch_id, slm_pattern_gv);
            slm.update;
            
            % To start position
            hSI.hMotors.motorPosition = [0 0 p.piezo_start_um + p.z_backlash_distance_um];
            current_piezo_z = p.piezo_start_um;
            hSI.hMotors.motorPosition = [0 0 current_piezo_z];

            % Volume acquisition
            for iz = 1:p.num_zslices
                frames_ao(:, :, iz) = grabSIFrame(hSI, hSICtl);
                hSI.hMotors.motorPosition = [0 0 p.piezo_range_um(iz)];
            end

            % Random pattern prevents bleaching
            slm.setData(p.pattern_patch_id, 255*rand(300));
            slm.update
        end

        feedback = p.feedback_func(frames_ao - frames_dark_mean) ./ p.feedback_func(frames_flatslm - frames_dark_mean);
        all_feedback(i_mode1, i_mode2) = feedback;

        if do_plot_scan
            figure(fig_scan)
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
    
                % Plot aberrated image
                subplot(2,2,3)
                if office_mode
                    imagesc(max(frames_ao(end/2-20:end/2+20, end/2-20:end/2+20), [], 3))
                    title('Aberrated image (zoomed)')
                else
                    imagesc(max(frames_ao, [], 3))
                    title('Aberrated image')
                end
                xlabel('x (pix)')
                ylabel('y (pix)')
                axis image
                colorbar
            end

            % Feedback signal
            subplot(2,2,4)
            imagesc(p.phase_range_mode2, p.phase_range_mode1, log(all_feedback))
            title('Log of feedback signals')
            xlabel('Mode 2 phase amplitude')
            ylabel('Mode 1 phase amplitude')
            
            if office_mode
                hold on
                plot(fake_amplitude_mode2, fake_amplitude_mode1, '+r')
                legend('Secret aberration')
                hold off
            end
            colorbar

            drawnow
            
            % Save gif animation
            if do_save_plot_scan
                im = frame2im(getframe(gcf));
                [A,map] = rgb2ind(im,256);
                if i_mode2 == 1 && i_mode1 == 1
                    imwrite(A,map,filename_gif,"gif","LoopCount",Inf,"DelayTime",delaytime_gif);
                else
                    imwrite(A,map,filename_gif,"gif","WriteMode","append","DelayTime",delaytime_gif);
                end
            end
        end
    
        if do_save
            % Save intermediate frames
            savepath = fullfile(p.savedir, sprintf('%s_%03i_%03i.mat', p.savename, i_mode2, i_mode1));
            disp('Saving...')
            save(savepath, '-v7.3', 'p', 'frames_flatslm', 'frames_ao', 'i_mode2', 'i_mode1', ...
                'slm_pattern_2pi', 'slm_pattern_gv')
        end

        eta(count, p.num_patterns_mode2.*p.num_patterns_mode1,  starttime, 'cmd', 'Scanning Zernike modes', 0);
        count = count+1;
    end
end

if ~office_mode
    % Random pattern prevents bleaching
    slm.setData(p.pattern_patch_id, 255*rand(300));
    slm.update
end

%% Plot final
% Upscale image and find coordinates of maximum
all_feedback_upscaled = imresize(all_feedback, p.upscale_factor);
[~, imax] = max(all_feedback_upscaled(:));
[peak_index_mode1_upscaled, peak_index_mode2_upscaled] = ind2sub(size(all_feedback_upscaled), imax);
peak_index_mode1 = downscale(peak_index_mode1_upscaled, p.upscale_factor);
peak_index_mode2 = downscale(peak_index_mode2_upscaled, p.upscale_factor);

% Compute optimal coordinates
phase_amp_mode1_optimal = interp1(1:p.num_patterns_mode1, p.phase_range_mode1, peak_index_mode1);
phase_amp_mode2_optimal = interp1(1:p.num_patterns_mode2, p.phase_range_mode2, peak_index_mode2);

if isnan(phase_amp_mode2_optimal) || isnan(phase_amp_mode1_optimal)
    warning('Could not find phase pattern. NaN phase amplitude.')
end

% Compute optimal pattern
slm_pattern_2pi_optimal = phase_amp_mode1_optimal.*p.Zcart_mode1 + phase_amp_mode2_optimal.*p.Zcart_mode2;
slm_pattern_gv_optimal = (128/pi) * slm_pattern_2pi_optimal;

if do_plot_final
    fig_final = figure;
    fig_resize(800, 1);
    movegui('center')
    colormap magma

    % Plot parameter landscape
    subplot(2,2,1)
    imagesc(p.phase_range_mode2, p.phase_range_mode1, all_feedback); hold on
    plot(phase_amp_mode2_optimal, phase_amp_mode1_optimal, '+r'); hold off
    title(sprintf('Zernike mode\namplitude landscape'))
    xlabel('Mode 2 phase amplitude')
    ylabel('Mode 1 phase amplitude')
    colorbar

    % Plot upscaled parameter landscape
    subplot(2,2,2)
    imagesc(all_feedback_upscaled); hold on
    plot(peak_index_mode2_upscaled, peak_index_mode1_upscaled, '+r'); hold off
    title(sprintf('Zernike mode\namplitude landscape upscaled'))
    xlabel('Index Mode 2')
    ylabel('Index Mode 1')
    colorbar

    % Plot optimal SLM pattern
    subplot(2,2,3)
    imagesc(angle(exp(1i .* slm_pattern_2pi_optimal)))
    title('Phase pattern optimized by AO (rad)')
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

%% === Save ===
if do_save
    % Save intermediate frames
    savepath = fullfile(p.savedir, sprintf('%s_optimal_pattern.mat', p.savename));
    disp('Saving...')
    save(savepath, '-v7.3', 'p', 'all_feedback', 'all_feedback_upscaled', ...
        'phase_amp_mode1_optimal', 'phase_amp_mode2_optimal', ...
        'slm_pattern_2pi_optimal', 'slm_pattern_gv_optimal')
    fprintf('\nSaved optimized pattern to:\n%s\n', savepath)
end

% Compute downscaled image coordinates
function x_down = downscale(x_up, upscale_factor)
    x_down = x_up ./ upscale_factor + 0.5*(1 - 1./upscale_factor);
end


function abort_if_required(hSI, hSICtl)
    %% Press abort button if required
    if ~strcmp(hSI.acqState, 'idle')
        disp('Scanimage not idle. Trying to abort...')
        hSICtl.abortButton
        pause(0.2)
        if ~strcmp(hSI.acqState, 'idle')
            error('Could not abort current scanimage operation')
        end
        disp('Succesfully aborted current operation.')
    end
end