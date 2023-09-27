% Test rotation influence on tube correction

%% === Settings ===
% Flags
office_mode = 0;
do_save = 1;
do_plot_final = 1;
do_plot_scan = 0;
force_reset = 0;

% Scan settings
p.num_scales = 15;
p.num_angles = 11;
p.scale_range = linspace(0.85, 1.00, p.num_scales);
p.angle_range_deg = linspace(1.0, 5.00, p.num_angles);

% Other settings
p.samplename = 'tube500nL-bottom-rotatescalescan';
p.sampleid = 'DCOX-2023-8-C';

p.truncate = false;                         % Truncate Zernike modes at circle edge
p.pattern_patch_id = 1;
p.feedback_func = @contrast_enhancement;

% Image acquisition
p.zstep_um = 1.00;                          % Size of a z-step for the volume acquisition
p.num_zslices = 15;                         % Number of z-slices per volume acquisition
p.z_backlash_distance_um = -10;             % Backlash distance piezo (must be negative!)
assert(p.z_backlash_distance_um < 0, 'Z backlash distance must be negative.')

% Define test range astigmatism
p.upscale_factor = 40;

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
else
    if force_reset || ~exist('slm', 'var')
        close all; clear; clc
        setup_raylearn_exptpm
    end
    calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
    offset_center_slm = calibrationdata.sopt.offset_center_slm;

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
    filenameprefix = 'tube_angle_scan';
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

% Fetch SLM pattern data
disp('Loading pattern...')
patterndata = load(replace("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat", '\', filesep));

% Coords for rescaling pattern
Nslmpattern = size(patterndata.phase_SLM, 2);
x_slmpattern = linspace(-1, 1, Nslmpattern);
y_slmpattern = x_slmpattern';

% Initialize feedback
all_feedback = zeros(1, p.num_scales);

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
end

frames_dark = grabSIFrame(hSI, hSICtl);
frames_dark_mean = mean(frames_dark(:));
frames_dark_var = var(frames_dark(:));

p.piezo_center_um = hSI.hMotors.motorPosition;
p.piezo_start_um = p.piezo_center_um(3) - p.zstep_um * p.num_zslices/2;
p.piezo_range_um = p.piezo_start_um + (0:p.num_zslices-1) * p.zstep_um;

frames_flatslm = [];
frames_ao = [];

%% === Scan scaling ===
starttime = now;
count = 1;
for i_angle = 1:p.num_angles                    % Scan angle

    angle_slm_deg = p.angle_range_deg(i_angle);
    phase_SLM_rot_gv = 255 / (2*pi) * imrotate(patterndata.phase_SLM, angle_slm_deg, "bilinear", "crop");

    for i_scale = 1:p.num_scales                % Scan scale
    
        % Prepare scaled SLM pattern
        scale_slm = p.scale_range(i_scale);
        xq = x_slmpattern * scale_slm;
        yq = xq';
        
        extrapval = 0;
        slm_pattern_gv = interp2(x_slmpattern, y_slmpattern, phase_SLM_rot_gv, xq, yq, 'bilinear', extrapval);
    
        % Measure
        if office_mode
            secret_pattern_gv = patterndata.phase_SLM .* (128/pi);
            fake_wavefront = exp(1i .* (pi/128) .* (secret_pattern_gv - slm_pattern_gv)) ...
                .* ((coord_x.^2) + (coord_y.^2) < 1);
            fake_wavefront_flatslm = exp(1i .* (secret_pattern_gv) .* (pi/128)) ...
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
            pause(0.05)
            hSI.hMotors.motorPosition = [0 0 p.piezo_start_um];
    
            % Volume acquisition
            for iz = 1:p.num_zslices
                frames_flatslm(:, :, iz) = grabSIFrame(hSI, hSICtl);
                hSI.hMotors.motorPosition = [0 0 p.piezo_range_um(iz)];
            end
            
            % === SLM test pattern ===
            slm.setData(p.pattern_patch_id, slm_pattern_gv);
            slm.update;
            
            % To start position
            hSI.hMotors.motorPosition = [0 0 p.piezo_start_um + p.z_backlash_distance_um];
            pause(0.05)
            hSI.hMotors.motorPosition = [0 0 p.piezo_start_um];
    
            % Volume acquisition
            for iz = 1:p.num_zslices
                frames_ao(:, :, iz) = grabSIFrame(hSI, hSICtl);
                hSI.hMotors.motorPosition = [0 0 p.piezo_range_um(iz)];
            end
    
            % Random pattern prevents bleaching
            slm.setData(p.pattern_patch_id, 255*rand(300));
            slm.update
        end
    
        % Compute feedback, save signal variances
        feedback = p.feedback_func(frames_ao, frames_flatslm, frames_dark);
        all_feedback(i_angle, i_scale) = feedback;
        all_signal_std_flat(i_angle, i_scale) = sqrt( var(frames_flatslm, [], [1 2 3]) - var(frames_dark, [], [1 2 3]) );
        all_signal_std_corrected(i_angle, i_scale) = sqrt( var(frames_ao, [], [1 2 3]) - var(frames_dark, [], [1 2 3]) );
    
    
        if do_plot_scan
            figure(fig_scan)
            % Plot pattern on SLM
            subplot(2,2,1)
            imagesc(angle(exp(1i .* (pi/128) .* slm_pattern_gv)))
            title('SLM pattern')
            xticklabels([])
            yticklabels([])
            colorbar
            axis image
    
            if office_mode
                % Plot secret pattern (office mode only)
                subplot(2,2,2)
                imagesc(angle(exp(1i .* (pi/128) .* secret_pattern_gv)))
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
            plot(p.scale_range, all_feedback, '.-')
            title('Feedback signals')
            xlabel('Tube Angle (deg)')
            ylabel('Feedback')
    
            drawnow
        end
    
        if do_save
            % Save intermediate frames
            savepath = fullfile(p.savedir, sprintf('%s_%03i_%03i.mat', p.savename, i_angle, i_scale));
            disp('Saving...')
            save(savepath, '-v7.3', 'p', 'frames_flatslm', 'frames_ao', 'i_angle', 'i_scale', 'slm_pattern_gv')
        end
    
        eta(count, p.num_scales*p.num_angles,  starttime, 'cmd', 'Scanning scale and angles', 0);
        count = count+1;
    end
end

if ~office_mode
    % Random pattern prevents bleaching
    slm.setData(p.pattern_patch_id, 255*rand(300));
    slm.update

    hSI.hMotors.motorPosition = [0 0 p.piezo_start_um + p.z_backlash_distance_um];
    pause(0.1)
    hSI.hMotors.motorPosition = [0 0 p.piezo_start_um];
end

%% Plot final
% Upscale image and find coordinates of maximum
all_feedback_meanflat = all_signal_std_corrected ./ mean(all_signal_std_flat);
all_feedback_meanflat_upscaled = imresize(all_feedback_meanflat, p.upscale_factor);
[~, imax] = max(all_feedback_meanflat_upscaled(:));
[peak_index_angle_upscaled, peak_index_scale_upscaled] = ind2sub(size(all_feedback_meanflat_upscaled), imax);
peak_index_angle = downscale(peak_index_angle_upscaled, p.upscale_factor);
peak_index_scale = downscale(peak_index_scale_upscaled, p.upscale_factor);

% Compute optimal coordinates
angle_optimal = interp1(1:p.num_angles, p.angle_range_deg, peak_index_angle);
scale_optimal = interp1(1:p.num_scales, p.scale_range, peak_index_scale);

if isnan(scale_optimal) || isnan(angle_optimal)
    warning('Could not find phase pattern. NaN phase amplitude.')
end


if do_plot_final
    fig_final = figure;
    fig_resize(800, 1);
    movegui('center')
    colormap magma

    % Plot parameter landscape
    subplot(1,2,1)
    imagesc(p.scale_range, p.angle_range_deg, all_feedback_meanflat); hold on
    plot(scale_optimal, angle_optimal, '+r'); hold off
    title(sprintf('Contrast enhancement parameter landscape\n(no photobleaching compensation)'))
    if office_mode
        title(sprintf('Simulated parameter landscape'))
    end
    xlabel('Scale')
    ylabel('Angle (deg)')
    set(gca, 'fontsize', 14)
    colorbar

    % Plot upscaled parameter landscape
    subplot(1,2,2)
    imagesc(all_feedback_meanflat_upscaled); hold on
    plot(peak_index_scale_upscaled, peak_index_angle_upscaled, '+r'); hold off
    title(sprintf('Parameter landscape upscaled\nOptimal: angle=%.3fdeg, scale=%.3f', angle_optimal, scale_optimal))
    xlabel('Index Scale')
    ylabel('Index Angle')
    set(gca, 'fontsize', 14)
    colorbar
end



%% === Save ===
if do_save
    % Save intermediate frames
    savepath = fullfile(p.savedir, sprintf('%s_angle_scan.mat', p.savename));
    disp('Saving...')
    save(savepath, '-v7.3', 'p', 'all_feedback', 'all_feedback_meanflat', 'all_feedback_meanflat_upscaled', ...
        'all_signal_std_corrected', 'all_signal_std_flat', 'angle_optimal', 'scale_optimal')
    fprintf('\nSaved optimized pattern to:\n%s\n', savepath)
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

% Contrast enhancement, corrected for background std
function contrast_enh = contrast_enhancement(corrected, uncorrected, background)
    contrast_enh = sqrt( (var(corrected, [], [1 2 3]) - var(background, [], [1 2 3])) ...
        ./ (var(uncorrected, [], [1 2 3]) - var(background, [], [1 2 3])) );
end


% Compute downscaled image coordinates
function x_down = downscale(x_up, upscale_factor)
    x_down = x_up ./ upscale_factor + 0.5*(1 - 1./upscale_factor);
end