% Test rotation influence on tube correction

%% === Settings ===
% Flags
office_mode = 0;
do_save = 1;
do_plot_final = 1;
do_plot_scan = 0;
force_reset = 0;

% Other settings
p.samplename = 'tube500nL-bottom-anglescan';
p.sampleid = 'DCOX-2023-8-C';

p.truncate = false;                         % Truncate Zernike modes at circle edge
p.pattern_patch_id = 1;
p.feedback_func = @contrast_enhancement;

% Image acquisition
p.zstep_um = 2;                             % Size of a z-step for the volume acquisition
p.num_zslices = 5;                          % Number of z-slices per volume acquisition
p.z_backlash_distance_um = -10;             % Backlash distance piezo (must be negative!)
assert(p.z_backlash_distance_um < 0, 'Z backlash distance must be negative.')

% Define test range astigmatism
p.min_angle_deg = -12;                      % Min tube angle in degree
p.max_angle_deg =  12;                      % Max tube angle in degree
p.num_angles = 41;                          % Number of angles to test
p.angle_range = linspace(p.min_angle_deg, p.max_angle_deg, p.num_angles);
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

% Fetch SLM pattern
patterndata = load(replace("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat", '\', filesep));
p.SLM_pattern_base = angle(patterndata.field_SLM) * 255 / (2*pi);

% Initialize feedback
all_feedback = zeros(1, p.num_angles);

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

%% === Scan Zernike modes ===
starttime = now;
count = 1;
for i_angle = 1:p.num_angles              % Scan mode 1
    test_angle = p.angle_range(i_angle);

    slm_pattern_gv = imrotate(p.SLM_pattern_base, test_angle, "bilinear", "crop");

    if office_mode
        secret_pattern = p.SLM_pattern_base;
        fake_wavefront = exp(1i .* (pi/128) .* (secret_pattern - slm_pattern_gv)) ...
            .* ((coord_x.^2) + (coord_y.^2) < 1);
        fake_wavefront_flatslm = exp(1i .* (secret_pattern) .* (pi/128)) ...
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
        for iz = 1:p.num_zslices
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

    % Compute feedback, save signal variances
    feedback = p.feedback_func(frames_ao, frames_flatslm, frames_dark);
    all_feedback(i_angle) = feedback;
    all_signal_std_flat(i_angle) = sqrt( var(frames_flatslm, [], [1 2 3]) - var(frames_dark, [], [1 2 3]) );
    all_signal_std_corrected(i_angle) = sqrt( var(frames_ao, [], [1 2 3]) - var(frames_dark, [], [1 2 3]) );


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
            imagesc(angle(exp(1i .* (pi/128) .* secret_pattern)))
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
        plot(p.angle_range, all_feedback, '.-')
        title('Feedback signals')
        xlabel('Tube Angle (deg)')
        ylabel('Feedback')

        drawnow
    end

    if do_save
        % Save intermediate frames
        savepath = fullfile(p.savedir, sprintf('%s_%03i.mat', p.savename, i_angle));
        disp('Saving...')
        save(savepath, '-v7.3', 'p', 'frames_flatslm', 'frames_ao', 'i_angle', 'slm_pattern_gv')
    end

    eta(count, p.num_angles,  starttime, 'cmd', 'Scanning angle', 0);
    count = count+1;
end

if ~office_mode
    % Random pattern prevents bleaching
    slm.setData(p.pattern_patch_id, 255*rand(300));
    slm.update
end

%%
if do_plot_final
    p.angle_range_interp = linspace(p.angle_range(1), p.angle_range(end), p.num_angles * p.upscale_factor);
    all_feedback_interp = interp1(p.angle_range, all_feedback, p.angle_range_interp, 'spline');
    [max_feedback, i_max_feedback] = max(all_feedback_interp);

    figure;
    plot(p.angle_range_interp, all_feedback_interp, '-k'); hold on
    plot(p.angle_range, all_feedback, '.b');
    plot(p.angle_range_interp(i_max_feedback), max_feedback, '+r'); hold off
    title(sprintf('Feedback signals\nmax at: %.2f deg', p.angle_range_interp(i_max_feedback)))
    xlabel('Tube Angle (deg)')
    ylabel('Contrast enhancement')

    all_signal_std_corrected_interp = interp1(p.angle_range, all_signal_std_corrected, p.angle_range_interp, 'spline');
    [max_signal_std, i_max_signal_std] = max(all_signal_std_corrected_interp);

    figure;
    plot(p.angle_range_interp, all_signal_std_corrected_interp, '-k'); hold on
    plot(p.angle_range, all_signal_std_corrected, '.b');
    plot(p.angle_range, all_signal_std_flat, '.-g');
    plot(p.angle_range_interp(i_max_signal_std), max_signal_std, '+r'); hold off
    xlabel('Tube Angle (deg)')
    ylabel('Signal \sigma')
    legend({'corrected std', 'uncorrected std'}, 'Location', 'best')
    title(sprintf('Signal std\nmax at: %.2f deg', p.angle_range_interp(i_max_signal_std)))
end


%% === Save ===
if do_save
    % Save intermediate frames
    savepath = fullfile(p.savedir, sprintf('%s_angle_scan.mat', p.savename));
    disp('Saving...')
    save(savepath, '-v7.3', 'p', 'all_feedback', 'all_feedback_interp', ...
        'all_signal_std_corrected', 'all_signal_std_flat', 'max_feedback', ...
        'i_max_feedback', 'max_signal_std', 'i_max_signal_std')
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
