% Compare scanimage frames with different correction patterns
% This script uses corrections found with Adaptive Optics scans (tube_adaptive_optics)
% and Ray Tracing (learn_glass_tube) to demonstrate enhanced imaging

abort_if_required(hSI, hSICtl)

% Scan settings
p.slm_rotation_deg = 3.4;                   % Can be found with a calibration
p.scale_slm_x = 0.9827;                     % Scale SLM 'canvas' by this amount in x
p.scale_slm_y = 0.9557;                     % Scale SLM 'canvas' by this amount in y
hSI.hStackManager.numSlices = 172;
hSI.hStackManager.stackZStepSize = 1;
hSI.hChannels.loggingEnable = true;                 % True = Enable save

zoom = hSICtl.hModel.hRoiManager.scanZoomFactor;    % Get zoom
zstep = hSI.hStackManager.stackZStepSize;
basefilename = sprintf('tube-500nL-zoom%i-zstep%.3fum-', zoom, zstep);
savedir = ['C:\LocalData\raylearn-data\TPM\TPM-3D-scans\' sprintf('%s_tube-500nL', date)];

% Load calibration for SLM center
calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
p.offset_center_slm = calibrationdata.sopt.offset_center_slm;

% Load system aberration correction
p.system_aberration_data = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\28-Sep-2023-system-aberration\tube_ao_739157.585029_system-aberration\tube_ao_739157.585029_system-aberration_optimal_pattern.mat");
p.system_aberration_pattern = p.system_aberration_data.slm_pattern_gv_optimal;

% Save directory
mkdir(savedir);
hSI.hScan2D.logFilePath = savedir;
fprintf('\nSaving in:\n%s\n\n', hSI.hScan2D.logFilePath)


%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% ===== AO 1 ===== %%
%% Adaptive optics top pattern
set_AO_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\05-Sep-2023-tube500nL-top\tube_ao_739134.619924_tube500nL-top\tube_ao_739134.619924_tube500nL-top_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics bottom pattern
set_AO_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\05-Sep-2023-tube500nL-bottom\tube_ao_739134.524176_tube500nL-bottom\tube_ao_739134.524176_tube500nL-bottom_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics center pattern
set_AO_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\06-Sep-2023-tube500nL-center\tube_ao_739135.438888_tube500nL-center\tube_ao_739135.438888_tube500nL-center_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'center-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics side pattern
set_AO_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\06-Sep-2023-tube500nL-side\tube_ao_739135.560600_tube500nL-side\tube_ao_739135.560600_tube500nL-side_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'side-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% ===== RT 1 ===== %%
%% Ray traced top pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-top-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced bottom pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced center pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-center-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'center-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced side pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'side-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced side2 pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side2-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'side2-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-3');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% ===== AO 2 ===== %%
%% Adaptive optics top pattern
set_AO_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\05-Sep-2023-tube500nL-top\tube_ao_739134.619924_tube500nL-top\tube_ao_739134.619924_tube500nL-top_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics bottom pattern
set_AO_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\05-Sep-2023-tube500nL-bottom\tube_ao_739134.524176_tube500nL-bottom\tube_ao_739134.524176_tube500nL-bottom_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics center pattern
set_AO_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\06-Sep-2023-tube500nL-center\tube_ao_739135.438888_tube500nL-center\tube_ao_739135.438888_tube500nL-center_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'center-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics side pattern
set_AO_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\06-Sep-2023-tube500nL-side\tube_ao_739135.560600_tube500nL-side\tube_ao_739135.560600_tube500nL-side_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'side-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-4');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% ===== RT 2 ===== %%
%% Ray traced top pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-top-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced bottom pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced center pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-center-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'center-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced side pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'side-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced side2 pattern
set_RT_pattern(slm, p, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side2-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'side2-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% No correction
slm.setRect(1, [0 0 1 1]); slm.setData(1, 0); slm.update
hSI.hScan2D.logFileStem = strcat(basefilename, 'no-correction-5');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

% Random pattern to annihilate the focus
slm.setRect(1, [0 0 1 1]); slm.setData(1, 255*rand(300)); slm.update


function set_RT_pattern(slm, p, matpath)
    patterndata = load(matpath);                        % Load pattern data
    SLM_pattern_rad = -patterndata.phase_SLM';          % Pattern in rad
    SLM_pattern_gv = SLM_pattern_rad * 255 / (2*pi);    % Pattern in gray value
    SLM_pattern_gv_rot = imrotate(SLM_pattern_gv, p.slm_rotation_deg, "bilinear", "crop"); % Rotate

    % Prepare scaled SLM pattern (slm pattern must be square!)
    Nslmpattern = size(SLM_pattern_gv_rot, 2);
    x_slmpattern = linspace(-1, 1, Nslmpattern);
    y_slmpattern = x_slmpattern';
    xq = x_slmpattern * p.scale_slm_x;
    yq = y_slmpattern * p.scale_slm_y;
    
    extrapval = 0;
    SLM_pattern_gv_rot_scaled = interp2(x_slmpattern, y_slmpattern, SLM_pattern_gv_rot, xq, yq, 'bilinear', extrapval);

    SLM_pattern_gv_rot_scaled_bg = SLM_pattern_gv_rot_scaled + p.system_aberration_pattern;

    slm.setRect(1, [p.offset_center_slm(1) p.offset_center_slm(2) 1 1]);
    slm.setData(1, SLM_pattern_gv_rot_scaled); slm.update;
end


function set_AO_pattern(slm, p, matpath)
    patterndata = load(matpath);

    SLM_pattern_gv_rot_scaled_bg = patterndata.slm_pattern_gv_optimal + p.system_aberration_pattern;
    
    slm.setRect(1, [p.offset_center_slm(1) p.offset_center_slm(2) 1 1]);
    slm.setData(1, SLM_pattern_gv_rot_scaled_bg); slm.update
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