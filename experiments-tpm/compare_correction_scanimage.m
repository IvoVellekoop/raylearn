% Compare scanimage frames with different correction patterns
% This script uses corrections found with Adaptive Optics scans (tube_adaptive_optics)
% and Ray Tracing (learn_glass_tube) to demonstrate enhanced imaging

abort_if_required(hSI, hSICtl)

% Scan settings
slm_rotation_deg = 2.7;
hSI.hStackManager.numSlices = 172;
hSI.hStackManager.stackZStepSize = 1;
hSI.hChannels.loggingEnable = true;                 % True = Enable save

zoom = hSICtl.hModel.hRoiManager.scanZoomFactor;    % Get zoom
zstep = hSI.hStackManager.stackZStepSize;
basefilename = sprintf('tube-500nL-zoom%i-zstep%.3fum-', zoom, zstep);
savedir = ['C:\LocalData\raylearn-data\TPM\TPM-3D-scans\' sprintf('%s_tube-500nL', date)];

calibrationdata = load("\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\calibration\calibration_matrix_parabola\calibration_values.mat");
offset_center_slm = calibrationdata.sopt.offset_center_slm;

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
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\05-Sep-2023-tube500nL-top\tube_ao_739134.619924_tube500nL-top\tube_ao_739134.619924_tube500nL-top_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics bottom pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\05-Sep-2023-tube500nL-bottom\tube_ao_739134.524176_tube500nL-bottom\tube_ao_739134.524176_tube500nL-bottom_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics center pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\06-Sep-2023-tube500nL-center\tube_ao_739135.438888_tube500nL-center\tube_ao_739135.438888_tube500nL-center_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'center-AO-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics side pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\06-Sep-2023-tube500nL-side\tube_ao_739135.560600_tube500nL-side\tube_ao_739135.560600_tube500nL-side_optimal_pattern.mat")
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
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-top-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced bottom pattern
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced center pattern
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-center-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'center-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced side pattern
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'side-RT-1');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced side2 pattern
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side2-λ808.0nm.mat")
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
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\05-Sep-2023-tube500nL-top\tube_ao_739134.619924_tube500nL-top\tube_ao_739134.619924_tube500nL-top_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics bottom pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\05-Sep-2023-tube500nL-bottom\tube_ao_739134.524176_tube500nL-bottom\tube_ao_739134.524176_tube500nL-bottom_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics center pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\06-Sep-2023-tube500nL-center\tube_ao_739135.438888_tube500nL-center\tube_ao_739135.438888_tube500nL-center_optimal_pattern.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'center-AO-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Adaptive optics side pattern
set_AO_pattern(slm, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\TPM\adaptive-optics\06-Sep-2023-tube500nL-side\tube_ao_739135.560600_tube500nL-side\tube_ao_739135.560600_tube500nL-side_optimal_pattern.mat")
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
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-top-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'top-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced bottom pattern
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-bottom-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'bottom-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced center pattern
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-center-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'center-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced side pattern
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side-λ808.0nm.mat")
hSI.hScan2D.logFileStem = strcat(basefilename, 'side-RT-2');
fprintf('Start measurement %s\n', hSI.hScan2D.logFileStem)
grabSIFrame(hSI, hSICtl);

%% Ray traced side2 pattern
set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, "\\ad.utwente.nl\TNW\BMPI\Data\Daniel Cox\ExperimentalData\raylearn-data\pattern-0.5uL-tube-side2-λ808.0nm.mat")
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


function set_RT_pattern(slm, slm_rotation_deg, offset_center_slm, matpath)
    patterndata = load(matpath);                        % Load pattern data
    SLM_pattern_rad = patterndata.phase_SLM;            % Pattern in rad
    SLM_pattern_gv = SLM_pattern_rad * 255 / (2*pi);    % Pattern in gray value
    SLM_pattern_gv_rot = imrotate(SLM_pattern_gv, slm_rotation_deg, "bilinear", "crop"); % Rotate
    slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);
    slm.setData(1, SLM_pattern_gv_rot); slm.update;
end


function set_AO_pattern(slm, offset_center_slm, matpath)
    patterndata = load(matpath);
    slm.setRect(1, [offset_center_slm(1) offset_center_slm(2) 1 1]);
    slm.setData(1, patterndata.slm_pattern_gv_optimal); slm.update
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